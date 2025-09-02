import copy
from dataclasses import asdict
from typing import Callable, Optional, Dict, Any

import mlflow
import numpy as np
import pandas as pd
import torch.optim.lr_scheduler
from torch.func import jacrev, vmap
from torchinfo import summary

from config import Config
import logging
from utils.modelInferenceFunctions import get_optimal_betas_from_factors_IV_numObsPerGroup, \
    predict_IV_from_betas_factors_numObsPerGroup
from utils.models import SplitDeepFactorNN


logger = logging.getLogger(__name__)
def train_deep_factor_model(model: SplitDeepFactorNN, train_loader: torch.utils.data.DataLoader,
                            valid_loader: torch.utils.data.DataLoader,
                            optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
                            epochs: int, loss_fn: Callable, verbose: bool = True,
                            use_mlflow: bool = True,
                            clipping=None):
    best_model = copy.copy(model)
    best_loss = None
    model.train()
    factors_l = []
    for epoch in range(epochs):
        n = 0
        last_epoch = epoch == epochs-1
        train_RMSE_sum = 0
        for features, labels, num_obs_per_group, groups in train_loader:
            n += 1
            factors = model(features)

            optimal_betas = get_optimal_betas_from_factors_IV_numObsPerGroup(factors, labels, num_obs_per_group, )

            IV_pred = predict_IV_from_betas_factors_numObsPerGroup(optimal_betas, factors, num_obs_per_group)
            loss = loss_fn(IV_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            if clipping is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipping)
            optimizer.step()

            train_RMSE_sum += loss.detach().item()

            if last_epoch:
                factors_l.append(factors.detach().cpu())

        train_epoch_RMSE = np.sqrt(train_RMSE_sum / n)

        scheduler.step()

        if valid_loader is not None:
            with torch.no_grad():
                model.eval()
                n = 0
                valid_total_loss = 0
                for features, labels, num_obs_per_group, groups in valid_loader:
                    n += 1
                    factors = model(features)
                    optimal_betas = get_optimal_betas_from_factors_IV_numObsPerGroup(factors, labels, num_obs_per_group)

                    IV_pred = predict_IV_from_betas_factors_numObsPerGroup(optimal_betas, factors, num_obs_per_group)
                    loss = loss_fn(IV_pred, labels)

                    valid_total_loss += loss.detach().item()

                valid_epoch_RMSE = np.sqrt(valid_total_loss / n)
        else:
            valid_epoch_RMSE = None

        if verbose: print(
            f"epoch: {epoch}, train RMSE: {train_epoch_RMSE: .5f}, valid RMSE: {valid_epoch_RMSE: .5f}")
        if use_mlflow:
            mlflow.log_metric('train_RMSE', train_epoch_RMSE, step=epoch)
            mlflow.log_metric('valid_RMSE', valid_epoch_RMSE, step=epoch)
            mlflow.log_metric('lr', optimizer.param_groups[0]['lr'], step=epoch)

        if best_loss is None or (valid_epoch_RMSE < best_loss):
            best_loss = valid_epoch_RMSE
            best_model = copy.copy(model)

    if use_mlflow:
        # Makes sure every factor has std of 1 and mean of 0. If the last batch norm layer did not have the time to converge.
        # It changes nothing in the predictions, only in visualisation
        factors_l = torch.concat(factors_l, dim=0)
        mu = torch.mean(factors_l, dim=0)
        sigma = torch.std(factors_l, dim=0)

        model.L[:, 1:] = torch.cat([-mu[1:].unsqueeze(0), torch.eye(4)], dim=0) / sigma[1:].unsqueeze(0)
        mlflow.pytorch.log_model(best_model.to('cpu'), "model")




    return best_model


def loss_fn(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y) ** 2)

def mlflow_log_run_start(cfg: Config, model: torch.nn.Module, descriptive_stats: pd.DataFrame) -> str:
    # Start or resume a run
    mlflow.set_experiment(f"{cfg.stock} Factor Model")
    mlflow.start_run()
    run_id = mlflow.active_run().info.run_id

    # Log parameters (guard for None/classes)
    params: Dict[str, Any] = asdict(cfg)
    params["activation"] = cfg.activation.__name__ if hasattr(cfg.activation, "__name__") else str(cfg.activation)
    # Convert enums/paths/tuples
    params["data_dir"] = str(cfg.data_dir)
    params["features"] = ",".join(cfg.features)
    params["labels"] = ",".join(cfg.labels)
    mlflow.log_params(params)

    # Log descriptive stats
    try:
        mlflow.log_table(data=descriptive_stats.reset_index(), artifact_file="descriptive_statistics.json")
    except Exception:
        # Older MLflow: fallback to JSON artifact
        mlflow.log_text(descriptive_stats.reset_index().to_json(orient="records"), artifact_file="descriptive_statistics.json")

    # Log model summary safely
    try:
        s = summary(model, verbose=0)
        mlflow.log_text(str(s), "model_summary.txt")
    except Exception as e:
        logger.warning("torchinfo.summary failed: %s", e)

    return run_id


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")