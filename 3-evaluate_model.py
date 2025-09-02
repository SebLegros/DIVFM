import logging

import mlflow
import numpy as np
import torch

from config import Config
from utils.evaluationFunctions import compute_iv_statistics, plot_obs_per_date, \
    get_daily_betas_IV_pred_and_RMSE, attach_iv_and_prices, compute_split_daily_rmse_arpe, \
    make_daily_information_dataframe, plot_betas_dynamics_from_df, plot_feedforward_rmse_dynamics, plot_model_factors, \
    plot_rnd_moneyness_surfaces, compute_vix_from_rnd, plot_computed_vix_timeseries

from utils.models import build_model, SplitDeepFactorNN
from utils.optionsDataset import make_datasets, make_loaders
from utils.options_processing import add_earning_dates, compute_time_to_earnings, \
    add_scaled_moneyness_and_filter_ttm, setup_logging, load_options_data, keep_OTM
from utils.trainingFunctions import train_deep_factor_model, get_device, mlflow_log_run_start, loss_fn

# -----------------------------
# Logger setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Main script
# -----------------------------

cfg = Config()
setup_logging(cfg.log_level)
device = get_device()
logger.info("Using device: %s", device)

# Load + feature engineering
logger.info("Loading options data...")
options_data = load_options_data(cfg.data_dir, cfg.stock)
options_data = keep_OTM(options_data, options_data.loc[:, 'forward_price'])

logger.info("Adding earnings dates...")
options_data, earning_dates = add_earning_dates(options_data, str(cfg.data_dir), cfg.stock)

logger.info("Scaling moneyness & filtering by TTM…")
options_data = add_scaled_moneyness_and_filter_ttm(df=options_data, min_ttm_days=cfg.min_ttm_days)

logger.info("Computing time to earnings...")
options_data = compute_time_to_earnings(options_data, earning_dates)

logger.info("Computing descriptive statistics...")
descriptive_stats = compute_iv_statistics(options_data, 'ttm_scaled_moneyness', 'ttm')

# Datasets / loaders
logger.info("Splitting dataset...")
train_ds, valid_ds, test_ds = make_datasets(options_data, cfg)

logger.info("Loading Model...")
model: SplitDeepFactorNN = mlflow.pytorch.load_model(f"runs:/{cfg.model_run_id}/model")
model.eval()
##########
# factors_l = []
# for x, _, _ in train_ds:
#     factors_l.append(model(x))
#
# factors_l = torch.cat(factors_l, dim=0)
# mu = torch.mean(factors_l, dim=0)
# sigma = torch.std(factors_l, dim=0)
#
# model.L[:, 1:] = torch.cat([-mu[1:].unsqueeze(0), torch.eye(4)], dim=0) / sigma[1:].unsqueeze(0)
##########

client = mlflow.tracking.MlflowClient()


logger.info("Saving Number of observation plot...")
fig = plot_obs_per_date(train_ds.data,
                      valid_ds.data,
                      test_ds.data,
                      figsize=(10, 2),
                      linewidth=0.5)
client.log_figure(run_id=cfg.model_run_id, figure=fig, artifact_file='numberOptions.png')

logger.info("Getting best betas per day...")
feedforward_train_betas_from_ols, train_RMSE_feedforward, train_IV_pred_feedforward, train_dates = get_daily_betas_IV_pred_and_RMSE(model, train_ds)
feedforward_valid_betas_from_ols, valid_RMSE_feedforward, valid_IV_pred_feedforward, valid_dates = get_daily_betas_IV_pred_and_RMSE(model, valid_ds)
feedforward_test_betas_from_ols, test_RMSE_feedforward, test_IV_pred_feedforward, test_dates = get_daily_betas_IV_pred_and_RMSE(model, test_ds)

logger.info("Attaching IV and prices to dataframes...")
attach_iv_and_prices(datasets=[train_ds.data, valid_ds.data, test_ds.data],
                     iv_preds=[train_IV_pred_feedforward, valid_IV_pred_feedforward, test_IV_pred_feedforward],
                     )
logger.info("Computing RMSE and ARPE...")
daily_train_metrics, daily_valid_metrics, daily_test_metrics = compute_split_daily_rmse_arpe(
    train_ds.data, valid_ds.data, test_ds.data,
    pred_iv_col='feedforward_iv_pred',
    label_col='OM_IV'
)

logger.info("Merging daily information...")
daily_information_df = make_daily_information_dataframe(
    feedforward_train_betas_from_ols,
    feedforward_valid_betas_from_ols,
    feedforward_test_betas_from_ols,
    train_dates=train_dates,
    valid_dates=valid_dates,
    test_dates=test_dates,
    earning_dates=earning_dates,
    # Optional: pass your daily metrics if you want the final merge
    daily_train_metrics=daily_train_metrics,
    daily_valid_metrics=daily_valid_metrics,
    daily_test_metrics=daily_test_metrics,
)
logger.info("Creating and saving timeseries plots...")
fig = plot_betas_dynamics_from_df(daily_information_df)
client.log_figure(run_id=cfg.model_run_id, figure=fig, artifact_file='betas_timeseries/betas_earnings_all_dates.png')

fig = plot_feedforward_rmse_dynamics(daily_information_df)
client.log_figure(run_id=cfg.model_run_id, figure=fig, artifact_file='performance_timeseries/RMSE_timeseries.png')

# figs = plot_factor_surfaces_dynamic(feedforward=model)
logger.info("Creating and saving factors plots...")
view_angles_list = [[(20, 40), (10, 320), (20, 210), (20, 135), (20, 10)],
                    [(20, 40), (20, 40), (20, 40), (20, 40), (20, 40)],
                    [(20, 40), (20, 210), (20, 210), (20, 210), (20, 210)],
                    [(20, 40), (20, 40), (20, 40), (20, 40), (20, 230)]]
figures = plot_model_factors(model, shared_input_shape=cfg.shared_input_shape, ttm_input_shape=cfg.input_ttm, view_angles_list=view_angles_list)
for idx, fig in enumerate(figures):
    client.log_figure(run_id=cfg.model_run_id, figure=fig, artifact_file=f"factors/factors_{idx}.png")


logger.info("Creating and saving risk neutral density plots...")
chosen_earnings_date = daily_information_df.loc[daily_information_df.is_earnings * daily_information_df.train, :].index[-2]
idx_date = np.where(daily_information_df.index == chosen_earnings_date)[0]
chosen_dates = daily_information_df.iloc[idx_date[0]-2:idx_date[0]+2, :].index

fig = plot_rnd_moneyness_surfaces(daily_information_df, train_ds, chosen_dates, model)
client.log_figure(run_id=cfg.model_run_id, figure=fig, artifact_file=f"risk_neutral_density.png")

logger.info("Creating and saving VIX figure...")
daily_information_df = compute_vix_from_rnd(
    options_data,
    daily_information_df,
    model)

fig = plot_computed_vix_timeseries(
    daily_information_df,
    cfg.stock,)

client.log_figure(run_id=cfg.model_run_id, figure=fig, artifact_file=f"model_VIX.png")
print('ok')

# client.log_figure(run_id=cfg.model_run_id, figure=fig, artifact_file=f"RMSE_timeseries.png")
# print('ok')


# plot_metrics_dynamics(axs, feedforward_valid_betas_from_ols[mask[len(train_dates):(len(train_dates)+len(valid_dates))], :],
#                       valid_dates[mask[len(train_dates):(len(train_dates)+len(valid_dates))]],
#                       label='Validation set', color='thistle')
# plot_metrics_dynamics(axs, feedforward_test_betas_from_ols[mask[-len(test_dates):], :],
#                       test_dates[mask[-len(test_dates):]],
#                       label='Test set', color='deepskyblue')