import logging
import torch

from config import Config
from utils.evaluationFunctions import compute_iv_statistics
from utils.models import build_model
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
if __name__ == "__main__":
    cfg = Config()
    cfg.model_run_id = None
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

    train_loader, valid_loader, test_loader = make_loaders(train_ds, valid_ds, test_ds, cfg, device)

    # Model / optimizer / scheduler
    logger.info("Building model…")
    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)

    # MLflow
    run_id = None
    if cfg.use_mlflow:
        run_id = mlflow_log_run_start(cfg, model, descriptive_stats)

    logger.info("Training…")
    model = train_deep_factor_model(
        model,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        cfg.epochs,
        loss_fn,
        verbose=True,
        use_mlflow=cfg.use_mlflow,
        clipping=cfg.clipping,
    )

    logger.info("Processing completed.")
