import logging
from pathlib import Path

from utils.options_processing import setup_logging, process_options_pipeline

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logging("INFO")

    base_dir = Path("../data")
    logger.debug("Base directory resolved to %s", base_dir.resolve())

    df_final = process_options_pipeline(base_dir)

    out_path = base_dir / "transformedData/stockDataClean.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_path, index=False)

    logger.info("Saved cleaned dataset to %s", out_path.resolve())