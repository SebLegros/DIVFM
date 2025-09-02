from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class Config:
    # Paths & data
    data_dir: Path = Path("data")
    stock: str = "AAPL"
    min_ttm_days: Optional[int] = None

    # Features/labels
    features: tuple[str, ...] = ("ttm", "moneyness", "time_to_earnings")  # Important to keep in this order
    labels: tuple[str, ...] = ("OM_IV",)
    dtype: torch.dtype = torch.float32

    # Model
    num_factors: int = 4
    shared_input_shape: int = 2
    input_ttm: int = 2
    num_neurons: int = 64
    num_layers: int = 5
    activation: type[torch.nn.Module] = torch.nn.Sigmoid
    dropout: float = 0.0
    batch_norm: bool = True
    out_batch_norm: bool = True
    add_level_as_factor: bool = True

    # Training
    lr: float = 1e-4
    epochs: int = 200
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.1
    batch_size: int = 64
    clipping: float = 0.1

    # Dates
    begin_train_date: str = "2000-01-01"
    begin_valid_date: str = "2021-01-01"
    begin_test_date: str = "2022-01-01"

    # Infra
    use_mlflow: bool = True
    model_run_id: Optional[str] = '4b4f633cf3474aa9ab457351dbbdb0f5'  # Only useful in '3-evaluate_model.py'
    log_level: str = "INFO"
    seed: Optional[int] = 1337
    num_workers: int = 4