Deep Option-Implied Volatility Surfaces with Earnings Dates
==========================================================

This repository contains the code implementation for the paper:

"Deep Factor Models for Option-Implied Volatility Surfaces with Earnings Dates"

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5283770

----------------------------------------------------------------------
⚠️ Data Disclaimer
----------------------------------------------------------------------

- The dataset included in this repository is SYNTHETIC and was generated
  only for demonstration purposes.
- The real option data used in the paper comes from WRDS OptionMetrics
  and cannot be redistributed.
- If you have access to WRDS, replace the synthetic dataset with the
  real one in the `data/` folder and rerun preprocessing.

----------------------------------------------------------------------
Repository Structure
----------------------------------------------------------------------
```
  Repository Structure
├─ config.py                # Global configuration parameters
├─ 1_process_data.py        # Preprocess raw option data
├─ 2_train_model.py         # Train neural networks (requires MLflow)
├─ 3_evaluate_model.py      # Evaluate trained models and generate plots
├─ utils/                   # Utility modules
│  ├─ logging.py            # Custom logging and experiment helpers
│  ├─ modelInferenceFunctions.py  # Functions for inference and prediction
│  ├─ models.py             # Model architectures (Feedforward, SplitDeepFactorNN, etc.)
│  ├─ options_processing.py # Option data cleaning and feature engineering
│  ├─ optionsDataset.py     # Dataset classes and DataLoader utilities
├─ data/                    # Placeholder for synthetic dataset
└─ README.txt               # This file
```
----------------------------------------------------------------------
Requirements
----------------------------------------------------------------------

- Python 3.9+
- PyTorch
- NumPy, pandas
- Matplotlib, seaborn
- SciPy
- tqdm
- **MLflow** (mandatory for training and experiment tracking)

Install dependencies with:

    pip install -r requirements.txt

----------------------------------------------------------------------
Usage
----------------------------------------------------------------------

1. Prepare data (synthetic data already included):
       python 1_process_data.py

2. Train the model (logs are tracked with MLflow):
       python 2_train_model.py

3. Evaluate the model:
       python 3_evaluate_model.py

This produces:
- Factors surface plots
- RMSE timeseries plots
- Risk-neutral density surfaces
- Computed VIX timeseries

----------------------------------------------------------------------
Notes
----------------------------------------------------------------------

- The provided synthetic dataset is for demonstration only.
- Results will not match the paper exactly unless you use WRDS OptionMetrics.
- Replace the fake dataset with WRDS data under `data/` if you have access.
- **MLflow is required** to run training and track experiments, parameters, 
  and generated figures.

----------------------------------------------------------------------
Citation
----------------------------------------------------------------------

If you use this code, please cite the paper:

@article{gauthier2025deep,
  title={Deep Implied Volatility Factor Models for Stock Options},
  author={Gauthier, Genevi{\`e}ve and Godin, Fr{\'e}d{\'e}ric and Legros, Sebastien},
  journal={Available at SSRN 5283770},
  year={2025},
  url={https://ssrn.com/abstract=5283770}
}
