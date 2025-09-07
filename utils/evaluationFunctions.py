from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from typing import Optional, List, Sequence, Mapping, Dict, Union, Callable, Iterable

from matplotlib.figure import Figure
from torch.distributions import Normal


def compute_iv_statistics(df, moneyness_column: str = 'ttm_scaled_moneyness', ttm_column: str = 'ttm'):

    # Convert date column to datetime if not already
    df["date"] = pd.to_datetime(df["date"])

    # Define Moneyness Bins
    moneyness_bins = [-float("inf"), -0.2, 0, 0.2, 0.8, float("inf")]
    moneyness_labels = ["M ≤ -0.2", "-0.2 < M ≤ 0", "0 < M ≤ 0.2", "0.2 < M ≤ 0.8", "M ≥ 0.8"]
    df["moneyness_bin"] = pd.cut(df[moneyness_column], bins=moneyness_bins, labels=moneyness_labels)

    # Define Time to Maturity Bins (adjusted to match τ format)
    ttm_bins = [0, 30 / 365, 90 / 365, 180 / 365, 365 / 365, float("inf")]
    ttm_labels = ["τ ≤ 30", "30 < τ ≤ 90", "90 < τ ≤ 180", "180 < τ ≤ 365", "τ ≥ 365"]
    df["ttm_bin"] = pd.cut(df[ttm_column], bins=ttm_bins, labels=ttm_labels)

    # Function to compute statistics safely
    def descriptive_stats(grouped_df):
        if grouped_df.empty:
            return pd.DataFrame(
                columns=["Average IV (%)", "Standard deviation IV (%)", "Number of contracts", "Avg contracts per day"])

        total_contracts = grouped_df["OM_IV"].count()
        unique_days = grouped_df["date"].nunique()  # Number of unique days in this group
        avg_contracts_per_day = total_contracts / unique_days if unique_days > 0 else 0  # Avoid division by zero

        return pd.DataFrame({
            "Average IV (%)": [grouped_df["OM_IV"].mean() * 100],
            "Standard deviation IV (%)": [grouped_df["OM_IV"].std() * 100],
            "Number of contracts": [total_contracts],
            "Avg contracts per day": [avg_contracts_per_day]
        })

    # Compute Statistics for Moneyness Bins
    moneyness_stats = df.groupby("moneyness_bin").apply(descriptive_stats).reset_index(level=1, drop=True)

    # Compute Statistics for Time to Maturity Bins
    ttm_stats = df.groupby("ttm_bin").apply(descriptive_stats).reset_index(level=1, drop=True)

    # Compute Total Row (aggregating across all categories)
    total_stats = descriptive_stats(df)
    total_stats.index = ["Total"]  # Rename row index to "Total"

    # Combine both tables with the Total row at the bottom
    final_table = pd.concat([moneyness_stats, ttm_stats, total_stats])

    return final_table


def plot_obs_per_date(train_data: pd.DataFrame,
                      valid_data: pd.DataFrame,
                      test_data: pd.DataFrame,
                      figsize=(10, 2),
                      linewidth=0.5):
    """
    Plot the number of observed contracts per day for train, validation, and test sets.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Training set
    obs_per_date = train_data.groupby('date').size()
    ax.plot(obs_per_date.index, obs_per_date.values,
            linestyle='solid', color='darkblue',
            label='Training set', linewidth=linewidth)

    # Validation set
    obs_per_date = valid_data.groupby('date').size()
    ax.plot(obs_per_date.index, obs_per_date.values,
            linestyle='solid', color='thistle',
            label='Validation set', linewidth=linewidth)

    # Test set
    obs_per_date = test_data.groupby('date').size()
    ax.plot(obs_per_date.index, obs_per_date.values,
            linestyle='solid', color='deepskyblue',
            label='Test set', linewidth=linewidth)

    # Labels and formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of contracts')
    plt.legend(fontsize='small', loc='upper left')
    plt.tight_layout()

    return fig


def compute_betas_using_ols_from_factors_and_IV(factors: torch.Tensor, IV: torch.Tensor):
    return torch.linalg.lstsq(factors, IV).solution


def get_optimal_betas_from_factors_IV(factors: torch.Tensor, IV: torch.Tensor):
    x = torch.cat([factors, IV], dim=-1)
    num_obs_per_group = torch.tensor([factors.shape[0]])
    sub_tensors = torch.split(x, split_size_or_sections=list(num_obs_per_group.numpy()), dim=0)

    stacked = torch.stack(
        [compute_betas_using_ols_from_factors_and_IV(tensor[..., :-1], tensor[..., -1]) for tensor in sub_tensors],
        dim=0)

    return stacked


def get_daily_betas_IV_pred_and_RMSE(model: torch.nn.Module, dataset: torch.utils.data.Dataset):
    with torch.no_grad():
        model.eval()
        betas = []
        RMSE = []
        IV_pred = []
        groups = []

        for inputs, IV, group in dataset:
            factors = model(inputs)

            betas.append(get_optimal_betas_from_factors_IV(factors, IV))
            IV_pred.append(factors @ betas[-1].T)
            RMSE.append(((IV_pred[-1] - IV) ** 2).mean().sqrt().item())
            groups.append(group)

        betas = torch.cat(betas, dim=0)
        IV_pred = torch.cat(IV_pred, dim=0)

        return betas, RMSE, IV_pred, groups


def d1_forward(S, K, r, F, T, sigma):
    '''Calculate d1 from the Black, Merton and Scholes formula'''
    return (torch.log(F / K) + 0.5 * (sigma ** 2) * T) / (sigma * torch.sqrt(T))


def get_option_price_from_forward(S, K, r, F, T, sigma, is_call, device='cpu'):
    '''Return Black, Merton, Scholes price of the European option'''

    norm = torch.distributions.Normal(0, 1)

    _d1 = d1_forward(S, K, r, F, T, sigma)
    _d2 = _d1 - sigma * torch.sqrt(T)

    # d_sign: Sign of the the option's delta
    d_sign = (is_call - 0.5) * 2
    premium = torch.full_like(_d1, np.nan, device=device)  # Initialize with zeros
    mask = torch.isnan(_d1) == False
    delta = d_sign * norm.cdf(d_sign * _d1[mask])
    premium[mask] = torch.exp(-r[mask] * T[mask]) * (F[mask] * delta - d_sign * K[mask] * norm.cdf(d_sign * _d2[mask]))
    return premium


def attach_iv_and_prices(
        datasets: Sequence[pd.DataFrame],
        iv_preds: Sequence[torch.Tensor],
        *,
        inplace: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        iv_pred_col: str = "feedforward_iv_pred",
        price_pred_col: str = "feedforward_price_pred",
        price_obs_col: str = "OM_price_pred",
) -> List[pd.DataFrame]:

    if len(datasets) != len(iv_preds):
        raise ValueError("datasets and iv_preds must have the same length.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dfs: List[pd.DataFrame] = []

    for df, iv_pred in zip(datasets, iv_preds):
        # Choose working frame
        work_df = df if inplace else df.copy()

        # Normalize iv_pred to 1D numpy
        iv_pred = iv_pred.detach().cpu().squeeze()
        if iv_pred.ndim != 1:
            raise ValueError(f"iv_pred must be 1D after squeeze; got shape {tuple(iv_pred.shape)}.")
        if len(iv_pred) != len(work_df):
            raise ValueError("Length of iv_pred does not match number of rows in dataset.")
        work_df.loc[:, iv_pred_col] = iv_pred.numpy()

        # Convert necessary columns once
        def to_tensor(series_name: str) -> torch.Tensor:
            return torch.as_tensor(work_df[series_name].values, dtype=dtype, device=device)

        strike = to_tensor("strike_price")
        rate = to_tensor("rate")
        fwd = to_tensor("forward_price")
        ttm = to_tensor("ttm")
        iv_hat = torch.as_tensor(work_df[iv_pred_col].values, dtype=dtype, device=device)
        iv_obs = to_tensor("OM_IV")
        # Ensure is_call is {0,1}
        is_call = torch.as_tensor((work_df["is_call"].astype(int)).values, dtype=dtype, device=device)

        with torch.no_grad():
            # Predicted price using predicted IV
            price_pred = get_option_price_from_forward(
                None, strike, rate, fwd, ttm, iv_hat, is_call
            ).detach().cpu().numpy()

            # “Observed” price using OM_IV
            price_obs = get_option_price_from_forward(
                None, strike, rate, fwd, ttm, iv_obs, is_call
            ).detach().cpu().numpy()

        work_df.loc[:, price_pred_col] = price_pred
        work_df.loc[:, price_obs_col] = price_obs
        work_df.dropna(subset=['rate'], axis=0, inplace=True)

        out_dfs.append(work_df)

    return out_dfs


def make_daily_information_dataframe(
        feedforward_train_betas_from_ols: torch.Tensor,
        feedforward_valid_betas_from_ols: torch.Tensor,
        feedforward_test_betas_from_ols: torch.Tensor,
        *,
        # Optional extras (provide if you want identical behavior to your snippet)
        train_dates: Optional[np.ndarray] = None,
        valid_dates: Optional[np.ndarray] = None,
        test_dates: Optional[np.ndarray] = None,
        earning_dates: Optional[pd.Index, pd.Series] = None,  # index/array of datetimes matching the final index
        rename_cols: Optional[Mapping[int, str]] = None,  # e.g. {0:'beta_1', 1:'beta_2', ...}
        daily_train_metrics: Optional[pd.DataFrame] = None,
        daily_valid_metrics: Optional[pd.DataFrame] = None,
        daily_test_metrics: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a single betas DataFrame with train/valid/test flags, optional outlier mask,
    earnings flags, column renaming, and merged daily metrics.
    """

    def _to_df(betas: torch.Tensor, dates: Optional[np.ndarray], split: str) -> pd.DataFrame:
        arr = betas.detach().cpu().numpy()
        df = pd.DataFrame(arr)
        if dates is not None:
            df.index = pd.DatetimeIndex(dates)
        df["train"] = split == "train"
        df["valid"] = split == "valid"
        df["test"] = split == "test"
        return df

    # 1) Per-split frames
    betas_train = _to_df(feedforward_train_betas_from_ols, train_dates, "train")
    betas_valid = _to_df(feedforward_valid_betas_from_ols, valid_dates, "valid")
    betas_test = _to_df(feedforward_test_betas_from_ols, test_dates, "test")

    # 2) Combine
    betas_df = pd.concat([betas_train, betas_valid, betas_test]).sort_index()

    # 4) Earnings marker (optional)
    betas_df["is_earnings"] = False
    if earning_dates is not None:
        # Align types
        earning_idx = pd.DatetimeIndex(earning_dates)
        betas_df.loc[betas_df.index.intersection(earning_idx), "is_earnings"] = True

    # 5) Rename beta columns
    # Identify numeric beta columns (start from 0..F-1)
    n_factors = feedforward_train_betas_from_ols.shape[1]
    default_mapping = {i: f"beta_{i + 1}" for i in range(n_factors)}
    mapping = dict(default_mapping)
    if rename_cols:
        mapping.update(rename_cols)
    betas_df = betas_df.rename(columns=mapping)

    # 6) Merge daily metrics (optional)
    if any(df is not None for df in (daily_train_metrics, daily_valid_metrics, daily_test_metrics)):
        parts = [df for df in (daily_train_metrics, daily_valid_metrics, daily_test_metrics) if df is not None]
        metrics_all = pd.concat(parts).sort_index()
        betas_df = betas_df.merge(metrics_all, left_index=True, right_index=True, how="left")

    return betas_df


import numpy as np
import pandas as pd
from typing import Tuple


def compute_daily_rmse_arpe(
        df: pd.DataFrame,
        *,
        date_col: str = "date",
        label_col: str = "OM_IV",
        pred_iv_col: str = "feedforward_iv_pred",
        pred_price_col: str = "feedforward_price_pred",
        mid_price_col: str = "midPrice",
        eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Compute per-date RMSE (between predicted IV and label IV) and ARPE
    (between predicted price and mid price).

    Returns a DataFrame indexed by date with columns:
        - feedforward_RMSE
        - feedforward_ARPE

    NaNs are ignored via nanmean. Division-by-zero in ARPE is guarded by `eps`.
    """
    if not np.issubdtype(pd.Series(df[date_col]).dtype, np.datetime64):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

    def _metrics_for_group(g: pd.DataFrame) -> pd.Series:
        # RMSE on IV
        rmse = np.sqrt(np.nanmean((g[pred_iv_col] - g[label_col]) ** 2))

        # ARPE on price with safe denominator
        denom = np.maximum(np.abs(g[mid_price_col].to_numpy(dtype=float)), eps)
        arpe = np.nanmean(np.abs(g[pred_price_col] - g[mid_price_col]) / denom)

        return pd.Series(
            {"feedforward_RMSE": rmse, "feedforward_ARPE": arpe},
            dtype="float64",
        )

    return df.groupby(date_col, sort=True, dropna=False).apply(_metrics_for_group)


def compute_split_daily_rmse_arpe(
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        test_data: pd.DataFrame,
        *,
        date_col: str = "date",
        label_col: str = "OM_IV",
        pred_iv_col: str = "feedforward_iv_pred",
        pred_price_col: str = "feedforward_price_pred",
        mid_price_col: str = "midPrice",
        eps: float = 1e-12,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    daily_train = compute_daily_rmse_arpe(
        train_data, date_col=date_col, label_col=label_col,
        pred_iv_col=pred_iv_col, pred_price_col=pred_price_col,
        mid_price_col=mid_price_col, eps=eps,
    )
    daily_valid = compute_daily_rmse_arpe(
        valid_data, date_col=date_col, label_col=label_col,
        pred_iv_col=pred_iv_col, pred_price_col=pred_price_col,
        mid_price_col=mid_price_col, eps=eps,
    )
    daily_test = compute_daily_rmse_arpe(
        test_data, date_col=date_col, label_col=label_col,
        pred_iv_col=pred_iv_col, pred_price_col=pred_price_col,
        mid_price_col=mid_price_col, eps=eps,
    )
    return daily_train, daily_valid, daily_test


def plot_betas_dynamics_from_df(
        daily_information_df: pd.DataFrame,
        *,
        figsize: Tuple[int, int] = (10, 8),
        linewidth: float = 0.7,
        colors: Optional[Dict[str, str]] = None,  # {'train': ..., 'valid': ..., 'test': ...}
        earnings_marker_color: str = "red",
        earnings_marker_size: Union[int, float] = 10,
) -> plt.Figure:
    """
    Plot beta dynamics from a single DataFrame containing columns:
    ['beta_1', ..., 'beta_K', 'train', 'valid', 'test', 'is_earnings',
     'feedforward_RMSE', 'feedforward_ARPE'].
    The DataFrame index must be the date.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Identify beta columns and number of factors (nrows)
    beta_cols = [c for c in daily_information_df.columns if isinstance(c, str) and c.startswith("beta_")]
    if not beta_cols:
        raise ValueError("betas_df must contain columns starting with 'beta_' (e.g., 'beta_1').")
    num_factors = len(beta_cols)

    if colors is None:
        colors = {"train": "darkblue", "valid": "thistle", "test": "deepskyblue"}

    # Create subplots (one row per factor)
    fig, axs = plt.subplots(num_factors, 1, figsize=figsize, sharex=True)
    if num_factors == 1:
        axs = [axs]  # normalize to list for consistent indexing

    def _plot_subset(subset: pd.DataFrame, label: str, color: str) -> None:
        if subset.empty:
            return
        dates = subset.index.values
        for i, beta_col in enumerate(beta_cols):
            series = subset[beta_col].to_numpy()
            axs[i].plot(dates, series, linewidth=linewidth, color=color, label=label)
            axs[i].set_title(rf"$\beta_{i + 1}$")

    # Draw train / valid / test (if columns present)
    if "train" in daily_information_df.columns:
        _plot_subset(daily_information_df[daily_information_df["train"].astype(bool)], "Training set", colors["train"])
    if "valid" in daily_information_df.columns:
        _plot_subset(daily_information_df[daily_information_df["valid"].astype(bool)], "Validation set",
                     colors["valid"])
    if "test" in daily_information_df.columns:
        _plot_subset(daily_information_df[daily_information_df["test"].astype(bool)], "Test set", colors["test"])

    # Earnings markers (optional)
    if "is_earnings" in daily_information_df.columns:
        earnings_df = daily_information_df[daily_information_df["is_earnings"].astype(bool)]
        if not earnings_df.empty:
            for i, beta_col in enumerate(beta_cols):
                axs[i].scatter(
                    earnings_df.index,
                    earnings_df[beta_col].to_numpy(),
                    color=earnings_marker_color,
                    s=earnings_marker_size,
                    zorder=3,
                    label="Earnings date",
                )

    # Legend on first axis, x-label on figure
    axs[0].legend(ncol=2, loc="upper right", fontsize=8, handletextpad=0.3, labelspacing=0.3)
    plt.xlabel("Date")
    plt.tight_layout()
    return fig


def plot_feedforward_rmse_dynamics(
        daily_information_df: pd.DataFrame,
        *,
        figsize: tuple[int, int] = (10, 2),
) -> plt.Figure:
    """
    Plot feedforward_RMSE dynamics across train/valid/test sets
    with earnings markers.

    Parameters
    ----------
    daily_information_df : pd.DataFrame
        Must contain columns:
        ['train', 'valid', 'test', 'is_earnings', 'feedforward_RMSE'].
        The index must be the date.

    figsize : tuple[int, int]
        Size of the matplotlib figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Training
    ax.plot(
        pd.to_datetime(daily_information_df.index[daily_information_df.train]),
        daily_information_df.loc[daily_information_df.train, "feedforward_RMSE"],
        label="Training set",
        linestyle="solid",
        color="darkblue",
        alpha=1,
        linewidth=1.0,
    )

    # Validation
    ax.plot(
        pd.to_datetime(daily_information_df.index[daily_information_df.valid]),
        daily_information_df.loc[daily_information_df.valid, "feedforward_RMSE"],
        label="Validation set",
        linestyle="solid",
        color="thistle",
        alpha=1,
        linewidth=1.0,
    )

    # Test
    ax.plot(
        pd.to_datetime(daily_information_df.index[daily_information_df.test]),
        daily_information_df.loc[daily_information_df.test, "feedforward_RMSE"],
        label="Test set",
        linestyle="solid",
        color="deepskyblue",
        alpha=1,
        linewidth=1.0,
    )

    # Earnings markers
    ax.scatter(
        pd.to_datetime(daily_information_df.index[daily_information_df.is_earnings]),
        daily_information_df.loc[daily_information_df.is_earnings, "feedforward_RMSE"],
        color="red",
        zorder=3,
        label="Earnings date",
        s=10,
    )

    ax.legend(ncol=2, fontsize="small")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Date")

    plt.tight_layout()
    return fig


def plot_model_factors(
    feedforward: torch.nn.Module,
    *,
    shared_input_shape: int = 2,
    ttm_input_shape: int = 1,
    # grid sizes
    N_TTM: int = 1_000,
    N_MONEYNESS: int = 1_000,
    N_TTE: int = 500,
    # grid ranges
    ttm_min: float = 0.02,
    ttm_max: float = 1.0,
    M_min: float = -3.0,
    M_max: float = 1.0,
    tte_min: float = 0.0,
    tte_max: float = 0.25,
    # plotting
    view_angles_list: Optional[Sequence[Sequence[Tuple[float, float]]]] = None,
    cmap: str = "gray",
    figsize: Tuple[int, int] = (12, 6),
    ncols: int = 3,
) -> List[plt.Figure]:
    """
    Build input grids, construct features, evaluate the model to get factor surfaces,
    and plot them as 3D surfaces. Returns a list of figures (one per angle set).

    - Factors (except factor 2) are shown over (TTM, M), where moneyness = exp(M * sqrt(TTM)).
    - Factor 2 is shown over (TTM, TTE).


    """
    # ---- device ----
    try:
        device = next(feedforward.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # ---- 1) Grid construction ----
    ttm_1d = torch.exp(torch.linspace(np.log(ttm_min), np.log(ttm_max), N_TTM, device=device))
    M_1d = torch.linspace(M_min, M_max, N_MONEYNESS, device=device)
    tte_1d = torch.linspace(tte_min, tte_max, N_TTE, device=device)

    # 2D meshes
    TTM_M_ttm, TTM_M_M = torch.meshgrid(ttm_1d, M_1d, indexing="ij")        # (TTM, M)
    TTM_TTE_ttm, TTM_TTE_tte = torch.meshgrid(ttm_1d, tte_1d, indexing="ij")# (TTM, TTE)

    # ---- 2) Features construction ----
    # moneyness from scaled log-moneyness M: exp(M * sqrt(TTM))
    moneyness = torch.exp(TTM_M_M * torch.sqrt(TTM_M_ttm))

    # Base (TTM, M) features: [TTM, moneyness] (+ optional extra const)
    features_tm = torch.stack([TTM_M_ttm, moneyness], dim=-1)  # [..., 2]

    in_shape = 2
    if shared_input_shape == 3 or ttm_input_shape == 2:
        in_shape = 3
        extra = torch.full_like(features_tm[..., :1], 0.1)
        features_tm = torch.cat([features_tm, extra], dim=-1)  # [..., 3]

    # (TTM, TTE) path: if in_shape==3, use [TTM, 1.0, TTE]; else [TTM, TTE]
    if in_shape == 3:
        features_tte = torch.stack([TTM_TTE_ttm, torch.ones_like(TTM_TTE_ttm), TTM_TTE_tte], dim=-1)
    else:
        features_tte = torch.stack([TTM_TTE_ttm, TTM_TTE_tte], dim=-1)

    # ---- 3) Get factors from features (model eval) ----
    feedforward.eval()
    with torch.no_grad():
        f_tm = feedforward(features_tm.reshape(-1, in_shape)).reshape(*features_tm.shape[:-1], -1)   # [N_TTM, N_M, K]
        f_tte = feedforward(features_tte.reshape(-1, in_shape)).reshape(*features_tte.shape[:-1], -1)# [N_TTM, N_TTE, K]
        K = f_tm.shape[-1]  # number of outputs
        titles = [f"Factor {i+1}" for i in range(K)]

        # split per factor
        f_tm_split = torch.split(f_tm, 1, dim=-1)
        f_tte_split = torch.split(f_tte, 1, dim=-1)
        factors = list(f_tm_split)
        if K >= 2:
            factors[1] = f_tte_split[1]  # Factor 2 uses (TTM, TTE)

    # ---- 4) Plotting ----
    if view_angles_list is None:
        view_angles_list = [tuple((20.0, 40.0) for _ in range(K))]

    figures: List[plt.Figure] = []
    for views in view_angles_list:
        if len(views) != K:
            raise ValueError(f"Each angle set must have {K} pairs (elev, azim); got {len(views)}.")

        nrows = math.ceil(K / ncols)
        fig = plt.figure(figsize=figsize)

        for idx, ((elev, azim), factor) in enumerate(zip(views, factors)):
            r = idx // ncols
            c = idx % ncols
            ax = plt.subplot2grid((nrows, ncols), (r, c), projection="3d")

            if idx == 1:  # Factor 2 over (TTM, TTE)
                X = TTM_TTE_ttm.detach().cpu().numpy()
                Y = TTM_TTE_tte.detach().cpu().numpy()
                Z = factor.squeeze(-1).detach().cpu().numpy()
                ax.set_ylabel("TTEA")
                ax.set_yticks([0.0, 0.10, 0.20])
            else:         # others over (TTM, M)
                X = TTM_M_ttm.detach().cpu().numpy()
                Y = TTM_M_M.detach().cpu().numpy()
                Z = factor.squeeze(-1).detach().cpu().numpy()
                ax.set_ylabel("Moneyness")

            ax.set_xlabel("Time-to-maturity")
            ax.set_xticks([0.0, 0.5, 1.0])
            surf = ax.plot_surface(X, Y, Z, cmap=cmap)
            ax.set_title(titles[idx])
            ax.view_init(elev=elev, azim=azim)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.05)

        plt.tight_layout()
        figures.append(fig)

    return figures


from typing import Iterable
import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from typing import Iterable, Optional
import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def plot_rnd_moneyness_surfaces(
    daily_information_df: pd.DataFrame,   # index = dates, contains beta_1..beta_n
    train_ds,                             # has .data with ['date','ttm','rate','forward_price','time_to_earnings']
    dates: Iterable,                      # iterable of datelike values
    model: torch.nn.Module,               # maps [TTM, K_F, TTE] → factors; dot with betas → σ
    *,
    steps: int = 2,
    n_moneyness: int = 5000,
    figsize=(16, 4),
    elev: float = 5.0,
    azim: float = -50.0,
) -> plt.Figure:
    """
    Plot risk–neutral density (RND) surfaces vs. scaled moneyness for selected dates.

    """
    # --- ensure datetime types ---
    df_betas = daily_information_df.copy()
    if not np.issubdtype(df_betas.index.dtype, np.datetime64):
        df_betas.index = pd.to_datetime(df_betas.index)

    df_train = train_ds.data.copy()
    df_train["date"] = pd.to_datetime(df_train["date"])

    # identify beta columns
    beta_cols = [c for c in df_betas.columns if c.startswith("beta_")]
    if not beta_cols:
        raise KeyError("daily_information_df must have beta_1,... columns")

    dates = [pd.to_datetime(d) for d in dates]

    fig, axs = plt.subplots(1, len(dates), subplot_kw={"projection": "3d"}, figsize=figsize)
    if len(dates) == 1:
        axs = np.array([axs])

    normal = dist.Normal(0.0, 1.0)
    was_training = model.training
    model.eval()

    eps = 1e-8

    for date, ax in zip(dates, axs.flat):
        # --- betas ---
        if date not in df_betas.index:
            ax.text2D(0.5, 0.5, f"No betas for {date.date()}", transform=ax.transAxes,
                      ha="center", va="center")
            continue
        betas_date = torch.tensor(
            df_betas.loc[date, beta_cols].to_numpy(dtype=np.float32)
        )

        # --- market curves from train_ds.data ---
        day_mkt = df_train.loc[df_train["date"] == date,
                               ["ttm", "rate", "forward_price", "time_to_earnings"]]
        if day_mkt.empty:
            ax.text2D(0.5, 0.5, f"No market data for {date.date()}", transform=ax.transAxes,
                      ha="center", va="center")
            continue

        # time-to-earnings (assumed constant for that date)
        tte = float(day_mkt["time_to_earnings"].iloc[0])

        # unique maturities with rate/forward
        unique = day_mkt.drop_duplicates(subset="ttm").sort_values("ttm")
        ttm_rate_fwd = torch.tensor(unique[["ttm", "rate", "forward_price"]].to_numpy(),
                                    dtype=torch.float32)

        ttm_vec  = ttm_rate_fwd[::steps, 0]
        rate_vec = ttm_rate_fwd[::steps, 1]
        fwd_vec  = ttm_rate_fwd[::steps, 2]

        N_TTM = int(ttm_vec.shape[0])
        N_M   = int(n_moneyness)
        if N_TTM == 0:
            continue

        # --- grid in (TTM, scaled-moneyness) ---
        m_scaled_vec = torch.linspace(-1.0, 1.0, N_M, requires_grad=False)
        TTM, m_scaled = torch.meshgrid(ttm_vec, m_scaled_vec, indexing="ij")

        K_F = torch.exp(m_scaled * torch.sqrt(TTM.clamp_min(eps))).clone().detach().requires_grad_(True)
        forward, _ = torch.meshgrid(fwd_vec, m_scaled_vec, indexing="ij")
        rate, _    = torch.meshgrid(rate_vec, m_scaled_vec, indexing="ij")
        K = K_F * forward

        # --- model features and sigma ---
        features = torch.stack([TTM, K_F, torch.full_like(TTM, tte)], dim=-1)
        factors = model(features.reshape(-1, 3))
        if factors.shape[-1] != betas_date.shape[0]:
            raise ValueError(f"Model output dim {factors.shape[-1]} != betas dim {betas_date.shape[0]}")
        sigma = (factors @ betas_date).reshape(N_TTM, N_M).clamp_min(eps)

        # --- discounted Black call ---
        sqrt_T = torch.sqrt(TTM.clamp_min(eps))
        d1 = (-torch.log(K_F) + 0.5 * sigma**2 * TTM) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        call = torch.exp(-rate * TTM) * (forward * normal.cdf(d1) - K_F * forward * normal.cdf(d2))
        call = call.reshape(-1)

        # --- derivatives wrt K_F and RND ---
        dC_dK_F = torch.autograd.grad(call, K_F, torch.ones_like(call), create_graph=True)[0]
        d2C_dK2 = torch.autograd.grad(dC_dK_F, K_F, torch.ones_like(dC_dK_F))[0].reshape(N_TTM, N_M)
        density_K = d2C_dK2 * torch.exp(rate * TTM) / (forward**2)


        Z = torch.sqrt(TTM) * K * density_K

        # --- plot curves ---
        X = m_scaled.detach().cpu().numpy()
        Y = TTM.detach().cpu().numpy()
        Z_np = Z.detach().cpu().numpy()

        cmap = plt.get_cmap("gist_gray")
        norm_colors = plt.Normalize(0, N_TTM - 1)
        for i in range(N_TTM - 1, -1, -1):
            ax.plot(X[i], Y[i], Z_np[i], color=cmap(norm_colors(i)))

        ax.text2D(0.5, 0.9, date.strftime("%Y-%m-%d"), transform=ax.transAxes,
                  ha="center", va="top", fontsize="small")
        ax.set_xlabel("Moneyness")
        ax.set_ylabel("Time-to-maturity (years)")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)

    if was_training:
        model.train()

    return fig

from typing import Iterable, Optional, Sequence
import numpy as np
import pandas as pd
import torch


import numpy as np
import pandas as pd
import torch
from typing import Iterable, Sequence, Optional


def compute_rn_density_nogrid(
        ttm: torch.Tensor,
        K: torch.Tensor,
        forward: torch.Tensor,
        rate: torch.Tensor,
        tte: float,
        feedforward: torch.nn.Module,
        betas_date: torch.Tensor
) -> torch.Tensor:
    """
    Compute the risk-neutral density for each (ttm, K) observation without building a grid.

    """
    # ensure K/F is differentiable
    K_F = (K / forward).detach().clone().requires_grad_(True)  # shape (N,)

    # stack features and predict implied vol
    feats = torch.stack([ttm, K_F, torch.full_like(ttm, tte)], dim=1)  # (N, 3)
    sigma = (feedforward(feats) @ betas_date).squeeze()  # (N,)

    # Black formula components
    sqrt_t = torch.sqrt(ttm)
    norm = Normal(0., 1.)
    d1 = (-torch.log(K_F) + 0.5 * sigma ** 2 * ttm) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    # discounted call price ∂C/∂K_F later -> use K = K_F * forward
    call = torch.exp(-rate * ttm) * (
            forward * norm.cdf(d1) - (K_F * forward) * norm.cdf(d2)
    )  # shape (N,)

    # first and second derivatives w.r.t. K_F
    dC_dKF = torch.autograd.grad(call, K_F, torch.ones_like(call), create_graph=True)[0]
    d2C_dKF2 = torch.autograd.grad(dC_dKF, K_F, torch.ones_like(dC_dKF))[0]

    # transform to density w.r.t. K
    density = d2C_dKF2 * torch.exp(rate * ttm) / forward ** 2

    return density


def truncate_density_and_K(K, density, forward, tol=0.15):
    """
    Truncate the density and strike arrays outside the region of interest,
    keeping only the parts of the distribution that increase before a lower bound
    and decrease after an upper bound.

    """

    def is_decreasing(x):
        x_diff = x[1:] - x[:-1]
        decreasing = (np.append((x_diff <= 0) * 1, 1.)).cumprod() == 1
        return decreasing

    def is_increasing(x):
        x = x[::-1]
        x_diff = x[1:] - x[:-1]
        decreasing = (np.append((x_diff <= 0) * 1, 1.)).cumprod() == 1
        return decreasing[::-1]

    K_min = (1 - tol) * forward
    K_max = (1 + tol) * forward
    idx_min = np.argmin(np.abs(K - K_min))
    idx_max = np.argmin(np.abs(K - K_max))

    density = np.clip(density, 0., None)
    density_min = density[:idx_min]
    density_max = density[idx_max:]

    cummin_min = np.minimum.accumulate(density_min[::-1])[::-1]
    is_min_so_far_min = density_min == cummin_min
    cummin_max = np.minimum.accumulate(density_max)
    is_min_so_far_max = density_max == cummin_max

    keep_density = np.full_like(density, fill_value=True)
    keep_density[idx_max:] = is_min_so_far_max
    keep_density[:idx_min] = is_min_so_far_min

    return K[keep_density == 1], density[keep_density == 1], None

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def interpolate_forward(df: pd.DataFrame,
                        tau: float,
                        kind: str = "linear",
                        column_to_interpolate:str = 'forward_price') -> float:
    """
    Interpolate the forward price at a given time-to-maturity.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'days_to_maturity' and 'forward_price'.
    tau : float
        Target days_to_maturity at which to interpolate forward price.
    kind : str, default "linear"
        Type of interpolation. Passed to scipy.interpolate.interp1d,
        so you can use "nearest", "cubic", etc. If kind=="linear",
        uses numpy.interp for speed.

    """
    # Ensure sorted by maturity
    df_sorted = df.sort_values("days_to_maturity")
    x = df_sorted["days_to_maturity"].to_numpy()
    y = df_sorted[column_to_interpolate].to_numpy()

    if kind == "linear":
        # numpy.interp does linear interpolation & constant extrapolation at ends
        return float(np.interp(tau, x, y))
    else:
        # scipy interp1d allows other methods + extrapolation
        f = interp1d(x, y, kind=kind, fill_value="extrapolate", assume_sorted=True)
        return float(f(tau))
def compute_vix_from_rnd(
    options_data: pd.DataFrame,
    betas_df: pd.DataFrame,
    model: torch.nn.Module,
    *,
    taus: Sequence[int] = (30,),                  # target maturities in *days* for which to compute VIX
    forward_col: str = "forward_price",
    rate_col: str = "rate",
    tte_col: str = "time_to_earnings",
    days_col: str = "days_to_maturity",
    beta_cols: Optional[Sequence[str]] = None,    # e.g. ["beta_1", ..., "beta_5"]; autodetected if None
    strike_span: tuple[float, float] = (0.5, 1.5),# K range as multiples of forward: [0.5F, 1.5F]
    num_strikes: int = 200,                       # granularity for strike integration
    interpolation_kind: str = "linear",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Compute a VIX-like index from a model-implied risk-neutral density and store it in `betas_df`.
    """
    # --- Safety & setup ---
    if "date" not in options_data.columns:
        raise KeyError("`options_data` must contain a 'date' column.")
    opt = options_data.copy()
    opt["date"] = pd.to_datetime(opt["date"])

    if beta_cols is None:
        beta_cols = [c for c in betas_df.columns if c.startswith("beta_")]
        if not beta_cols:
            raise KeyError("No beta_* columns found in `betas_df` and `beta_cols` not supplied.")

    # Progress wrapper
    def _prog(iterable, **kwargs):
        if show_progress:
            try:
                from tqdm import tqdm
                return tqdm(iterable, **kwargs)
            except Exception:
                return iterable
        return iterable

    # Group market data by date for interpolation
    grouped_by_date = opt.groupby("date", sort=True)

    # Ensure VIX columns exist
    for tau in taus:
        col = f"VIX{tau}"
        if col not in betas_df.columns:
            betas_df[col] = np.nan

    # Evaluate in eval mode to stabilize BN/Dropout
    was_training = model.training
    model.eval()

    # Main loop
    for tau in taus:
        ttm_years = tau / 365.0
        for date, df_date in _prog(grouped_by_date, desc=f"Computing VIX{tau}"):
            # Skip if we don't have betas for this date
            if date not in betas_df.index:
                continue

            # Interpolate needed term-structure scalars at tau
            F_tau = interpolate_forward(df_date, tau=tau, kind=interpolation_kind, column_to_interpolate=forward_col)
            r_tau = interpolate_forward(df_date, tau=tau, kind=interpolation_kind, column_to_interpolate=rate_col)
            tte_tau = interpolate_forward(df_date, tau=tau, kind=interpolation_kind, column_to_interpolate=tte_col)
            S0      = interpolate_forward(df_date, tau=0,   kind=interpolation_kind, column_to_interpolate=forward_col)

            # Strike grid [a*F, b*F]
            K_min = strike_span[0] * F_tau
            K_max = strike_span[1] * F_tau
            # Use uniform spacing similar to your original delta_k=F/200:
            K = torch.linspace(K_min, K_max, steps=num_strikes, dtype=torch.float32)  # (N,)

            # Vectorize market inputs to match K
            forward_v = torch.full_like(K, fill_value=float(F_tau))
            rate_v    = torch.full_like(K, fill_value=float(r_tau))
            ttm_v     = torch.full_like(K, fill_value=float(ttm_years))

            # Betas for this date (ensure 1D tensor)
            betas_date = torch.tensor(
                betas_df.loc[date, beta_cols].to_numpy(dtype=np.float32)
            )  # shape (num_factors,)

            # Compute risk-neutral density at each K
            density = compute_rn_density_nogrid(
                ttm=ttm_v,
                K=K,
                forward=forward_v,
                rate=rate_v,
                tte=float(tte_tau),
                feedforward=model,
                betas_date=betas_date,
            )  # torch.Tensor (N,)
            density = density.clamp_min(0)  # ensure non-negative

            # Normalize density over K
            area = torch.trapz(density, K)
            if area <= 0:
                # If area is degenerate, skip this date
                continue
            density = density / area

            # E[K] under f(K)
            EK = torch.trapz(K * density, K)

            # VIX formula from your code:
            # VIX = 100 * sqrt( -2/T * (S0/F) * ∫ log(K / E[K]) f(K) dK )
            # guard against numerical issues
            log_term = torch.log((K / EK).clamp_min(1e-12))
            integral = torch.trapz(log_term * density, K)
            radicand = (-2.0 / ttm_years) * (S0 / F_tau) * integral
            vix = (radicand.clamp_min(0.0)).sqrt() * 100.0

            betas_df.loc[date, f"VIX{tau}"] = float(vix.item())

    # Restore model mode if needed
    if was_training:
        model.train()

    return betas_df


import matplotlib.pyplot as plt
import pandas as pd


def plot_computed_vix_timeseries(
    daily_information_df: pd.DataFrame,
    stock: str,
    tau: int = 30,
    start_date: str = "2000-01-01",
    figsize=(10, 3),
    log_scale: bool = False,
) -> plt.Figure:
    """
    Plot computed VIX vs. market VIX time series, highlighting earnings dates.


    """
    col_vix_model = f"VIX{tau}"
    cols = ["is_earnings", col_vix_model]
    if stock.upper() == "AAPL" and f"{stock}_VIX" in daily_information_df.columns:
        cols.append(f"{stock}_VIX")

    df_plot = daily_information_df.loc[daily_information_df.index >= start_date, cols].copy()
    df_plot.dropna(subset=[col_vix_model], inplace=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    # Computed model VIX
    ax.plot(
        df_plot.index,
        df_plot[col_vix_model],
        label=f"DIVFM - {stock.upper()} VIX",
        color="darkblue",
        linewidth=2.0,
    )

    # # Market VIX
    # ax.plot(
    #     df_plot.index,
    #     df_plot["IndexVIX"],
    #     label="CBOE - VIX",
    #     linewidth=1.0,
    # )

    # Single-stock VIX if available (AAPL only)
    if stock.upper() == "AAPL" and f"{stock}_VIX" in df_plot.columns:
        ax.plot(
            df_plot.index,
            df_plot[f"{stock}_VIX"],
            label="CBOE - VXAPL",
            linewidth=1.0,
            alpha=0.9,
        )

    # Mark earnings dates
    ax.scatter(
        df_plot.index[df_plot["is_earnings"]],
        df_plot.loc[df_plot["is_earnings"], col_vix_model],
        color="red",
        s=10,
        zorder=3,
        label="Earnings date",
        linewidth=1.0,
    )

    if log_scale:
        ax.set_yscale("log")

    ncols = 2 if stock.upper() == "AAPL" else 1
    ax.legend(loc="upper left", fontsize="small", ncols=ncols)

    return fig