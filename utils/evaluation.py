from __future__ import annotations

import numpy as np
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from utils.staticModels import SplitDeepFactorNNList
import re
import math
from torch.distributions import Normal
from scipy.interpolate import interp1d
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from typing import List, Mapping, Dict, Union, Iterable, Optional, Sequence, Tuple, Any


def get_predictions_on_dataset(model, dataset):
    model.eval()
    betas_l = []
    logpred_l = []
    groups_l = []
    factors_l = []
    with torch.no_grad():
        for features, labels, group in dataset:
            factors, betas, logpred = model(features, labels, torch.tensor([features.shape[0]]))
            betas_l.append(betas)
            factors_l.append(factors)
            logpred_l.append(logpred)
            groups_l.append(group)

        logpred_l = torch.concat(logpred_l, dim=0)
        betas_l = torch.concat(betas_l, dim=0)
        factors_l = torch.concat(factors_l, dim=0)
        return factors_l.numpy(), logpred_l[:, 0].numpy(), betas_l.numpy(), np.array(groups_l)


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


def compute_daily_RMSE_and_ARPE(df):
    feedforward_RMSE = np.sqrt(np.mean((df['IVpred'] - df['OM_IV']) ** 2))
    feedforward_ARPE = np.mean(np.abs(df['price_pred'] - df['OM_price_pred']) / df['OM_price_pred'])

    return pd.Series({
        'feedforward_RMSE': feedforward_RMSE,
        'feedforward_ARPE': feedforward_ARPE,
    })


#
# 
def create_full_and_daily_datasets(model, train_dataset, valid_dataset, test_dataset, earning_dates,
                                   VIX_data: Optional[pd.DataFrame] = None):
    train_factors, train_logpred, train_betas, train_groups = get_predictions_on_dataset(model, train_dataset)
    valid_factors, valid_logpred, valid_betas, valid_groups = get_predictions_on_dataset(model, valid_dataset)
    test_factors, test_logpred, test_betas, test_groups = get_predictions_on_dataset(model, test_dataset)

    full_factors = np.concatenate([train_factors, valid_factors, test_factors], axis=0)
    full_data = pd.concat([train_dataset.data, valid_dataset.data, test_dataset.data])
    full_data.loc[:, 'logIVpred'] = np.concatenate([train_logpred, valid_logpred, test_logpred], axis=0)
    full_data.loc[:, 'IVpred'] = np.exp(np.concatenate([train_logpred, valid_logpred, test_logpred], axis=0))

    for idx in range(full_factors.shape[1]):
        full_data.loc[:, f'factor_{idx + 1}'] = full_factors[:, idx]

    full_data.loc[:, 'price_pred'] = get_option_price_from_forward(None,
                                                                   torch.tensor(full_data['strike_price'].values),
                                                                   torch.tensor(full_data['rate'].values),
                                                                   torch.tensor(full_data['forward_price'].values),
                                                                   torch.tensor(full_data['ttm'].values),
                                                                   torch.tensor(full_data['IVpred'].values),
                                                                   torch.tensor(
                                                                       full_data['is_call'].values * 1)).numpy()

    full_data.loc[:, 'OM_price_pred'] = get_option_price_from_forward(None,
                                                                      torch.tensor(full_data['strike_price'].values),
                                                                      torch.tensor(full_data['rate'].values),
                                                                      torch.tensor(full_data['forward_price'].values),
                                                                      torch.tensor(full_data['ttm'].values),
                                                                      torch.tensor(full_data['OM_IV'].values),
                                                                      torch.tensor(
                                                                          full_data['is_call'].values * 1)).numpy()
    full_data['train'] = full_data['date'].isin(train_groups)
    full_data['valid'] = full_data['date'].isin(valid_groups)
    full_data['test'] = full_data['date'].isin(test_groups)

    # Daily data
    full_groups = np.concatenate([train_groups, valid_groups, test_groups])
    columns = {'date': full_groups}
    full_betas = np.concatenate([train_betas, valid_betas, test_betas], axis=0)

    for idx in range(full_betas.shape[1]):
        beta = full_betas[:, idx]
        columns[f'beta_{idx + 1}'] = beta
    daily_data = pd.DataFrame(columns)
    daily_data['train'] = daily_data['date'].isin(train_groups)
    daily_data['valid'] = daily_data['date'].isin(valid_groups)
    daily_data['test'] = daily_data['date'].isin(test_groups)
    daily_data['is_earning_date'] = daily_data['date'].isin(earning_dates)

    daily_metrics = full_data.groupby('date').apply(compute_daily_RMSE_and_ARPE)
    daily_data = pd.merge(daily_data, daily_metrics, on='date')
    if VIX_data is not None: daily_data = pd.merge(daily_data, VIX_data, on='date', how='left')

    return full_data, daily_data


#
# 
#
# 
def plot_betas_timeseries(
        df: pd.DataFrame,
        *,
        alpha: float = 0.01,  # y-limits use [alpha, 1-alpha] percentiles per beta (NO date removal)
        n_std: float = 0.0,  # NEW: expand y-limits by n_std * std(kept-data) per beta
        date_col: str = "date",
        factor_prefix: str = "beta_",
        train_col: str = "train",
        valid_col: str = "valid",
        test_col: str = "test",
        earnings_col: str = "is_earning_date",
        colors: Optional[dict] = None,
        figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Plot beta time series by split (train/valid/test) WITHOUT removing any dates.

    For each beta axis:
      - compute lo/hi from percentiles [alpha, 1-alpha] on ALL rows for that beta
      - compute std on the "kept" data within [lo, hi]
      - set ylim to [lo - n_std*std_kept, hi + n_std*std_kept]
      - plot full time series per split (all dates in that split)
      - overlay earnings markers (red dots)

    Notes:
      - This does not filter the plotted data; it only controls y-axis scaling.
      - If alpha=0, lo/hi are min/max and "kept" is all finite data.
    """
    # ---- validation
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}' in dataframe.")
    if not {train_col, valid_col, test_col}.issubset(df.columns):
        raise ValueError(f"Dataframe must contain boolean split columns '{train_col}', '{valid_col}', '{test_col}'.")
    if earnings_col not in df.columns:
        raise ValueError(f"Missing earnings flag column '{earnings_col}' in dataframe.")
    if not (0.0 <= alpha < 0.5):
        raise ValueError(f"'alpha' must be in [0, 0.5). Got {alpha}.")
    if n_std < 0:
        raise ValueError(f"'n_std' must be >= 0. Got {n_std}.")

    factor_cols = [c for c in df.columns if c.startswith(factor_prefix)]
    if not factor_cols:
        raise ValueError(f"No columns starting with '{factor_prefix}' found.")

    def beta_idx(c: str) -> int:
        m = re.search(rf"{re.escape(factor_prefix)}(\d+)$", c)
        return int(m.group(1)) if m else 10 ** 9

    factor_cols = sorted(factor_cols, key=beta_idx)
    num_factors = len(factor_cols)

    if colors is None:
        colors = {"train": "darkblue", "valid": "thistle", "test": "deepskyblue"}

    # ---- sort by date
    df_sorted = df.copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors="coerce")
    df_sorted = df_sorted.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    dates = df_sorted[date_col].to_numpy()
    betas = df_sorted[factor_cols].to_numpy(dtype=float)

    split_masks = {
        "Training": df_sorted[train_col].to_numpy(dtype=bool),
        "Validation": df_sorted[valid_col].to_numpy(dtype=bool),
        "Test": df_sorted[test_col].to_numpy(dtype=bool),
    }
    earnings_mask = df_sorted[earnings_col].to_numpy(dtype=bool)

    # ---- percentile ylims computed ONCE per beta from ALL data, then expanded by n_std*std_kept
    ylims: list[Optional[Tuple[float, float]]] = []
    for i in range(num_factors):
        v = betas[:, i]
        v = v[np.isfinite(v)]
        if v.size == 0:
            ylims.append(None)
            continue

        if alpha == 0:
            lo, hi = float(np.min(v)), float(np.max(v))
            kept = v
        else:
            lo = float(np.nanpercentile(v, 100.0 * alpha))
            hi = float(np.nanpercentile(v, 100.0 * (1.0 - alpha)))
            kept = v[(v >= lo) & (v <= hi)]

        if kept.size == 0:
            # fallback if something weird happens
            kept = v

        std_kept = float(np.nanstd(kept))  # population std; use ddof=1 if you prefer sample std
        pad = n_std * std_kept

        lo2 = lo - pad
        hi2 = hi + pad

        if lo2 == hi2:
            eps = 1e-6
            lo2 -= eps
            hi2 += eps

        ylims.append((lo2, hi2))

    # ---- create subplots
    fig, axs = plt.subplots(nrows=num_factors, ncols=1, sharex=True, figsize=figsize)
    if num_factors == 1:
        axs = [axs]

    # ---- plot full series per split (NO filtering)
    split_color = {
        "Training": colors.get("train"),
        "Validation": colors.get("valid"),
        "Test": colors.get("test"),
    }

    for split_name in ["Training", "Validation", "Test"]:
        m = split_masks[split_name]
        if not m.any():
            continue

        for i in range(num_factors):
            axs[i].plot(
                dates[m],
                betas[m, i],
                linewidth=0.7,
                color=split_color[split_name],
                alpha=0.8,
                label=f"{split_name} set" if i == 0 else None,
            )

    # ---- apply ylims (once) and titles
    for i in range(num_factors):
        if ylims[i] is not None:
            axs[i].set_ylim(ylims[i])
        axs[i].set_title(rf"$\beta_{i + 1}$", fontsize=10)

    # ---- earnings markers
    if earnings_mask.any():
        for i in range(num_factors):
            axs[i].scatter(
                dates[earnings_mask],
                betas[earnings_mask, i],
                s=10,
                color="red",
                zorder=3,
                label="Earnings date" if i == 0 else None,
            )

    axs[0].legend(ncol=2, loc="upper right", fontsize=8, handletextpad=0.3, labelspacing=0.3)
    axs[-1].set_xlabel("Date")
    fig.tight_layout()
    return fig


#
# 
def plot_rmse_timeseries(
        df: pd.DataFrame,
        *,
        alpha: float = 0.01,  # percentile range for y-limits
        n_std: float = 0.0,  # expand y-limits by n_std * std(kept data)
        date_col: str = "date",
        rmse_col: str = "feedforward_RMSE",
        earnings_col: str = "is_earning_date",
        figsize=(10, 2),
        colors: Optional[dict] = None,
) -> plt.Figure:
    """
    Plot RMSE time series for training, validation, and test sets.

    Keeps ALL dates.
    Y-limits are computed as:
      [q_alpha - n_std*std, q_(1-alpha) + n_std*std]
    where std is computed on the kept percentile range.

    Earnings dates are overlaid as red markers.
    """
    if colors is None:
        colors = {"train": "darkblue", "valid": "thistle", "test": "deepskyblue"}

    if not (0.0 <= alpha < 0.5):
        raise ValueError(f"'alpha' must be in [0, 0.5). Got {alpha}.")
    if n_std < 0:
        raise ValueError(f"'n_std' must be >= 0. Got {n_std}.")

    # ---- extract dates
    if date_col is None:
        dates = pd.to_datetime(df.index, errors="coerce")
    else:
        dates = pd.to_datetime(df[date_col], errors="coerce")

    # ---- RMSE values (all rows)
    v_all = df[rmse_col].to_numpy(dtype=float)
    v = v_all[np.isfinite(v_all)]

    # ---- compute y-limits
    if v.size > 0:
        if alpha == 0:
            lo, hi = float(np.min(v)), float(np.max(v))
            kept = v
        else:
            lo = float(np.nanpercentile(v, 100.0 * alpha))
            hi = float(np.nanpercentile(v, 100.0 * (1.0 - alpha)))
            kept = v[(v >= lo) & (v <= hi)]

        if kept.size == 0:
            kept = v

        std_kept = float(np.nanstd(kept))
        pad = n_std * std_kept

        lo2 = lo - pad
        hi2 = hi + pad

        if lo2 == hi2:
            eps = 1e-6
            lo2 -= eps
            hi2 += eps

        ylim = (lo2, hi2)
    else:
        ylim = None

    # ---- plot
    fig, ax = plt.subplots(figsize=figsize)

    name_mapping = {"train": "Training", "valid": "Validation", "test": "Test"}

    for split_name in ["train", "valid", "test"]:
        if split_name not in df.columns:
            continue

        mask = df[split_name].to_numpy(bool)
        if not mask.any():
            continue

        ax.plot(
            dates[mask],
            df.loc[mask, rmse_col],
            label=f"{name_mapping[split_name]} set",
            linestyle="solid",
            color=colors[split_name],
            linewidth=1.0,
            alpha=1.0,
        )

    # ---- earnings markers
    if earnings_col in df.columns:
        m = df[earnings_col].to_numpy(bool)
        if m.any():
            ax.scatter(
                dates[m],
                df.loc[m, rmse_col],
                color="red",
                zorder=3,
                label="Earnings date",
                s=10,
            )

    # ---- apply y-limits
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_ylabel("RMSE")
    ax.set_xlabel("Date")
    ax.legend(ncol=2, fontsize="small")
    fig.tight_layout()

    return fig


#
# 
# import numpy as np
# import matplotlib.pyplot as plt
# 
# 
def _bin_interval_labels(bins, *, symbol="M"):
    labels = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        lo_str = "-inf" if np.isneginf(lo) else f"{lo:g}"
        hi_str = "inf" if np.isposinf(hi) else f"{hi:g}"

        # Make endpoints look like your example
        if np.isneginf(lo) and np.isposinf(hi):
            labels.append(f"-inf < {symbol} < inf")
        elif np.isneginf(lo):
            labels.append(f"-inf < {symbol} ≤ {hi_str}")
        elif np.isposinf(hi):
            labels.append(f"{lo_str} < {symbol} < inf")
        else:
            labels.append(f"{lo_str} < {symbol} ≤ {hi_str}")
    return labels


#
# 
def _get_all_df(summary):
    return summary["all"] if isinstance(summary, dict) and "all" in summary else summary


def _pick_rmse_row(df):
    rmse_rows = [idx for idx in df.index if isinstance(idx, str) and "rmse" in idx.lower()]
    if not rmse_rows:
        raise ValueError("Could not find an RMSE row in summary DataFrame.")
    return rmse_rows[0]


#
# 
def plot_rmse_by_bins(
        summary_a,
        summary_b,
        *,
        bins,
        symbol="M",
        title="RMSE by bins",
        label_a="Model A",
        label_b="Model B",
        figsize=(13, 3),
):
    """
    Plot RMSE-by-bin comparison and RETURN the matplotlib Figure.

    Works for moneyness (M) and TTM / tau.
    """
    df_a = _get_all_df(summary_a)
    df_b = _get_all_df(summary_b)

    row_a = _pick_rmse_row(df_a)
    row_b = _pick_rmse_row(df_b)

    cols = [c for c in df_a.columns if c != "All" and c in df_b.columns]

    y_a = df_a.loc[row_a, cols].astype(float).values
    y_b = df_b.loc[row_b, cols].astype(float).values

    x = np.arange(len(cols))
    tick_labels = _bin_interval_labels(bins, symbol=symbol)

    if len(tick_labels) != len(cols):
        tick_labels = cols  # fallback

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y_a, marker="o", label=label_a)
    ax.plot(x, y_b, marker="o", label=label_b)

    ax.set_title(title)
    ax.set_ylabel("RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.legend()

    fig.tight_layout()
    return fig


#
# 
def plot_rmse_by_bins_two_panel(
        *,
        # moneyness summaries
        m_summary_a,
        m_summary_b,
        m_bins,
        m_symbol="M",
        m_title="RMSE by moneyness",
        # ttm summaries
        tau_summary_a,
        tau_summary_b,
        tau_bins,
        tau_symbol="TTM",
        tau_title="RMSE by time-to-maturity",
        # legend labels
        label_a="DIVFM without TTEA",
        label_b="DIVFM with TTEA",
        # styling
        color_a="red",
        color_b="darkblue",
        figsize=(10, 4),
):
    """
    Create a 2-panel figure:
      Panel 1: RMSE by moneyness bins
      Panel 2: RMSE by TTM/tau bins

    Each panel plots BOTH models (A and B): scatter + line.
    Returns: matplotlib Figure
    """

    def _extract_xy(summary_a, summary_b, bins, symbol):
        df_a = _get_all_df(summary_a)
        df_b = _get_all_df(summary_b)

        row_a = _pick_rmse_row(df_a)
        row_b = _pick_rmse_row(df_b)

        cols = [c for c in df_a.columns if c != "All" and c in df_b.columns]

        y_a = df_a.loc[row_a, cols].astype(float).values
        y_b = df_b.loc[row_b, cols].astype(float).values

        x = np.arange(len(cols))
        tick_labels = _bin_interval_labels(bins, symbol=symbol)
        if len(tick_labels) != len(cols):
            tick_labels = list(cols)  # fallback

        return x, tick_labels, y_a, y_b

    # extract data for both panels
    m_x, m_ticks, m_y_a, m_y_b = _extract_xy(m_summary_a, m_summary_b, m_bins, m_symbol)
    t_x, t_ticks, t_y_a, t_y_b = _extract_xy(tau_summary_a, tau_summary_b, tau_bins, tau_symbol)

    # figure
    fig, axs = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=False)

    # ---- Panel 1: Moneyness ----
    axs[0].scatter(m_x, m_y_a, label=label_a, color=color_a)
    axs[0].plot(m_x, m_y_a, color=color_a)

    axs[0].scatter(m_x, m_y_b, label=label_b, color=color_b)
    axs[0].plot(m_x, m_y_b, color=color_b)

    axs[0].set_xticks(m_x)
    axs[0].set_xticklabels(m_ticks, rotation=20, ha="center")
    axs[0].set_ylabel("RMSE")
    axs[0].legend(fontsize="small")
    axs[0].set_ylim(bottom=0)
    axs[0].set_title(m_title, fontsize=12)

    # ---- Panel 2: TTM/Tau ----
    axs[1].scatter(t_x, t_y_a, label=label_a, color=color_a)
    axs[1].plot(t_x, t_y_a, color=color_a)

    axs[1].scatter(t_x, t_y_b, label=label_b, color=color_b)
    axs[1].plot(t_x, t_y_b, color=color_b)

    axs[1].set_xticks(t_x)
    axs[1].set_xticklabels(t_ticks, rotation=20, ha="center")
    axs[1].set_ylabel("RMSE")
    axs[1].legend(fontsize="small")
    axs[1].set_ylim(bottom=0)
    axs[1].set_title(tau_title, fontsize=12)

    return fig


#
# 
def render_factor_surfaces(
        model,
        VIEW_ANGLES_LIST: List[List[Tuple[float, float]]],
        *,
        n_ttm: int = 1000,
        n_moneyness: int = 1000,
        n_tte: int = 500,
        ttm_min: float = 0.02,
        ttm_max: float = 1.0,
        moneyness_range: Tuple[float, float] = (-3.0, 1.0),
        const_third_feature: float = 0.1,
        fixed_moneyness_for_tte: float = 1.0,
        max_factors_to_plot: int = 5,
        cmap: str = "gray",
) -> List[plt.Figure]:
    """
    Build two input grids (TTM×Moneyness and TTM×TTE), evaluate model factors,
    swap Factor 3 with the TTM×TTE version, and render 3D surfaces using the provided view angles.

    Returns a list of matplotlib Figures (one per row in VIEW_ANGLES_LIST).
    """

    # ---- device & eval mode
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.eval()

    # ---- axes for inputs
    ttm_axis = torch.exp(torch.linspace(math.log(ttm_min), math.log(ttm_max), n_ttm, device=device))
    mon_axis = torch.linspace(moneyness_range[0], moneyness_range[1], n_moneyness, device=device)
    tte_axis = torch.linspace(0.0, 0.25, n_tte, device=device)

    # ---- grids
    # Grid A: TTM × Moneyness (+ constant third feature)
    ttm_grid, mon_grid = torch.meshgrid(ttm_axis, mon_axis, indexing="ij")  # (n_ttm, n_moneyness)
    featA = torch.stack(
        [
            ttm_grid,  # TTM
            mon_grid,  # Moneyness
            torch.full_like(ttm_grid, const_third_feature),
        ],
        dim=-1,
    )  # (n_ttm, n_moneyness, 3)

    # Grid B: TTM × TTE (+ moneyness fixed to 1)
    ttm_tte_grid, tte_grid = torch.meshgrid(ttm_axis, tte_axis, indexing="ij")  # (n_ttm, n_tte)
    featB = torch.stack(
        [
            ttm_tte_grid,  # TTM
            torch.full_like(ttm_tte_grid, fixed_moneyness_for_tte),  # Moneyness fixed at 1
            tte_grid,  # TTE
        ],
        dim=-1,
    )  # (n_ttm, n_tte, 3)

    # ---- run model once per grid
    with torch.no_grad():
        outA = model.get_factors(featA.reshape(-1, 3)).reshape(n_ttm, n_moneyness, -1)  # (n_ttm, n_mon, F)
        outB = model.get_factors(featB.reshape(-1, 3)).reshape(n_ttm, n_tte, -1)  # (n_ttm, n_tte, F)

    num_factors = outA.shape[-1]
    n_plot = min(max_factors_to_plot, num_factors)

    # Split into a list of factor surfaces
    factors_list = [outA[..., i] for i in range(num_factors)]  # each (n_ttm, n_moneyness)
    # Replace Factor 3 (index 2) with the TTM×TTE version if available
    if num_factors >= 3:
        factors_list[2] = outB[..., 2]  # shape (n_ttm, n_tte)

    # Convert plotting axes to numpy (CPU)
    ttm_grid_np = ttm_grid.detach().cpu().numpy()
    mon_grid_np = mon_grid.detach().cpu().numpy()
    ttm_tte_np = ttm_tte_grid.detach().cpu().numpy()
    tte_np = tte_grid.detach().cpu().numpy()

    titles = [f"Factor {i + 1}" for i in range(n_plot)]
    figs: List[plt.Figure] = []

    # layout positions (we’ll use up to 5 axes)
    def make_axes(fig):
        gs = GridSpec(2, 6, figure=fig)
        slots = [
            fig.add_subplot(gs[0, 0:2], projection="3d"),
            fig.add_subplot(gs[0, 2:4], projection="3d"),
            fig.add_subplot(gs[0, 4:6], projection="3d"),
            fig.add_subplot(gs[1, 1:3], projection="3d"),
            fig.add_subplot(gs[1, 3:5], projection="3d"),
        ]
        return slots[:n_plot]

    # Render one figure per list of view angles
    for view_angles in VIEW_ANGLES_LIST:
        fig = plt.figure(figsize=(12, 6))
        axes = make_axes(fig)

        for i, (ax, title) in enumerate(zip(axes, titles)):
            # choose grid depending on factor index (3rd factor uses TTM×TTE)
            if i == 2:
                Z = factors_list[i].detach().cpu().numpy()  # (n_ttm, n_tte)
                surf = ax.plot_surface(ttm_tte_np, tte_np, Z, cmap=cmap, linewidth=0, antialiased=True)
                ax.set_xlabel("Time-to-maturity")
                ax.set_ylabel("TTEA")
                ax.set_yticks([0.0, 0.10, 0.20])
            else:
                Z = factors_list[i].detach().cpu().numpy()  # (n_ttm, n_moneyness)
                surf = ax.plot_surface(ttm_grid_np, mon_grid_np, Z, cmap=cmap, linewidth=0, antialiased=True)
                ax.set_xlabel("Time-to-maturity")
                ax.set_ylabel("Moneyness")

            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_title(title)

            # view angles for this factor (fallback to first if not enough provided)
            angles = view_angles[i] if i < len(view_angles) else view_angles[0]
            ax.view_init(elev=angles[0], azim=angles[1])

            fig.colorbar(surf, ax=ax, shrink=0.4, aspect=15, pad=0.1)

        plt.tight_layout()
        figs.append(fig)

    return figs


#
# 
def static_arbitrage_checks(
        df: pd.DataFrame,
        price_col: str,
        *,
        strike_col: str = "strike_price",
        is_call_col: str = "is_call",
        ttm_col: str = "ttm",  # time-to-maturity in YEARS
        rate_col: str = "rate",  # continuous risk-free rate
        fwd_col: str = "forward_price",  # forward price for that maturity
        group_by: Iterable[str] = ("ttm", "forward_price", "rate"),
        dedupe: str = "mean",  # how to handle duplicate strikes within a group: "mean" or "first" or "last"
) -> pd.DataFrame:
    """
    Compute monotonicity (call-spread) and convexity (butterfly) violations on call prices
    derived from a mixed call/put dataset via put–call parity for forwards.

    Steps per group (by default: (ttm, forward_price, rate)):
      1) Convert puts to calls: C = P + (F - K) * exp(-r * T)
      2) Sort by strike
      3) For consecutive strikes (K_i, K_{i+1}), compute Call Spread:
            CS_i = ( C(K_{i+1}) - C(K_i) ) / ( K_{i+1} - K_i )
         Count how many CS_i > 0  (monotonicity violations; should be <= 0 in theory)
      4) For consecutive call spreads, compute Butterfly:
            BF_i = CS_{i-1} - CS_i
         Count how many BF_i > 0  (convexity violations; should be <= 0 in theory)

    Parameters
    ----------
    df : DataFrame
        Input data with at least these columns:
        price_col, is_call_col, ttm_col, strike_col, rate_col, fwd_col.
    price_col : str
        Name of the column containing the option price (call or put).
    strike_col, is_call_col, ttm_col, rate_col, fwd_col : str
        Column names for strike, call/put flag, time-to-maturity (years),
        continuous rate, and forward price.
    group_by : Iterable[str]
        Columns that define a homogeneous set (same market inputs); spreads
        and butterflies are computed within each group.
    dedupe : {"mean","first","last"}
        How to handle duplicate strikes within a group.

    Returns
    -------
    summary : DataFrame
        One row per group with:
            - the group-by keys
            - n_strikes: number of unique strikes used
            - n_call_spreads: number of CS computed
            - n_call_spreads_pos: # of CS > 0
            - n_butterflies: number of BF computed
            - n_butterflies_pos: # of BF > 0
    """
    # --- 0) Basic validation
    required = {price_col, is_call_col, ttm_col, strike_col, rate_col, fwd_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- 1) Select only the needed columns and drop rows with missing key values
    cols = list(set(required) | set(group_by))
    work = df[cols].dropna(subset=[price_col, is_call_col, ttm_col, strike_col, rate_col, fwd_col]).copy()

    # Ensure dtypes are sane
    work[is_call_col] = work[is_call_col].astype(bool)
    for c in [price_col, ttm_col, strike_col, rate_col, fwd_col]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[price_col, ttm_col, strike_col, rate_col, fwd_col])

    # --- 2) Convert all prices to call prices using put–call parity for forwards
    # C = P + (F - K) * exp(-rT)   if row is a put
    # C = price                    if row is already a call
    P = work[price_col].to_numpy()
    F = work[fwd_col].to_numpy()
    K = work[strike_col].to_numpy()
    r = work[rate_col].to_numpy()
    T = work[ttm_col].to_numpy()
    is_call = work[is_call_col].to_numpy()

    call_price = np.where(is_call, P, P + (F - K) * np.exp(-r * T))
    work["call_price"] = call_price

    # --- 3) Group, sort by strike, (optionally) dedupe, and compute spreads
    results: list[Dict[str, Any]] = []

    # If group_by is empty, treat all rows as one group
    if not group_by:
        group_iter = [((None,), work)]
        group_names = [None]
    else:
        group_iter = work.groupby(list(group_by), sort=False)
        group_names = list(group_by)

    for keys, g in group_iter:
        # keys can be a scalar if single key, normalize to tuple for safe unpack
        if not isinstance(keys, tuple):
            keys = (keys,)

        # Sort by strike (ascending)
        g_sorted = g.sort_values(by=strike_col, ascending=True)

        # Dedupe on strike if needed
        if dedupe == "mean":
            # Average call price for duplicate strikes
            g_uni = (
                g_sorted.groupby(strike_col, as_index=False, sort=True)["call_price"]
                .mean()
                .sort_values(by=strike_col, ascending=True)
            )
        elif dedupe == "first":
            g_uni = g_sorted.drop_duplicates(subset=strike_col, keep="first")
        elif dedupe == "last":
            g_uni = g_sorted.drop_duplicates(subset=strike_col, keep="last")
        else:
            raise ValueError("dedupe must be one of {'mean','first','last'}")

        Ks = g_uni[strike_col].to_numpy()
        Cs = g_uni["call_price"].to_numpy()

        # Need at least 2 unique strikes for call spreads
        if Ks.size >= 2:
            dK = np.diff(Ks)  # K_{i+1} - K_i
            dC = np.diff(Cs)  # C(K_{i+1}) - C(K_i)
            # Avoid divide-by-zero (shouldn't happen if strikes are unique & sorted)
            with np.errstate(divide="ignore", invalid="ignore"):
                CS = np.where(dK != 0.0, dC / dK, np.nan)
            # Count CS > 0 (monotonicity violations; calls should be non-increasing in K)
            cs_pos = np.nansum(CS > 0.0)
        else:
            CS = np.array([], dtype=float)
            cs_pos = 0

        # Need at least 3 unique strikes for butterflies
        if CS.size >= 2:
            BF = CS[:-1] - CS[1:]  # consecutive differences of CS
            bf_pos = np.nansum(BF > 0.0)  # convexity violations
        else:
            BF = np.array([], dtype=float)
            bf_pos = 0

        # Build result row
        row: Dict[str, Any] = {
            "n_strikes": Ks.size,
            "n_call_spreads": CS.size,
            "n_call_spreads_pos": int(cs_pos),
            "n_butterflies": BF.size,
            "n_butterflies_pos": int(bf_pos),
        }
        # Add group keys to the row
        for name, val in zip(group_names, keys):
            row[name] = val

        results.append(row)

    # --- 4) Return tidy summary DataFrame with group columns first
    summary = pd.DataFrame(results)
    if group_by:
        # Reorder columns: group_by keys first
        summary = summary[list(group_by) + [c for c in summary.columns if c not in group_by]]

    return summary


#
# 
def static_arbitrage_by_date(
        df: pd.DataFrame,
        price_col: str,
        *,
        date_col: str = "date",
        **kwargs
) -> pd.DataFrame:
    """
    Apply static_arbitrage_checks() to each unique date and
    concatenate all results into a single DataFrame with a 'date' column.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset containing a 'date' column.
    price_col : str
        Column name of the option price.
    date_col : str
        Name of the date column.
    **kwargs :
        Additional arguments passed to static_arbitrage_checks()
        (e.g. group_by, strike_col, etc.)

    Returns
    -------
    all_results : pd.DataFrame
        Concatenated output with an added 'date' column.
    """
    # Make sure the date column exists
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' column not found in dataframe.")

    results = []
    # Group by each date
    for date_val, group in df.groupby(date_col):
        res = static_arbitrage_checks(group, price_col=price_col, **kwargs)
        res[date_col] = date_val  # keep the date label
        results.append(res)

    # Concatenate all results
    all_results = pd.concat(results, ignore_index=True)

    # Reorder columns so 'date' comes first
    cols = [date_col] + [c for c in all_results.columns if c != date_col]
    return all_results[cols]


#
# 
def aggregate_by_date(
        summary_df: pd.DataFrame,
        *,
        date_col: str = "date",
        weight_col: str = "n_strikes",
        weighted_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate the per-(date, group) summary into per-day totals.

    Inputs
    ------
    summary_df : DataFrame
        Output of `static_arbitrage_by_date(...)`, i.e. has columns like:
        ['date', 'n_strikes', 'n_call_spreads', 'n_call_spreads_pos',
         'n_butterflies', 'n_butterflies_pos', ...]
    date_col : str
        Date column name.
    weight_col : str
        Column holding weights for weighted sums (e.g., 'n_strikes' or 'n_call_spreads').
    weighted_cols : list[str] | None
        Columns for which to compute *weighted sums* per day:
          weighted_sum(col) = sum(col * weight_col) within each date.
        If None or empty, no weighted sums are produced.

    Returns
    -------
    DataFrame
        One row per date with:
          - total_weight: sum of weight_col
          - sum_call_spreads_pos: plain sum of n_call_spreads_pos
          - sum_butterflies_pos: plain sum of n_butterflies_pos
          - <col>_wsum: weighted sum for each requested column
    """

    if date_col not in summary_df.columns:
        raise ValueError(f"'{date_col}' not found in summary_df.")
    if weight_col not in summary_df.columns:
        raise ValueError(f"'{weight_col}' not found in summary_df.")

    # Columns we always sum (plain sums)
    must_have = ["n_call_spreads_pos", "n_butterflies_pos"]
    missing = [c for c in must_have if c not in summary_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in summary_df: {missing}")

    # Work copy to avoid mutating caller's df
    df = summary_df.copy()

    # Build aggregations:
    # - total_weight = sum(weight_col)
    # - plain sums for the two *_pos counts
    agg_dict = {
        weight_col: "sum",
        "n_call_spreads_pos": "sum",
        "n_butterflies_pos": "sum",
    }

    # Compute per-date totals first (for plain sums and total weight)
    totals = df.groupby(date_col, as_index=False).agg(agg_dict)
    totals = totals.rename(columns={
        weight_col: "total_weight",
        "n_call_spreads_pos": "sum_call_spreads_pos",
        "n_butterflies_pos": "sum_butterflies_pos",
    })

    # If user asked for weighted sums for other columns, compute them
    if weighted_cols:
        # Validate requested columns
        missing_w = [c for c in weighted_cols if c not in df.columns]
        if missing_w:
            raise ValueError(f"weighted_cols not found in summary_df: {missing_w}")

        # Create product columns col * weight_col
        prod_cols = {}
        for col in weighted_cols:
            prod_name = f"{col}__x__{weight_col}"
            df[prod_name] = df[col] * df[weight_col]
            prod_cols[col] = prod_name

        # Aggregate the products per date
        wsum = (
            df.groupby(date_col, as_index=False)[list(prod_cols.values())]
            .sum()
            .rename(columns={v: f"{k}_wsum" for k, v in prod_cols.items()})
        )

        # Join weighted sums back to totals
        out = totals.merge(wsum, on=date_col, how="left")
    else:
        out = totals

    # Optional: sort by date for nice ordering
    # If your 'date' column is not datetime, consider converting:
    # out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(by=date_col).reset_index(drop=True)

    return out


#
# 
def plot_arb_aggregates(
        agg_pred: pd.DataFrame,
        agg_om: pd.DataFrame,
        value_cols: Iterable[str],
        *,
        date_col: str = "date",
        labels: Tuple[str, str] = ("Pred", "Observed"),
        figsize: Optional[Tuple[float, float]] = (10, 8),
        colors: Tuple[str, str] = ("tab:blue", "tab:orange"),
        linewidth: float = 1.2,
        markersize: float = 0.0,  # set >0 to show markers
        ylabels: Optional[Tuple[str, ...]] = None,
) -> List[plt.Figure]:
    """
    Plot time series of arbitrage aggregates for two sources (Pred vs Observed),
    returning one separate figure per metric.

    Parameters
    ----------
    agg_pred, agg_om : pd.DataFrame
        Aggregated per-day DataFrames (e.g., from `aggregate_by_date(...)`).
    value_cols : iterable of str
        Columns to plot on Y (e.g., ['sum_call_spreads_pos', 'sum_butterflies_pos']).
    date_col : str
        Name of the date column.
    labels : (str, str)
        Legend labels for (agg_pred, agg_om).
    figsize : (w, h)
        Figure size for each individual figure.
    colors : (str, str)
        Line colors for (agg_pred, agg_om).
    linewidth : float
        Line width for the time series.
    markersize : float
        Marker size; if 0, no markers.
    ylabels : tuple of str, optional
        Custom y-axis labels for each plot (must match len(value_cols)).

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        List of figures created (one per metric).
    """

    value_cols = list(value_cols)
    if not value_cols:
        raise ValueError("`value_cols` must contain at least one column name.")

    # --- Ensure date is datetime and sorted
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        if date_col not in df.columns:
            raise ValueError(f"'{date_col}' not found in DataFrame.")
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.sort_values(by=date_col).reset_index(drop=True)
        return out

    pred = _prep(agg_pred)
    om = _prep(agg_om)

    # --- Validate requested columns exist
    present_cols = [c for c in value_cols if c in pred.columns and c in om.columns]
    if not present_cols:
        raise ValueError("None of the `value_cols` are present in both DataFrames.")

    # Adjust ylabels if not provided
    if ylabels is None or len(ylabels) != len(present_cols):
        ylabels = tuple(present_cols)

    figs = []

    # --- Create a separate figure for each metric
    for col, ylabel in zip(present_cols, ylabels):
        fig, ax = plt.subplots(figsize=figsize)

        # Observed data
        ax.plot(
            om[date_col], om[col],
            label=labels[1], color=colors[1],
            linewidth=linewidth, marker="o" if markersize > 0 else None, markersize=markersize
        )
        # Predicted data
        ax.plot(
            pred[date_col], pred[col],
            label=labels[0], color=colors[0],
            linewidth=linewidth, marker="o" if markersize > 0 else None, markersize=markersize
        )

        # ax.set_title(col)
        ax.set_xlabel("Date")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize="small")

        fig.tight_layout()
        figs.append(fig)

    return figs


#
# 
def plot_rnd_surfaces_by_date(
        dates: Iterable,
        model,
        full_data: pd.DataFrame,  # every option observation
        daily_data: pd.DataFrame,  # one row per date with betas (and maybe time_to_earnings)
        *,
        n_moneyness: int = 5000,
        thin_every: int = 2,
        scaled_moneyness_range: Optional[Tuple[float, float]] = None,
        cmap_name: str = "gist_gray",
        view: Tuple[float, float] = (5, -50),
        title_prefix: Optional[str] = None,
) -> plt.Figure:
    """
    Plot risk-neutral 'moneyness density' curves per date.

    If scaled_moneyness_range is None, automatically uses the min/max of
    'ttm_scaled_moneyness' for that date in full_data (if available).

    Parameters
    ----------
    dates : iterable of date-like
        Dates to render.
    model : nn.Module-like
        Provides .get_factors() and .get_predictions().
    full_data : DataFrame
        Option-level data; must contain: 'date','ttm','rate','forward_price'.
        If 'ttm_scaled_moneyness' exists, its per-day min/max defines the plotting range.
    daily_data : DataFrame
        Per-date betas; must contain: 'date', 'beta_*' columns.
        Optionally 'time_to_earnings'.
    n_moneyness : int
        Number of scaled-moneyness grid points.
    thin_every : int
        Use every Nth maturity to reduce clutter.
    scaled_moneyness_range : (float,float) or None
        If None, automatically computed from data.
    cmap_name : str
        Colormap name.
    view : (elev, azim)
        3D viewing angles.
    title_prefix : str, optional
        Label prefix in subplot titles.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.eval()

    include_tte = "time_to_earnings" in daily_data.columns
    has_scaled_moneyness = "ttm_scaled_moneyness" in full_data.columns

    dates = list(dates)
    fig, axs = plt.subplots(1, len(dates), subplot_kw={'projection': '3d'},
                            figsize=(max(4 * len(dates), 4), 4))
    if len(dates) == 1:
        axs = [axs]

    beta_candidates = [c for c in daily_data.columns if c.startswith("beta_")]
    if not beta_candidates:
        raise ValueError("No beta_* columns found in daily_data.")

    for date, ax in zip(dates, axs):
        fd_d = full_data.loc[full_data["date"] == date].copy()
        if fd_d.empty:
            ax.text(0.5, 0.5, "No full_data for date", transform=ax.transAxes, ha="center")
            continue

        dd_d = daily_data.loc[daily_data["date"] == date].copy()
        if dd_d.empty:
            ax.text(0.5, 0.45, "No daily_data for date", transform=ax.transAxes, ha="center")
            continue

        # Betas + optional TTE
        betas = torch.tensor(
            dd_d.iloc[0][beta_candidates].to_numpy(dtype=float),
            dtype=torch.float32, device=device
        ).unsqueeze(0)
        tte = float(dd_d.iloc[0]["time_to_earnings"]) if include_tte else None

        # --- dynamic moneyness range ---
        if scaled_moneyness_range is not None:
            sm_min, sm_max = scaled_moneyness_range
        elif has_scaled_moneyness:
            sm_min = fd_d["ttm_scaled_moneyness"].min()
            sm_max = fd_d["ttm_scaled_moneyness"].max()
            # If degenerate or missing, fallback to default range
            if not np.isfinite(sm_min) or not np.isfinite(sm_max) or sm_min == sm_max:
                sm_min, sm_max = -1.0, 1.0
        else:
            sm_min, sm_max = -1.0, 1.0

        # --- Build TTM grid ---
        per_ttm = (
            fd_d.groupby("ttm", as_index=False)[["rate", "forward_price"]]
            .median()
            .sort_values("ttm")
            .reset_index(drop=True)
        )
        per_ttm = per_ttm.iloc[::max(1, thin_every)].reset_index(drop=True)
        if per_ttm.empty:
            ax.text(0.5, 0.5, "No TTM data", transform=ax.transAxes, ha="center")
            continue

        ttm_vec = torch.tensor(per_ttm["ttm"].to_numpy(dtype=float), dtype=torch.float32, device=device)
        rate_vec = torch.tensor(per_ttm["rate"].to_numpy(dtype=float), dtype=torch.float32, device=device)
        fwd_vec = torch.tensor(per_ttm["forward_price"].to_numpy(dtype=float), dtype=torch.float32, device=device)
        N_TTM = ttm_vec.shape[0]

        # --- Grid for scaled moneyness ---
        sm_vec = torch.linspace(sm_min, sm_max, n_moneyness, device=device)
        TTM, SM = torch.meshgrid(ttm_vec, sm_vec, indexing='ij')

        K_over_F = torch.exp(SM * torch.sqrt(TTM)).detach().clone().requires_grad_(True)
        FWD, _ = torch.meshgrid(fwd_vec, sm_vec, indexing='ij')
        RATE, _ = torch.meshgrid(rate_vec, sm_vec, indexing='ij')
        K = K_over_F * FWD

        feats_list = [TTM, torch.log(K_over_F) / torch.sqrt(TTM)]
        if include_tte:
            feats_list.append(torch.full_like(TTM, tte))
        feats = torch.stack(feats_list, dim=-1)
        d_in = feats.shape[-1]

        # --- Model forward ---
        with torch.no_grad():
            factors = model.get_factors(feats.reshape(-1, d_in))
        log_sigma = model.get_predictions(factors, betas, num_obs_per_group=torch.tensor([factors.shape[0]]))
        sigma = torch.exp(log_sigma).view(N_TTM, n_moneyness)

        # --- Black (forward) discounted call ---
        norm01 = dist.Normal(0.0, 1.0)
        sqrtT = torch.sqrt(TTM)
        d1 = (-torch.log(K_over_F) + 0.5 * sigma ** 2 * TTM) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        call = torch.exp(-RATE * TTM) * (FWD * norm01.cdf(d1) - K_over_F * FWD * norm01.cdf(d2))
        call_flat = call.reshape(-1)

        # --- Risk-neutral density ---
        dC_dKF = torch.autograd.grad(call_flat, K_over_F, torch.ones_like(call_flat), create_graph=True)[0]
        d2C_dKF2 = torch.autograd.grad(dC_dKF, K_over_F, torch.ones_like(dC_dKF))[0].view(N_TTM, n_moneyness)
        density_K = d2C_dKF2 * torch.exp(RATE * TTM) / (FWD ** 2)

        X = SM.detach().cpu().numpy()
        Y = TTM.detach().cpu().numpy()
        Z = (torch.sqrt(TTM) * K * density_K).detach().cpu().numpy()

        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(0, N_TTM - 1)
        for i in range(N_TTM - 1, -1, -1):
            ax.plot(X[i], Y[i], Z[i], color=cmap(norm(i)))

        title_date = pd.to_datetime(date).strftime("%Y-%m-%d")
        title = f"{title_prefix} - {title_date}" if title_prefix else title_date
        ax.text2D(0.5, 0.92, title, transform=ax.transAxes, ha="center", va="top", fontsize="small")
        ax.set_xlabel("Scaled moneyness")
        ax.set_ylabel("Time-to-maturity")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune=None))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune=None))
        ax.view_init(*view)

    plt.tight_layout()
    return fig


#
# 
def merge_vix_data(vix_path: str, vix_aapl_path: str) -> pd.DataFrame:
    """
    Load and merge the VIX and Apple VIX datasets on date.

    Keeps only the 'CLOSE' column from each file, renaming them to
    'IndexVIX' and 'AAPLVIX'.

    Parameters
    ----------
    vix_path : str
        Path to the VIX index CSV file (e.g. "VIX.csv").
    vix_aapl_path : str
        Path to the Apple-specific VIX CSV file (e.g. "VIXApple.csv").

    Returns
    -------
    merged_df : pd.DataFrame
        DataFrame indexed by date, with columns ['IndexVIX', 'AAPLVIX'].
    """

    # --- Load both datasets ---
    vix = pd.read_csv(vix_path)
    vix_aapl = pd.read_csv(vix_aapl_path)

    # --- Standardize date columns ---
    vix['DATE'] = pd.to_datetime(vix['DATE'])
    vix_aapl['DATE'] = pd.to_datetime(vix_aapl['DATE'])

    # --- Keep only the 'CLOSE' column and rename ---
    vix = vix[['DATE', 'CLOSE']].rename(columns={'DATE': 'date', 'CLOSE': 'IndexVIX'})
    vix_aapl = vix_aapl[['DATE', 'CLOSE']].rename(columns={'DATE': 'date', 'CLOSE': 'AAPLVIX'})

    # --- Merge on 'date' ---
    merged = pd.merge(vix, vix_aapl, on='date', how='outer')

    # --- Set date as index ---
    merged.set_index('date', inplace=True)

    return merged


#
# 
def select_window_dates(
        daily_data: pd.DataFrame,
        date,
        window_below: int = 0,
        window_above: int = 0,
        use_earnings: bool = False,
        *,
        date_col: str = "date",
        earnings_col: str = "is_earning_date",
) -> List[pd.Timestamp]:
    """
    Parameters
    ----------
    daily_data : DataFrame
        Must contain a date column and (optionally) an earnings boolean column.
    date : any date-like
        Anchor reference date (will be converted to pandas.Timestamp).
    window_below : int
        Number of dates before the anchor to include (only used when use_earnings=False).
    window_above : int
        Number of dates after the anchor to include (only used when use_earnings=False).
    use_earnings : bool
        If True, return ONLY the closest earnings date to `date`.
        If False, return a window around the closest calendar date to `date`.

    Returns
    -------
    List[pd.Timestamp]
        List of dates (sorted). May be empty if no earnings dates exist when use_earnings=True.
    """
    if date_col not in daily_data.columns:
        raise ValueError(f"'{date_col}' column is required in daily_data.")
    if use_earnings and earnings_col not in daily_data.columns:
        raise ValueError(f"'{earnings_col}' column is required when use_earnings=True.")

    # Normalize and sort unique dates
    df = daily_data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    dates_sorted = (
        df[[date_col] + ([earnings_col] if earnings_col in df.columns else [])]
        .drop_duplicates(subset=[date_col])
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    target = pd.to_datetime(date)

    # Choose the candidate set: earnings-only or all dates
    if use_earnings:
        earn_df = dates_sorted[dates_sorted.get(earnings_col, False) == True]
        if earn_df.empty:
            return []
        # Find closest earnings date
        idx = (earn_df[date_col] - target).abs().idxmin()
        return [pd.Timestamp(earn_df.loc[idx, date_col])]

    # use_earnings == False → work on all dates
    all_dates = dates_sorted[date_col].to_list()
    if not all_dates:
        return []

    # Find index of closest date in all dates
    ser = dates_sorted[date_col]
    idx_closest = (ser - target).abs().idxmin()

    # Slice window [idx_closest - window_below, idx_closest + window_above]
    start = max(0, idx_closest - window_below)
    end = min(len(ser) - 1, idx_closest + window_above)
    window = ser.iloc[start:end + 1].to_list()

    # Ensure list of pd.Timestamp
    return [pd.Timestamp(d) for d in window]


#
# 
def compute_rn_density_nogrid(
        model: SplitDeepFactorNNList,
        K: torch.Tensor,  # (N,) strike grid (torch)
        ttm: Union[float, torch.Tensor],  # scalar or (N,) time to maturity in years
        forward_price: Union[float, torch.Tensor],  # scalar or (N,)
        rate: Union[float, torch.Tensor],  # scalar or (N,) continuous r
        betas: torch.Tensor,  # (F,) or (1,F)
        tte: Optional[Union[float, torch.Tensor]] = None,  # ignored unless include_tte=True
        *,
        return_numpy: bool = True  # return np.ndarray for convenience
):
    """
    Compute the risk-neutral density f(K, T) for a given strike grid K at maturity T,
    using your model pipeline:

        features = [TTM, log(K/F)/sqrt(TTM)] (+ TTE if include_tte=True)
        factors  = model.get_factors(features)
        log_sigma= model.get_predictions(factors, betas, num_obs_per_group=[N])
        sigma    = exp(log_sigma)
        C        = discounted forward Black call(F, K/F, sigma, r, T)
        f(K,T)   = exp(rT) * (1/F^2) * d^2 C / d(K/F)^2

    Returns
    -------
    f : np.ndarray or torch.Tensor with shape (N,)
        Risk-neutral density evaluated on the input K grid.
    """

    # ---- device / dtype
    include_tte = tte is not None
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    dtype = torch.float32

    # ---- ensure tensors on the right device/shape
    K = K.to(device=device, dtype=dtype).view(-1)  # (N,)
    N = K.shape[0]

    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.tensor(x, device=device, dtype=dtype)

    TTM = _to_tensor(ttm)
    FWD = _to_tensor(forward_price)
    RATE = _to_tensor(rate)

    # Broadcast scalars to (N,)
    if TTM.ndim == 0:  TTM = TTM.repeat(N)
    if FWD.ndim == 0:  FWD = FWD.repeat(N)
    if RATE.ndim == 0: RATE = RATE.repeat(N)

    # betas must be (1, F)
    if betas.ndim == 1:
        betas = betas.unsqueeze(0)
    betas = betas.to(device=device, dtype=dtype)  # (1, F)

    # Optional TTE (constant across grid if provided)
    if include_tte:
        if tte is None:
            raise ValueError("include_tte=True but `tte` not provided.")
        TTE = _to_tensor(tte)
        if TTE.ndim == 0:
            TTE = TTE.repeat(N)
    else:
        TTE = None

    # ---- core variables
    sqrtT = torch.sqrt(TTM)
    # IMPORTANT: use K/F as the autograd leaf
    K.requires_grad_(True)
    K_over_F = (K / FWD)  # (N,)

    # ---- build model features
    # Your latest setup uses log(K/F)/sqrt(T) as the moneyness-like input
    m_scaled = torch.log(K_over_F) / (sqrtT)  # (N,)
    if include_tte:
        feats = torch.stack([TTM, m_scaled, TTE], dim=-1)  # (N, 3)
    else:
        feats = torch.stack([TTM, m_scaled], dim=-1)  # (N, 2)

    # ---- model forward: factors -> log_sigma -> sigma
    model.eval()
    with torch.no_grad():
        factors = model.get_factors(feats)  # (N, F)
    # Per your interface: num_obs_per_group should be a Tensor
    log_sigma = model.get_predictions(
        factors, betas,
        num_obs_per_group=torch.tensor([factors.shape[0]], device=factors.device)
    ).view(-1)  # (N,)
    sigma = torch.exp(log_sigma)  # (N,)

    # ---- discounted forward Black call
    norm01 = dist.Normal(0.0, 1.0)
    d1 = (-torch.log(K_over_F) + 0.5 * sigma ** 2 * TTM) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    call = torch.exp(-RATE * TTM) * (FWD * norm01.cdf(d1) - K_over_F * FWD * norm01.cdf(d2))  # (N,)

    # ---- risk-neutral density in K via 2nd derivative wrt (K/F)
    dC_dKF = torch.autograd.grad(call, K, torch.ones_like(call), create_graph=True)[0]  # (N,)
    d2C_dKF2 = torch.autograd.grad(dC_dKF, K, torch.ones_like(dC_dKF))[0]  # (N,)

    # f(K,T) = e^{rT} * ∂²C/∂K²
    density_K = d2C_dKF2 * torch.exp(RATE * TTM)  # (N,)

    # Return as numpy by default (to match your downstream usage)
    if return_numpy:
        return density_K.detach().cpu().numpy()
    return density_K


#
# 
def interpolate_data(df: pd.DataFrame,
                     tau: float,
                     kind: str = "linear",
                     column_to_interpolate: str = 'forward_price') -> float:
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

    Returns
    -------
    float
        Interpolated forward price at time tau.
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


def compute_vix_timeseries(
        full_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        model,
        *,
        TAU: int = 30,  # days
        step_per_forward: int = 1000,  # K step = F / this
        vix_col_out=None,  # default: f"VIX{TAU}"
) -> pd.DataFrame:
    """
    Compute a VIX-like index per date using:
      - scalar inputs at TAU (interpolated from per-date panel `full_data`)
      - strike grid around forward
      - model-based RN density via compute_rn_density_nogrid(...)
      - normalized density and variance integral

    Returns a DataFrame with ['date', vix_col_out].
    """
    if vix_col_out is None:
        vix_col_out = f"VIX{TAU}"

    # Ensure date is a column (not just index)
    if "date" not in full_data.columns:
        raise ValueError("full_data must contain a 'date' column.")
    if "date" not in daily_data.columns:
        # if it's the index, reset
        daily_data = daily_data.reset_index().rename(
            columns={"index": "date"}) if daily_data.index.name == "date" else daily_data

    out = []
    grouped_by_date = full_data.groupby("date")
    T_years = TAU / 365.0

    for date, y in tqdm(grouped_by_date, desc=f"Computing {vix_col_out}"):
        # Slice daily_data row for this date
        dd = daily_data.loc[daily_data["date"] == date]
        if dd.empty:
            continue

        # ---- betas (unsqueeze(0))
        beta_cols = [c for c in dd.columns if c.startswith("beta_")]
        if not beta_cols:
            continue
        betas = torch.tensor(dd.iloc[0][beta_cols].to_numpy(dtype=float), dtype=torch.float32).unsqueeze(0)

        # ---- Interpolate inputs at TAU (days)
        forward = interpolate_data(y, tau=TAU, kind="linear", column_to_interpolate="forward_price")
        rate = interpolate_data(y, tau=TAU, kind="linear", column_to_interpolate="rate")
        # tte may or may not exist; if missing, set None and we won't include it
        tte = interpolate_data(y, tau=TAU, kind="linear", column_to_interpolate="time_to_earnings") \
            if "time_to_earnings" in y.columns else None

        # S0 = interpolate_data(y, tau=0, kind="linear", column_to_interpolate="forward_price")
        # S0 = y['close'].iloc[0]

        # ---- Strike grid
        K_min = np.exp(y['ttm_scaled_moneyness'].min() * np.sqrt(T_years)) * forward
        K_max = np.exp(y['ttm_scaled_moneyness'].max() * np.sqrt(T_years)) * forward
        # K_min = forward * (1-k_span)
        # K_max = forward * (1+k_span)
        delta_k = (K_max - K_min) / step_per_forward

        K = torch.arange(K_min, K_max, delta_k, dtype=torch.float32)  # (N,)

        # Broadcast scalars
        ttm_t = torch.full_like(K, fill_value=T_years)
        fwd_t = torch.full_like(K, fill_value=forward)
        r_t = torch.full_like(K, fill_value=rate)

        # ---- Density (numpy)
        density = compute_rn_density_nogrid(
            model=model,
            K=K,
            ttm=ttm_t,
            forward_price=fwd_t,
            rate=r_t,
            betas=betas,
            tte=tte if tte is not None else None,
            return_numpy=True
        )
        density = np.clip(density, 0.0, None)

        # ---- Normalize and compute E[K]
        K_np = K.numpy()
        area = np.trapz(density, x=K_np)
        density /= area
        EK = np.trapz(K_np * density, x=K_np)

        # ---- Variance → VIX
        if T_years <= 0:
            continue
        integral = np.trapz(np.log(K_np / EK) * density, x=K_np)
        var = (-2.0 / T_years) * integral
        # var = max(var, 0.0)
        vix_val = 100.0 * np.sqrt(var)

        out.append((pd.to_datetime(date), vix_val))

    return pd.DataFrame(out, columns=["date", vix_col_out]).sort_values("date").reset_index(drop=True)


def plot_vix_timeseries(
        daily_data: pd.DataFrame,
        *,
        tau_col: str = "VIX30",  # your model’s VIX column
        index_vix_col: str = "IndexVIX",  # CBOE VIX
        stock_vix_col: Optional[str] = "AAPLVIX",  # optional VXAPL
        date_col: str = "date",
        figsize: Tuple[int, int] = (10, 3),
        constrained_layout: bool = True,
        start_date: Optional[str] = None,
        logy: bool = False,  # set True to use log scale
) -> plt.Figure:
    """
    Plot model-implied VIX vs market indices (VIX, VXAPL) with optional earnings markers.

    Parameters
    ----------
    daily_data : pd.DataFrame
        Must contain at least 'date', 'VIX30', 'IndexVIX'.
        Can optionally contain 'AAPLVIX', 'is_earnings' or 'is_earning_date'.
    """
    df = daily_data.copy()
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' column is required.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    if start_date is not None:
        df = df[df[date_col] >= pd.to_datetime(start_date)]

    if tau_col not in df.columns:
        raise ValueError(f"'{tau_col}' column not found in data.")
    df = df.dropna(subset=[tau_col])

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=constrained_layout)

    # --- Plot model VIX ---
    ax.plot(df[date_col], df[tau_col], label=f"Model - {tau_col}", color="darkblue", linewidth=2.0)

    # --- Plot CBOE VIX ---
    if index_vix_col in df.columns:
        ax.plot(df[date_col], df[index_vix_col], label="CBOE - VIX", linewidth=1.0)

    # --- Plot AAPL VXAPL if present ---
    has_stock_vix = stock_vix_col is not None and stock_vix_col in df.columns
    if has_stock_vix:
        ax.plot(df[date_col], df[stock_vix_col], label="CBOE - VXAPL", linewidth=1.0, alpha=0.9)

    # --- Handle earnings column (either 'is_earnings' or 'is_earning_date') ---
    earnings_col = None
    for possible_col in ["is_earnings", "is_earning_date"]:
        if possible_col in df.columns:
            earnings_col = possible_col
            break

    if earnings_col:
        earn_mask = df[earnings_col].astype(bool)
        if earn_mask.any():
            ax.scatter(
                df.loc[earn_mask, date_col],
                df.loc[earn_mask, tau_col],
                color="red",
                s=10,
                zorder=3,
                label="Earnings date",
                linewidth=1.0,
            )

    if logy:
        ax.set_yscale("log")

    ncols = 2 if has_stock_vix else 1
    ax.legend(loc="upper left", fontsize="small", ncols=ncols)
    ax.set_xlabel("Date")
    ax.set_ylabel("VIX Level")

    return fig


def plot_daily_rmse_timeseries_with_diff(
        ttea_daily_data: pd.DataFrame,
        no_ttea_daily_data: pd.DataFrame,
        *,
        date_col: str = "date",
        split_cols=("train", "valid", "test"),
        rmse_col: str = "feedforward_RMSE",
        label_a_suffix: str = 'DIVFM with TTEA',
        label_b_suffix: str = "DIVFM without TTEA",
        figsize=(10, 4),
):
    required = {date_col, rmse_col, *split_cols}
    for name, df in [("ttea_daily_data", ttea_daily_data), ("no_ttea_daily_data", no_ttea_daily_data)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")

    ttea = ttea_daily_data.copy()
    no_ttea = no_ttea_daily_data.copy()
    ttea[date_col] = pd.to_datetime(ttea[date_col], errors="coerce")
    no_ttea[date_col] = pd.to_datetime(no_ttea[date_col], errors="coerce")

    key_cols = [date_col, *split_cols]
    merged = (
        ttea[key_cols + [rmse_col]]
        .rename(columns={rmse_col: "rmse_with"})
        .merge(
            no_ttea[key_cols + [rmse_col]].rename(columns={rmse_col: "rmse_without"}),
            on=key_cols,
            how="inner",
        )
    )
    merged["rmse_diff"] = merged["rmse_without"] - merged["rmse_with"]

    def _series(df, split_name, col):
        out = df[df[split_name] == True][[date_col, col]].sort_values(date_col)
        return out[date_col], out[col]

    fig, axs = plt.subplots(nrows=2, figsize=figsize, sharex=True)

    split_style = {
        "train": dict(color="darkblue", alpha=1.0, linewidth=1.0),
        "valid": dict(color="thistle", alpha=1.0, linewidth=1.0),
        "test": dict(color="deepskyblue", alpha=1.0, linewidth=1.0),
    }

    split_name = {
        "train": "Training set",
        "valid": "Validation set",
        "test": "Test set",
    }

    # ---------- Top panel ----------
    for s in split_cols:
        x, y = _series(merged, s, "rmse_with")
        axs[0].plot(
            x, y,
            linestyle="solid",
            **split_style[s],
            label=f"{split_name[s]}: {label_a_suffix}",
        )

        x2, y2 = _series(merged, s, "rmse_without")
        axs[0].plot(
            x2, y2,
            linestyle="solid",
            color="red",
            linewidth=0.3,
            alpha=1.0 if s == "train" else 0.8,
            label=label_b_suffix if s == "train" else None,
        )

    axs[0].set_ylabel("RMSE")
    axs[0].set_title("Daily RMSE timeseries")
    axs[0].legend(ncol=2, fontsize="small")

    # ---------- Bottom panel ----------
    for s in split_cols:
        x, y = _series(merged, s, "rmse_diff")
        axs[1].plot(
            x, y,
            linestyle="solid",
            color=split_style[s]["color"],
            linewidth=1.0,
            label=f"{split_name[s]}: RMSE difference",
        )

    axs[1].axhline(y=0, color="red")
    axs[1].set_ylabel("RMSE")
    axs[1].set_title("Daily RMSE timeseries difference")
    axs[1].legend(ncol=2, fontsize="small")

    axs[1].set_xlabel("Date")
    fig.tight_layout()
    return fig, axs
