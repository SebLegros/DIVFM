import numpy as np

import pandas as pd
from typing import Dict, Sequence, Optional

def add_tau_moneyness_bins(
    df: pd.DataFrame,
    *,
    tau_col: str = "days_to_maturity",
    moneyness_col: str = "ttm_scaled_moneyness",
    # Default 10 moneyness bins:
    m_bins: Sequence[float] = (-np.inf, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, np.inf),
    # Default 10 tau bins (in days):
    tau_bins: Sequence[float] = (0, 5, 35, 75, 110, 145, 220, 290, 400, 620, np.inf),
    out_m_col: str = "moneyness_bin",
    out_tau_col: str = "tau_bin",
) -> pd.DataFrame:
    """
    Add categorical bins for moneyness and tau (days_to_maturity) to a copy of df.
    """
    m_labels = tuple(f'M{i+1}' for i in range(len(m_bins)-1))
    tau_labels = tuple(f'tau{i+1}' for i in range(len(tau_bins)-1))
    out = df.copy()
    if moneyness_col not in out.columns:
        raise ValueError(f"Column '{moneyness_col}' not found.")
    if tau_col not in out.columns:
        raise ValueError(f"Column '{tau_col}' not found.")

    out[out_m_col] = pd.cut(out[moneyness_col], bins=m_bins, labels=m_labels, include_lowest=True)
    out[out_tau_col] = pd.cut(out[tau_col],    bins=tau_bins, labels=tau_labels, include_lowest=True)
    return out




def summarize_by_moneyness_or_tau_full(
    full_data: pd.DataFrame,
    *,
    split_data: bool = True,  # NEW
    split_cols: Sequence[str] = ("train", "valid", "test"),
    bin_col: str = "moneyness_bin",
    moneyness_or_tau_col: str = "ttm_scaled_moneyness",
    date_col: str = "date",
    # auto-create bins if missing:
    auto_make_bins_if_missing: bool = True,
    bins: Sequence[float] = (-np.inf, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, np.inf),
    labels_order: Sequence[str] = ("M1","M2","M3","M4","M5","M6","M7","M8","M9","M10"),
    # metric columns:
    pred_col1: str = "IVpred",                 # e.g., with TTEA
    pred_col2: Optional[str] = None,           # optional: without TTEA
    observed_col: str = "OM_IV",
    # row labels:
    label1: str = "RMSE - DIVFM with TTEA",
    label2: Optional[str] = "RMSE - DIVFM without TTEA",
) -> Dict[str, pd.DataFrame]:
    """
    Summarize DATE-AVERAGED RMSE by bins (moneyness or tau).

    If split_data=True (default): returns tables for train/valid/test using split_cols.
    If split_data=False: returns a single table for all rows (key: "all").

    For each bin:
      - compute RMSE per date within the bin
      - then average those daily RMSEs across dates (equal weight per date)
    """
    df = full_data.copy()

    # Check split columns only if we actually split
    if split_data:
        missing_splits = [c for c in split_cols if c not in df.columns]
        if missing_splits:
            raise ValueError(f"Missing split columns in full_data: {missing_splits}")

    # Ensure date col and coerce to datetime
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' not in DataFrame (required for date-averaged RMSE).")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Ensure / create bin column
    if bin_col not in df.columns:
        if not auto_make_bins_if_missing:
            raise ValueError(f"'{bin_col}' not found and auto_make_bins_if_missing=False.")
        if moneyness_or_tau_col not in df.columns:
            raise ValueError(f"'{bin_col}' missing and '{moneyness_or_tau_col}' not found to build it.")
        df[bin_col] = pd.cut(df[moneyness_or_tau_col], bins=bins, labels=labels_order, include_lowest=True)

    # If bin column exists and is categorical, use its category order unless user passed a custom labels_order.
    if pd.api.types.is_categorical_dtype(df[bin_col]):
        existing_cats = list(df[bin_col].cat.categories)
        if tuple(labels_order) != tuple(existing_cats):
            labels_order = tuple(existing_cats)

    # Validate metric columns
    needed = [pred_col1, observed_col, date_col, bin_col]
    if pred_col2 is not None:
        needed.append(pred_col2)
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in full_data.")

    def _date_avg_rmse(sub: pd.DataFrame, pred_col: str) -> float:
        s = sub[[date_col, pred_col, observed_col]].dropna()
        if s.empty:
            return np.nan

        def _rmse_for_date(g: pd.DataFrame) -> float:
            e2 = (g[pred_col] - g[observed_col]) ** 2
            return float(np.sqrt(e2.mean()))

        daily = s.groupby(date_col, sort=True).apply(_rmse_for_date)
        return float(daily.mean())

    def _one_split(df_split: pd.DataFrame) -> pd.DataFrame:
        rows = {label1: [], "Number of contracts": []}
        use_second = (pred_col2 is not None)
        if use_second:
            rows[label2 if label2 is not None else "RMSE (second)"] = []

        # per-bin
        for lab in labels_order:
            sub = df_split[df_split[bin_col] == lab]
            rows[label1].append(_date_avg_rmse(sub, pred_col1))
            if use_second:
                rows[label2 if label2 is not None else "RMSE (second)"].append(_date_avg_rmse(sub, pred_col2))
            rows["Number of contracts"].append(int(sub.shape[0]))

        # All
        rows[label1].append(_date_avg_rmse(df_split, pred_col1))
        if use_second:
            rows[label2 if label2 is not None else "RMSE (second)"].append(_date_avg_rmse(df_split, pred_col2))
        rows["Number of contracts"].append(int(df_split.shape[0]))

        cols = list(labels_order) + ["All"]
        return pd.DataFrame(rows, index=cols).T

    if not split_data:
        return {"all": _one_split(df)}

    split_map = {
        "train": df[df[split_cols[0]] == True],
        "valid": df[df[split_cols[1]] == True],
        "test":  df[df[split_cols[2]] == True],
    }
    return {k: _one_split(v) for k, v in split_map.items()}


def bins_table_to_latex(
    summary: Dict[str, pd.DataFrame],
    *,
    # Row labels to pull from the summary frames:
    label1: str = "RMSE - DIVFM with TTEA",
    label2: Optional[str] = "RMSE - DIVFM without TTEA",  # set None to skip or auto-skip if missing
    count_label: str = "Number of contracts",
    # Column headers to show (e.g., moneyness or tau headers):
    bin_headers_math: Sequence[str] = (
        r"$\text{M}_1$", r"$\text{M}_2$", r"$\text{M}_3$", r"$\text{M}_4$",
        r"$\text{M}_5$", r"$\text{M}_6$", r"$\text{M}_7$", r"$\text{M}_8$",
        r"$\text{M}_9$", r"$\text{M}_{10}$", "All"
    ),
    # Titles for the three blocks:
    block_titles: Sequence[str] = ("Training Set", "Validation Set", "Test Set"),
    # Optional table wrapper:
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """
    Convert {split: DataFrame} (from your summarize_* function) into a LaTeX tabularx.
    - Works for moneyness or tau bins (headers are passed via `bin_headers_math`).
    - Includes the second RMSE row only if `label2` is provided AND present in frames.
    """
    # Helpers
    def _format_row_rmse(vals):
        return " & ".join([
            (fr"\num[round-mode=places,round-precision=4]{{{v}}}" if pd.notna(v) else "")
            for v in vals
        ])

    def _format_row_count(vals):
        return " & ".join([
            (fr"\num{{{int(v)}}}" if pd.notna(v) else "")
            for v in vals
        ])

    def _block(df: pd.DataFrame, title: str) -> str:
        # Use the actual column order from df, but ensure it matches header count
        cols = list(df.columns)  # e.g., ["M1",...,"M10","All"] or ["tau1",...,"tau10","All"]

        # Rows: label1 (required)
        rmse_1 = df.loc[label1, cols].tolist()

        # Optional label2 row (only if both requested and present)
        has_label2 = label2 is not None and (label2 in df.index)
        rmse_2 = df.loc[label2, cols].tolist() if has_label2 else None

        counts = df.loc[count_label, cols].tolist()

        lines = [
            r"\midrule",
            fr"\multicolumn{{{1+len(cols)}}}{{c}}{{\textbf{{{title}}}}} \\",
            r"\midrule",
            f"{label1} & " + _format_row_rmse(rmse_1) + r" \\",
        ]
        if has_label2:
            lines.append(f"{label2} & " + _format_row_rmse(rmse_2) + r" \\")
        lines.append("Number of contracts & " + _format_row_count(counts) + r" \\")
        return "\n".join(lines)

    # Build head/body/tail
    N = len(bin_headers_math)  # number of X columns after the leading 'l'
    head = (
        rf"\begin{{tabularx}}{{\textwidth}}{{l *{{{N}}}{{>{{\centering\arraybackslash}}X}}}}" + "\n"
        r"\toprule" + "\n"
        r"& " + " & ".join(bin_headers_math) + r" \\" + "\n"
        r"\midrule" + "\n"
    )

    # Expect keys "train", "valid", "test" in summary
    body = (
        _block(summary["train"], block_titles[0]) + "\n" +
        _block(summary["valid"], block_titles[1]) + "\n" +
        _block(summary["test"],  block_titles[2])
    )

    tail = "\n" + r"\bottomrule" + "\n" + r"\end{tabularx}"

    # Optional table wrapper
    if caption or label:
        wrapper_head = r"\begin{table}[!ht]\centering" + "\n"
        if caption:
            wrapper_head += fr"\caption{{{caption}}}" + "\n"
        if label:
            wrapper_head += fr"\label{{{label}}}" + "\n"
        wrapper_tail = "\n" + r"\end{table}"
        return wrapper_head + head + body + tail + wrapper_tail

    return head + body + tail


def summary_to_long_df(summary: dict) -> pd.DataFrame:
    """
    Convert a summary dict {split: DataFrame} into a long DataFrame
    with columns: split, metric, bin, value.
    Each DataFrame has rows like 'RMSE - DIVFM with TTEA', 'Number of contracts'
    and columns like M1..M10, 'All'.
    """
    frames = []
    for split, df in summary.items():
        # df: rows = metrics, cols = bins (M1.., All)
        tmp = df.copy()
        tmp.index.name = "metric"
        tmp = tmp.reset_index().melt(id_vars="metric", var_name="bin", value_name="value")
        tmp["split"] = split
        frames.append(tmp)
    out = pd.concat(frames, ignore_index=True)
    # ensure numeric where possible
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    # order columns nicely
    return out[["split", "metric", "bin", "value"]]