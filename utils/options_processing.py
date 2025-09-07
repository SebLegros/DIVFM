from __future__ import annotations

import numpy as np
from pandas import DataFrame, Series, Index
from scipy.interpolate import interp1d

"""options_processing.py

Utility functions to load, merge and clean stock **option** and **forward** price data
with *built-in structured logging* so that you can see exactly what the pipeline is
doing at runtime.

Example (library usage)
-----------------------
>>> from options_processing import process_options_pipeline, setup_logging
>>> setup_logging("DEBUG")  # or "INFO", "WARNING", …
>>> df_clean = process_options_pipeline("../data")

Command-line usage
------------------
$ python options_processing.py --base-dir ../data --log-level DEBUG
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Union, Any, Optional

import pandas as pd

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def setup_logging(level: str | int = "INFO") -> None:
    """Configure *root* logger with a sensible default format.

    Parameters
    ----------
    level : str | int, default "INFO"
        Logging threshold. Accepts both the symbolic name (``"DEBUG"``) or the
        numeric constant (``logging.DEBUG``).
    """

    if isinstance(level, str):
        level = level.upper()
        level = getattr(logging, level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from noisy dependencies if any
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# -----------------------------------------------------------------------------
# 1. IO helpers
# -----------------------------------------------------------------------------

def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV file with :func:`pandas.read_csv` and raise a helpful error."""

    logger.debug("Reading CSV: %s", path)
    try:
        df = pd.read_csv(path, **kwargs)
        logger.info("Loaded %d rows from %s", len(df), path.name)
        return df
    except FileNotFoundError as err:
        logger.error("File not found: %s", path)
        raise FileNotFoundError(f"CSV file not found: {path.resolve()}") from err


def load_raw_data(base_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw *options* and *forwards* prices as dataframes."""

    base = Path(base_dir)
    options_path = base / "raw_options.csv"
    forwards_path = base / "stock_forwards.csv"

    logger.info("Loading raw data from %s", base.resolve())
    options_df = _read_csv(options_path)
    forwards_df = _read_csv(forwards_path)

    return options_df, forwards_df


# -----------------------------------------------------------------------------
# 2. Data wrangling
# -----------------------------------------------------------------------------

def _coerce_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ensure the listed columns are dtype *datetime64[ns]* (in-place)."""

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    logger.debug("Coerced columns %s to datetime[ns]", cols)
    return df

def load_zero_curve(path: str = 'data/US_zeroCoupon.csv') -> pd.DataFrame:
    """Load rates. Makes sure that within a date, rates are sorted by maturity."""
    rf = pd.read_csv(path)
    rf.loc[:, 'date'] = pd.to_datetime(rf['date'])
    return rf.sort_values(by=['date', 'days'])

def mergeOptionsWithZeroCurve(options_data, zero_curve_data):
    """Interpolate zero curve rates for stock options based on their days to maturity."""
    # Merge options with zero_curve on 'date'

    # zero_curve_data.loc[:, ['date']] = pd.to_datetime(zero_curve_data['date'])
    # options_data.loc[:, ['date']] = pd.to_datetime(options_data['date'])

    grouped_by_date_zero_curve = zero_curve_data.groupby('date')

    def interpolate_group_rate(group):
        group_unique_days_to_maturity = group['days_to_maturity'].unique()
        try:
            current_date_zero_curve = grouped_by_date_zero_curve.get_group(group.name)
            interpolation_function = interp1d(current_date_zero_curve['days'], current_date_zero_curve['rate'] / 100,
                                              kind='linear', bounds_error=False,
                                              fill_value=(current_date_zero_curve['rate'].iloc[0] / 100,
                                                          current_date_zero_curve['rate'].iloc[-1] / 100))

            group_interpolated_rates = interpolation_function(group_unique_days_to_maturity)
            return pd.DataFrame({'days_to_maturity': group_unique_days_to_maturity, 'rate': group_interpolated_rates})
        except KeyError:
            return pd.DataFrame({'days_to_maturity': group_unique_days_to_maturity,
                                 'rate': [np.nan] * len(group_unique_days_to_maturity)})

    # options_data.loc[:, 'date'] = pd.to_datetime(options_data.date)
    interpolated_zero_curve = options_data.groupby('date').apply(interpolate_group_rate).reset_index()
    interpolated_zero_curve.drop(columns='level_1', inplace=True)
    # It gives an error if I don't convert the date to string
    interpolated_zero_curve_dates = interpolated_zero_curve.loc[:, 'date'].astype(str)
    interpolated_zero_curve.drop(columns='date', inplace=True)
    interpolated_zero_curve.loc[:, 'date'] = interpolated_zero_curve_dates
    interpolated_zero_curve['date'] = pd.to_datetime(interpolated_zero_curve['date'])
    ##
    merged_data = pd.merge(options_data, interpolated_zero_curve, on=['date', 'days_to_maturity'], how='left')

    return merged_data

def merge_option_forward_dfs(options_df: pd.DataFrame, forwards_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join forward prices onto option quotes, aligning by *date*, *expiration*, *ticker*."""

    logger.info("Merging options and forwards dataframes …")

    # Normalise date types
    forwards_df = _coerce_dates(forwards_df, ["date", "expiration"])
    options_df = _coerce_dates(options_df, ["date", "exdate"])

    # Guarantee required column for downstream logic
    if "AMSettlement" not in forwards_df.columns:
        forwards_df["AMSettlement"] = 1
        logger.debug("Added missing AMSettlement column with default=1")

    merged = pd.merge(
        options_df,
        forwards_df[["date", "expiration", "ForwardPrice", "AMSettlement", "ticker"]],
        left_on=["date", "exdate", "ticker"],
        right_on=["date", "expiration", "ticker"],
        how="left",
    )

    merged = merged.drop(columns=["expiration", "AMSettlement", "forward_price"],errors="ignore").rename(
        columns={"ForwardPrice": "forward_price"}
    )

    logger.info("Merged dataframe shape: %s", merged.shape)
    return merged

def keep_OTM(df: pd.DataFrame, forward_prices) -> pd.DataFrame:
    keep = np.log(df['strike_price'] / forward_prices) * np.where(df.loc[:, 'is_call'], 1, -1) > 0
    return df.loc[keep, :]

def prepare_options_prices(df, remove_days_to_maturity_equal_to_zero=True, remove_volume_equal_to_zero=True,
                           remove_bid_equal_to_zero=True, remove_bid_lower_than=3 / 8, remove_spread_gt=1.75):
    """Prepare the options prices dataframe by filtering and adding necessary columns."""
    # Converting dates from string to datetime
    df.loc[:, 'date'] = pd.to_datetime(df['date'], errors='coerce')
    df.loc[:, 'exdate'] = pd.to_datetime(df['exdate'], errors='coerce')

    # Calculate days to maturity
    df.loc[:, 'days_to_maturity'] = pd.to_timedelta(df['exdate'] - df['date']).dt.days.astype(int)

    if remove_days_to_maturity_equal_to_zero:
        # Filter out options with days to maturity <= 0
        df = df.loc[df['days_to_maturity'] > 0, :]

    if remove_volume_equal_to_zero:
        # Filter out options with volume <= 0
        df = df.loc[df['volume'] > 0, :]

    if remove_bid_equal_to_zero:
        # Filter out options with bid price <= 0
        df = df.loc[df['best_bid'] > 0, :]

    if remove_bid_lower_than is not None:
        df = df.loc[df['best_bid'] > remove_bid_lower_than, :]
    if remove_spread_gt is not None:
        bid_ask_spread_smaller = ((df.best_offer - df.best_bid) * 2 / (
                df.best_offer + df.best_bid)) < remove_spread_gt
        df = df.loc[bid_ask_spread_smaller, :]

    # Keep only essential columns plus days_to_maturity for further calculations

    df.loc[:, 'strike_price'] = df['strike_price'] / 1000
    df.loc[:, 'is_call'] = df['cp_flag'] == 'C'

    essential_columns = ['date', 'ticker', 'exdate', 'strike_price', 'best_bid',
                         'best_offer', 'volume', 'impl_volatility', 'am_settlement',
                         'days_to_maturity', 'is_call', 'open_interest', 'forward_price']

    if 'am_settlement' not in df.columns:
        df.loc[:, 'am_settlement'] = 1

    df = df[essential_columns]
    df = compute_ttm(df)

    df.rename(columns={'impl_volatility': 'OM_IV'}, inplace=True)
    df = add_mid_price(df)

    return df


def compute_ttm(df):
    """compute the time to maturity (ttm)."""

    # Compute time to maturity in years
    df.loc[:, 'ttm'] = df['days_to_maturity'] / 365.0

    # Adjust ttm for 'am_settlement' options
    df.loc[df['am_settlement'] == 1, 'ttm'] -= (6.5 / (24 * 365))

    # Adjust ttm for options with 0 days to maturity (i.e. one minute to maturity)
    df.loc[df['days_to_maturity'] == 0, 'ttm'] = 1 / (365.0 * 24 * 60)

    return df


def add_mid_price(dfOptions):
    """
    Adds a column for the mid price of options based on best bid and best offer.

    """
    dfOptions['midPrice'] = (dfOptions['best_bid'] + dfOptions['best_offer']) / 2
    return dfOptions


def filter_options_data(
        df: pd.DataFrame,
        *,
        min_ttm: Optional[int] = None,
        remove_days_to_maturity_equal_to_zero: bool = True,
        remove_volume_equal_to_zero: bool = True,
        remove_bid_equal_to_zero: bool = True,
        remove_bid_lower_than: float | None = 3 / 8,
        remove_spread_gt: float | None = 1.75,
) -> pd.DataFrame:
    """Execute domain-specific filters via *prepare_options_prices*."""

    logger.info("Cleaning options data via prepare_options_prices …")
    cleaned = prepare_options_prices(
        df,
        remove_days_to_maturity_equal_to_zero=remove_days_to_maturity_equal_to_zero,
        remove_volume_equal_to_zero=remove_volume_equal_to_zero,
        remove_bid_equal_to_zero=remove_bid_equal_to_zero,
        remove_bid_lower_than=remove_bid_lower_than,
        remove_spread_gt=remove_spread_gt,
    )
    cleaned = keep_OTM(cleaned, cleaned.loc[:, 'forward_price'])
    cleaned["moneyness"] = cleaned["strike_price"] / cleaned["forward_price"]
    cleaned = cleaned.dropna(subset=["OM_IV"])

    logger.info("Options dataframe after cleaning: %s", cleaned.shape)
    return cleaned


def add_scaled_moneyness_and_filter_ttm(
        df: pd.DataFrame,
        min_ttm_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Add a column 'ttm_scaled_moneyness' to the options dataset and optionally filter
    out rows with time-to-maturity (TTM) below a threshold.


    """
    if 'moneyness' not in df.columns or 'ttm' not in df.columns:
        raise ValueError("DataFrame must contain 'moneyness' and 'ttm' columns.")

    df = df.copy()
    df['ttm_scaled_moneyness'] = np.log(df['moneyness']) / np.sqrt(df['ttm'])

    if min_ttm_days is not None:
        df = df[df['ttm'] >= min_ttm_days / 365]

    return df

def attach_zero_curve(df: pd.DataFrame, base_dir: str | Path) -> pd.DataFrame:
    """Merge US zero-coupon yield curve with options data for discounting."""

    curve_path = Path(base_dir) / "US_zeroCoupon.csv"
    logger.info("Loading zero-coupon curve from %s", curve_path)

    curve = load_zero_curve(curve_path)
    merged = mergeOptionsWithZeroCurve(df, curve)

    logger.info("Attached zero curve → dataframe shape: %s", merged.shape)
    return merged




def process_options_pipeline(base_dir: str | Path = "../data", **kwargs) -> pd.DataFrame:
    logger.info("Starting options processing pipeline …")

    # 1. Load raw
    options_df, forwards_df = load_raw_data(base_dir)

    # 2. Merge
    merged_df = merge_option_forward_dfs(options_df, forwards_df)

    # 3. Filter / enrich
    options_clean = filter_options_data(merged_df, **kwargs)

    # 4. Yield-curve attachment
    final_df = attach_zero_curve(options_clean, base_dir)

    #5 Keep only OTM options
    final_df = keep_OTM(final_df, final_df.loc[:, 'forward_price'])

    logger.info("Pipeline completed. Final dataframe shape: %s", final_df.shape)
    return final_df




def keep_OTM(df: pd.DataFrame, forward_prices) -> pd.DataFrame:
    keep = np.log(df['strike_price'] / forward_prices) * np.where(df.loc[:, 'is_call'], 1, -1) > 0
    return df.loc[keep, :]


def compute_log_moneyness_ttm_scaled(forward_price, strike_price, ttm):
    return np.log(forward_price / strike_price) / np.sqrt(ttm)


def restrict_dataset(dt, forward_column_name='PCP_forward_mean'):
    # Compute moneyness
    moneyness_to_filter = np.log(dt[forward_column_name].to_numpy() / dt.strike_price.to_numpy()) / np.sqrt(
        dt.ttm.to_numpy())
    # Only keep the following: ttm > 6 days, bid > 3/8, bid-ask spread < 175%, ttm < 3 years
    is_OTM = (dt.is_call == 1) * (moneyness_to_filter < 0) + (dt.is_call == 0) * (
            moneyness_to_filter > 0) == 1
    ttm_larger_than_6_days = (dt.days_to_maturity > 6)
    bid_larger_than = (dt.best_bid > (3 / 8))
    bid_ask_spread_smaller = ((dt.best_offer - dt.best_bid) * 2 / (
            dt.best_offer + dt.best_bid)) < 1.75
    ttm_smaller_than_3years = dt.ttm < 3
    restrictions = (
                           is_OTM * ttm_larger_than_6_days * bid_larger_than * bid_ask_spread_smaller * ttm_smaller_than_3years) == 1
    dt_restricted = dt.loc[restrictions, :]

    return dt_restricted


def add_earning_dates(options_df: pd.DataFrame, path: str, stock: str) -> tuple[DataFrame, pd.Series]:
    """
    Add a column indicating if a date is an earnings date.

    Args:
        options_df (pd.DataFrame): Options dataset.
        path (str): Path to the earnings CSV.
        stock (str): Stock ticker.

    Returns:
        pd.DataFrame: Updated options dataset with 'is_earning_date' flag.
    """
    earning_df = pd.read_csv(
        f'{path}/earningDatesPerStock.csv',
        usecols=['datadate', 'rdq', 'tic', 'dvpspq', 'dvpsxq']
    )
    earning_df.rename(
        columns={
            'datadate': 'date',
            'tic': 'ticker',
            'rdq': 'earning_date',
            'dvpspq': 'is_div_pay_date',
            'dvpsxq': 'is_div_ex_date'
        },
        inplace=True
    )

    earning_dates = pd.to_datetime(
        earning_df.loc[earning_df.ticker == stock, 'earning_date'].unique()
    )
    earning_dates = earning_dates[earning_dates < '2023-08-31']

    options_df['is_earning_date'] = options_df['date'].isin(earning_dates).astype(float)
    return options_df, earning_dates


def compute_time_to_earnings(df: pd.DataFrame, earning_dates: pd.Series) -> pd.DataFrame:
    """
    Compute time in years until the next earnings date.

    Args:
        df (pd.DataFrame): Options dataset.
        earning_dates (pd.Series): Sorted earnings dates.

    Returns:
        pd.DataFrame: Updated dataset with 'time_to_earnings'.
    """
    earning_dates = earning_dates.sort_values()

    # Find index of the next earning date for each option date
    indices = np.searchsorted(earning_dates, df['date'], side='right')
    indices = np.clip(indices, 0, len(earning_dates) - 1)

    # Get the next earning date and calculate difference
    next_dates = earning_dates[indices].values
    df['time_to_earnings'] = (
            (next_dates - df['date'].values).astype('timedelta64[D]').astype(float) / 365
    )
    df.loc[df['is_earning_date'] == 1, 'time_to_earnings'] = 0

    return df


def split_dataset(df: pd.DataFrame, begin_train: str, end_train: str, begin_test: str):
    """
    Split dataset into train, validation, and test sets.

    Args:
        df (pd.DataFrame): Full dataset.
        begin_train (str): Start date for training.
        end_train (str): End date for training.
        begin_test (str): Start date for testing.

    Returns:
        tuple: (train, valid, test)
    """
    train = df[(df.date > begin_train) & (df.date < end_train)].copy()
    valid = df[(df.date > end_train) & (df.date < begin_test)].copy()
    test = df[df.date > begin_test].copy()
    return train, valid, test


def load_options_data(data_dir: Path, stock: str) -> pd.DataFrame:
    path = data_dir / "transformedData" / "stockDataClean.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["OM_IV"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.loc[df["ticker"] == stock]
    # safety checks
    req = {"ttm", "moneyness", "OM_IV", "date"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if (df["ttm"] <= 0).any() or (df["moneyness"] <= 0).any():
        logger.warning("Found non-positive ttm or moneyness; consider filtering these rows earlier.")
    return df


