import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.optionsDataset import DailyDataset, batch_daily_observations_with_group_and_number_of_obs_per_group
from utils.options_processing import keep_OTM


def import_data(path_to_database: str, stock: str):
    if stock == 'SPX':
        options_data = pd.read_csv(f'{path_to_database}/transformedData/PCP_IV.csv')
    else:
        options_data = pd.read_csv(f'{path_to_database}/transformedData/stockDataClean.csv')

    options_data.dropna(subset=['OM_IV'], inplace=True)
    options_data['date'] = pd.to_datetime(options_data['date'])

    if 'ticker' not in options_data.columns:
        options_data.loc[:, 'ticker'] = stock
    if stock == 'SPX':
        options_data.drop(columns=['OM_IV', 'OM_forward_price'], inplace=True)
        options_data.rename(columns={'PCP_IV_mid': 'OM_IV', 'PCP_forward_mean': 'forward_price'}, inplace=True)

    options_data.loc[:, 'LOG_OM_IV'] = np.log(options_data['OM_IV'])
    options_data = options_data.loc[options_data.ticker == stock, :]

    return options_data

def import_earnings_date(path_to_database:str, STOCK: str):
    earning_dates = pd.read_csv(f'{path_to_database}/earning_dates/earningDatesPerStock.csv', usecols=['datadate', 'rdq', 'tic' ])
    earning_dates.rename(columns={'datadate':'date', 'tic': 'ticker', 'rdq': 'earning_date'}, inplace=True)
    earning_dates = earning_dates.loc[earning_dates.ticker == STOCK, 'earning_date'].unique()
    earning_dates = pd.to_datetime(earning_dates)
    earning_dates = earning_dates[earning_dates < '2023-08-31']
    return earning_dates


def merge_options_earnings(options_data: pd.DataFrame, earning_dates: pd.DataFrame):
    options_data['is_earning_date'] = options_data['date'].isin(earning_dates).astype(float)
    options_data['date'] = pd.to_datetime(options_data['date'])
    return options_data

def apply_filters_to_options(options_data: pd.DataFrame):
    options_data = keep_OTM(options_data, options_data.loc[:, 'forward_price'])
    options_data = options_data.loc[options_data.loc[:, 'volume'] > 0]
    options_data = options_data.loc[options_data.loc[:, 'best_bid'] > 3 / 8]
    bid_ask_spread_smaller = ((options_data.best_offer - options_data.best_bid) * 2 / (
            options_data.best_offer + options_data.best_bid)) < 1.75
    options_data = options_data.loc[bid_ask_spread_smaller, :]
    options_data.loc[:, 'ttm_scaled_moneyness'] = np.log(options_data['moneyness']) / np.sqrt(options_data['ttm'])
    return options_data


def generate_tte(options_data:pd.DataFrame, earning_dates:pd.DataFrame):
    # Sort the earning_dates just to be safe
    earning_dates = earning_dates.sort_values()

    # Use searchsorted to find index of the next earning date for each option date
    indices = np.searchsorted(earning_dates, options_data['date'], side='right')

    # Clip indices to stay within bounds
    indices = np.clip(indices, 0, len(earning_dates) - 1)

    # Get the next earning date
    next_earning_dates = earning_dates[indices].values

    # Calculate the difference in days
    options_data['time_to_earnings'] = (next_earning_dates - options_data['date'].values).astype(
        'timedelta64[D]').astype(float) / 365
    options_data.loc[options_data['is_earning_date'] == 1, 'time_to_earnings'] = 0
    return options_data

def split_in_sets(options_data: pd.DataFrame, begin_train_date: str, end_train_date: str, begin_test_date: str):
    train_data = options_data[(options_data.date > begin_train_date) & (options_data.date < end_train_date)].copy()
    valid_data = options_data[(options_data.date > end_train_date) & (options_data.date < begin_test_date)].copy()
    test_data = options_data[(options_data.date > begin_test_date)].copy()

    return train_data, valid_data, test_data

def get_datasets(train_data, valid_data, test_data, features_name, labels_name, dtype=torch.float32):
    train_dataset = DailyDataset(train_data, features_name=features_name, labels_name=labels_name, group_by='date',
                                 dtype=dtype, return_group=True)

    valid_dataset = DailyDataset(valid_data, features_name=features_name, labels_name=labels_name, group_by='date',
                                 dtype=dtype, return_group=True)

    test_dataset = DailyDataset(test_data, features_name=features_name, labels_name=labels_name, group_by='date',
                                dtype=dtype, return_group=True)
    return train_dataset, valid_dataset, test_dataset
def get_dataloader(train_dataset, valid_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=batch_daily_observations_with_group_and_number_of_obs_per_group)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=batch_daily_observations_with_group_and_number_of_obs_per_group)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=batch_daily_observations_with_group_and_number_of_obs_per_group)

    return train_loader, valid_loader, test_loader

