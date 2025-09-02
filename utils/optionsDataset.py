from typing import List, Union, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config
from utils.options_processing import split_dataset


class DailyDataset(Dataset):
    def __init__(self, data: pd.DataFrame, features_name: List[str], labels_name: List[str],
                 group_by: Union[str, List[str]] = 'date', dtype=torch.float32, return_group: bool = True,
                 successive_groups: Optional[int] = None):
        self.data = data.sort_values(by='date')
        self.gb = self.data.groupby(group_by)
        self.groups = list(self.gb.groups.keys())
        self.features_name = features_name
        self.labels_name = labels_name
        self.dtype = dtype
        self.return_group = return_group

        self.return_list = successive_groups is not None
        self.successive_groups = successive_groups if successive_groups else 1

    def __len__(self):
        return len(self.groups) - (self.successive_groups - 1)

    def __getitem__(self, idx):

        # Get the data for a specific day
        features = []
        labels = []
        groups = []
        for i in range(self.successive_groups):
            group_to_get = self.groups[idx + i]
            date_df = self.gb.get_group(group_to_get)
            features.append(torch.tensor(date_df.loc[:, self.features_name].values, dtype=self.dtype, device='cpu'))
            labels.append(torch.tensor(date_df.loc[:, self.labels_name].values, dtype=self.dtype, device='cpu'))
            groups.append(group_to_get)

        if not self.return_list:
            features = features[0]
            labels = labels[0]
            groups = groups[0]

        if self.return_group:
            return features, labels, groups
        else:
            return features, labels


def batch_daily_observations_with_group_and_number_of_obs_per_group(batch):
    groups = []
    num_obs_per_group = []
    full_features = []
    full_labels = []
    for features, labels, group in batch:
        groups.append(group)
        num_obs_per_group.append(features.shape[0])
        full_features.append(features)
        full_labels.append(labels)

    return torch.cat(full_features, dim=0), torch.cat(full_labels, dim=0), torch.tensor(num_obs_per_group), groups


def make_datasets(
        df: pd.DataFrame, cfg: Config
) -> tuple[DailyDataset, DailyDataset, DailyDataset]:
    train_df, valid_df, test_df = split_dataset(
        df, cfg.begin_train_date, cfg.begin_valid_date, cfg.begin_test_date
    )
    train_ds = DailyDataset(
        train_df,
        features_name=list(cfg.features),
        labels_name=list(cfg.labels),
        group_by="date",
        dtype=cfg.dtype,
        return_group=True,
        successive_groups=None,
    )
    valid_ds = DailyDataset(
        valid_df,
        features_name=list(cfg.features),
        labels_name=list(cfg.labels),
        group_by="date",
        dtype=cfg.dtype,
        return_group=True,
    )
    test_ds = DailyDataset(
        test_df,
        features_name=list(cfg.features),
        labels_name=list(cfg.labels),
        group_by="date",
        dtype=cfg.dtype,
        return_group=True,
    )
    return train_ds, valid_ds, test_ds

def make_loaders(
    train_ds: DailyDataset, valid_ds: DailyDataset, test_ds: DailyDataset, cfg: Config, device: torch.device
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # pin_memory if using CUDA for faster host->device transfers
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=batch_daily_observations_with_group_and_number_of_obs_per_group,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=batch_daily_observations_with_group_and_number_of_obs_per_group,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=batch_daily_observations_with_group_and_number_of_obs_per_group,
    )
    return train_loader, valid_loader, test_loader
