from typing import List, Union, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


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


def batch_daily_observations_with_group_and_number_of_obs_per_group(batch, successive_groups: Optional[int] = None):
    groups = []
    num_obs_per_group = []
    full_features = []
    full_labels = []
    if successive_groups is None:
        for features, labels, group in batch:
            groups.append(group)
            num_obs_per_group.append(features.shape[0])
            full_features.append(features)
            full_labels.append(labels)

        return torch.cat(full_features, dim=0), torch.cat(full_labels, dim=0), torch.tensor(num_obs_per_group), groups
    else:
        features_per_successive = {key: [] for key in range(successive_groups)}
        labels_per_successive = {key: [] for key in range(successive_groups)}
        group_name_per_successive = {key: [] for key in range(successive_groups)}
        num_obs_per_group_per_successive = {key: [] for key in range(successive_groups)}
        for grouped_features, grouped_labels, grouped_group in batch:
            for key in range(successive_groups):
                group_name_per_successive[key].append(grouped_group[key])
                num_obs_per_group_per_successive[key].append(grouped_features[key].shape[0])
                features_per_successive[key].append(grouped_features[key])
                labels_per_successive[key].append(grouped_labels[key])

        features_per_successive = [torch.cat(features, dim=0) for features in features_per_successive.values()]
        labels_per_successive = [torch.cat(labels, dim=0) for labels in labels_per_successive.values()]
        num_obs_per_group_per_successive = [torch.tensor(num_obs_per_group) for num_obs_per_group in num_obs_per_group_per_successive.values()]
        group_name_per_successive = [group_name for group_name in group_name_per_successive.values()]

        return features_per_successive, labels_per_successive, num_obs_per_group_per_successive, group_name_per_successive

