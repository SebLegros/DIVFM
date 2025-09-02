from typing import List, Optional

import torch


def compute_betas_using_ols_from_factors_and_IV(factors: torch.Tensor, IV: torch.Tensor):
    return torch.linalg.lstsq(factors, IV).solution


def inverse_matrix(X: torch.Tensor):
    return torch.linalg.solve(X, torch.eye(X.shape[0]))


def predict_IV_from_factors_and_betas(factors: torch.Tensor, betas: torch.Tensor):
    return factors @ betas


def predict_IV_from_betas_factors_numObsPerGroup(betas, factors, num_obs_per_group):
    sub_tensors = torch.split(factors, split_size_or_sections=list(num_obs_per_group.numpy()), dim=0)
    return torch.cat([tensor @ betas[idx] for idx, tensor in enumerate(sub_tensors)], dim=0).unsqueeze(-1)


def get_optimal_betas_from_factors_IV_numObsPerGroup(factors: torch.Tensor, IV: torch.Tensor,
                                                     num_obs_per_group: Optional[List[int]] = None,
                                                     options_dropout=None):
    x = torch.cat([factors, IV], dim=-1)
    if num_obs_per_group is None:
        num_obs_per_group = torch.tensor([factors.shape[0]])
    sub_tensors = torch.split(x, split_size_or_sections=list(num_obs_per_group.numpy()), dim=0)
    if options_dropout is None:
        stacked = torch.stack(
            [compute_betas_using_ols_from_factors_and_IV(tensor[..., :-1], tensor[..., -1]) for tensor in sub_tensors],
            dim=0)
        if torch.isnan(stacked).any():
            print('stop')
        return stacked
    else:
        l = []
        for tensor in sub_tensors:
            N = tensor.shape[0]
            idx_keep = torch.randperm(N)[:max(int(N * (1 - options_dropout)), 6)]
            l.append(compute_betas_using_ols_from_factors_and_IV(tensor[idx_keep, :-1], tensor[idx_keep, -1]))

        return torch.stack(l, dim=0)


