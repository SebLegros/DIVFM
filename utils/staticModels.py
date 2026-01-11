from typing import Optional, List, Callable, Tuple

import numpy as np
from torch import nn, Tensor
import torch
from typing import Optional


class FeedforwardNN(nn.Module):
    def __init__(self, input_shape, output_shape, num_neurons, num_layers, activation_fn=nn.ReLU, dropout=0.1,
                 output_activation: Optional[nn.Module] = None, batch_norm=False):
        """
        Initialize the Feedforward Neural Network.

        :param input_shape: int, the number of input features
        :param output_shape: int, the number of output features
        :param num_neurons: int, the number of neurons in each hidden layer
        :param num_layers: int, the number of hidden layers
        :param activation_fn: PyTorch activation function (default is nn.ReLU)
        :param output_activation: PyTorch output activation function (default is nn.Identity)
        """
        super(FeedforwardNN, self).__init__()
        self.first_layer = nn.Linear(input_shape, num_neurons)
        self.layers = nn.ModuleList(
            [nn.Linear(num_neurons, num_neurons) for _ in range(num_layers)]
        )
        self.out_layer = nn.Linear(num_neurons, output_shape)
        self.activation = activation_fn()
        self.dropout = nn.Dropout(dropout)
        self.output_activation = output_activation() if output_activation is not None else nn.Identity()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.ModuleList([nn.BatchNorm1d(num_neurons) for _ in range(num_layers)])

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: input tensor
        :return: output tensor
        """
        has_no_batch = False
        x = self.first_layer(x)
        x = self.activation(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norm:
                if len(x.shape) == 1:
                    has_no_batch = True
                    x = x.unsqueeze(0)
                x = self.bn[i](x)
                if has_no_batch:
                    x = x.squeeze(0)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.out_layer(x)
        x = self.output_activation(x)
        return x


class SplitFeedforwardNN(nn.Module):
    def __init__(self, output_shape, num_neurons, num_layers,
                 shared_outputs, input1_outputs, input2_outputs, activation_fn=nn.ReLU, dropout=0.1,
                 output_activation: Optional[nn.Module] = None, batch_norm=False, input_shape=2, ttm_input=1):
        """
        A feedforward neural network with two inputs and configurable output dependencies.

        :param input1_shape: int, the number of features in the first input
        :param input2_shape: int, the number of features in the second input
        :param output_shape: int, total number of output features
        :param num_neurons: int, number of neurons in each hidden layer
        :param num_layers: int, number of hidden layers in each network
        :param shared_outputs: int, number of outputs that depend on both inputs
        :param input1_outputs: int, number of outputs that depend only on the first input
        :param input2_outputs: int, number of outputs that depend only on the second input
        :param activation_fn: PyTorch activation function (default is nn.ReLU)
        :param dropout: float, dropout probability
        :param output_activation: PyTorch output activation function (default is nn.Identity)
        :param batch_norm: bool, whether to use batch normalization
        """
        super(SplitFeedforwardNN, self).__init__()
        self.input_shape = input_shape
        self.ttm_input = ttm_input
        self.batch_norm = batch_norm

        self.input1_nn = FeedforwardNN(ttm_input, input1_outputs, num_neurons, num_layers, activation_fn, dropout,
                                       output_activation, batch_norm)
        self.input2_nn = FeedforwardNN(1, input2_outputs, num_neurons, num_layers, activation_fn, dropout,
                                       output_activation, batch_norm)
        self.shared_nn = FeedforwardNN(self.input_shape, shared_outputs, num_neurons, num_layers,
                                       activation_fn, dropout, output_activation, batch_norm)

    def forward(self, x):
        """
        Forward pass through the network.

        :param x1: Tensor, first input
        :param x2: Tensor, second input
        :return: Concatenated output tensor
        """

        ttm = x[..., :1]
        M = torch.log(x[..., 1:2]) / torch.sqrt(ttm)

        shared_out = self.shared_nn(torch.cat([ttm, M] if self.input_shape == 2 else [ttm, M, x[:, 2:]], dim=-1))
        input1_out = self.input1_nn(ttm if self.ttm_input == 1 else torch.cat([ttm, x[..., 2:]], dim=-1))
        input2_out = self.input2_nn(M)

        out = torch.cat([input1_out, input2_out, shared_out], dim=-1)

        return out


class SplitFeedforwardNNList(nn.Module):
    def __init__(self, input_size: List[int], output_size: List[int], num_neurons: int, num_layers: int,
                 activation_fn=nn.ReLU, dropout=0.1, output_activation: Optional[nn.Module] = None, batch_norm=False):
        super().__init__()
        self.input_shape = input_size
        self.output_shape = output_size
        self.batch_norm = batch_norm
        self.x_dim = len(input_size)
        self.output_shape = np.sum(output_size)

        self.modules_list = nn.ModuleList(
            [FeedforwardNN(in_size, out_size, num_neurons, num_layers, activation_fn, dropout,
                           output_activation, batch_norm) for in_size, out_size in
             zip(input_size, output_size)])

    def forward(self, x: List[Tensor]):
        assert len(x) == self.x_dim

        outs = [m(xi) for m, xi in zip(self.modules_list, x)]

        return torch.cat(outs, dim=-1)


class SplitDeepFactorNN(SplitFeedforwardNN):
    def __init__(self, output_shape, num_neurons, num_layers, activation_fn=nn.ReLU, dropout=0.1,
                 output_activation: Optional[nn.Module] = None, batch_norm: bool = False,
                 add_level_as_factor: bool = False, add_ttm_slope_as_factor: bool = False, L=None, input_shape=2,
                 input_ttm=1, out_batch_norm=False):
        if input_shape == 3:
            super().__init__(output_shape, num_neurons, num_layers, output_shape - 2, 1, 1, activation_fn, dropout,
                             output_activation, batch_norm, input_shape=input_shape, ttm_input=input_ttm)
        else:
            super().__init__(output_shape, num_neurons, num_layers, output_shape - 2, 1, 1, activation_fn, dropout,
                             output_activation, batch_norm, input_shape=input_shape, ttm_input=input_ttm)
        self.input_shape = input_shape
        self.add_level_as_factor = add_level_as_factor
        self.add_ttm_slope_as_factor = add_ttm_slope_as_factor
        self.log_T_conv = torch.nn.Parameter(torch.log(torch.tensor([0.25])))
        self.out_batch_norm = out_batch_norm
        if out_batch_norm:
            self.factor_bn = nn.BatchNorm1d(output_shape, affine=False)
        if L is None:
            self.L = torch.eye(output_shape + add_level_as_factor + add_ttm_slope_as_factor)
        else:
            self.L = L

    def forward(self, x):
        x_out = super().forward(x)
        if self.out_batch_norm:
            x_out = self.factor_bn(x_out)
        given_factors = torch.empty((x.shape[0], 0))
        if self.add_level_as_factor:
            given_factors = torch.cat((given_factors, torch.ones([x.shape[0], 1])), dim=-1)
        if self.add_ttm_slope_as_factor:
            given_factors = torch.cat(
                (given_factors, torch.exp(-torch.sqrt(x[:, 0] / torch.exp(self.log_T_conv))).unsqueeze(-1)),
                dim=-1)

        x_out = torch.cat((given_factors, x_out), dim=-1)
        return x_out @ self.L


class SplitDeepFactorNNList(nn.Module):
    def __init__(self, splitFeedforwardNNList: SplitFeedforwardNNList, factors_inputs_index: List[List[int]],
                 L=None, out_batch_norm=False, output_activation: Optional[Callable] = None):

        super().__init__()
        self.factors_inputs_index = factors_inputs_index
        self.split_model = splitFeedforwardNNList
        self.add_level_as_factor = True
        self.out_batch_norm = out_batch_norm
        self.feedforward_output_shape = self.split_model.output_shape
        if self.out_batch_norm:
            self.factor_bn = nn.BatchNorm1d(self.feedforward_output_shape, affine=False)
        self.L = L
        self.output_activation = output_activation

    def get_factors(self, x) -> Tensor:
        Bsz = x.shape[0]
        x_list = [x[..., idx] for idx in self.factors_inputs_index]
        x_out = self.split_model(x_list)
        if self.out_batch_norm:
            x_out = self.factor_bn(x_out)
        level = torch.ones((Bsz, 1), device=x.device)
        x_out = torch.cat((level, x_out), dim=-1)
        if self.L is None:
            return x_out
        else:
            return x_out @ self.L

    def get_betas(self, factors, y, num_obs_per_group) -> Tensor:
        x = torch.cat([factors, y], dim=-1)
        sub_tensors = torch.split(x, split_size_or_sections=list(num_obs_per_group.numpy()), dim=0)
        betas = torch.stack(
            [torch.linalg.lstsq(tensor[..., :-1], tensor[..., -1]).solution for tensor in sub_tensors],
            dim=0)  # shape: (B, n)
        return betas

    def get_predictions(self, factors, betas, num_obs_per_group) -> Tensor:
        betas_rep = torch.repeat_interleave(betas, num_obs_per_group, dim=0)  # shape (N, n)
        x_out = (factors * betas_rep).sum(-1, keepdims=True)
        if self.output_activation is not None:
            x_out = self.output_activation(x_out)
        return x_out

    def forward(self, x, y, num_obs_per_group) -> Tuple[Tensor, Tensor, Tensor]:
        factors = self.get_factors(x)
        betas = self.get_betas(factors, y, num_obs_per_group)
        x_out = self.get_predictions(factors, betas, num_obs_per_group)

        return factors, betas, x_out


class SplitFrancois(SplitDeepFactorNNList):
    def __init__(self, moneyness_idx, ttm_idx):
        nn.Module.__init__(self)
        self.moneyness_idx = moneyness_idx
        self.ttm_idx = ttm_idx
        self.T_max_raw = torch.nn.Parameter(self.softplus_inverse(torch.tensor(5.0, dtype=torch.float32)),
                                            requires_grad=False)
        self.T_conv_raw = torch.nn.Parameter(self.softplus_inverse(torch.tensor(0.25, dtype=torch.float32)),
                                             requires_grad=True)
        self.output_activation = None

    @staticmethod
    def softplus_inverse(y):
        return torch.log(torch.expm1(y))

    @property
    def T_max(self):
        # positive, differentiable w.r.t. T_max_raw
        return torch.nn.functional.softplus(self.T_max_raw)

    @property
    def T_conv(self):
        return torch.nn.functional.softplus(self.T_conv_raw)

    def get_factors(self, x) -> Tensor:
        M = -x[:, self.moneyness_idx].unsqueeze(-1)
        ttm = x[:, self.ttm_idx].unsqueeze(-1)

        x_out = torch.empty((*ttm.shape[:-1], 5))

        x_out[..., 0] = torch.ones_like(ttm[..., 0])
        x_out[..., 1] = torch.exp(-torch.sqrt(ttm / self.T_conv)).squeeze(-1)
        x_out[..., 2] = torch.where(M > 0, M, (torch.exp(2 * M) - 1) / (torch.exp(2 * M) + 1)).squeeze(-1)
        x_out[..., 3] = ((1 - torch.exp(-M ** 2)) * torch.log(ttm / self.T_max)).squeeze(-1)
        x_out[..., 4] = torch.where(M < 0, (1 - torch.exp((3 * M) ** 3)) * torch.log(ttm / self.T_max), 0).squeeze(-1)
        return x_out

    def get_betas(self, factors, y, num_obs_per_group) -> Tensor:
        return super().get_betas(factors, torch.exp(y), num_obs_per_group)

    def get_predictions(self, factors, betas, num_obs_per_group) -> Tensor:
        return torch.log(torch.clamp(super().get_predictions(factors, betas, num_obs_per_group), torch.tensor(1e-12)))
