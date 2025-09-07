from typing import Optional, Type, Callable

import torch
from torch import nn

from config import Config


class FeedForward_FrancoisEtAl(nn.Module):
    def __init__(self, T_max=5, T_conv=0.25, batch_norm=False):
        super(FeedForward_FrancoisEtAl, self).__init__()
        self.T_max = T_max
        self.T_conv = T_conv
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.factor_bn = nn.BatchNorm1d(5, affine=False)

    def forward(self, x: torch.Tensor):
        ttm, moneyness = torch.split(x[:, :2], 1, dim=-1)
        factors = torch.empty((*ttm.shape[:-1], 5))
        M = (torch.log(1 / moneyness) / torch.sqrt(ttm))

        factors[..., 0] = torch.ones_like(ttm[..., 0])
        factors[..., 1] = torch.exp(-torch.sqrt(ttm / self.T_conv)).squeeze(-1)
        factors[..., 2] = torch.where(M > 0, M, (torch.exp(2 * M) - 1) / (torch.exp(2 * M) + 1)).squeeze(-1)
        factors[..., 3] = ((1 - torch.exp(-M ** 2)) * torch.log(ttm / self.T_max)).squeeze(-1)
        factors[..., 4] = torch.where(M < 0, (1 - torch.exp((3 * M) ** 3)) * torch.log(ttm / self.T_max), 0).squeeze(-1)
        if self.batch_norm:
            factors = self.factor_bn(factors)
        return factors


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



class SplitDeepFactorNN(nn.Module):
    """
    SplitDeepFactorNN producing 5 factors as:
        [1, head1(moneyness)->1, head2(ttm,time_to_earnings)->1, head3(ttm,moneyness)->2]

    All heads share the *same* hyperparameters (num_neurons, num_layers, activation),
    as requested.

    Final per-head outputs are passed through BatchNorm1d with affine=False (no learnable
    scale/shift). The constant '1' is concatenated as the first factor.

    Expected input features (dim=3, feature order fixed):
        x[..., 0] = ttm
        x[..., 1] = moneyness
        x[..., 2] = time_to_earnings
    """
    def __init__(
        self,
        num_neurons: int,
        num_layers: int,
        activation_fn: Callable[..., nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        head_hidden_batchnorm: bool = False,
            L = None,
            **kwargs
    ):
        """
        Args:
            num_neurons: hidden width for each head
            num_layers: number of hidden layers for each head
            activation_fn: activation class for all heads (e.g., nn.ReLU)
            dropout: dropout prob inside heads
            head_hidden_batchnorm: if True, apply BN on hidden layers *inside* heads
                                   (independent of the final BN on head outputs)
        """
        super().__init__()

        if L is None:
            self.L = nn.Parameter(torch.eye(5),
                                  requires_grad=False)
        else:
            self.L = nn.Parameter(L)

        # Head 1: moneyness -> 1
        self.head1 = FeedforwardNN(
            input_shape=1,
            output_shape=1,
            num_neurons=num_neurons,
            num_layers=num_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            output_activation=None,
            batch_norm=head_hidden_batchnorm,
        )
        self.head1_bn = nn.BatchNorm1d(1, affine=False)

        # Head 2: (ttm, time_to_earnings) -> 1
        self.head2 = FeedforwardNN(
            input_shape=2,
            output_shape=1,
            num_neurons=num_neurons,
            num_layers=num_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            output_activation=None,
            batch_norm=head_hidden_batchnorm,
        )
        self.head2_bn = nn.BatchNorm1d(1, affine=False)

        # Head 3: (ttm, moneyness) -> 2
        self.head3 = FeedforwardNN(
            input_shape=2,
            output_shape=2,
            num_neurons=num_neurons,
            num_layers=num_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            output_activation=None,
            batch_norm=head_hidden_batchnorm,
        )
        self.head3_bn = nn.BatchNorm1d(2, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (..., 3) with features [ttm, moneyness, time_to_earnings]

        Returns:
            factors: tensor of shape (..., 5) = concat([1, out1, out2, out3]) along last dim
        """
        # Ensure we have a batch dim for BatchNorm
        squeeze_back = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_back = True

        # Split inputs by spec
        ttm = x[..., 0:1]                  # shape (N,1)
        moneyness = x[..., 1:2]            # shape (N,1)
        time_to_earn = x[..., 2:3]         # shape (N,1)

        # Head 1: moneyness -> 1
        h1 = self.head1(moneyness)         # (N,1)
        h1 = self.head1_bn(h1)

        # Head 2: (ttm, time_to_earnings) -> 1
        h2_in = torch.cat([ttm, time_to_earn], dim=-1)  # (N,2)
        h2 = self.head2(h2_in)             # (N,1)
        h2 = self.head2_bn(h2)

        # Head 3: (ttm, moneyness) -> 2
        h3_in = torch.cat([ttm, moneyness], dim=-1)     # (N,2)
        h3 = self.head3(h3_in)             # (N,2)
        h3 = self.head3_bn(h3)


        # Constant 1 factor
        ones = torch.ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)

        # Concatenate to 5 factors: [1, h1, h2, h3(2)]
        factors = torch.cat([ones, h2, h1, h3], dim=-1)  # (N,5)

        factors = factors @ self.L

        if squeeze_back:
            factors = factors.squeeze(0)
        return factors


def build_model(cfg: Config, device: torch.device) -> SplitDeepFactorNN:
    model = SplitDeepFactorNN(
        output_shape=cfg.num_factors,
        num_neurons=cfg.num_neurons,
        num_layers=cfg.num_layers,
        activation_fn=cfg.activation,
        dropout=cfg.dropout,
        head_hidden_batchnorm=cfg.batch_norm,
        add_level_as_factor=cfg.add_level_as_factor,
        shared_input_shape=cfg.shared_input_shape,
        input_ttm=cfg.input_ttm,
        out_batch_norm=cfg.out_batch_norm,
    )
    return model.to(device=device, dtype=cfg.dtype)





