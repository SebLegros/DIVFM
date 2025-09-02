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


class SplitFeedforwardNN(nn.Module):
    """
    A three-headed MLP that splits the feature processing into:
      1) a TTM-only head
      2) an M-only head, where M = log(moneyness) / sqrt(TTM)
      3) a shared head that sees [TTM, M] and (optionally) extra features

    Final output is the concatenation: [input1_out, input2_out, shared_out],
    whose total width must equal `output_shape`.

    ASCII FLOW (feature routing and concatenation)
    ----------------------------------------------
    Inputs per sample (x):
        [ TTM | moneyness | extra... ]
          x[...,0]  x[...,1]   x[...,2:]

    Compute scaled moneyness:
        M = log(moneyness) / sqrt(TTM)

                              ┌───────────────────────┐
                              │   Shared head (Hs)    │
                              │   in: [TTM, M] or     │
                              │        [TTM, M, extra]│
                              └───────────────────────┘
                                        │
                                        ▼
                              shared_out ∈ ℝ^{k-2}

    ┌───────────────────────┐                       ┌───────────────────────┐
    │   TTM head (H1)       │                       │    M head (H2)        │
    │   in: TTM or          │                       │    in: M              │
    │       [TTM, extra]    │                       └───────────────────────┘
    └───────────────────────┘                                   │
                │                                               ▼
                ▼                                      input2_out ∈ ℝ^{1}
      input1_out ∈ ℝ^{1}

                             ┌─────────────────────────────────────────────┐
                             │   concat along channel dim: [H1, H2, Hs]    │
                             └─────────────────────────────────────────────┘
                                                   │
                                                   ▼
                                             out ∈ ℝ^{k}

    Requirements / assumptions:
    - TTM > 0 and moneyness > 0 (to avoid log(0) / divide-by-zero).
    - input_shape must match what you pass to the shared head in forward().
    - output_shape = shared_outputs + input1_outputs + input2_outputs.

    Inputs
    ------
    x : Tensor of shape [..., F]
        Expected first two columns:
          - x[..., 0] = TTM (must be > 0)
          - x[..., 1] = moneyness (must be > 0)
        Optional extra columns x[..., 2:] if `input_shape == 3` and your
        FeedforwardNN is sized accordingly.

    Parameters
    ----------
    output_shape : int
        Final output width after concatenation of the three heads.
    num_neurons : int
        Hidden width of each head MLP.
    num_layers : int
        Number of hidden layers in each head MLP.
    shared_outputs : int
        Output width of the shared head.
    input1_outputs : int
        Output width of the TTM-only head.
    input2_outputs : int
        Output width of the M-only head.
    activation_fn : nn.Module type, default nn.ReLU
        Activation for hidden layers.
    dropout : float
        Dropout probability for hidden layers.
    output_activation : nn.Module or None
        Optional activation applied to head outputs.
    batch_norm : bool
        Whether to use BatchNorm within the FeedforwardNN heads.
    shared_input_shape : int, {2, 3}
        Number of input features fed into the shared head:
          2 -> [TTM, M]
          3 -> [TTM, M, extra_features]  (assumes x[..., 2:] matches what your
                                          FeedforwardNN expects!)
    ttm_input : int, {1, >1}
        Dimensionality of input to the TTM head:
          1 -> TTM only
          >1 -> [TTM, extra_features] (again assumes x[..., 2:] exists)
    """

    def __init__(
            self,
            output_shape: int,
            num_neurons: int,
            num_layers: int,
            shared_outputs: int,
            input1_outputs: int,
            input2_outputs: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            dropout: float = 0.1,
            output_activation: Optional[nn.Module] = None,
            batch_norm: bool = False,
            shared_input_shape: int = 2,
            ttm_input: int = 1,
    ) -> None:
        super().__init__()

        # Correctness check: final concat size must equal output_shape
        assert output_shape == (shared_outputs + input1_outputs + input2_outputs), (
            "output_shape must equal shared_outputs + input1_outputs + input2_outputs"
        )
        assert shared_input_shape in (2, 3), "input_shape must be 2 or 3"
        assert ttm_input >= 1, "ttm_input must be >= 1"

        self.output_shape = output_shape
        self.input_shape = shared_input_shape
        self.ttm_input = ttm_input
        self.batch_norm = batch_norm

        # Head 1: TTM-only (or TTM + extra features if ttm_input > 1)
        self.input1_nn = FeedforwardNN(
            input_shape=ttm_input,
            output_shape=input1_outputs,
            num_neurons=num_neurons,
            num_layers=num_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            output_activation=output_activation,
            batch_norm=batch_norm,
        )

        # Head 2: M-only
        self.input2_nn = FeedforwardNN(
            input_shape=1,  # only M
            output_shape=input2_outputs,
            num_neurons=num_neurons,
            num_layers=num_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            output_activation=output_activation,
            batch_norm=batch_norm,
        )

        # Shared head: [TTM, M] (and optional extra features)
        # IMPORTANT: `input_shape` must match the number of features you actually feed
        # into this head in forward(). If you plan to pass [TTM, M, x[..., 2:]],
        # make sure your FeedforwardNN in_features matches that dimensionality.
        self.shared_nn = FeedforwardNN(
            input_shape=self.input_shape,
            output_shape=shared_outputs,
            num_neurons=num_neurons,
            num_layers=num_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            output_activation=output_activation,
            batch_norm=batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor, shape [..., F]
            x[..., 0] = TTM > 0
            x[..., 1] = moneyness > 0
            Optional extra features x[..., 2:] if configured.

        Returns
        -------
        Tensor, shape [..., output_shape]
        """
        # Split basic inputs
        ttm = x[..., :1]  # shape [..., 1]
        moneyness = x[..., 1:2]  # shape [..., 1]

        # Build scaled moneyness M = log(moneyness) / sqrt(ttm)
        # NOTE: caller must ensure positive values to avoid NaNs/infs.
        M = torch.log(moneyness) / torch.sqrt(ttm)

        # Shared head input:
        #   if input_shape == 2 -> [TTM, M]
        #   if input_shape == 3 -> [TTM, M, x[..., 2:]]  (assumes shapes match self.shared_nn)
        shared_in = torch.cat([ttm, M], dim=-1)
        if self.input_shape == 3:
            shared_in = torch.cat([shared_in, x[..., 2:]], dim=-1)

        shared_out = self.shared_nn(shared_in)

        # TTM head input:
        #   if ttm_input == 1 -> TTM only
        #   else -> [TTM, x[..., 2:]]
        if self.ttm_input == 1:
            ttm_in = ttm
        else:
            ttm_in = torch.cat([ttm, x[..., 2:]], dim=-1)
        input1_out = self.input1_nn(ttm_in)

        # M head input is just M
        input2_out = self.input2_nn(M)

        # Concatenate in a fixed order
        out = torch.cat([input1_out, input2_out, shared_out], dim=-1)  # [..., output_shape]
        return out


# class SplitDeepFactorNN(SplitFeedforwardNN):
#     """
#     Extends SplitFeedforwardNN by:
#       - Treating the concatenated head outputs as a factor vector.
#       - Optionally adding hand-crafted factors:
#           * A constant "level" factor (column of ones).
#           * A TTM "slope" factor: exp(-sqrt(TTM / T)), where T is a learnable scalar.
#       - Optionally normalizing factor vector with BatchNorm (affine=False).
#       - Applying a linear mixing via a factor loading matrix L at the end.
#
#     Final output = concat([given_factors, model_factors]) @ L
#
#     Notes
#     -----
#     - If you enable extra factors, your output width becomes (output_shape + num_given_factors)
#       *before* multiplying by L. If you want the final width to still be `output_shape`,
#       supply an L of shape [(output_shape + num_given_factors) x output_shape].
#
#     - `L` is only needed if you want to:
#         1. Perform a **linear combination** of the factors to produce your final outputs.
#         2. **Swap or reorder** the factors after concatenation.
#       If you don’t need to mix/reorder, `L` can just be the identity matrix of the
#       appropriate size, in which case the output is exactly the factor vector (plus
#       any given factors).
#     """
#
#     def __init__(
#             self,
#             output_shape: int,
#             num_neurons: int,
#             num_layers: int,
#             activation_fn: Type[nn.Module] = nn.ReLU,
#             dropout: float = 0.1,
#             output_activation: Optional[nn.Module] = None,
#             batch_norm: bool = False,
#             add_level_as_factor: bool = False,
#             add_ttm_slope_as_factor: bool = False,
#             L: Optional[torch.Tensor] = None,
#             shared_input_shape: int = 2,
#             input_ttm: int = 1,
#             out_batch_norm: bool = True,
#     ) -> None:
#         # We enforce the split: shared gives (output_shape - 2), two 1-D dedicated heads.
#         super().__init__(
#             output_shape=output_shape,
#             num_neurons=num_neurons,
#             num_layers=num_layers,
#             shared_outputs=output_shape - 2,
#             input1_outputs=1,
#             input2_outputs=1,
#             activation_fn=activation_fn,
#             dropout=dropout,
#             output_activation=output_activation,
#             batch_norm=batch_norm,
#             shared_input_shape=shared_input_shape,
#             ttm_input=input_ttm,
#         )
#
#         self.input_shape = shared_input_shape
#         self.add_level_as_factor = add_level_as_factor
#         self.add_ttm_slope_as_factor = add_ttm_slope_as_factor
#
#         # Learnable T used inside exp(-sqrt(TTM / T)). Stored as log(T) for positivity.
#         self.log_T_conv = nn.Parameter(torch.log(torch.tensor([0.25])))
#
#         # Optional BN on the factor vector; affine=False to avoid scaling/shifting
#         self.out_batch_norm = out_batch_norm
#         if out_batch_norm:
#             self.factor_bn = nn.BatchNorm1d(output_shape, affine=False)
#
#         # Factor loading matrix L (applied after adding given factors)
#         # If not provided, default to Identity with appropriate width.
#         extra = (1 if add_level_as_factor else 0) + (1 if add_ttm_slope_as_factor else 0)
#         if L is None:
#             self.L = nn.Parameter(torch.eye(output_shape + extra),
#                                   requires_grad=False)
#         else:
#             self.L = nn.Parameter(L)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Parameters
#         ----------
#         x : Tensor, shape [..., F]
#             Same assumptions as the parent class:
#               x[..., 0] = TTM > 0
#               x[..., 1] = moneyness > 0
#
#         Returns
#         -------
#         Tensor
#             If no given factors: shape [..., output_shape]
#             If given factors added: shape [..., output_shape + num_given_factors] @ L
#             (final width depends on L’s second dimension)
#         """
#         # Run parent: produces a factor vector of size `output_shape`
#         x_out = super().forward(x)  # [..., output_shape]
#
#         # Optionally normalize factors across the batch
#         if self.out_batch_norm:
#             x_out = self.factor_bn(x_out)
#
#         # Build optional given factors on the same device/dtype
#         device = x_out.device
#         dtype = x_out.dtype
#         B = x.shape[0]
#
#         given = []
#         if self.add_level_as_factor:
#             # Column of ones
#             given.append(torch.ones((B, 1), device=device, dtype=dtype))
#
#         if self.add_ttm_slope_as_factor:
#             # exp(-sqrt(TTM / T)), where T = exp(log_T_conv) > 0
#             T = torch.exp(self.log_T_conv)  # scalar > 0
#             ttm = x[..., 0:1]  # shape [B, 1]
#             slope = torch.exp(-torch.sqrt(ttm / T))
#             given.append(slope)
#
#         # Concatenate given factors (if any) in front of model factors
#         if given:
#             given_factors = torch.cat(given, dim=-1)  # [B, g]
#             x_out = torch.cat([given_factors, x_out], dim=-1)  # [B, g + output_shape]
#
#         # Apply linear mixing with L
#         # Ensure L matches the current device/dtype and width
#         # L = self.L.to(device=device, dtype=dtype)
#
#         # x_out @ L  ->  last-dim matrix multiply
#         return x_out @ self.L

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





