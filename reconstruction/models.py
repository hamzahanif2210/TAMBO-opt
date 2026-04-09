import math

import torch
from torch import nn

__all__ = [
    "FullyConnected",
    "ConcatSquash",
    "Transformer",
]


class FullyConnected(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_condition: int = 0,
        dim_time: int = 0,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        layers = []
        prev_dim = dim_input + dim_condition + dim_time
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, dim_input))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input = [t, x]
        if condition is not None:
            input.append(condition)
        input = torch.cat(input, dim=-1)
        return self.network(input)


class ConcatSquash(nn.Module):
    class __ConcatSquashLayer(nn.Module):
        def __init__(
            self,
            dim_input: int,
            dim_output: int,
            dim_cond: int,
            activation: type[nn.Module],
        ) -> None:
            super().__init__()
            self.cond_embed = nn.Sequential(
                nn.Linear(dim_cond, dim_input),
            )
            self.network = nn.Sequential(
                nn.Linear(dim_input, dim_output),
                activation(),
            )

        def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            cond_embed = self.cond_embed(condition)
            return self.network(x + cond_embed)

    def __init__(
        self,
        dim_input: int,
        dim_condition: int = 0,
        dim_time: int = 0,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        layers = []
        prev_dim = dim_input
        dim_cond_time = dim_condition + dim_time
        for dim in hidden_dims:
            layers.append(
                self.__ConcatSquashLayer(prev_dim, dim, dim_cond_time, activation)
            )
            prev_dim = dim
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(prev_dim, dim_input)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = t if condition is None else torch.cat([t, condition], dim=-1)
        for layer in self.layers:
            x = layer(x, condition)
        return self.output(x)


class Transformer(nn.Module):
    """Transformer-based velocity field for reconstruction flow matching.

    Projects condition + time + x into a sequence of tokens, processes
    through standard transformer encoder blocks with self-attention,
    and reads out the velocity prediction.

    Architecture:
        1. Condition → linear → (batch, num_tokens, d_model)  [tokenize condition]
        2. Time + x  → linear → (batch, 1, d_model)           [query token]
        3. Concatenate → (batch, num_tokens + 1, d_model)
        4. Transformer encoder (self-attention)
        5. Read out query token → linear → dim_input

    Parameters
    ----------
    dim_input : int
        Dimension of x (and output velocity), e.g. 5.
    dim_condition : int
        Dimension of raw condition vector, e.g. 72.
    dim_time : int
        Dimension of Fourier time embedding from CNF, e.g. 6.
    d_model : int
        Hidden dimension of transformer.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dim_feedforward : int
        Feedforward dimension inside each transformer layer.
    dropout : float
        Dropout rate.
    num_tokens : int
        Number of condition tokens (condition is chunked into this many tokens).
    activation : str
        Activation function for transformer feedforward ("relu" or "gelu").
    """

    def __init__(
        self,
        dim_input: int,
        dim_condition: int = 0,
        dim_time: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        num_tokens: int = 8,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.dim_input = dim_input
        self.d_model = d_model
        self.num_tokens = num_tokens

        # Project condition into num_tokens tokens
        self.cond_proj = nn.Linear(dim_condition, num_tokens * d_model)

        # Project time + x into a single query token
        self.query_proj = nn.Linear(dim_time + dim_input, d_model)

        # Learnable positional encoding for (num_tokens + 1) positions
        self.pos_encoding = nn.Parameter(
            torch.randn(1, num_tokens + 1, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)

        # Output projection: read from query token
        self.output_proj = nn.Linear(d_model, dim_input)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        # Build condition tokens: (batch, num_tokens, d_model)
        if condition is not None:
            cond_tokens = self.cond_proj(condition)
            cond_tokens = cond_tokens.reshape(batch_size, self.num_tokens, self.d_model)
        else:
            cond_tokens = torch.zeros(
                batch_size, self.num_tokens, self.d_model,
                device=x.device, dtype=x.dtype,
            )

        # Build query token from time + x: (batch, 1, d_model)
        query_input = torch.cat([t, x], dim=-1)
        query_token = self.query_proj(query_input).unsqueeze(1)

        # Assemble full sequence: [cond_tokens, query_token]
        tokens = torch.cat([cond_tokens, query_token], dim=1)  # (B, T+1, d_model)
        tokens = tokens + self.pos_encoding

        # Transformer encoder
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        # Read out from the query token (last position)
        query_out = tokens[:, -1, :]  # (batch, d_model)
        return self.output_proj(query_out)  # (batch, dim_input)
