import torch
from torch import nn

__all__ = [
    "FullyConnected",
    "ConcatSquash",
]


def _resolve_activation(activation: type[nn.Module] | str = nn.ReLU) -> type[nn.Module]:
    """Resolve activation from string name or class."""
    if isinstance(activation, str):
        if not hasattr(nn, activation):
            raise ValueError(f"Unknown activation: {activation}")
        return getattr(nn, activation)
    return activation


class FullyConnected(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_condition: int = 0,
        dim_time: int = 0,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] | str = nn.ReLU,
    ) -> None:
        super().__init__()
        activation = _resolve_activation(activation)
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
            layer_norm: bool = False,
        ) -> None:
            super().__init__()
            self.cond_embed = nn.Sequential(
                nn.Linear(dim_cond, dim_cond * 2),
                activation(),
                nn.Linear(dim_cond * 2, dim_input),
            )
            layers = []
            if layer_norm:
                layers.append(nn.LayerNorm(dim_input))
            layers.append(nn.Linear(dim_input, dim_output))
            layers.append(activation())
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            cond_embed = self.cond_embed(condition)
            return self.network(x + cond_embed)

    def __init__(
        self,
        dim_input: int,
        dim_condition: int = 0,
        dim_time: int = 0,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] | str = nn.ReLU,
        residual: bool = False,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        activation = _resolve_activation(activation)
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self.residual = residual

        layers = []
        prev_dim = dim_input
        dim_cond_time = dim_condition + dim_time
        for dim in hidden_dims:
            layers.append(
                self.__ConcatSquashLayer(
                    prev_dim, dim, dim_cond_time, activation, layer_norm,
                )
            )
            prev_dim = dim
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(prev_dim, dim_input)

        # Precompute residual pairs: connect layers with matching dimensions
        # e.g. [256, 512, 1024, 512, 256] → pairs [(0,4), (1,3)]
        self._residual_targets: dict[int, int] = {}
        if residual:
            dims = [dim_input] + hidden_dims
            n = len(dims)
            left = 0
            right = n - 1
            while left < right:
                # dims[left] is input to layer[left], dims[right] is output of layer[right-1]
                if dims[left] == dims[right]:
                    # layer[right-1] output += layer[left] input (skip connection)
                    self._residual_targets[right - 1] = left
                    left += 1
                    right -= 1
                else:
                    break

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = t if condition is None else torch.cat([t, condition], dim=-1)

        if self.residual and self._residual_targets:
            # Store intermediate activations for skip connections
            intermediates: dict[int, torch.Tensor] = {}
            # dims[i] = input to layer i; dims[0] = x before any layer
            intermediates[0] = x
            for i, layer in enumerate(self.layers):
                x = layer(x, condition)
                intermediates[i + 1] = x
                if i in self._residual_targets:
                    src = self._residual_targets[i]
                    x = x + intermediates[src]
        else:
            for layer in self.layers:
                x = layer(x, condition)

        return self.output(x)
