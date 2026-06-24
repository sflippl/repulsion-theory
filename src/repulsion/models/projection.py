"""Fixed (frozen) random projection layer."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from repulsion.models.activations import build_activation


class RandomProjection(nn.Module):
    """Frozen random linear projection followed by an activation.

    Weights are drawn from N(0, 1/sqrt(input_dim)) at construction time and
    are never updated (``requires_grad=False``).  No bias term.

    A typical use case is reservoir computing: project to a high-dimensional
    space and apply kWTA before feeding a trainable MLP.

    Args:
        input_dim: Input dimensionality.
        output_dim: Output dimensionality (may exceed input_dim).
        activation: Activation applied after the linear map.
        **activation_kwargs: Passed to :func:`build_activation`.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "identity",
        **activation_kwargs,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.normal_(self.linear.weight, std=1.0 / math.sqrt(input_dim))
        self.linear.weight.requires_grad_(False)
        self.activation = build_activation(activation, **activation_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))
