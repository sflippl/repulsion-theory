"""Activation functions used across repulsion models."""
from __future__ import annotations

import torch
import torch.nn as nn


class KWinnerTakesAll(nn.Module):
    """K-winner-takes-all: keeps the top-k units per sample, zeros the rest.

    k = max(1, round(frac * width))
    Values of winning units are preserved unchanged.

    Args:
        frac: Fraction of units to keep active. Default 0.1.
    """

    def __init__(self, frac: float = 0.1) -> None:
        super().__init__()
        self.frac = frac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = max(1, round(self.frac * x.shape[-1]))
        topk = x.topk(k, dim=-1).indices
        mask = torch.zeros_like(x).scatter_(-1, topk, 1.0)
        return x * mask


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "sigmoid":    nn.Sigmoid,
    "relu":       nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "kwta":       KWinnerTakesAll,
    "identity":   nn.Identity,
}


def build_activation(name: str, **kwargs) -> nn.Module:
    """Instantiate an activation module by name.

    Args:
        name: One of ``'sigmoid'``, ``'relu'``, ``'leaky_relu'``,
              ``'kwta'``, ``'identity'``.
        **kwargs: Forwarded to the constructor.
            - ``leaky_relu``: ``negative_slope`` (default 0.01)
            - ``kwta``: ``frac`` (default 0.1)

    Raises:
        ValueError: If *name* is not a known activation.
    """
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {sorted(ACTIVATIONS)}"
        )
    return ACTIVATIONS[name](**kwargs)
