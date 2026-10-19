"""Fixed (frozen) random projection layer."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from repulsion.models.activations import build_activation


def _kwta_squared_gain(k: int, n: int, chunk: int = 256) -> float:
    """E[sum of the k largest z_i^2] / n for n i.i.d. N(0, 1) values.

    Equals E[z^2; fewer than k of the other n-1 units exceed z], i.e. the
    integral of z^2 * pdf(z) * P(Binomial(n-1, sf(z)) <= k-1), evaluated on a
    grid. The binomial CDF is summed in log space, in chunks over the count.
    """
    z = torch.linspace(-10.0, 10.0, 8001, dtype=torch.float64)
    p = torch.special.ndtr(-z).clamp(1e-300, 1.0 - 1e-16)  # P(other unit > z)
    log_p, log_q = torch.log(p)[:, None], torch.log1p(-p)[:, None]
    log_cdf = torch.full_like(z, -math.inf)
    for start in range(0, k, chunk):
        j = torch.arange(start, min(start + chunk, k), dtype=torch.float64)
        log_pmf = (math.lgamma(n) - torch.lgamma(j + 1) - torch.lgamma(n - j)
                   + j * log_p + (n - 1 - j) * log_q)
        log_cdf = torch.logaddexp(log_cdf, torch.logsumexp(log_pmf, dim=1))
    pdf = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    return torch.trapezoid(z ** 2 * pdf * log_cdf.exp(), z).item()


def expected_squared_gain(activation: str, output_dim: int, **activation_kwargs) -> float:
    """E[||act(h)||^2] / output_dim for ``h`` with i.i.d. N(0, 1) entries.

    For ``kwta`` this is exact at finite width (see :func:`_kwta_squared_gain`).

    Raises:
        ValueError: For activations that are not positively homogeneous
            (e.g. ``sigmoid``), where rescaling the weights cannot set the
            output norm.
    """
    if activation == "identity":
        return 1.0
    if activation == "relu":
        return 0.5
    if activation == "leaky_relu":
        slope = activation_kwargs.get("negative_slope", 0.01)
        return 0.5 * (1.0 + slope ** 2)
    if activation == "kwta":
        k = max(1, round(activation_kwargs.get("frac", 0.1) * output_dim))
        return _kwta_squared_gain(k, output_dim)
    raise ValueError(
        f"Cannot normalize a projection with activation '{activation}' "
        "(only identity, relu, leaky_relu and kwta are supported)."
    )


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
        normalize: If True, rescale the weights so that the projected and
            activated features have the same expected squared norm as the
            input (E||act(Wx)||^2 = ||x||^2), i.e. the stream sees inputs of
            the same magnitude as a stream without a projection.  Exact for
            an ungrouped projection; with ``slot_dims`` it assumes each slot's
            squared norm is proportional to its width (otherwise kWTA favours
            the larger slot and the norm comes out somewhat larger).
        **activation_kwargs: Passed to :func:`build_activation`.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "identity",
        slot_dims: Sequence[int] | None = None,
        normalize: bool = False,
        **activation_kwargs,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.slot_dims = list(slot_dims) if slot_dims is not None else None

        self.linear: nn.Linear | None = None
        self.slot_linears: nn.ModuleList | None = None
        self.slot_output_dims: list[int] | None = None

        if self.slot_dims is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=False)
            nn.init.normal_(self.linear.weight, std=1.0 / math.sqrt(input_dim))
            self.linear.weight.requires_grad_(False)
        else:
            self._validate_slot_dims(input_dim, output_dim, self.slot_dims)
            self.slot_output_dims = self._allocate_slot_output_dims(self.slot_dims, output_dim)
            self.slot_linears = nn.ModuleList(
                [
                    nn.Linear(in_dim, out_dim, bias=False)
                    for in_dim, out_dim in zip(self.slot_dims, self.slot_output_dims)
                ]
            )
            for in_dim, layer in zip(self.slot_dims, self.slot_linears):
                nn.init.normal_(layer.weight, std=1.0 / math.sqrt(in_dim))
                layer.weight.requires_grad_(False)

        if normalize:
            # Each projected unit has variance ||x||^2 / input_dim, so the
            # activated features have E||.||^2 = ||x||^2 * output_dim / input_dim * gain.
            gain = expected_squared_gain(activation, output_dim, **activation_kwargs)
            scale = math.sqrt(input_dim / (output_dim * gain))
            with torch.no_grad():
                for p in self.parameters():
                    p.mul_(scale)

        self.activation = build_activation(activation, **activation_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear is not None:
            proj = self.linear(x)
        else:
            assert self.slot_dims is not None and self.slot_linears is not None
            parts = torch.split(x, self.slot_dims, dim=-1)
            proj = torch.cat(
                [layer(part) for layer, part in zip(self.slot_linears, parts)],
                dim=-1,
            )
        return self.activation(proj)

    @staticmethod
    def _validate_slot_dims(input_dim: int, output_dim: int, slot_dims: list[int]) -> None:
        if len(slot_dims) == 0:
            raise ValueError("slot_dims must be non-empty when provided.")
        if any(d <= 0 for d in slot_dims):
            raise ValueError(f"All slot_dims must be positive; got {slot_dims}.")
        if sum(slot_dims) != input_dim:
            raise ValueError(
                f"slot_dims must sum to input_dim ({input_dim}); got sum={sum(slot_dims)}."
            )
        if output_dim < len(slot_dims):
            raise ValueError(
                f"output_dim ({output_dim}) must be >= number of slots ({len(slot_dims)}) "
                "for slot-grouped projection."
            )

    @staticmethod
    def _allocate_slot_output_dims(slot_dims: list[int], output_dim: int) -> list[int]:
        """Allocate output dims proportionally, guaranteeing >=1 per slot."""
        n_slots = len(slot_dims)
        if n_slots == 1:
            return [output_dim]

        remaining = output_dim - n_slots
        total_in = float(sum(slot_dims))
        raw = [remaining * (d / total_in) for d in slot_dims]
        floors = [int(v) for v in raw]
        frac = [v - f for v, f in zip(raw, floors)]
        leftover = remaining - sum(floors)

        order = sorted(range(n_slots), key=lambda i: (-frac[i], i))
        for i in order[:leftover]:
            floors[i] += 1

        return [1 + f for f in floors]
