"""Build :class:`MultiNetwork` models for single-slot task scripts.

Task scripts present every stimulus in one shared input space (e.g. a
superposition of two items) and read out into one shared output space.  The
model is the regular :class:`MultiNetwork` with a single ``"stimulus"`` slot
on each side, so network specs use the same keys as ``configs/model/*.yaml``.
"""
from __future__ import annotations

import torch

from repulsion.models import MultiNetwork, build_single_network

SLOT = "stimulus"


def build_model(network_specs: list[dict], in_dim: int, out_dim: int | None = None) -> MultiNetwork:
    """Build a :class:`MultiNetwork` whose streams all read/write one slot.

    Args:
        network_specs: List of stream spec dicts (same keys as the ``networks``
            list in ``configs/model/*.yaml``).
        in_dim: Input dimensionality.
        out_dim: Output dimensionality (defaults to *in_dim*).
    """
    out_dim = in_dim if out_dim is None else out_dim
    networks = [
        build_single_network(
            spec,
            input_slots=[SLOT],
            output_slots=[SLOT],
            input_slot_offsets={SLOT: 0},
            input_slot_dims={SLOT: in_dim},
            output_pred_offsets={SLOT: 0},
            output_pred_dims={SLOT: out_dim},
            global_prediction_dim=out_dim,
        )
        for spec in network_specs
    ]
    return MultiNetwork(networks=networks, global_prediction_dim=out_dim)


@torch.no_grad()
def hidden_reps(model: MultiNetwork, x: torch.Tensor, layer: str = "hidden_0") -> list[torch.Tensor]:
    """Per-stream activations at *layer* for inputs *x*."""
    x = model.prepare_input(x)
    return [net.extract(x, None, layer) for net in model.networks]
