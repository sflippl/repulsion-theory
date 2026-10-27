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


def build_model(
    network_specs: list[dict],
    in_dim: int,
    out_dim: int | None = None,
    n_input_slots: int = 1,
) -> MultiNetwork:
    """Build a :class:`MultiNetwork` whose streams all read/write one slot.

    Args:
        network_specs: List of stream spec dicts (same keys as the ``networks``
            list in ``configs/model/*.yaml``).
        in_dim: Input dimensionality (per input slot).
        out_dim: Output dimensionality (defaults to *in_dim*).
        n_input_slots: Number of input slots of width *in_dim*, laid out side
            by side (``SLOT_0, SLOT_1, ...``).  With more than one slot, specs
            default to ``sum_input_slots=True``, so each stream sees the sum of
            its (projected) slots rather than their concatenation.  Summed
            slots are already projected one at a time, so
            ``fixed_projection_slot_grouping`` (set in most ``configs/model``
            entries) is dropped.
    """
    out_dim = in_dim if out_dim is None else out_dim
    in_slots = [SLOT] if n_input_slots == 1 else [f"{SLOT}_{i}" for i in range(n_input_slots)]
    specs = [{"sum_input_slots": n_input_slots > 1, **spec} for spec in network_specs]
    for spec in specs:
        if spec["sum_input_slots"]:
            spec.pop("fixed_projection_slot_grouping", None)
    networks = [
        build_single_network(
            spec,
            input_slots=in_slots,
            output_slots=[SLOT],
            input_slot_offsets={s: i * in_dim for i, s in enumerate(in_slots)},
            input_slot_dims={s: in_dim for s in in_slots},
            output_pred_offsets={SLOT: 0},
            output_pred_dims={SLOT: out_dim},
            global_prediction_dim=out_dim,
        )
        for spec in specs
    ]
    return MultiNetwork(networks=networks, global_prediction_dim=out_dim)


@torch.no_grad()
def hidden_reps(model: MultiNetwork, x: torch.Tensor, layer: str = "hidden_0") -> list[torch.Tensor]:
    """Per-stream activations at *layer* for inputs *x*."""
    x = model.prepare_input(x)
    return [net.extract(x, None, layer) for net in model.networks]
