"""Softmax-based multiplicative attention layer.

Each input dimension is scaled by a learned non-negative weight.
Weights are computed as ``softmax(gating * logits) * n_logits``, so the
mean weight is always 1.0.  When logits are all zero the layer is an
identity (no attention effect).

The ``gating`` scalar plays the same role as the gate_factor in the old
intrepul GatedMLP: higher values sharpen the distribution (more selective
attention), lower values flatten it toward uniform (1.0 everywhere).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """Learned multiplicative attention applied to the input.

    Args:
        input_dim: Total dimensionality of the vector this layer operates on.
        slot_dims: If given, one logit per slot; the slot's scalar is broadcast
            to all its dimensions.  ``sum(slot_dims)`` must equal ``input_dim``.
            If ``None``, one logit per input dimension.
        gating: Multiplier on logits before softmax (inverse temperature).
        per_sample: If ``True``, each sample has its own logit vector looked up
            from a learned table keyed by integer sample ID.
        row_index_to_id: Required when ``per_sample=True``.  Maps
            ``(subgroup, group_id, item_id)`` tuples to integer row indices.
    """

    def __init__(
        self,
        input_dim: int,
        slot_dims: Optional[list[int]] = None,
        gating: float = 1.0,
        per_sample: bool = False,
        row_index_to_id: Optional[dict[tuple, int]] = None,
    ) -> None:
        super().__init__()
        self.gating = gating
        self.per_sample = per_sample
        self.input_dim = input_dim

        if slot_dims is not None:
            if sum(slot_dims) != input_dim:
                raise ValueError(
                    f"sum(slot_dims)={sum(slot_dims)} must equal input_dim={input_dim}."
                )
            n_logits = len(slot_dims)
            self.register_buffer(
                "_repeat_counts",
                torch.tensor(slot_dims, dtype=torch.long),
            )
        else:
            n_logits = input_dim
            self._repeat_counts = None

        self.n_logits = n_logits

        if per_sample:
            if row_index_to_id is None:
                raise ValueError("row_index_to_id is required when per_sample=True.")
            self.row_index_to_id: Optional[dict[tuple, int]] = row_index_to_id
            self.logits = nn.Parameter(torch.zeros(len(row_index_to_id), n_logits))
        else:
            self.row_index_to_id = None
            self.logits = nn.Parameter(torch.zeros(n_logits))

    def row_indices_to_ids(self, collated_row_indices: tuple) -> torch.Tensor:
        """Convert a collated batch of row indices to integer IDs.

        Args:
            collated_row_indices: Tuple ``(list[str], int_tensor, int_tensor)``
                as produced by PyTorch's default collate on
                ``(subgroup, group_id, item_id)`` tuples.

        Returns:
            1-D ``torch.long`` tensor of sample IDs (same device as logits).
        """
        subgroups, group_ids, item_ids = collated_row_indices
        ids = [
            self.row_index_to_id[(sg, int(g), int(i))]  # type: ignore[index]
            for sg, g, i in zip(subgroups, group_ids.tolist(), item_ids.tolist())
        ]
        return torch.tensor(ids, dtype=torch.long, device=self.logits.device)

    def forward(
        self,
        x: torch.Tensor,
        sample_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(batch, input_dim)``
            sample_ids: ``(batch,)`` integer IDs; required when ``per_sample=True``.
        """
        if self.per_sample:
            if sample_ids is None:
                raise ValueError("sample_ids is required for per-sample attention.")
            logits = self.logits[sample_ids]        # (batch, n_logits)
        else:
            logits = self.logits.unsqueeze(0)       # (1, n_logits)

        weights = torch.softmax(self.gating * logits, dim=-1) * self.n_logits

        if self._repeat_counts is not None:
            weights = torch.repeat_interleave(
                weights, self._repeat_counts, dim=-1
            )   # (batch or 1, input_dim)

        return x * weights
