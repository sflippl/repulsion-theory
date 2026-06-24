"""Core network classes: MLP, SingleNetwork, MultiNetwork.

The overall model prediction is the element-wise sum of all SingleNetwork
outputs, each scattered into the global prediction space.

Prediction space layout
-----------------------
For MSE output slots  : dim = slot vector dim.
For classify slots    : dim = n_classes (logits, not the stored label scalar).

This means the model output tensor has a different width than the target
tensor from the DataLoader — the ``LossSpec`` in ``training.py`` maps between
the two.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from repulsion.models.activations import build_activation
from repulsion.models.attention import AttentionLayer
from repulsion.models.projection import RandomProjection


# ---------------------------------------------------------------------------
# Internal MLP
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """Feedforward network with configurable depth and activation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: list[int],
        activation: str,
        activation_kwargs: dict,
        init_scale: float,
    ) -> None:
        super().__init__()
        sizes = [input_dim] + list(hidden_sizes) + [output_dim]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            linear = nn.Linear(sizes[i], sizes[i + 1], bias=False)
            size = sizes[i+1] if i+1 < len(sizes) - 1 else sizes[i]
            nn.init.normal_(linear.weight, std=init_scale/math.sqrt(size))
            layers.append(linear)
            if i < len(sizes) - 2:
                layers.append(build_activation(activation, **activation_kwargs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# SingleNetwork
# ---------------------------------------------------------------------------

class SingleNetwork(nn.Module):
    """One network stream: attention → projection → MLP, scattered to output space.

    Args:
        input_slot_names: Ordered list of input slot labels this network reads.
        output_slot_names: Ordered list of output slot labels this network writes.
        all_input_slot_offsets: Mapping from every input slot label to its
            start column in the full input tensor.
        all_input_slot_dims: Mapping from every input slot label to its width.
        all_output_pred_offsets: Mapping from every output slot label to its
            start column in the global prediction tensor.
        all_output_pred_dims: Mapping from every output slot label to its width
            in prediction space (n_classes for classify slots, dim for mse).
        global_prediction_dim: Total width of the prediction tensor.
        hidden_sizes: Hidden layer widths for the MLP.
        activation: Activation name for MLP hidden layers.
        activation_kwargs: Kwargs forwarded to the activation constructor.
        init_scale: Standard deviation for MLP weight initialisation.
        attention_layer: Optional pre-MLP attention layer.
        projection_layer: Optional frozen random projection (applied after
            attention, before MLP).
    """

    def __init__(
        self,
        input_slot_names: list[str],
        output_slot_names: list[str],
        all_input_slot_offsets: dict[str, int],
        all_input_slot_dims: dict[str, int],
        all_output_pred_offsets: dict[str, int],
        all_output_pred_dims: dict[str, int],
        global_prediction_dim: int,
        hidden_sizes: list[int],
        activation: str,
        activation_kwargs: dict,
        init_scale: float,
        attention_layer: Optional[AttentionLayer] = None,
        projection_layer: Optional[RandomProjection] = None,
    ) -> None:
        super().__init__()
        self.input_slot_names = input_slot_names
        self.output_slot_names = output_slot_names
        self.all_input_slot_offsets = all_input_slot_offsets
        self.all_input_slot_dims = all_input_slot_dims
        self.all_output_pred_offsets = all_output_pred_offsets
        self.all_output_pred_dims = all_output_pred_dims
        self.global_prediction_dim = global_prediction_dim

        self.attention = attention_layer    # None or AttentionLayer (registered auto)
        self.projection = projection_layer  # None or RandomProjection (frozen)

        this_input_dim = sum(all_input_slot_dims[s] for s in input_slot_names)
        mlp_input_dim = (
            projection_layer.linear.out_features
            if projection_layer is not None
            else this_input_dim
        )
        mlp_output_dim = sum(all_output_pred_dims[s] for s in output_slot_names)

        self.mlp = _MLP(
            input_dim=mlp_input_dim,
            output_dim=mlp_output_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            activation_kwargs=activation_kwargs,
            init_scale=init_scale,
        )

    @property
    def hidden_layer_names(self) -> list[str]:
        """Names of all hidden activation stages (``hidden_0``, ``hidden_1``, …)."""
        n = sum(1 for m in self.mlp.net if not isinstance(m, nn.Linear))
        return [f"hidden_{i}" for i in range(n)]

    def _extract_input(self, full_input: torch.Tensor) -> torch.Tensor:
        """Slice this network's input slots; zero-fill any NaN (disabled slots)."""
        parts = []
        for slot in self.input_slot_names:
            off = self.all_input_slot_offsets[slot]
            dim = self.all_input_slot_dims[slot]
            x_slot = full_input[:, off:off + dim]
            parts.append(torch.nan_to_num(x_slot, nan=0.0))
        return torch.cat(parts, dim=-1)

    def _scatter_output(self, out: torch.Tensor) -> torch.Tensor:
        """Place this network's outputs into the global prediction tensor."""
        scattered = torch.zeros(
            out.shape[0], self.global_prediction_dim,
            device=out.device, dtype=out.dtype,
        )
        local_off = 0
        for slot in self.output_slot_names:
            pred_off = self.all_output_pred_offsets[slot]
            pred_dim = self.all_output_pred_dims[slot]
            scattered[:, pred_off:pred_off + pred_dim] = out[:, local_off:local_off + pred_dim]
            local_off += pred_dim
        return scattered

    def forward(
        self,
        full_input: torch.Tensor,
        sample_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self._extract_input(full_input)
        if self.attention is not None:
            x = self.attention(x, sample_ids)
        if self.projection is not None:
            x = self.projection(x)
        return self._scatter_output(self.mlp(x))

    def extract(
        self,
        full_input: torch.Tensor,
        sample_ids: Optional[torch.Tensor],
        layer: str,
    ) -> torch.Tensor:
        """Run the forward pipeline and return activations at the named stage.

        Pipeline order (each stage is a valid ``layer`` value):

        * ``"input"``           – after slot extraction and NaN zero-fill.
        * ``"post_attention"``  – after the attention layer (error if absent).
        * ``"post_projection"`` – after the fixed projection (error if absent).
        * ``"hidden_N"``        – after the N-th hidden activation (0-indexed).
        * ``"output"``          – MLP logits, before scatter.
        * ``"scattered"``       – output placed into the global prediction space.

        Args:
            full_input: ``(batch, total_input_dim)`` — the full concatenated input.
            sample_ids: ``(batch,)`` integer IDs for per-sample attention, or ``None``.
            layer: Named stage at which to stop and return activations.

        Raises:
            ValueError: If *layer* is unknown or names a stage that does not
                exist in this network (e.g. ``"post_attention"`` with no attention).
        """
        x = self._extract_input(full_input)
        if layer == "input":
            return x

        if layer == "post_attention":
            if self.attention is None:
                raise ValueError("Layer 'post_attention' requested but this network has no attention layer.")
            return self.attention(x, sample_ids)

        if self.attention is not None:
            x = self.attention(x, sample_ids)

        if layer == "post_projection":
            if self.projection is None:
                raise ValueError("Layer 'post_projection' requested but this network has no projection layer.")
            return self.projection(x)

        if self.projection is not None:
            x = self.projection(x)

        # Step through the MLP, collecting hidden-layer outputs at non-Linear modules.
        hidden_idx = 0
        n_hidden = sum(1 for m in self.mlp.net if not isinstance(m, nn.Linear))
        for module in self.mlp.net:
            x = module(x)
            if not isinstance(module, nn.Linear):
                if layer == f"hidden_{hidden_idx}":
                    return x
                hidden_idx += 1

        if layer == "output":
            return x

        scattered = self._scatter_output(x)
        if layer == "scattered":
            return scattered

        valid = (
            ["input"]
            + (["post_attention"] if self.attention is not None else [])
            + (["post_projection"] if self.projection is not None else [])
            + [f"hidden_{i}" for i in range(n_hidden)]
            + ["output", "scattered"]
        )
        raise ValueError(
            f"Unknown layer '{layer}'. Valid stages for this network: {valid}."
        )


# ---------------------------------------------------------------------------
# MultiNetwork
# ---------------------------------------------------------------------------

class MultiNetwork(nn.Module):
    """Additive ensemble of :class:`SingleNetwork` streams.

    The final prediction is the element-wise sum of each stream's scattered
    output.  All streams share the same global prediction space layout.

    Args:
        networks: Component streams.
        global_prediction_dim: Width of the shared prediction tensor.
        row_index_to_id: Global mapping from ``(subgroup, group_id, item_id)``
            to integer row ID used by per-sample attention layers.  ``None``
            if no network uses per-sample attention.
    """

    def __init__(
        self,
        networks: list[SingleNetwork],
        global_prediction_dim: int,
        row_index_to_id: Optional[dict[tuple, int]] = None,
    ) -> None:
        super().__init__()
        self.networks = nn.ModuleList(networks)
        self.global_prediction_dim = global_prediction_dim
        self.row_index_to_id = row_index_to_id

    def network_params(self) -> list[nn.Parameter]:
        """MLP parameters only (excludes attention; projection is always frozen)."""
        params: list[nn.Parameter] = []
        for net in self.networks:
            params.extend(net.mlp.parameters())
        return params

    def attention_params(self) -> list[nn.Parameter]:
        """Attention layer parameters across all streams."""
        params: list[nn.Parameter] = []
        for net in self.networks:
            if net.attention is not None:
                params.extend(net.attention.parameters())
        return params

    def has_attention(self) -> bool:
        return any(net.attention is not None for net in self.networks)

    def get_sample_ids(
        self,
        collated_row_indices: tuple,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Convert collated row indices to sample-ID tensor, or None."""
        if self.row_index_to_id is None:
            return None
        subgroups, group_ids, item_ids = collated_row_indices
        ids = [
            self.row_index_to_id[(sg, int(g), int(i))]
            for sg, g, i in zip(subgroups, group_ids.tolist(), item_ids.tolist())
        ]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def forward(
        self,
        full_input: torch.Tensor,
        sample_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        result: Optional[torch.Tensor] = None
        for net in self.networks:
            out = net(full_input, sample_ids)
            result = out if result is None else result + out
        if result is None:
            raise ValueError("MultiNetwork has no component networks.")
        return result
