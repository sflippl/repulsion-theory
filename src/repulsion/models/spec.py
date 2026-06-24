"""Parse model specification dicts into a :class:`MultiNetwork`.

Example spec for a single network with slot-grouped attention::

    specs = [
        {
            "name": "model",
            "hidden_sizes": [256],
            "activation": "identity",
            "attention_layer": True,
            "attention_layer_slot_grouping": True,
            "attention_layer_sample_grouping": True,
            "attention_layer_gating": 5.0,
        }
    ]

    # Slot routing lives in the dataset config under ``model_slots``; the
    # network above will default to all declared input and output slots.

Example spec for two-stream network with a fixed random projection::

    specs = [
        {"name": "net1"},
        {
            "name": "net2",
            "fixed_projection": True,
            "fixed_projection_hidden_size": 1000,
            "fixed_projection_activation": "kwta",
            "fixed_projection_kwta_frac": 0.03,
        },
    ]
"""
from __future__ import annotations

from repulsion.dataset import DatasetCollection
from repulsion.dataset.spec import DatasetSpec
from repulsion.models.attention import AttentionLayer
from repulsion.models.network import MultiNetwork, SingleNetwork
from repulsion.models.projection import RandomProjection


def parse_model_spec(
    specs: list[dict],
    collection: DatasetCollection,
    dataset_spec: DatasetSpec | None = None,
) -> MultiNetwork:
    """Build a :class:`MultiNetwork` from specification dicts and a dataset.

    Reads slot layout (dims, prediction-space offsets, output types) from
    the first task of *collection*.  All tasks in the collection must share
    the same slot structure.

    Slot routing (which input/output slots each network sees) is resolved from
    *dataset_spec*.``model_slots``.  If a network name is not present there,
    it defaults to all declared input slots and all declared output slots in
    declaration order.  When *dataset_spec* is ``None``, all networks default
    to all slots (backward-compatible behaviour; slot lists may also be
    provided directly in each spec dict via ``"input"``/``"output"`` keys).

    Args:
        specs: List of network specification dicts.  Required key: ``name``.
            Slot routing is taken from *dataset_spec*; legacy ``"input"``/
            ``"output"`` keys are still accepted as a fallback when
            *dataset_spec* is ``None`` or the network is absent from
            ``model_slots``.
        collection: Built dataset collection used to derive slot layout.
        dataset_spec: Parsed dataset specification carrying ``model_slots``.

    Returns:
        Configured :class:`MultiNetwork` ready for training.

    Raises:
        ValueError: For unknown slot names or missing required spec fields.
    """
    first_task = collection[collection.task_names()[0]]

    # --- Input slot layout ---
    input_slot_dims: dict[str, int] = dict(first_task.input_slot_dims)
    input_slot_offsets: dict[str, int] = {}
    off = 0
    for label, dim in first_task.input_slot_dims.items():
        input_slot_offsets[label] = off
        off += dim

    # --- Output prediction-space layout ---
    output_pred_dims: dict[str, int] = dict(first_task.output_prediction_dims)
    output_pred_offsets: dict[str, int] = {}
    off = 0
    for label, dim in output_pred_dims.items():
        output_pred_offsets[label] = off
        off += dim
    global_prediction_dim = off

    all_input_labels = list(input_slot_dims)   # ordered
    all_output_labels = list(output_pred_dims)  # ordered

    # Default routing: all input slots, all output slots (in declaration order)
    _model_slots: dict[str, dict] = {}
    if dataset_spec is not None and dataset_spec.model_slots:
        _model_slots = dataset_spec.model_slots

    # --- Global row-index → ID mapping for per-sample attention ---
    all_rows: set = set()
    for task_name in collection.task_names():
        all_rows.update(collection[task_name].rows)
    sorted_rows = sorted(all_rows)
    row_index_to_id: dict[tuple, int] = {r: i for i, r in enumerate(sorted_rows)}
    has_per_sample_attn = False

    # --- Build each stream ---
    networks: list[SingleNetwork] = []
    for spec in specs:
        name = spec.get("name", f"network_{len(networks)}")

        # Resolve slot routing: dataset_spec.model_slots > legacy spec keys > all slots
        routing = _model_slots.get(name, {})
        if "input" in routing:
            input_slots: list[str] = list(routing["input"])
        else:
            input_slots = spec.get("input", all_input_labels)
        if "output" in routing:
            output_slots: list[str] = list(routing["output"])
        else:
            output_slots = spec.get("output", all_output_labels)

        for s in input_slots:
            if s not in input_slot_dims:
                raise ValueError(
                    f"Network '{name}': input slot '{s}' is not declared in the dataset. "
                    f"Available input slots: {all_input_labels}."
                )
        for s in output_slots:
            if s not in output_pred_dims:
                raise ValueError(
                    f"Network '{name}': output slot '{s}' is not declared in the dataset. "
                    f"Available output slots: {all_output_labels}."
                )

        hidden_sizes: list[int] = spec.get("hidden_sizes", [256])
        activation: str = spec.get("activation", "identity")
        activation_kwargs: dict = {}
        if activation == "leaky_relu":
            activation_kwargs["negative_slope"] = spec.get("activation_leaky_relu_slope", 0.01)
        elif activation == "kwta":
            activation_kwargs["frac"] = spec.get("activation_kwta_frac", 0.1)
        init_scale: float = float(spec.get("init_scale", 0.01))

        # -- Attention layer --
        attention_layer: AttentionLayer | None = None
        if spec.get("attention_layer", False):
            this_input_dim = sum(input_slot_dims[s] for s in input_slots)
            slot_grouped: bool = spec.get("attention_layer_slot_grouping", False)
            # attention_layer_sample_grouping=True means SHARED across samples (default)
            per_sample: bool = not spec.get("attention_layer_sample_grouping", True)
            gating: float = float(spec.get("attention_layer_gating", 1.0))

            slot_dims_for_attn = (
                [input_slot_dims[s] for s in input_slots] if slot_grouped else None
            )
            attention_layer = AttentionLayer(
                input_dim=this_input_dim,
                slot_dims=slot_dims_for_attn,
                gating=gating,
                per_sample=per_sample,
                row_index_to_id=row_index_to_id if per_sample else None,
            )
            if per_sample:
                has_per_sample_attn = True

        # -- Fixed projection --
        projection_layer: RandomProjection | None = None
        if spec.get("fixed_projection", False):
            this_input_dim = sum(input_slot_dims[s] for s in input_slots)
            proj_out_dim: int = int(spec.get("fixed_projection_hidden_size", 1000))
            proj_act: str = spec.get("fixed_projection_activation", "identity")
            proj_act_kw: dict = {}
            if proj_act == "kwta":
                proj_act_kw["frac"] = float(spec.get("fixed_projection_kwta_frac", 0.1))
            elif proj_act == "leaky_relu":
                proj_act_kw["negative_slope"] = float(
                    spec.get("fixed_projection_leaky_relu_slope", 0.01)
                )
            projection_layer = RandomProjection(
                input_dim=this_input_dim,
                output_dim=proj_out_dim,
                activation=proj_act,
                **proj_act_kw,
            )

        networks.append(
            SingleNetwork(
                input_slot_names=input_slots,
                output_slot_names=output_slots,
                all_input_slot_offsets=input_slot_offsets,
                all_input_slot_dims=input_slot_dims,
                all_output_pred_offsets=output_pred_offsets,
                all_output_pred_dims=output_pred_dims,
                global_prediction_dim=global_prediction_dim,
                hidden_sizes=hidden_sizes,
                activation=activation,
                activation_kwargs=activation_kwargs,
                init_scale=init_scale,
                attention_layer=attention_layer,
                projection_layer=projection_layer,
            )
        )

    return MultiNetwork(
        networks=networks,
        global_prediction_dim=global_prediction_dim,
        row_index_to_id=row_index_to_id if has_per_sample_attn else None,
    )
