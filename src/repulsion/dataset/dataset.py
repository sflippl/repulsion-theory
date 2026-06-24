"""TaskDataset, DatasetCollection, and build_datasets.

Data model
----------
Each row in a dataset corresponds to one unique ``(subgroup, group_id, item_id)``
triple, represented as a :class:`RowIndex`.  All active slots in a task must
reference item types that share the same row structure (same subgroup names,
group counts, and item counts); this is validated at construction time.

The ``input`` and ``output`` arrays are built by concatenating per-slot arrays
in the slot-definition order.  Disabled slots are filled with ``off_value``
(NaN by default), giving a clear distinction between "not predicted" and
"predicted as zero".

Accessing individual slots
--------------------------
Use ``input_slot_arrays["Face1"]`` and ``output_slot_arrays["Face2"]`` to
retrieve the (N, dim) array for a specific slot before concatenation.
Use ``input_slot_dims`` / ``output_slot_dims`` to slice back out of the
concatenated arrays::

    face_start = 0
    face_end   = face_start + ds.input_slot_dims["Face1"]
    face_cols  = ds.input[:, face_start:face_end]
"""
from __future__ import annotations

from __future__ import annotations

from typing import Iterator, NamedTuple, Optional

import numpy as np

from repulsion.dataset.spec import DatasetSpec, SlotConfig, SlotDef, TaskSpec, parse_dataset_spec
from repulsion.stimgen.generator import ItemSet


# ---------------------------------------------------------------------------
# Row index
# ---------------------------------------------------------------------------

class RowIndex(NamedTuple):
    """Identity of one data row: (subgroup, group_id, item_id), all 1-indexed."""
    subgroup: str
    group_id: int
    item_id: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_row_structure(item_set: ItemSet, item_type: str) -> list[RowIndex]:
    """Return the ordered list of RowIndexes for one item type.

    Order mirrors the generation order: subgroups in spec order, then
    groups 1..n_groups, then items 1..n_items within each group.
    """
    rows: list[RowIndex] = []
    seen_sgs: list[str] = []
    for it in item_set.by_type(item_type):
        if it.subgroup not in seen_sgs:
            seen_sgs.append(it.subgroup)

    for sg in seen_sgs:
        seen_groups: list[int] = []
        for it in item_set.by_subgroup(item_type, sg):
            if it.group_id not in seen_groups:
                seen_groups.append(it.group_id)
        for g in seen_groups:
            for it in sorted(item_set.by_group(item_type, sg, g), key=lambda x: x.item_id):
                rows.append(RowIndex(subgroup=sg, group_id=g, item_id=it.item_id))
    return rows


def _validate_slot_compatibility(
    item_set: ItemSet,
    item_types: set[str],
    task_name: str,
) -> None:
    """Ensure all active item types share the same (subgroup, group_id, item_id) structure.

    Raises:
        ValueError: If any two active item types have different row structures.
    """
    if len(item_types) <= 1:
        return

    structures: dict[str, list[RowIndex]] = {
        t: _get_row_structure(item_set, t) for t in item_types
    }
    types = list(structures)
    ref_type = types[0]
    ref_rows = structures[ref_type]

    for other_type in types[1:]:
        other_rows = structures[other_type]
        if other_rows != ref_rows:
            raise ValueError(
                f"Task '{task_name}': active slots reference item types "
                f"'{ref_type}' and '{other_type}' which have incompatible "
                f"structures (different subgroup names, group counts, or item counts). "
                f"All active slots in a task must share the same row structure."
            )


def _apply_manipulation(
    item_set: ItemSet,
    item_type: str,
    row: RowIndex,
    cfg: SlotConfig,
) -> np.ndarray:
    """Compute the vector for one slot at one row.

    ``"default"``  — item vector × magnitude.
    ``"group"``    — mean of all item vectors in (item_type, subgroup, group_id) × magnitude.
    """
    if cfg.manipulation == "default":
        item = item_set.by_name(f"{item_type}_{row.subgroup}_{row.group_id}_{row.item_id}")
        return item.vector * cfg.magnitude

    elif cfg.manipulation == "group":
        group_items = item_set.by_group(item_type, row.subgroup, row.group_id)
        if not group_items:
            raise ValueError(
                f"No items found for ({item_type}, {row.subgroup}, group_id={row.group_id})."
            )
        mean_vec = np.mean(np.stack([it.vector for it in group_items]), axis=0)
        return mean_vec * cfg.magnitude

    else:
        raise ValueError(f"Unknown manipulation: {cfg.manipulation!r}")


# ---------------------------------------------------------------------------
# TaskDataset
# ---------------------------------------------------------------------------

class TaskDataset:
    """A single task's input/output arrays, built from an :class:`ItemSet`.

    Attributes:
        task_name:           Name of the task.
        rows:                Ordered list of :class:`RowIndex` — one per data point.
        input:               ``(N, total_input_dim)`` float64 array.
                             Disabled slots are NaN (or ``off_value``).
        output:              ``(N, total_output_dim)`` float64 array.
                             Disabled slots are NaN (or ``off_value``).
        input_slot_dims:     ``{label: dim}`` for each input slot, in slot order.
        output_slot_dims:    ``{label: dim}`` for each output slot, in slot order.
        input_slot_arrays:   ``{label: (N, dim) array}`` before concatenation.
        output_slot_arrays:  ``{label: (N, dim) array}`` before concatenation.
    """

    def __init__(
        self,
        item_set: ItemSet,
        spec: DatasetSpec,
        task: TaskSpec,
        off_value: float = float("nan"),
    ) -> None:
        self.task_name = task.name

        # Collect active item types in input-slot, then output-slot definition order
        # (deterministic: first occurrence wins for choosing the reference row structure)
        active_item_types: set[str] = set()
        first_type: str | None = None

        for slot_def in spec.input_slots:
            if task.input_config[slot_def.label] is not None:
                if first_type is None:
                    first_type = slot_def.item_type
                active_item_types.add(slot_def.item_type)

        for slot_def in spec.output_slots:
            if task.output_config[slot_def.label] is not None:
                if first_type is None:
                    first_type = slot_def.item_type
                active_item_types.add(slot_def.item_type)

        if first_type is None:
            raise ValueError(f"Task '{task.name}' has no active slots in input or output.")

        _validate_slot_compatibility(item_set, active_item_types, task.name)

        self.rows: list[RowIndex] = _get_row_structure(item_set, first_type)
        N = len(self.rows)

        # Precompute row structure metadata for classification label computation
        _unique_sgs: list[str] = list(dict.fromkeys(r.subgroup for r in self.rows))
        _sg_to_idx: dict[str, int] = {sg: i for i, sg in enumerate(_unique_sgs)}
        _n_groups: int = max((r.group_id for r in self.rows), default=1)
        _n_items: int = max((r.item_id for r in self.rows), default=1)
        _n_sgs: int = len(_unique_sgs)

        # Build per-slot arrays and dims
        def _build_side(
            slot_defs: tuple[SlotDef, ...],
            config: dict[str, SlotConfig | None],
        ) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, str], dict[str, Optional[int]]]:
            arrays: dict[str, np.ndarray] = {}
            dims: dict[str, int] = {}
            slot_types: dict[str, str] = {}
            n_classes_out: dict[str, Optional[int]] = {}

            for slot_def in slot_defs:
                loss_type = slot_def.loss_type
                slot_types[slot_def.label] = loss_type
                cfg = config[slot_def.label]

                if loss_type == "mse":
                    type_items = item_set.by_type(slot_def.item_type)
                    if not type_items:
                        raise ValueError(
                            f"No items found for item type '{slot_def.item_type}' "
                            f"(referenced by slot '{slot_def.label}')."
                        )
                    dim = type_items[0].dim
                    dims[slot_def.label] = dim
                    n_classes_out[slot_def.label] = None

                    if cfg is None:
                        arrays[slot_def.label] = np.full((N, dim), off_value)
                    else:
                        arrays[slot_def.label] = np.stack([
                            _apply_manipulation(item_set, slot_def.item_type, row, cfg)
                            for row in self.rows
                        ])

                else:  # classify_group or classify_item
                    if loss_type == "classify_group":
                        nc = _n_sgs * _n_groups
                        raw_labels = np.array(
                            [_sg_to_idx[r.subgroup] * _n_groups + (r.group_id - 1)
                             for r in self.rows],
                            dtype=np.float32,
                        )
                    else:  # classify_item
                        nc = _n_sgs * _n_groups * _n_items
                        raw_labels = np.array(
                            [_sg_to_idx[r.subgroup] * _n_groups * _n_items
                             + (r.group_id - 1) * _n_items
                             + (r.item_id - 1)
                             for r in self.rows],
                            dtype=np.float32,
                        )
                    dims[slot_def.label] = 1
                    n_classes_out[slot_def.label] = nc

                    if cfg is None:
                        arrays[slot_def.label] = np.full((N, 1), off_value)
                    else:
                        arrays[slot_def.label] = raw_labels[:, np.newaxis]

            return arrays, dims, slot_types, n_classes_out

        self.input_slot_arrays, self.input_slot_dims, _, _ = _build_side(
            spec.input_slots, task.input_config
        )
        _out_arrays, _out_dims, _out_types, _out_n_classes = _build_side(
            spec.output_slots, task.output_config
        )
        self.output_slot_arrays: dict[str, np.ndarray] = _out_arrays
        self.output_slot_dims: dict[str, int] = _out_dims
        self.output_slot_types: dict[str, str] = _out_types
        self.output_slot_n_classes: dict[str, Optional[int]] = _out_n_classes
        # prediction dim = n_classes for classify slots, slot_dim for mse
        self.output_prediction_dims: dict[str, int] = {
            label: (_out_n_classes[label] if _out_n_classes[label] is not None else _out_dims[label])
            for label in _out_dims
        }

        def _concat(arrays_dict, slot_defs):
            if not slot_defs:
                return np.empty((N, 0), dtype=np.float64)
            return np.concatenate([arrays_dict[s.label] for s in slot_defs], axis=1)

        self.input: np.ndarray = _concat(self.input_slot_arrays, spec.input_slots)
        self.output: np.ndarray = _concat(self.output_slot_arrays, spec.output_slots)

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"TaskDataset(task={self.task_name!r}, "
            f"n_rows={len(self.rows)}, "
            f"input_shape={self.input.shape}, "
            f"output_shape={self.output.shape})"
        )


# ---------------------------------------------------------------------------
# DatasetCollection
# ---------------------------------------------------------------------------

class DatasetCollection:
    """Dict-like container of :class:`TaskDataset` objects keyed by task name."""

    def __init__(
        self,
        item_set: ItemSet,
        spec: DatasetSpec,
        off_value: float = float("nan"),
    ) -> None:
        self.spec: DatasetSpec = spec
        self._datasets: dict[str, TaskDataset] = {
            task.name: TaskDataset(item_set, spec, task, off_value)
            for task in spec.tasks
        }

    def __getitem__(self, task_name: str) -> TaskDataset:
        try:
            return self._datasets[task_name]
        except KeyError:
            raise KeyError(
                f"No task named {task_name!r}. Available: {list(self._datasets)}."
            )

    def __iter__(self) -> Iterator[str]:
        return iter(self._datasets)

    def __len__(self) -> int:
        return len(self._datasets)

    def __repr__(self) -> str:  # noqa: D401
        return f"DatasetCollection(tasks={list(self._datasets)})"

    def task_names(self) -> list[str]:
        """Return task names in definition order."""
        return list(self._datasets)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def build_datasets(
    item_set: ItemSet,
    slots: dict,
    tasks: list[dict],
    off_value: float = float("nan"),
    model_slots: dict | None = None,
) -> DatasetCollection:
    """Parse a slot/task spec and build all task datasets in one call.

    Args:
        item_set:    Generated items from :class:`~repulsion.stimgen.ItemGenerator`.
        slots:       Dict with ``"input"`` and ``"output"`` keys, each an ordered
                     mapping ``{slot_label: item_type}``.
        tasks:       List of task specification dicts.  Required: ``"name"``.
                     Optional: ``"input"`` / ``"output"`` sub-dicts.
        off_value:   Value used to fill disabled slots.  Defaults to ``NaN``.
        model_slots: Optional per-model slot routing, passed to
                     :func:`~repulsion.dataset.spec.parse_dataset_spec`.

    Returns:
        :class:`DatasetCollection` with one :class:`TaskDataset` per task.
        The collection also exposes ``.spec`` (:class:`~repulsion.dataset.spec.DatasetSpec`)
        which carries the ``model_slots`` routing for use by
        :func:`~repulsion.models.parse_model_spec`.

    Example::

        collection = build_datasets(
            item_set,
            slots={
                "input":  {"Face1": "face", "Object": "object"},
                "output": {"Face1": "face", "Object": "object", "Face2": "face"},
            },
            tasks=[
                {
                    "name": "autoencoding",
                    "input":  {"Face1": {"manipulation": "default"},
                               "Object": {"manipulation": "default"}},
                    "output": {"Face1": {"manipulation": "default"},
                               "Object": {"manipulation": "default"}},
                },
                {
                    "name": "pairmate_prediction",
                    "input":  {"Face1": {"manipulation": "default"}},
                    "output": {"Face2": {"manipulation": "group"}},
                },
            ],
        )
        ae  = collection["autoencoding"]
        pm  = collection["pairmate_prediction"]
    """
    spec = parse_dataset_spec(slots, tasks, model_slots)
    return DatasetCollection(item_set, spec, off_value)