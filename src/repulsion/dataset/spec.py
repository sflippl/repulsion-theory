"""Specification dataclasses and parsing for dataset/task construction.

Terminology
-----------
slot
    A named position in an input or output vector. Every slot is bound to one
    item type (e.g. ``"face"``).  Slot labels must be unique within input and
    within output but may repeat across the two sides.

task
    A named combination of active slots with per-slot configurations
    (magnitude and manipulation).  Slots not mentioned in a task are disabled
    (their portion of the concatenated vector is filled with ``off_value``).

manipulation
    How the raw item vector is used for a slot in a specific task:
      ``"default"``  — use the item's own vector as-is (times magnitude).
      ``"group"``    — use the mean of all item vectors in the same group
                       (times magnitude).  The resulting vector is the same
                       for every item_id within a group.
"""
from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

VALID_MANIPULATIONS: frozenset[str] = frozenset({"default", "group"})
VALID_LOSS_TYPES: frozenset[str] = frozenset({"mse", "classify_group", "classify_item"})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlotDef:
    """Definition of one slot: its label and the item type it carries.

    ``loss_type`` controls how the output slot is used in the loss:
      ``"mse"``             — mean-squared error against the item vector.
      ``"classify_group"`` — cross-entropy; target is a global group label
                             (subgroup_idx * n_groups + group_id − 1).
      ``"classify_item"``  — cross-entropy; target is a global item label
                             (subgroup_idx * n_groups * n_items + ...).
    Only output slots may use classification loss types.
    """
    label: str
    item_type: str
    loss_type: str = "mse"


@dataclass(frozen=True)
class SlotConfig:
    """Configuration of one active slot within a specific task."""
    magnitude: float = 1.0
    manipulation: str = "default"
    noise_std: float = 0.0
    """Standard deviation of Gaussian noise added to this slot's vector at training time.
    Applied independently on input and output. Zero means no noise (default)."""


@dataclass
class TaskSpec:
    """Fully parsed specification for one task.

    ``input_config`` and ``output_config`` have one key per defined slot.
    A value of ``None`` means that slot is disabled for this task.
    """
    name: str
    input_config: dict[str, Optional[SlotConfig]]
    output_config: dict[str, Optional[SlotConfig]]


@dataclass
class DatasetSpec:
    """Complete dataset specification: ordered slots, all tasks, and per-model slot routing.

    ``model_slots`` maps network names to their input/output slot label lists::

        {"net1": {"input": ["Face", "Object"], "output": ["Face", "Object"]}}

    A network not present in ``model_slots`` defaults to all declared input
    slots and all declared output slots, in declaration order.
    """
    input_slots: tuple[SlotDef, ...]
    output_slots: tuple[SlotDef, ...]
    tasks: tuple[TaskSpec, ...]
    model_slots: dict[str, dict[str, list[str]]] = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_dataset_spec(slots: dict, tasks: list[dict], model_slots: dict | None = None) -> DatasetSpec:
    """Parse slot and task dicts into a resolved :class:`DatasetSpec`.

    Args:
        slots: Dict with ``"input"`` and ``"output"`` keys, each an ordered
            mapping of ``{slot_label: item_type}``.
        tasks: List of task dicts.  Required key: ``"name"``.  Optional keys:
            ``"input"`` and ``"output"``, each a mapping of
            ``{slot_label: {magnitude: float, manipulation: str}}``.
            Slots not mentioned in a task default to disabled.
        model_slots: Optional mapping from network name to a dict with
            ``"input"`` and/or ``"output"`` keys (each a list of slot labels).
            Labels must match those declared in *slots*.  Networks not
            mentioned default to all declared slots in order.

    Returns:
        :class:`DatasetSpec` with fully resolved slot definitions, tasks, and
        per-model slot routing.

    Raises:
        ValueError: For duplicate slot labels, unknown slot references, or
            invalid manipulation names.
    """
    raw_input = slots.get("input") or {}
    raw_output = slots.get("output") or {}

    def _parse_slot_def(label: str, value, allow_classification: bool) -> SlotDef:
        if isinstance(value, str):
            return SlotDef(label=label, item_type=value)
        if not isinstance(value, dict) or "item_type" not in value:
            raise ValueError(
                f"Slot '{label}' must be a string (item_type) or a dict with an 'item_type' key."
            )
        loss_type = value.get("loss_type", "mse")
        if loss_type not in VALID_LOSS_TYPES:
            raise ValueError(
                f"Slot '{label}': loss_type must be one of {sorted(VALID_LOSS_TYPES)}, "
                f"got {loss_type!r}."
            )
        if not allow_classification and loss_type != "mse":
            raise ValueError(f"Input slot '{label}' cannot have loss_type != 'mse'.")
        return SlotDef(label=label, item_type=value["item_type"], loss_type=loss_type)

    input_slots = tuple(
        _parse_slot_def(k, v, allow_classification=False) for k, v in raw_input.items()
    )
    output_slots = tuple(
        _parse_slot_def(k, v, allow_classification=True) for k, v in raw_output.items()
    )

    # Validate label uniqueness within each side
    in_labels = [s.label for s in input_slots]
    if len(set(in_labels)) != len(in_labels):
        dups = [l for l in in_labels if in_labels.count(l) > 1]
        raise ValueError(f"Duplicate input slot labels: {dups}")

    out_labels = [s.label for s in output_slots]
    if len(set(out_labels)) != len(out_labels):
        dups = [l for l in out_labels if out_labels.count(l) > 1]
        raise ValueError(f"Duplicate output slot labels: {dups}")

    valid_in = set(in_labels)
    valid_out = set(out_labels)

    def _parse_slot_config(label: str, d: dict, task_name: str) -> SlotConfig:
        manipulation = d.get("manipulation", "default")
        if manipulation not in VALID_MANIPULATIONS:
            raise ValueError(
                f"Task '{task_name}', slot '{label}': manipulation must be one of "
                f"{set(VALID_MANIPULATIONS)}, got {manipulation!r}."
            )
        noise_std = float(d.get("noise_std", 0.0))
        if noise_std < 0.0:
            raise ValueError(
                f"Task '{task_name}', slot '{label}': noise_std must be >= 0, got {noise_std}."
            )
        return SlotConfig(
            magnitude=float(d.get("magnitude", 1.0)),
            manipulation=manipulation,
            noise_std=noise_std,
        )

    parsed_tasks: list[TaskSpec] = []
    seen_task_names: set[str] = set()
    for task_dict in tasks:
        name = task_dict.get("name")
        if not name:
            raise ValueError("Each task must have a 'name' field.")
        if name in seen_task_names:
            raise ValueError(f"Duplicate task name: {name!r}.")
        seen_task_names.add(name)

        raw_in_cfg = task_dict.get("input") or {}
        raw_out_cfg = task_dict.get("output") or {}

        # Validate references point to declared slots
        for label in raw_in_cfg:
            if label not in valid_in:
                raise ValueError(
                    f"Task '{name}': input slot '{label}' is not defined in slots.input. "
                    f"Defined labels: {in_labels}."
                )
        for label in raw_out_cfg:
            if label not in valid_out:
                raise ValueError(
                    f"Task '{name}': output slot '{label}' is not defined in slots.output. "
                    f"Defined labels: {out_labels}."
                )

        # Build full config dicts (None for unmentioned = disabled)
        input_config: dict[str, SlotConfig | None] = {
            label: (
                _parse_slot_config(label, raw_in_cfg[label] or {}, name)
                if label in raw_in_cfg
                else None
            )
            for label in in_labels
        }
        output_config: dict[str, SlotConfig | None] = {
            label: (
                _parse_slot_config(label, raw_out_cfg[label] or {}, name)
                if label in raw_out_cfg
                else None
            )
            for label in out_labels
        }

        parsed_tasks.append(TaskSpec(name=name, input_config=input_config, output_config=output_config))

    # --- Validate and normalise model_slots ---
    resolved_model_slots: dict[str, dict[str, list[str]]] = {}
    for net_name, routing in (model_slots or {}).items():
        resolved: dict[str, list[str]] = {}
        if "input" in routing:
            for label in routing["input"]:
                if label not in valid_in:
                    raise ValueError(
                        f"model_slots['{net_name}']: input slot '{label}' is not declared. "
                        f"Declared input slots: {in_labels}."
                    )
            resolved["input"] = list(routing["input"])
        if "output" in routing:
            for label in routing["output"]:
                if label not in valid_out:
                    raise ValueError(
                        f"model_slots['{net_name}']: output slot '{label}' is not declared. "
                        f"Declared output slots: {out_labels}."
                    )
            resolved["output"] = list(routing["output"])
        resolved_model_slots[net_name] = resolved

    return DatasetSpec(
        input_slots=input_slots,
        output_slots=output_slots,
        tasks=tuple(parsed_tasks),
        model_slots=resolved_model_slots,
    )
