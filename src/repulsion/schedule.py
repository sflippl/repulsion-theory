"""Recursive training schedule parsing and expansion.

The schedule language supports three component forms under ``component_type``:

1) ``"task_name"``
   Atomic single-task phase.

2) ``["task_a", "task_b", ...]``
   Atomic multi-task phase (trained simultaneously with weighted losses).

3) ``[{...}, {...}, ...]``
   Recursive sequence of schedule components.

Each node can optionally override optimization parameters:
``batch_size``, ``lr``, ``momentum``.

Duration can be specified via exactly one of ``epochs`` or ``steps``:
- For atomic nodes: required.
- For sequence nodes: optional; when provided, it repeats the entire sequence.
  (e.g., sequence with ``epochs: 8`` repeats children 8 times.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class TrainingParams:
    """Resolved optimization parameters for one phase."""

    batch_size: Optional[int] = None
    lr: Optional[float] = None
    momentum: Optional[float] = None


@dataclass(frozen=True)
class TrainingPhase:
    """One concrete, executable training phase after schedule expansion."""

    tasks: tuple[str, ...]
    weights: tuple[float, ...]
    epochs: Optional[int]
    steps: Optional[int]
    params: TrainingParams
    joint: bool = False


@dataclass(frozen=True)
class TrainingSchedule:
    """Expanded schedule as a flat list of concrete phases."""

    phases: tuple[TrainingPhase, ...]

    @property
    def total_epochs(self) -> int:
        """Total epochs across phases that are epoch-based."""
        return sum(p.epochs or 0 for p in self.phases)

    @property
    def total_steps(self) -> int:
        """Total steps across phases that are step-based."""
        return sum(p.steps or 0 for p in self.phases)


def _normalize_batch_size(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"none", "null"}:
            return None
        value = int(value)
    b = int(value)
    if b <= 0:
        raise ValueError(f"batch_size must be positive or None, got {value!r}.")
    return b


def _resolve_params(node: dict, parent: TrainingParams) -> TrainingParams:
    return TrainingParams(
        batch_size=_normalize_batch_size(node.get("batch_size", parent.batch_size)),
        lr=float(node["lr"]) if "lr" in node and node["lr"] is not None else parent.lr,
        momentum=(
            float(node["momentum"])
            if "momentum" in node and node["momentum"] is not None
            else parent.momentum
        ),
    )


def _parse_duration(node: dict, *, required: bool, where: str) -> tuple[Optional[int], Optional[int]]:
    has_epochs = "epochs" in node and node["epochs"] is not None
    has_steps = "steps" in node and node["steps"] is not None

    if has_epochs and has_steps:
        raise ValueError(f"{where}: specify only one of 'epochs' or 'steps'.")
    if required and not (has_epochs or has_steps):
        raise ValueError(f"{where}: atomic component must define 'epochs' or 'steps'.")
    if not (has_epochs or has_steps):
        return None, None

    if has_epochs:
        epochs = int(node["epochs"])
        if epochs <= 0:
            raise ValueError(f"{where}: epochs must be > 0, got {epochs}.")
        return epochs, None

    steps = int(node["steps"])
    if steps <= 0:
        raise ValueError(f"{where}: steps must be > 0, got {steps}.")
    return None, steps


def _normalize_weights(raw_weights: object, n_tasks: int, where: str) -> tuple[float, ...]:
    if raw_weights is None:
        return tuple(1.0 for _ in range(n_tasks))

    if isinstance(raw_weights, (int, float)):
        if n_tasks == 1:
            return (float(raw_weights),)
        return tuple(float(raw_weights) for _ in range(n_tasks))

    if not isinstance(raw_weights, list):
        raise ValueError(f"{where}: weights must be a number or list of numbers.")

    if len(raw_weights) != n_tasks:
        raise ValueError(
            f"{where}: weights length ({len(raw_weights)}) must match number of tasks ({n_tasks})."
        )
    return tuple(float(w) for w in raw_weights)


def _validate_tasks(tasks: tuple[str, ...], available_tasks: Optional[set[str]], where: str) -> None:
    if available_tasks is None:
        return
    unknown = [t for t in tasks if t not in available_tasks]
    if unknown:
        raise ValueError(
            f"{where}: unknown task(s) {unknown}. Available tasks: {sorted(available_tasks)}."
        )


def _expand_node(
    node: dict,
    parent_params: TrainingParams,
    available_tasks: Optional[set[str]],
    path: str,
) -> list[TrainingPhase]:
    if "component_type" not in node:
        raise ValueError(f"{path}: missing required key 'component_type'.")

    component = node["component_type"]
    params = _resolve_params(node, parent_params)

    # Case 1: atomic single-task phase
    if isinstance(component, str):
        epochs, steps = _parse_duration(node, required=True, where=path)
        tasks = (component,)
        _validate_tasks(tasks, available_tasks, path)
        weights = _normalize_weights(node.get("weights"), n_tasks=1, where=path)
        return [TrainingPhase(tasks=tasks, weights=weights, epochs=epochs, steps=steps, params=params)]

    # Case 2/3: list - either atomic multi-task or recursive sequence
    if not isinstance(component, list) or not component:
        raise ValueError(
            f"{path}: component_type must be a non-empty string or list, got {component!r}."
        )

    # Atomic multi-task: list[str]
    if all(isinstance(x, str) for x in component):
        epochs, steps = _parse_duration(node, required=True, where=path)
        tasks = tuple(component)
        _validate_tasks(tasks, available_tasks, path)
        weights = _normalize_weights(node.get("weights"), n_tasks=len(tasks), where=path)
        joint = bool(node.get("joint", False))
        return [TrainingPhase(tasks=tasks, weights=weights, epochs=epochs, steps=steps, params=params, joint=joint)]

    # Recursive sequence: list[dict]
    if all(isinstance(x, dict) for x in component):
        if "weights" in node:
            raise ValueError(f"{path}: 'weights' is not valid for a sequence component.")

        # Optional repeat count for the entire sequence.
        rep_epochs, rep_steps = _parse_duration(node, required=False, where=path)
        if rep_steps is not None:
            raise ValueError(f"{path}: sequence component currently supports 'epochs' repeats, not 'steps'.")
        repeats = rep_epochs if rep_epochs is not None else 1

        out: list[TrainingPhase] = []
        for rep in range(repeats):
            for idx, child in enumerate(component):
                child_path = f"{path}.component_type[{idx}]#rep{rep + 1}"
                out.extend(_expand_node(child, params, available_tasks, child_path))
        return out

    raise ValueError(
        f"{path}: component_type list must be all strings (multi-task) or all dicts (sequence)."
    )


def build_training_schedule(
    schedule: dict,
    available_tasks: Optional[Iterable[str]] = None,
) -> TrainingSchedule:
    """Build and expand a recursive training schedule.

    Args:
        schedule: Root schedule node containing ``component_type``.
        available_tasks: Optional task-name whitelist for validation.

    Returns:
        Expanded :class:`TrainingSchedule` with flattened phases.

    Example (recursive sequence)::

        schedule = {
            "component_type": [
                {"component_type": "autoencoding", "epochs": 2},
                {"component_type": "pairmate_prediction", "epochs": 1, "weights": 0.1},
            ],
            "batch_size": 1,
            "lr": 0.1,
            "momentum": 0.9,
            "epochs": 8,
        }

    Example (simultaneous multi-task)::

        schedule = {
            "component_type": ["autoencoding", "pairmate_prediction"],
            "weights": [1.0, 0.1],
            "epochs": 10,
            "batch_size": "none",
        }
    """
    initial = TrainingParams(batch_size=None, lr=None, momentum=None)
    available = set(available_tasks) if available_tasks is not None else None
    phases = _expand_node(schedule, initial, available, path="schedule")
    return TrainingSchedule(phases=tuple(phases))
