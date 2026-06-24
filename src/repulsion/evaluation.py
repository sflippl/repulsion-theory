"""Evaluation of :class:`MultiNetwork` models during (or after) training.

Evaluations are specified as dicts — or :class:`EvalSpec` dataclasses — that
describe what to capture, from which task/network/layer, and how often.

What can be saved
-----------------
* ``"representation"`` — activations at a named pipeline stage of one stream.
  When ``network`` is ``None``, all streams are saved.  When ``layer`` is
  ``None``, all hidden layers (``hidden_0``, ``hidden_1``, …) of each stream
  are saved.  Files are keyed as *name*\_netN\_layerName.
  Valid stage names: ``input``, ``post_attention``, ``post_projection``,
  ``hidden_N`` (0-indexed), ``output``, ``scattered``.

* ``"output"`` — per-slot predictions for active slots only (slots whose
  target is not all-NaN for the task).  Written per-stream
  (*name*\\_netN_slotLabel) and as the combined sum
  (*name*\\_combined_slotLabel).  Shape: ``(n_checkpoints, n_samples, slot_pred_dim)``.
  A companion *name*\\_target_slotLabel.npy holds the ground-truth target
  for each active slot.

* ``"loss"`` — per-output-slot aggregated loss, written per-stream and combined.
  Shape: ``(n_checkpoints, n_slots)``.  A companion ``*_meta.json`` records
  slot names.

Eval frequency
--------------
* ``eval_every_steps`` — evaluate whenever ``step % N == 0``.
* ``eval_every_log_steps`` — evaluate at geometrically spaced steps.  The
  value is the geometric *factor* (e.g. ``2.0`` gives steps 0, 1, 2, 4, 8 …;
  ``1.01`` gives dense early sampling and progressively sparser later).

File layout
-----------
All files land in ``output_dir`` (created if missing).  For each spec named
``foo``:

* ``foo_steps.npy``         — 1-D int64 array of step numbers at each checkpoint.
* ``foo_data.npy``          — representations (n_checkpoints, n_samples, dim).
* ``foo_net0_slotLabel.npy`` … — per-stream per-slot prediction arrays.
* ``foo_combined_slotLabel.npy`` — sum-of-streams per-slot prediction array.
* ``foo_target_slotLabel.npy``  — ground-truth target for each active slot.
* ``foo_meta.json``         — dict with task, save type, slot names, etc.

Training loss (written by :func:`repulsion.training.train_schedule` when
*output_dir* is given):

* ``training_loss.csv``     — one row per gradient step: step, total_loss, <slots>.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from repulsion.dataset import DatasetCollection
from repulsion.models.network import MultiNetwork
from repulsion.training import LossSpec, compute_loss


# ---------------------------------------------------------------------------
# Spec and parsing
# ---------------------------------------------------------------------------

@dataclass
class EvalSpec:
    """Specification for one evaluation probe.

    Args:
        name: File-name stem (no extension) used for all output files.
        task: Task name to run the forward pass on.
        save: One of ``"representation"``, ``"output"``, ``"loss"``.
        layer: Pipeline stage for ``save="representation"``.  See module
            docstring for valid names.  Ignored for output/loss.
        network: 0-indexed stream index, required for ``save="representation"``.
        eval_every_steps: Evaluate when ``step % eval_every_steps == 0``.
        eval_every_log_steps: Geometric factor for logarithmically-spaced evals.
    """
    name: str
    task: str
    save: str
    layer: Optional[str] = None
    network: Optional[int] = None
    eval_every_steps: Optional[int] = None
    eval_every_log_steps: Optional[float] = None


def parse_eval_spec(d: dict) -> EvalSpec:
    """Build an :class:`EvalSpec` from a plain dict."""
    save = d.get("save", "output")
    if save not in ("representation", "output", "loss"):
        raise ValueError(
            f"EvalSpec 'save' must be 'representation', 'output', or 'loss'; got '{save!r}'."
        )
    # layer and network are optional for representation; None means "all"
    if "name" not in d:
        raise ValueError("EvalSpec requires 'name'.")
    if "task" not in d:
        raise ValueError("EvalSpec requires 'task'.")
    return EvalSpec(
        name=d["name"],
        task=d["task"],
        save=save,
        layer=d.get("layer"),
        network=d.get("network"),
        eval_every_steps=d.get("eval_every_steps"),
        eval_every_log_steps=d.get("eval_every_log_steps"),
    )


def parse_eval_specs(specs: list[dict]) -> list[EvalSpec]:
    """Parse a list of spec dicts into :class:`EvalSpec` objects."""
    return [parse_eval_spec(s) for s in specs]


# ---------------------------------------------------------------------------
# Geometric-step scheduler
# ---------------------------------------------------------------------------

class _LogSchedule:
    """Tracks geometrically-spaced evaluation steps.

    After evaluating at step *t*, the next evaluation is at the smallest
    integer > *t* that is >= ``t * factor``.  This ensures every step is
    covered when *factor* is close to 1 (e.g. 1.01) and produces clean
    doubling when *factor* = 2.

    Step 0 is always due on the first call to :meth:`due`.
    """

    def __init__(self, factor: float) -> None:
        if factor <= 1.0:
            raise ValueError(
                f"eval_every_log_steps factor must be > 1.0; got {factor}."
            )
        self.factor = factor
        self._next: int = 0       # step 0 is always the first eval
        self._float: float = 1.0  # running float used to compute next

    def due(self, step: int) -> bool:
        return step >= self._next

    def advance(self, step: int) -> None:
        """Advance the schedule after evaluating at *step*."""
        # Multiply until the integer part strictly exceeds the current step.
        while int(self._float) <= step:
            self._float *= self.factor
        self._next = int(self._float)


# ---------------------------------------------------------------------------
# Internal eval state
# ---------------------------------------------------------------------------

@dataclass
class _EvalState:
    spec: EvalSpec
    log_schedule: Optional[_LogSchedule]
    steps: list[int] = field(default_factory=list)
    # key (e.g. "data", "net0", "combined") → list of per-checkpoint arrays
    arrays: dict[str, list[np.ndarray]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------

def _save_incremental(
    output_dir: str,
    stem: str,
    steps: list[int],
    arrays: dict[str, list[np.ndarray]],
    meta: dict,
) -> None:
    """Rewrite .npy files with all checkpoints accumulated so far."""
    np.save(
        os.path.join(output_dir, f"{stem}_steps.npy"),
        np.array(steps, dtype=np.int64),
    )
    for key, arr_list in arrays.items():
        stacked = np.stack(arr_list, axis=0)   # (n_checkpoints, ...)
        np.save(os.path.join(output_dir, f"{stem}_{key}.npy"), stacked)
    with open(os.path.join(output_dir, f"{stem}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Runs scheduled evaluations and writes results incrementally to disk.

    Construct via :func:`build_evaluator` rather than calling directly.

    Args:
        specs: Evaluation specifications.
        collection: Dataset collection (for task data).
        model: The :class:`MultiNetwork` being trained.
        loss_spec: Precomputed loss layout from
            :func:`repulsion.training.build_loss_spec`.
        output_dir: Directory for .npy output files (created if absent).
        device: PyTorch device string.
    """

    def __init__(
        self,
        specs: list[EvalSpec],
        collection: DatasetCollection,
        model: MultiNetwork,
        loss_spec: LossSpec,
        output_dir: str,
        device: str = "cpu",
    ) -> None:
        self.collection = collection
        self.model = model
        self.loss_spec = loss_spec
        self.output_dir = output_dir
        self.device = torch.device(device)
        os.makedirs(output_dir, exist_ok=True)

        self._states: list[_EvalState] = []
        for spec in specs:
            log_sched = (
                _LogSchedule(spec.eval_every_log_steps)
                if spec.eval_every_log_steps is not None
                else None
            )
            self._states.append(_EvalState(spec=spec, log_schedule=log_sched))

        # Cache: task_name → (x, y, w, sample_ids)
        self._task_cache: dict[str, tuple] = {}
        # Track which probe names have already had their rows file written
        self._rows_written: set[str] = set()
        # Track which probe names have already had their target file written
        self._target_written: set[str] = set()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def maybe_run_all(self, step: int) -> None:
        """Check every spec and run evaluations that are due at *step*.

        Called by :func:`repulsion.training.train_schedule` at step 0
        (pre-training) and after each gradient update.
        """
        for state in self._states:
            if self._due(state, step):
                self._run(state, step)
                if state.log_schedule is not None:
                    state.log_schedule.advance(step)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _due(self, state: _EvalState, step: int) -> bool:
        spec = state.spec
        if spec.eval_every_steps is not None and step % spec.eval_every_steps == 0:
            return True
        if state.log_schedule is not None and state.log_schedule.due(step):
            return True
        return False

    # ------------------------------------------------------------------
    # Task data loading (cached, no shuffle, ordered)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _load_task(self, task_name: str) -> tuple:
        """Return ``(x, y, w, sample_ids)`` for all rows of *task_name*, ordered."""
        if task_name not in self._task_cache:
            ds = self.collection[task_name]
            x = torch.tensor(ds.input.tolist(),  dtype=torch.float32, device=self.device)
            y = torch.tensor(ds.output.tolist(), dtype=torch.float32, device=self.device)
            w = torch.ones(len(ds.rows), dtype=torch.float32, device=self.device)
            sample_ids: Optional[torch.Tensor] = None
            if self.model.row_index_to_id is not None:
                ids = [
                    self.model.row_index_to_id[(r.subgroup, r.group_id, r.item_id)]
                    for r in ds.rows
                ]
                sample_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
            self._task_cache[task_name] = (x, y, w, sample_ids)
        return self._task_cache[task_name]

    # ------------------------------------------------------------------
    # Evaluation runners
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run(self, state: _EvalState, step: int) -> None:
        spec = state.spec
        x, y, w, sample_ids = self._load_task(spec.task)
        state.steps.append(step)

        if spec.save == "representation":
            self._run_representation(state, x, sample_ids)
        elif spec.save == "output":
            self._run_output(state, x, y, sample_ids)
        elif spec.save == "loss":
            self._run_loss(state, x, y, w, sample_ids)

    def _save_rows_once(self, spec_name: str, task_name: str) -> None:
        """Write {name}_rows.json the first time a probe runs (rows are constant)."""
        if spec_name in self._rows_written:
            return
        rows = self.collection[task_name].rows
        rows_data = [
            {"subgroup": r.subgroup, "group_id": r.group_id, "item_id": r.item_id}
            for r in rows
        ]
        path = os.path.join(self.output_dir, f"{spec_name}_rows.json")
        with open(path, "w") as f:
            json.dump(rows_data, f)
        self._rows_written.add(spec_name)

    def _run_representation(
        self,
        state: _EvalState,
        x: torch.Tensor,
        sample_ids: Optional[torch.Tensor],
    ) -> None:
        spec = state.spec
        self._save_rows_once(spec.name, spec.task)
        network_indices = (
            [spec.network]
            if spec.network is not None
            else list(range(len(self.model.networks)))
        )
        saved: dict[str, list] = {}
        meta_entries = []
        for ni in network_indices:
            net = self.model.networks[ni]
            layers = (
                [spec.layer]
                if spec.layer is not None
                else net.hidden_layer_names
            )
            for layer in layers:
                key = f"net{ni}_{layer}"
                arr = np.array(net.extract(x, sample_ids, layer).tolist())
                state.arrays.setdefault(key, []).append(arr)
                saved[key] = state.arrays[key]
                meta_entries.append({"key": key, "network": ni, "layer": layer})
        _save_incremental(
            self.output_dir, spec.name,
            state.steps, saved,
            meta={
                "task": spec.task,
                "save": "representation",
                "layers": meta_entries,
            },
        )

    def _run_output(
        self,
        state: _EvalState,
        x: torch.Tensor,
        y: torch.Tensor,
        sample_ids: Optional[torch.Tensor],
    ) -> None:
        spec = state.spec
        self._save_rows_once(spec.name, spec.task)

        # Determine which slots are active for this task (have at least one
        # non-NaN target value).
        active_slots = [
            slot for slot in self.loss_spec.slots
            if not torch.isnan(y[:, slot.target_offset:slot.target_offset + slot.target_dim]).all()
        ]

        # Save ground truth per active slot once — targets never change.
        if spec.name not in self._target_written:
            for slot in active_slots:
                t_slice = y[:, slot.target_offset:slot.target_offset + slot.target_dim]
                np.save(
                    os.path.join(self.output_dir, f"{spec.name}_target_{slot.label}.npy"),
                    np.array(t_slice.tolist()),
                )
            self._target_written.add(spec.name)

        # Accumulate per-slot predictions for each network.
        combined_pred: Optional[torch.Tensor] = None
        for ni, net in enumerate(self.model.networks):
            pred = net(x, sample_ids)
            combined_pred = pred if combined_pred is None else combined_pred + pred
            for slot in active_slots:
                p_slice = np.array(
                    pred[:, slot.pred_offset:slot.pred_offset + slot.pred_dim].tolist()
                )
                state.arrays.setdefault(f"net{ni}_{slot.label}", []).append(p_slice)

        for slot in active_slots:
            p_slice = np.array(
                combined_pred[:, slot.pred_offset:slot.pred_offset + slot.pred_dim].tolist()
            )
            state.arrays.setdefault(f"combined_{slot.label}", []).append(p_slice)

        active_slot_names = [s.label for s in active_slots]
        _save_incremental(
            self.output_dir, spec.name,
            state.steps, state.arrays,
            meta={
                "task": spec.task,
                "save": "output",
                "n_networks": len(self.model.networks),
                "active_slots": active_slot_names,
            },
        )

    def _run_loss(
        self,
        state: _EvalState,
        x: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        sample_ids: Optional[torch.Tensor],
    ) -> None:
        spec = state.spec
        slot_names = [s.label for s in self.loss_spec.slots]
        combined_pred: Optional[torch.Tensor] = None
        for ni, net in enumerate(self.model.networks):
            pred_i = net(x, sample_ids)
            _, slot_losses = compute_loss(pred_i, y, w, self.loss_spec)
            row = np.array(
                [slot_losses.get(s, 0.0) for s in slot_names], dtype=np.float32
            )
            state.arrays.setdefault(f"net{ni}", []).append(row)
            combined_pred = pred_i if combined_pred is None else combined_pred + pred_i
        _, slot_losses_combined = compute_loss(combined_pred, y, w, self.loss_spec)
        row_c = np.array(
            [slot_losses_combined.get(s, 0.0) for s in slot_names], dtype=np.float32
        )
        state.arrays.setdefault("combined", []).append(row_c)
        _save_incremental(
            self.output_dir, spec.name,
            state.steps, state.arrays,
            meta={"task": spec.task, "save": "loss", "slots": slot_names},
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_evaluator(
    specs: list[dict | EvalSpec],
    collection: DatasetCollection,
    model: MultiNetwork,
    loss_spec: LossSpec,
    output_dir: str,
    device: str = "cpu",
) -> Evaluator:
    """Build an :class:`Evaluator` from a list of spec dicts or :class:`EvalSpec` objects.

    Args:
        specs: Each entry is either an :class:`EvalSpec` or a plain dict
            (passed to :func:`parse_eval_spec`).
        collection: Dataset collection.
        model: The :class:`MultiNetwork` to evaluate.
        loss_spec: Precomputed loss layout.
        output_dir: Directory for .npy output files.
        device: PyTorch device string.
    """
    parsed = [
        s if isinstance(s, EvalSpec) else parse_eval_spec(s)
        for s in specs
    ]
    return Evaluator(parsed, collection, model, loss_spec, output_dir, device)
