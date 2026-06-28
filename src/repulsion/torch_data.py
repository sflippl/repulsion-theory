"""PyTorch data utilities for repulsion datasets and schedules.

This module bridges:
- :class:`repulsion.dataset.DatasetCollection` (task-specific numpy arrays)
- :class:`repulsion.schedule.TrainingPhase` (active tasks + weights + params)

Each emitted sample contains:
- ``input``: ``torch.FloatTensor``
- ``output``: ``torch.FloatTensor``
- ``task``: task name (string)
- ``task_index``: integer index into ``phase.tasks``
- ``task_weight``: scalar float (for weighted multi-task losses)
- ``row_index``: ``(subgroup, group_id, item_id)`` tuple for traceability

Note: tensors are constructed via ``torch.tensor(array.tolist())`` rather than
``torch.from_numpy`` to avoid the NumPy-bridge incompatibility between
NumPy 2.x and older PyTorch builds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from repulsion.dataset import DatasetCollection
from repulsion.schedule import TrainingPhase, TrainingSchedule


@dataclass(frozen=True)
class PhaseSample:
    """One flattened sample from one task dataset within a training phase."""

    x: np.ndarray
    y: np.ndarray
    task_name: str
    task_index: int
    task_weight: float
    row_index: tuple[str, int, int]


class PhaseTorchDataset(Dataset):
    """Task-aware PyTorch Dataset for one training phase.

    Samples from all tasks in ``phase.tasks`` are concatenated in task order.
    """

    def __init__(
        self,
        collection: DatasetCollection,
        phase: TrainingPhase,
        *,
        dtype: str = "float32",
    ) -> None:
        self._dtype = np.float32 if dtype == "float32" else np.float64
        self._samples: list[PhaseSample] = []
        # Per-task noise std arrays (shape: total_dim); 0.0 where no noise.
        self._noise_stds_x: list[np.ndarray] = []
        self._noise_stds_y: list[np.ndarray] = []
        self._has_noise_x: list[bool] = []
        self._has_noise_y: list[bool] = []

        for task_idx, task_name in enumerate(phase.tasks):
            task_ds = collection[task_name]
            weight = float(phase.weights[task_idx])

            stds_x = task_ds.input_noise_stds.astype(self._dtype)
            stds_y = task_ds.output_noise_stds.astype(self._dtype)
            self._noise_stds_x.append(stds_x)
            self._noise_stds_y.append(stds_y)
            self._has_noise_x.append(bool(stds_x.any()))
            self._has_noise_y.append(bool(stds_y.any()))

            for row_idx, row in enumerate(task_ds.rows):
                self._samples.append(
                    PhaseSample(
                        x=np.asarray(task_ds.input[row_idx], dtype=self._dtype),
                        y=np.asarray(task_ds.output[row_idx], dtype=self._dtype),
                        task_name=task_name,
                        task_index=task_idx,
                        task_weight=weight,
                        row_index=(row.subgroup, row.group_id, row.item_id),
                    )
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        ti = s.task_index
        x = s.x
        y = s.y
        if self._has_noise_x[ti]:
            x = x + np.random.standard_normal(x.shape).astype(self._dtype) * self._noise_stds_x[ti]
        if self._has_noise_y[ti]:
            y = y + np.random.standard_normal(y.shape).astype(self._dtype) * self._noise_stds_y[ti]
        return {
            "input": torch.tensor(x.tolist()),
            "output": torch.tensor(y.tolist()),
            "task": s.task_name,
            "task_index": ti,
            "task_weight": float(s.task_weight),
            "row_index": s.row_index,
        }


def build_phase_dataloader(
    collection: DatasetCollection,
    phase: TrainingPhase,
    *,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    dtype: str = "float32",
):
    """Build a PyTorch DataLoader for one training phase.

    Batch-size resolution:
    1. explicit ``batch_size`` argument if provided
    2. ``phase.params.batch_size``
    3. full-batch fallback (dataset length)
    """
    ds = PhaseTorchDataset(collection, phase, dtype=dtype)
    if len(ds) == 0:
        raise ValueError("Cannot build DataLoader from an empty phase dataset.")

    resolved_bs = batch_size if batch_size is not None else phase.params.batch_size
    if resolved_bs is None:
        resolved_bs = len(ds)
    if resolved_bs <= 0:
        raise ValueError(f"batch_size must be positive or None, got {resolved_bs}.")

    return DataLoader(
        ds,
        batch_size=resolved_bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def build_schedule_dataloaders(
    collection: DatasetCollection,
    schedule: TrainingSchedule,
    *,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    dtype: str = "float32",
) -> list:
    """Build one DataLoader per schedule phase, preserving phase order."""
    loaders = []
    for phase in schedule.phases:
        loaders.append(
            build_phase_dataloader(
                collection,
                phase,
                batch_size=None,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                dtype=dtype,
            )
        )
    return loaders


# ---------------------------------------------------------------------------
# Joint multi-task dataset
# ---------------------------------------------------------------------------

class JointPhaseTorchDataset(Dataset):
    """Joint multi-task dataset: each sample contains all tasks' inputs/outputs.

    Each row emitted corresponds to one data point (shared row structure across
    all tasks).  The batch tensors ``inputs`` and ``outputs`` are shaped
    ``(batch, n_tasks, dim)`` so the training loop can do one forward pass per
    task and sum the weighted losses before a single ``backward()``.

    All tasks in the phase must share the same row structure (same subgroups,
    group counts, and item counts).  This is validated at construction time.
    """

    def __init__(
        self,
        collection: DatasetCollection,
        phase: TrainingPhase,
        *,
        dtype: str = "float32",
    ) -> None:
        self._dtype = np.float32 if dtype == "float32" else np.float64
        task_datasets = [collection[t] for t in phase.tasks]

        # Validate shared row structure across all tasks
        ref_rows = task_datasets[0].rows
        for i, ds in enumerate(task_datasets[1:], 1):
            if ds.rows != ref_rows:
                raise ValueError(
                    f"Joint training requires all tasks to share the same row structure. "
                    f"Task '{phase.tasks[0]}' and '{phase.tasks[i]}' have different rows."
                )

        self._rows = ref_rows
        self._task_inputs = [ds.input for ds in task_datasets]    # list of (N, input_dim)
        self._task_outputs = [ds.output for ds in task_datasets]  # list of (N, output_dim)
        self._weights = list(phase.weights)
        self._task_names = list(phase.tasks)
        # Per-task noise std arrays (shape: total_dim); 0.0 where no noise.
        self._noise_stds_x = [ds.input_noise_stds.astype(self._dtype) for ds in task_datasets]
        self._noise_stds_y = [ds.output_noise_stds.astype(self._dtype) for ds in task_datasets]
        self._has_noise_x = [bool(s.any()) for s in self._noise_stds_x]
        self._has_noise_y = [bool(s.any()) for s in self._noise_stds_y]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        xs = []
        ys = []
        for i, (task_x, task_y) in enumerate(zip(self._task_inputs, self._task_outputs)):
            x = np.asarray(task_x[idx], dtype=self._dtype)
            y = np.asarray(task_y[idx], dtype=self._dtype)
            if self._has_noise_x[i]:
                x = x + np.random.standard_normal(x.shape).astype(self._dtype) * self._noise_stds_x[i]
            if self._has_noise_y[i]:
                y = y + np.random.standard_normal(y.shape).astype(self._dtype) * self._noise_stds_y[i]
            xs.append(x)
            ys.append(y)
        inputs = np.stack(xs)   # (n_tasks, input_dim)
        outputs = np.stack(ys)  # (n_tasks, output_dim)
        return {
            "inputs": torch.tensor(inputs.tolist()),     # (n_tasks, input_dim)
            "outputs": torch.tensor(outputs.tolist()),   # (n_tasks, output_dim)
            "task_weights": torch.tensor(self._weights, dtype=torch.float32),  # (n_tasks,)
            "row_index": (row.subgroup, row.group_id, row.item_id),
        }


def build_joint_phase_dataloader(
    collection: DatasetCollection,
    phase: TrainingPhase,
    *,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    dtype: str = "float32",
):
    """Build a DataLoader for a joint multi-task phase.

    Each batch item carries all tasks' inputs/outputs stacked along a task
    dimension so the training loop can accumulate losses across tasks before
    a single ``backward()``.
    """
    ds = JointPhaseTorchDataset(collection, phase, dtype=dtype)
    if len(ds) == 0:
        raise ValueError("Cannot build DataLoader from an empty joint phase dataset.")

    resolved_bs = batch_size if batch_size is not None else phase.params.batch_size
    if resolved_bs is None:
        resolved_bs = len(ds)

    return DataLoader(
        ds,
        batch_size=resolved_bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
