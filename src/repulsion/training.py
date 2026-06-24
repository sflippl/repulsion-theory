"""Training loop for :class:`MultiNetwork` models.

Usage::

    from repulsion import build_training_schedule
    from repulsion.dataset import build_datasets
    from repulsion.models import parse_model_spec
    from repulsion.training import TrainingConfig, train_schedule

    collection = build_datasets(item_set, ...)
    model = parse_model_spec([{"input": [...], "output": [...]}], collection)
    schedule = build_training_schedule({"component_type": "task", "epochs": 10})
    history = train_schedule(model, collection, schedule, TrainingConfig())
"""
from __future__ import annotations

import csv
from logging import config
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from repulsion.evaluation import Evaluator

from repulsion.dataset import DatasetCollection
from repulsion.models.network import MultiNetwork
from repulsion.schedule import TrainingPhase, TrainingSchedule
from repulsion.torch_data import build_phase_dataloader, build_joint_phase_dataloader


# ---------------------------------------------------------------------------
# Loss specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LossSlot:
    """Per-slot metadata needed to compute the loss."""
    label: str
    loss_type: str          # "mse", "classify_group", "classify_item"
    pred_offset: int        # start col in prediction tensor
    pred_dim: int           # width in prediction tensor (n_classes for classify)
    target_offset: int      # start col in target tensor
    target_dim: int         # width in target tensor (1 for classify, slot_dim for mse)


@dataclass(frozen=True)
class LossSpec:
    """Precomputed layout needed by :func:`compute_loss`."""
    slots: list[LossSlot]
    global_pred_dim: int
    global_target_dim: int


def build_loss_spec(collection: DatasetCollection) -> LossSpec:
    """Build :class:`LossSpec` from a :class:`DatasetCollection`.

    Uses the first task in the collection; all tasks must share the same
    output slot definitions.
    """
    first_task = collection[collection.task_names()[0]]
    slots: list[LossSlot] = []
    target_offset = 0
    pred_offset = 0
    for label in first_task.output_slot_dims:
        target_dim = first_task.output_slot_dims[label]
        pred_dim = first_task.output_prediction_dims[label]
        loss_type = first_task.output_slot_types[label]
        slots.append(LossSlot(
            label=label,
            loss_type=loss_type,
            pred_offset=pred_offset,
            pred_dim=pred_dim,
            target_offset=target_offset,
            target_dim=target_dim,
        ))
        target_offset += target_dim
        pred_offset += pred_dim
    return LossSpec(slots=slots, global_pred_dim=pred_offset, global_target_dim=target_offset)


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    task_weights: torch.Tensor,
    loss_spec: LossSpec,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted, NaN-masked loss across all output slots.

    For each slot:
    * Active samples are those whose target is not NaN.
    * MSE slots: mean squared error averaged over dimensions, then
      weighted by ``task_weights`` and averaged over active samples.
    * Classification slots: cross-entropy loss, same weighting.
    * Disabled (all-NaN) slots contribute 0 to the total.

    Args:
        pred: ``(batch, global_pred_dim)`` — model output.
        target: ``(batch, global_target_dim)`` — target from dataloader.
        task_weights: ``(batch,)`` per-sample task weight.
        loss_spec: Precomputed layout from :func:`build_loss_spec`.

    Returns:
        ``(total_loss, per_slot_loss_dict)`` where ``total_loss`` is a
        differentiable scalar and per-slot values are plain floats.
    """
    total_loss: torch.Tensor = pred.new_zeros(())
    slot_losses: dict[str, float] = {}

    for slot in loss_spec.slots:
        p_slice = pred[:, slot.pred_offset:slot.pred_offset + slot.pred_dim]
        t_slice = target[:, slot.target_offset:slot.target_offset + slot.target_dim]

        active = ~torch.isnan(t_slice[:, 0])
        if not active.any():
            slot_losses[slot.label] = 0.0
            continue

        if slot.loss_type == "mse":
            per_sample = ((p_slice[active] - t_slice[active]) ** 2).sum(dim=-1)
        else:  # classify_group or classify_item
            labels = t_slice[active, 0].long()
            per_sample = F.cross_entropy(p_slice[active], labels, reduction="none")

        weighted = (per_sample * task_weights[active]).mean()
        total_loss = total_loss + weighted
        slot_losses[slot.label] = weighted.item()

    return total_loss, slot_losses


# ---------------------------------------------------------------------------
# Training configuration and history
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hyperparameters for the training loop.

    Args:
        optimizer: ``"adam"`` (default), ``"adamw"``, or ``"sgd"``.
        lr: Learning rate applied to the joint optimizer or the network
            optimizer when using separate attention training.
        momentum: SGD momentum (ignored for Adam/AdamW).
        separate_attention: If ``True``, train the MLP and the attention
            layer with separate optimizers and independent update steps.
        attention_optimizer: Optimizer for the attention parameters.
        attention_lr: Learning rate for the attention optimizer.
        attention_momentum: SGD momentum for the attention optimizer.
        network_steps: Number of network-parameter update steps per batch
            in separate-attention mode.
        attention_steps: Number of attention-parameter update steps per batch
            in separate-attention mode.
    """
    optimizer: str = "sgd"
    lr: float = 1e-3
    momentum: float = 0.9
    separate_attention: bool = False
    attention_optimizer: str = "sgd"
    attention_lr: float = 1e-3
    attention_momentum: float = 0.9
    network_steps: int = 1
    attention_steps: int = 1


@dataclass
class StepRecord:
    """One gradient-update step's loss values."""
    step: int
    total_loss: float
    per_slot_losses: dict[str, float]


@dataclass
class PhaseHistory:
    """Training history for one :class:`TrainingPhase`."""
    phase_idx: int
    phase: TrainingPhase
    steps: list[StepRecord] = field(default_factory=list)


@dataclass
class TrainingHistory:
    """Full training history returned by :func:`train_schedule`."""
    phases: list[PhaseHistory] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_optimizer(
    name: str,
    params: list[torch.nn.Parameter],
    lr: float,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    raise ValueError(f"Unknown optimizer '{name}'. Choose 'adam', 'adamw', or 'sgd'.")


def _do_step(
    model: MultiNetwork,
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    sample_ids: Optional[torch.Tensor],
    loss_spec: LossSpec,
    optimizers: tuple,
    config: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One or more optimiser steps; returns the final (loss, per_slot_losses)."""
    loss: torch.Tensor
    slot_losses: dict[str, float]

    if len(optimizers) == 1:
        # Joint training
        opt = optimizers[0]
        opt.zero_grad()
        pred = model(x, sample_ids)
        loss, slot_losses = compute_loss(pred, y, weights, loss_spec)
        loss.backward()
        opt.step()
    else:
        # Separate attention training
        net_opt, attn_opt = optimizers

        for _ in range(config.network_steps):
            net_opt.zero_grad()
            attn_opt.zero_grad()
            pred = model(x, sample_ids)
            loss, slot_losses = compute_loss(pred, y, weights, loss_spec)
            loss.backward()
            net_opt.step()

        for _ in range(config.attention_steps):
            net_opt.zero_grad()
            attn_opt.zero_grad()
            pred = model(x, sample_ids)
            loss, slot_losses = compute_loss(pred, y, weights, loss_spec)
            loss.backward()
            attn_opt.step()

    return loss, slot_losses


def _do_joint_step(
    model: MultiNetwork,
    batch: dict,
    loss_spec: LossSpec,
    optimizers: tuple,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint multi-task step: one forward pass per task, single backward.

    For each task, runs the model on that task's input and accumulates the
    weighted loss.  A single ``backward()`` is called on the sum so all
    tasks share one gradient update.
    """
    inputs = batch["inputs"].float().to(device)       # (batch, n_tasks, input_dim)
    outputs = batch["outputs"].float().to(device)     # (batch, n_tasks, output_dim)
    task_weights = batch["task_weights"].float().to(device)  # (batch, n_tasks)
    sample_ids = model.get_sample_ids(batch["row_index"], device)

    n_tasks = inputs.shape[1]
    total_loss: torch.Tensor = inputs.new_zeros(())
    all_slot_losses: dict[str, float] = {}

    for ti in range(n_tasks):
        x = inputs[:, ti, :]       # (batch, input_dim)
        y = outputs[:, ti, :]      # (batch, output_dim)
        w = task_weights[:, ti]    # (batch,)
        pred = model(x, sample_ids)
        task_loss, slot_losses = compute_loss(pred, y, w, loss_spec)
        total_loss = total_loss + task_loss
        for k, v in slot_losses.items():
            all_slot_losses[k] = all_slot_losses.get(k, 0.0) + v

    if len(optimizers) == 1:
        opt = optimizers[0]
        opt.zero_grad()
        total_loss.backward()
        opt.step()
    else:
        net_opt, attn_opt = optimizers
        for _ in range(config.network_steps):
            net_opt.zero_grad()
            attn_opt.zero_grad()
            # recompute to get fresh graph for each inner step
            total_loss_inner: torch.Tensor = inputs.new_zeros(())
            for ti in range(n_tasks):
                pred = model(inputs[:, ti, :], sample_ids)
                tl, _ = compute_loss(pred, outputs[:, ti, :], task_weights[:, ti], loss_spec)
                total_loss_inner = total_loss_inner + tl
            total_loss_inner.backward()
            net_opt.step()
        for _ in range(config.attention_steps):
            net_opt.zero_grad()
            attn_opt.zero_grad()
            total_loss_inner = inputs.new_zeros(())
            for ti in range(n_tasks):
                pred = model(inputs[:, ti, :], sample_ids)
                tl, _ = compute_loss(pred, outputs[:, ti, :], task_weights[:, ti], loss_spec)
                total_loss_inner = total_loss_inner + tl
            total_loss_inner.backward()
            attn_opt.step()

    return total_loss, all_slot_losses


# ---------------------------------------------------------------------------
# Training-loss CSV helpers
# ---------------------------------------------------------------------------

def _open_loss_csv(
    output_dir: str,
    loss_spec: LossSpec,
) -> tuple:
    """Create (or truncate) the training-loss CSV and return (file, csv.writer)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "training_loss.csv")
    f = open(path, "w", newline="")  # noqa: SIM115
    slot_names = [s.label for s in loss_spec.slots]
    writer = csv.writer(f)
    writer.writerow(["step", "total_loss"] + slot_names)
    return f, writer


def _write_loss_row(
    writer,
    step: int,
    total_loss: float,
    slot_losses: dict[str, float],
    loss_spec: LossSpec,
) -> None:
    slot_names = [s.label for s in loss_spec.slots]
    writer.writerow(
        [step, f"{total_loss:.8g}"]
        + [f"{slot_losses.get(s, 0.0):.8g}" for s in slot_names]
    )


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------

def train_schedule(
    model: MultiNetwork,
    collection: DatasetCollection,
    schedule: TrainingSchedule,
    config: TrainingConfig,
    device: str = "cpu",
    evaluator: Optional["Evaluator"] = None,
    output_dir: Optional[str] = None,
) -> TrainingHistory:
    """Train *model* for the full schedule defined by *schedule*.

    For each phase the function:
    1. Resolves the learning rate (phase ``params.lr`` overrides ``config.lr``).
    2. Builds the appropriate optimizer(s).
    3. Iterates over the phase's data loader for the specified number of
       epochs or gradient steps.
    4. Records per-step loss values in a :class:`TrainingHistory`.

    Args:
        model: :class:`MultiNetwork` to train (modified in place).
        collection: Dataset collection.
        schedule: Training schedule produced by
            :func:`repulsion.schedule.build_training_schedule`.
        config: Optimiser and training-mode configuration.
        device: PyTorch device string passed to ``tensor.to(device)``.
        evaluator: Optional :class:`~repulsion.evaluation.Evaluator` that is
            called at step 0 (pre-training) and after every gradient update.
        output_dir: If given, write ``training_loss.csv`` to this directory
            with one row per gradient step.

    Returns:
        :class:`TrainingHistory` containing per-step loss records for all
        phases.
    """
    torch_device = torch.device(device)
    model = model.to(torch_device)
    loss_spec = build_loss_spec(collection)
    history = TrainingHistory()
    global_step = 0

    # Open training-loss CSV (truncated fresh each run).
    _loss_csv_file = None
    _loss_writer = None
    if output_dir is not None:
        _loss_csv_file, _loss_writer = _open_loss_csv(output_dir, loss_spec)

    # Pre-training evaluation (step 0 = before any gradient updates).
    if evaluator is not None:
        evaluator.maybe_run_all(0)

    if config.separate_attention and model.has_attention():
        net_params = model.network_params()
        attn_params = model.attention_params()
        net_opt = _build_optimizer(config.optimizer, net_params, config.lr, config.momentum)
        attn_opt = _build_optimizer(
            config.attention_optimizer, attn_params,
            config.attention_lr, config.attention_momentum,
        )
        optimizers: tuple = (net_opt, attn_opt)
    else:
        trainable = model.parameters()
        joint_opt = _build_optimizer(config.optimizer, trainable, config.lr, config.momentum)
        optimizers = (joint_opt,)

    for phase_idx, phase in enumerate(schedule.phases):
        if phase.joint:
            loader = build_joint_phase_dataloader(collection, phase, shuffle=True)
        else:
            loader = build_phase_dataloader(collection, phase, shuffle=True)
        phase_hist = PhaseHistory(phase_idx=phase_idx, phase=phase)

        if phase.epochs is not None:
            # Epoch-based training
            for _epoch in range(phase.epochs):
                for batch in loader:
                    if phase.joint:
                        loss, slot_losses = _do_joint_step(
                            model, batch, loss_spec, optimizers, config, torch_device
                        )
                    else:
                        x = batch["input"].float().to(torch_device)
                        y = batch["output"].float().to(torch_device)
                        w = batch["task_weight"].float().to(torch_device)
                        sample_ids = model.get_sample_ids(batch["row_index"], torch_device)
                        loss, slot_losses = _do_step(
                            model, x, y, w, sample_ids, loss_spec, optimizers, config
                        )
                    phase_hist.steps.append(StepRecord(
                        step=global_step,
                        total_loss=loss.item(),
                        per_slot_losses=slot_losses,
                    ))
                    global_step += 1
                    if _loss_writer is not None:
                        _write_loss_row(_loss_writer, global_step, loss.item(), slot_losses, loss_spec)
                        _loss_csv_file.flush()
                    if evaluator is not None:
                        evaluator.maybe_run_all(global_step)
        else:
            # Step-based training
            total_steps = phase.steps
            step_count = 0
            while step_count < total_steps:
                for batch in loader:
                    if step_count >= total_steps:
                        break
                    if phase.joint:
                        loss, slot_losses = _do_joint_step(
                            model, batch, loss_spec, optimizers, config, torch_device
                        )
                    else:
                        x = batch["input"].float().to(torch_device)
                        y = batch["output"].float().to(torch_device)
                        w = batch["task_weight"].float().to(torch_device)
                        sample_ids = model.get_sample_ids(batch["row_index"], torch_device)
                        loss, slot_losses = _do_step(
                            model, x, y, w, sample_ids, loss_spec, optimizers, config
                        )
                    phase_hist.steps.append(StepRecord(
                        step=global_step,
                        total_loss=loss.item(),
                        per_slot_losses=slot_losses,
                    ))
                    global_step += 1
                    step_count += 1
                    if _loss_writer is not None:
                        _write_loss_row(_loss_writer, global_step, loss.item(), slot_losses, loss_spec)
                        _loss_csv_file.flush()
                    if evaluator is not None:
                        evaluator.maybe_run_all(global_step)

        history.phases.append(phase_hist)

    if _loss_csv_file is not None:
        _loss_csv_file.close()


    return history
