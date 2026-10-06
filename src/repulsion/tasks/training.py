"""Minimal training loop for task scripts.

Supports full-batch gradient descent (``batch_size=None``), online SGD
(``batch_size=1``, reshuffled every epoch) and minibatches in between, with
any optimizer known to :func:`repulsion.training._build_optimizer`.

Evaluation is a user-supplied ``eval_fn(model, probe)`` run at step 0 and then
either every ``eval_every_steps`` steps or on the geometric schedule used by
the main evaluator (``eval_every_log_steps``), plus always at the final step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch

from repulsion.evaluation import _LogSchedule
from repulsion.training import _build_optimizer

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    optimizer: str = "sgd"
    lr: float = 0.02
    momentum: float = 0.9
    batch_size: Optional[int] = 1  # None → full batch
    epochs: int = 100
    shuffle: bool = True
    loss: str = "mse"
    eval_every_steps: Optional[int] = None
    eval_every_log_steps: Optional[float] = 1.01
    log_every_epochs: int = 10


@dataclass
class TrainResult:
    eval_steps: np.ndarray        # (T,) step at which each eval ran (before that step's update)
    metrics: list[Any]            # eval_fn outputs, one per eval step
    losses: np.ndarray            # (n_steps,) training loss per update
    final_step: int


def _loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mse":
        return torch.mean((pred - target) ** 2)
    raise ValueError(f"Unknown loss '{kind}'. Available: 'mse'.")


def train(
    model: torch.nn.Module,
    X: np.ndarray | torch.Tensor,
    Y: np.ndarray | torch.Tensor,
    cfg: TrainConfig,
    probe: np.ndarray | torch.Tensor | None = None,
    eval_fn: Callable[[torch.nn.Module, torch.Tensor], Any] | None = None,
    device: str = "cpu",
    step_offset: int = 0,
) -> TrainResult:
    """Train *model* on ``(X, Y)``; call repeatedly with *step_offset* for blocked phases.

    The step-0 evaluation only runs when ``step_offset == 0`` so consecutive
    phases do not duplicate the boundary evaluation.
    """
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    Y = torch.as_tensor(Y, dtype=torch.float32, device=device)
    if probe is not None:
        probe = torch.as_tensor(probe, dtype=torch.float32, device=device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = _build_optimizer(cfg.optimizer, params, cfg.lr, cfg.momentum)

    n = X.shape[0]
    batch_size = n if cfg.batch_size is None else int(cfg.batch_size)
    steps_per_epoch = -(-n // batch_size)
    final_step = step_offset + cfg.epochs * steps_per_epoch

    log_sched = (
        _LogSchedule(cfg.eval_every_log_steps)
        if eval_fn is not None and cfg.eval_every_steps is None and cfg.eval_every_log_steps
        else None
    )

    def due(step: int) -> bool:
        if eval_fn is None:
            return False
        if step == step_offset:
            return step_offset == 0
        if step == final_step:
            return True
        if cfg.eval_every_steps is not None:
            return step % cfg.eval_every_steps == 0
        return log_sched is not None and log_sched.due(step)

    eval_steps: list[int] = []
    metrics: list[Any] = []
    losses = np.empty(final_step - step_offset, dtype=np.float32)

    def maybe_eval(step: int) -> None:
        if due(step):
            model.eval()
            metrics.append(eval_fn(model, probe))
            eval_steps.append(step)
            model.train()
            if log_sched is not None:
                log_sched.advance(step)

    step = step_offset
    model.train()
    for epoch in range(cfg.epochs):
        order = torch.randperm(n, device=device) if cfg.shuffle else torch.arange(n, device=device)
        for start in range(0, n, batch_size):
            maybe_eval(step)
            idx = order[start:start + batch_size]
            loss = _loss(model(X[idx]), Y[idx], cfg.loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses[step - step_offset] = loss.item()
            step += 1
        if cfg.log_every_epochs and (epoch + 1) % cfg.log_every_epochs == 0:
            log.info(
                "epoch %d/%d  step %d  loss %.4g",
                epoch + 1, cfg.epochs, step, losses[step - step_offset - steps_per_epoch:step - step_offset].mean(),
            )
    maybe_eval(step)

    return TrainResult(np.array(eval_steps), metrics, losses, final_step)
