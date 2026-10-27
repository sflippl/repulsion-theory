"""Evaluation helpers shared by pair-learning task scripts."""
from __future__ import annotations

import numpy as np
import torch

from repulsion.tasks.models import hidden_reps


def make_pair_corr_eval(n_pairs: int, layer: str = "hidden_0"):
    """``eval_fn`` returning ``corr(h(x_i), h(y_j))`` per stream, shape ``(S, n, n)``.

    Expects the probe to interleave the items as ``x_0, y_0, x_1, y_1, ...``.
    """
    def eval_fn(model, probe):
        out = []
        for rep in hidden_reps(model, probe, layer):
            rep = rep.cpu().numpy()
            out.append(np.corrcoef(rep[::2], rep[1::2])[:n_pairs, n_pairs:])
        return np.stack(out)
    return eval_fn


@torch.no_grad()
def readout_outputs(model, x: torch.Tensor) -> list[torch.Tensor]:
    """Outputs of the summed model followed by each stream alone."""
    full_input = model.prepare_input(x)
    return [model(x)] + [net(full_input) for net in model.networks]


def make_readout_eval(n_pairs: int, X, Y, items, layer: str = "hidden_0"):
    """``eval_fn`` returning hidden-layer pair correlations plus per-readout loss and recall.

    Readouts are the summed output followed by each stream alone.  Returns a
    dict with ``corr`` (S, n, n) as :func:`make_pair_corr_eval`, ``loss``
    (R,), the squared error on ``(X, Y)`` summed over output dimensions and
    averaged over samples, and ``recall_margin`` (R, 2n):
    ``||o - pairmate|| - ||o - own item||`` for the output ``o`` to each probed
    item (> 0 when the output is closer to the item than to its pairmate).

    Expects the probe and *items* (2n, out_dim) to interleave the items as
    ``x_0, y_0, x_1, y_1, ...``.
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.float32)
    items = torch.as_tensor(items, dtype=torch.float32)
    mates = items.view(n_pairs, 2, -1).flip(1).reshape(items.shape)
    corr_fn = make_pair_corr_eval(n_pairs, layer)

    def eval_fn(model, probe):
        device = probe.device
        losses = [((o - Y.to(device)) ** 2).sum(-1).mean().item()
                  for o in readout_outputs(model, X.to(device))]
        margins = [(torch.linalg.norm(o - mates.to(device), dim=-1)
                    - torch.linalg.norm(o - items.to(device), dim=-1)).cpu().numpy()
                   for o in readout_outputs(model, probe)]
        return {"corr": corr_fn(model, probe), "loss": np.array(losses),
                "recall_margin": np.stack(margins)}
    return eval_fn


def readout_arrays(metrics: list[dict]) -> dict[str, np.ndarray]:
    """Stack ``make_readout_eval`` outputs: pair correlations plus ``eval_loss`` and ``recall_margin``."""
    return {
        **pair_corr_arrays([m["corr"] for m in metrics]),
        "eval_loss": np.stack([m["loss"] for m in metrics]),               # (T, R)
        "recall_margin": np.stack([m["recall_margin"] for m in metrics]),  # (T, R, 2n)
    }


def pair_corr_arrays(metrics: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Stack ``make_pair_corr_eval`` outputs into ``corr_xy``, ``pair_corr`` and ``baseline``."""
    corr_xy = np.stack(metrics)  # (T, S, n, n)
    iu = np.triu_indices(corr_xy.shape[-1], k=1)
    return {
        "corr_xy": corr_xy,
        "pair_corr": np.diagonal(corr_xy, axis1=-2, axis2=-1),
        "baseline": corr_xy[..., iu[0], iu[1]],
    }


def stream_info(networks: list[dict]) -> dict[str, np.ndarray]:
    """Per-stream name and whether it has the fixed (kWTA) projection."""
    return {
        "stream_names": np.array([net.get("name", f"net{i}") for i, net in enumerate(networks)]),
        "stream_sparse": np.array([bool(net.get("fixed_projection", False)) for net in networks]),
    }
