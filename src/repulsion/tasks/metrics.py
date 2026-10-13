"""Evaluation helpers shared by pair-learning task scripts."""
from __future__ import annotations

import numpy as np

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
