"""wammes2021.py — pair learning on superposed, correlated item pairs.

Port of ``integration-and-repulsion/24_wammes.ipynb``.  ``n_pairs`` item pairs
``(x_i, y_i)`` with ``corr(x_i, y_i)`` spaced linearly in ``[rho_min, rho_max]``
are presented as superpositions ``sqrt(phi) x + sqrt(1 - phi) y`` (scaled by
``magnitude``).  The training set contains the true pairs ``(x_i, y_i)``
repeated ``pair_repeats`` times plus, optionally, every cross combination
``(x_j, y_i)`` (including the true pairs once more).  The target is the same
superposition built with ``phi_out``/``magnitude_out``.

Throughout training the individual items are probed, and for each stream the
correlation between the hidden representations of ``x_i`` and ``y_j`` is
recorded.

Outputs (``results.npz`` in the run directory)
----------------------------------------------
eval_steps  (T,)               step of each evaluation
corr_xy     (T, S, n, n)       corr(h(x_i), h(y_j)) per stream
pair_corr   (T, S, n)          diagonal of corr_xy (pairmates)
baseline    (T, S, n(n-1)/2)   upper triangle of corr_xy (non-pairmates)
loss        (n_steps,)         training loss per update
steps_per_epoch                updates per epoch (eval_steps / steps_per_epoch = epoch)
stream_names, stream_sparse    per-stream name and whether it has the fixed kWTA projection
rhos, items_x, items_y         stimulus ground truth

Usage
-----
    python scripts/tasks/wammes2021.py
    python scripts/tasks/wammes2021.py stimuli.phi_in=0.8 training.lr=0.05
    # notebook-style full-batch gradient descent
    python scripts/tasks/wammes2021.py training.batch_size=null training.momentum=0 training.lr=10 training.epochs=5000
    # single stream: drop the kWTA stream
    python scripts/tasks/wammes2021.py 'model.networks=[{name: net1, hidden_sizes: [512], init_scale: 0.01}]'
    # seed sweep
    python scripts/tasks/wammes2021.py seed=0,1,2 hydra.sweep.dir=data/wammes2021 --multirun
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from repulsion.tasks import (
    TrainConfig,
    build_model,
    correlated_pairs,
    hidden_reps,
    resolve_output_dir,
    save_results,
    seed_everything,
    superpose,
    train,
)

log = logging.getLogger(__name__)


# ── config ──────────────────────────────────────────────────────────────────

@dataclass
class StimuliConfig:
    n_pairs: int = 8
    rho_min: float = 0.0
    rho_max: float = 1.0
    dim: int = 128
    pair_repeats: int = 8
    include_nonpairs: bool = True
    phi_in: float = 0.5    # salience of the first item in the input
    phi_out: float = 0.5   # salience of the first item in the target
    magnitude_in: float = 1.0
    magnitude_out: float = 1.0


def _default_networks() -> list[dict]:
    return [
        {"name": "net1", "hidden_sizes": [512], "activation": "identity", "init_scale": 0.01},
        {
            "name": "net2",
            "hidden_sizes": [512],
            "activation": "identity",
            "init_scale": 0.01,
            "fixed_projection": True,
            "fixed_projection_hidden_size": 2000,
            "fixed_projection_activation": "kwta",
            "fixed_projection_kwta_frac": 0.005,
        },
    ]


@dataclass
class ModelConfig:
    networks: List[Any] = field(default_factory=_default_networks)
    probe_layer: str = "hidden_0"


@dataclass
class WammesConfig:
    seed: int = 0
    device: str = "cpu"
    output_dir: Optional[str] = None
    stimuli: StimuliConfig = field(default_factory=StimuliConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # online SGD with momentum; lr is large because the MSE averages over output dims
    training: TrainConfig = field(default_factory=lambda: TrainConfig(lr=0.5, epochs=20))


ConfigStore.instance().store(name="wammes2021", node=WammesConfig)


# ── task ────────────────────────────────────────────────────────────────────

def build_data(cfg: StimuliConfig, rng: np.random.Generator):
    """Return ``(X, Y, probe, rhos, items_x, items_y)``.

    ``probe`` interleaves the items as ``x_0, y_0, x_1, y_1, ...``.
    """
    rhos = np.linspace(cfg.rho_min, cfg.rho_max, cfg.n_pairs)
    pairs = correlated_pairs(rhos, dim=cfg.dim, rng=rng)
    items_x = np.stack([x for x, _ in pairs])
    items_y = np.stack([y for _, y in pairs])

    combos = [(i, i) for i in range(cfg.n_pairs)] * cfg.pair_repeats
    if cfg.include_nonpairs:
        combos += [(j, i) for i in range(cfg.n_pairs) for j in range(cfg.n_pairs)]
    xi, yi = np.array(combos).T
    X = superpose(items_x[xi], items_y[yi], cfg.phi_in, cfg.magnitude_in)
    Y = superpose(items_x[xi], items_y[yi], cfg.phi_out, cfg.magnitude_out)

    probe = np.stack([items_x, items_y], axis=1).reshape(-1, cfg.dim)
    return X, Y, probe, rhos, items_x, items_y


def make_eval_fn(n_pairs: int, layer: str):
    def eval_fn(model, probe):
        out = []
        for rep in hidden_reps(model, probe, layer):
            rep = rep.cpu().numpy()
            out.append(np.corrcoef(rep[::2], rep[1::2])[:n_pairs, n_pairs:])
        return np.stack(out)  # (S, n, n)
    return eval_fn


# ── main ────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../../configs", config_name="wammes2021", version_base=None)
def main(cfg: WammesConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    output_dir = resolve_output_dir(cfg.output_dir)
    rng = seed_everything(cfg.seed)

    X, Y, probe, rhos, items_x, items_y = build_data(cfg.stimuli, rng)
    log.info("Training set: %d stimuli, dim %d.", *X.shape)

    model = build_model(OmegaConf.to_container(cfg.model.networks, resolve=True), in_dim=cfg.stimuli.dim)
    log.info("Model: %d trainable parameters.", sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_cfg = TrainConfig(**OmegaConf.to_container(cfg.training, resolve=True))
    result = train(
        model, X, Y, train_cfg,
        probe=probe,
        eval_fn=make_eval_fn(cfg.stimuli.n_pairs, cfg.model.probe_layer),
        device=cfg.device,
    )

    corr_xy = np.stack(result.metrics)  # (T, S, n, n)
    iu = np.triu_indices(cfg.stimuli.n_pairs, k=1)
    networks = OmegaConf.to_container(cfg.model.networks, resolve=True)
    save_results(
        output_dir,
        eval_steps=result.eval_steps,
        corr_xy=corr_xy,
        pair_corr=np.diagonal(corr_xy, axis1=-2, axis2=-1),
        baseline=corr_xy[..., iu[0], iu[1]],
        loss=result.losses,
        steps_per_epoch=len(result.losses) // train_cfg.epochs,
        stream_names=np.array([net.get("name", f"net{i}") for i, net in enumerate(networks)]),
        stream_sparse=np.array([bool(net.get("fixed_projection", False)) for net in networks]),
        rhos=rhos,
        items_x=items_x,
        items_y=items_y,
    )
    log.info("Done. Final loss %.4g.", result.losses[-1])


if __name__ == "__main__":
    main()
