"""wammes2021.py — pair learning on superposed, correlated item pairs.

Port of ``integration-and-repulsion/24_wammes.ipynb``.  ``n_pairs`` item pairs
``(x_i, y_i)`` with ``corr(x_i, y_i)`` spaced linearly in ``[rho_min, rho_max]``
are presented as superpositions ``sqrt(phi) x + sqrt(1 - phi) y`` (scaled by
``magnitude``).  With ``separate_inputs`` (default) the two weighted items
arrive in separate input slots and each stream sums them after its fixed
projection: the dense stream sees ``x' + y'`` (identical to the superposition),
the kWTA stream ``kwta(P x') + kwta(P y')``.  The training set contains the true pairs ``(x_i, y_i)``
repeated ``pair_repeats`` times plus, optionally, every cross combination
``(x_j, y_i)`` (including the true pairs once more).  The target is the same
superposition built with ``phi_out``/``magnitude_out``.

Throughout training the individual items are probed (each alone in its own
slot when ``separate_inputs``), and for each stream the
correlation between the hidden representations of ``x_i`` and ``y_j`` is
recorded.  For the summed output and each stream alone ("readouts") the loss
on the training set and the recall of each probed item (is the output closer
to the item than to its pairmate?) are recorded as well.

Outputs (``results.npz`` in the run directory)
----------------------------------------------
eval_steps  (T,)               step of each evaluation
corr_xy     (T, S, n, n)       corr(h(x_i), h(y_j)) per stream
pair_corr   (T, S, n)          diagonal of corr_xy (pairmates)
baseline    (T, S, n(n-1)/2)   upper triangle of corr_xy (non-pairmates)
loss        (n_steps,)         training loss per update
eval_loss   (T, R)             training-set loss per readout (summed, then each stream)
recall_margin (T, R, 2n)       ||o - pairmate|| - ||o - item|| per readout and probed item
                               (x_0, y_0, x_1, ...); > 0 when the item is recalled
steps_per_epoch                updates per epoch (eval_steps / steps_per_epoch = epoch)
stream_names, stream_sparse    per-stream name and whether it has the fixed kWTA projection
rhos, items_x, items_y         stimulus ground truth

Usage
-----
    python scripts/tasks/wammes2021.py
    python scripts/tasks/wammes2021.py stimuli.phi_in=0.8 training.lr=0.05
    # kWTA applied to the superposition instead of to each item
    python scripts/tasks/wammes2021.py stimuli.separate_inputs=false
    # notebook-style full-batch gradient descent
    python scripts/tasks/wammes2021.py training.batch_size=null training.momentum=0 training.lr=0.1 training.epochs=5000
    # any model from configs/model (default: configs/model/complete_model_linear.yaml)
    python scripts/tasks/wammes2021.py model=complete_model_relu
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
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from repulsion.conf import ModelConf
from repulsion.tasks import (
    TrainConfig,
    build_model,
    correlated_pairs,
    make_readout_eval,
    readout_arrays,
    resolve_output_dir,
    save_results,
    seed_everything,
    stream_info,
    superpose,
    superpose_slots,
    train,
)

log = logging.getLogger(__name__)


# ── config ──────────────────────────────────────────────────────────────────

@dataclass
class StimuliConfig:
    n_pairs: int = 8
    rho_min: float = 0.0
    rho_max: float = 7/8
    dim: int = 128
    pair_repeats: int = 8
    include_nonpairs: bool = True
    phi_in: float = 0.5    # salience of the first item in the input
    phi_out: float = 0.5   # salience of the first item in the target
    magnitude_in: float = 1.0
    magnitude_out: float = 1.0
    separate_inputs: bool = True  # x and y in separate slots, summed after each stream's projection


@dataclass
class ModelConfig(ModelConf):
    """A ``configs/model`` entry plus the layer whose representations are probed."""
    probe_layer: str = "hidden_0"


@dataclass
class WammesConfig:
    # model: any configs/model/*.yaml, selected with ``model=<name>``
    defaults: List[Any] = field(default_factory=lambda: ["_self_", {"model": "complete_model_linear"}])
    seed: int = 0
    device: str = "cpu"
    output_dir: Optional[str] = None
    stimuli: StimuliConfig = field(default_factory=StimuliConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # online SGD with momentum
    training: TrainConfig = field(default_factory=lambda: TrainConfig(lr=0.1, momentum=0.5, epochs=20))


ConfigStore.instance().store(name="wammes2021", node=WammesConfig)


# ── task ────────────────────────────────────────────────────────────────────

def build_data(cfg: StimuliConfig, rng: np.random.Generator):
    """Return ``(X, Y, probe, rhos, items_x, items_y)``.

    ``probe`` interleaves the items as ``x_0, y_0, x_1, y_1, ...``.  With
    ``separate_inputs``, ``X`` and ``probe`` are ``[x-slot, y-slot]`` of width
    ``2 * dim``, and each probed item sits in its own slot with the other zeroed.
    """
    rhos = np.linspace(cfg.rho_min, cfg.rho_max, cfg.n_pairs)
    pairs = correlated_pairs(rhos, dim=cfg.dim, rng=rng)
    items_x = np.stack([x for x, _ in pairs])
    items_y = np.stack([y for _, y in pairs])

    combos = [(i, i) for i in range(cfg.n_pairs)] * cfg.pair_repeats
    if cfg.include_nonpairs:
        combos += [(j, i) for i in range(cfg.n_pairs) for j in range(cfg.n_pairs)]
    xi, yi = np.array(combos).T
    Y = superpose(items_x[xi], items_y[yi], cfg.phi_out, cfg.magnitude_out)
    if not cfg.separate_inputs:
        X = superpose(items_x[xi], items_y[yi], cfg.phi_in, cfg.magnitude_in)
        probe = np.stack([items_x, items_y], axis=1).reshape(-1, cfg.dim)
        return X, Y, probe, rhos, items_x, items_y

    X = superpose_slots(items_x[xi], items_y[yi], cfg.phi_in, cfg.magnitude_in)
    zeros = np.zeros_like(items_x)
    probe = np.stack([
        np.concatenate([items_x, zeros], axis=1),
        np.concatenate([zeros, items_y], axis=1),
    ], axis=1).reshape(-1, 2 * cfg.dim)
    return X, Y, probe, rhos, items_x, items_y


# ── main ────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../../configs", config_name="wammes2021", version_base=None)
def main(cfg: WammesConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    output_dir = resolve_output_dir(cfg.output_dir)
    rng = seed_everything(cfg.seed)

    X, Y, probe, rhos, items_x, items_y = build_data(cfg.stimuli, rng)
    log.info("Training set: %d stimuli, dim %d.", *X.shape)

    if cfg.model.shared_attention_layer:
        raise ValueError("The shared attention layer is not supported by this task.")
    model = build_model(
        OmegaConf.to_container(cfg.model.networks, resolve=True),
        in_dim=cfg.stimuli.dim,
        n_input_slots=2 if cfg.stimuli.separate_inputs else 1,
    )
    log.info("Model: %d trainable parameters.", sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_cfg = TrainConfig(**OmegaConf.to_container(cfg.training, resolve=True))
    result = train(
        model, X, Y, train_cfg,
        probe=probe,
        eval_fn=make_readout_eval(cfg.stimuli.n_pairs, X, Y,
                                  np.stack([items_x, items_y], axis=1).reshape(-1, cfg.stimuli.dim),
                                  cfg.model.probe_layer),
        device=cfg.device,
    )

    save_results(
        output_dir,
        eval_steps=result.eval_steps,
        **readout_arrays(result.metrics),
        loss=result.losses,
        steps_per_epoch=len(result.losses) // train_cfg.epochs,
        **stream_info(OmegaConf.to_container(cfg.model.networks, resolve=True)),
        rhos=rhos,
        items_x=items_x,
        items_y=items_y,
    )
    log.info("Done. Final loss %.4g.", result.losses[-1])


if __name__ == "__main__":
    main()
