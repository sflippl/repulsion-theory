"""schapiro2012.py — statistical learning of strong vs. weak pairs from a sequence.

Port of ``integration-and-repulsion/25_schapiro2012.ipynb``.  Items come in
``n_pairs`` pairs ``(A_i, B_i)`` with ``corr(A_i, B_i) = pair_corrs[i]`` and
``between_corr`` between items of different pairs.  A sequence of
``n_events`` events is sampled: with probability ``strong_prob`` a strong pair
occurs as ``A → B``; otherwise a weak pair occurs as ``A → B`` with probability
``weak_pair_prob`` and else as two separate events ``A`` and ``B``.  Events are
shuffled and concatenated.

Every training example is a pair of consecutive items ``(s_t, s_{t+1})``,
presented as ``sqrt(phi_in) s_t + sqrt(1 - phi_in) s_{t+1}``; the target is the
same superposition with ``phi_out`` (``null`` → ``1 - phi_in``).  ``phi_in = phi_out = 0.5`` is
autoencoding of the transition (as in the notebook, up to a factor sqrt(2));
``phi_in = 1, phi_out = 0`` is next-item prediction.  With ``separate_inputs``
(default) the two weighted items arrive in separate input slots and each stream
sums them after its fixed projection: the dense stream sees ``s_t' + s_{t+1}'``
(identical to the superposition), the kWTA stream
``kwta(P s_t') + kwta(P s_{t+1}')``.  Items are probed alone (in the first slot,
the second zeroed, when ``separate_inputs``).

Outputs (``results.npz`` in the run directory)
----------------------------------------------
eval_steps  (T,)               step of each evaluation
corr_xy     (T, S, n, n)       corr(h(A_i), h(B_j)) per stream
pair_corr   (T, S, n)          diagonal of corr_xy (pairmates)
baseline    (T, S, n(n-1)/2)   upper triangle of corr_xy (non-pairmates)
loss        (n_steps,)         training loss per update
steps_per_epoch                updates per epoch
phi_in, phi_out                saliences actually used
stream_names, stream_sparse    per-stream name and whether it has the fixed kWTA projection
pair_corrs, pair_strong        per-pair input correlation and strong/weak flag
items       (2n, dim)          item vectors, ordered A_0, B_0, A_1, B_1, ...
sequence    (L,)               item indices of the training sequence

Usage
-----
    python scripts/tasks/schapiro2012.py
    python scripts/tasks/schapiro2012.py stimuli.phi_in=1.0 stimuli.phi_out=0.0
    # kWTA applied to the superposition instead of to each item
    python scripts/tasks/schapiro2012.py stimuli.separate_inputs=false
    # notebook-style minibatch SGD
    python scripts/tasks/schapiro2012.py training.batch_size=32 training.momentum=0 training.lr=10 training.epochs=100
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

from repulsion.tasks import (
    TrainConfig,
    build_model,
    correlated_pairs_exact,
    make_pair_corr_eval,
    pair_corr_arrays,
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
    pair_corrs: List[float] = field(default_factory=lambda: [0.5] * 4 + [0.9] * 4)
    between_corr: float = 0.5
    strong_pairs: List[int] = field(default_factory=lambda: [0, 1, 4, 5])  # all other pairs are weak
    dim: int = 128
    n_events: int = 1000
    strong_prob: float = 0.5       # probability that an event is drawn from the strong pairs
    weak_pair_prob: float = 1 / 3  # probability that a weak pair occurs as A → B
    phi_in: float = 0.5            # salience of the current item in the input
    phi_out: Optional[float] = 0.5  # salience of the current item in the target; null → 1 - phi_in
    magnitude_in: float = 1.0
    magnitude_out: float = 1.0
    separate_inputs: bool = True   # current/next item in separate slots, summed after each stream's projection


def _default_networks() -> list[dict]:
    return [
        {"name": "net1", "hidden_sizes": [128], "activation": "identity", "init_scale": 0.01},
        {
            "name": "net2",
            "hidden_sizes": [128],
            "activation": "identity",
            "init_scale": 0.01,
            "fixed_projection": True,
            "fixed_projection_hidden_size": 1000,
            "fixed_projection_activation": "kwta",
            "fixed_projection_kwta_frac": 0.003,
        },
    ]


@dataclass
class ModelConfig:
    networks: List[Any] = field(default_factory=_default_networks)
    probe_layer: str = "hidden_0"


@dataclass
class SchapiroConfig:
    seed: int = 0
    device: str = "cpu"
    output_dir: Optional[str] = None
    stimuli: StimuliConfig = field(default_factory=StimuliConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    # online SGD with momentum; one epoch is one pass over the ~2000 transitions
    training: TrainConfig = field(default_factory=lambda: TrainConfig(lr=0.5, epochs=10, log_every_epochs=1))


ConfigStore.instance().store(name="schapiro2012", node=SchapiroConfig)


# ── task ────────────────────────────────────────────────────────────────────

def generate_sequence(cfg: StimuliConfig, rng: np.random.Generator) -> np.ndarray:
    """Item indices (``A_i = 2i``, ``B_i = 2i + 1``) of a shuffled event sequence."""
    n_pairs = len(cfg.pair_corrs)
    strong = list(cfg.strong_pairs)
    weak = [p for p in range(n_pairs) if p not in strong]
    if not strong or not weak:
        raise ValueError("Need at least one strong and one weak pair.")

    events = []
    for _ in range(cfg.n_events):
        if rng.random() < cfg.strong_prob:
            p = strong[rng.integers(len(strong))]
            events.append([2 * p, 2 * p + 1])
        else:
            p = weak[rng.integers(len(weak))]
            if rng.random() < cfg.weak_pair_prob:
                events.append([2 * p, 2 * p + 1])
            else:
                events.append([2 * p])
                events.append([2 * p + 1])
    order = rng.permutation(len(events))
    return np.concatenate([events[i] for i in order])


def resolve_phi_out(cfg: StimuliConfig) -> float:
    return round(1.0 - cfg.phi_in, 10) if cfg.phi_out is None else cfg.phi_out


def build_data(cfg: StimuliConfig, rng: np.random.Generator):
    """Return ``(X, Y, probe, items, sequence)``; ``probe`` is ``A_0, B_0, A_1, B_1, ...``.

    With ``separate_inputs``, ``X`` and ``probe`` are ``[s_t slot, s_{t+1} slot]``
    of width ``2 * dim``; probed items sit in the first slot, the second zeroed.
    """
    pairs = correlated_pairs_exact(cfg.pair_corrs, dim=cfg.dim, rng=rng, baseline=cfg.between_corr)
    items = np.stack([v for pair in pairs for v in pair])
    sequence = generate_sequence(cfg, rng)
    first, second = items[sequence[:-1]], items[sequence[1:]]
    Y = superpose(first, second, resolve_phi_out(cfg), cfg.magnitude_out)
    if not cfg.separate_inputs:
        X = superpose(first, second, cfg.phi_in, cfg.magnitude_in)
        return X, Y, items, items, sequence
    X = superpose_slots(first, second, cfg.phi_in, cfg.magnitude_in)
    probe = np.concatenate([items, np.zeros_like(items)], axis=1)
    return X, Y, probe, items, sequence


# ── main ────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../../configs", config_name="schapiro2012", version_base=None)
def main(cfg: SchapiroConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    output_dir = resolve_output_dir(cfg.output_dir)
    rng = seed_everything(cfg.seed)

    X, Y, probe, items, sequence = build_data(cfg.stimuli, rng)
    log.info("Training set: %d transitions, dim %d.", *X.shape)

    networks = OmegaConf.to_container(cfg.model.networks, resolve=True)
    model = build_model(networks, in_dim=cfg.stimuli.dim, n_input_slots=2 if cfg.stimuli.separate_inputs else 1)
    log.info("Model: %d trainable parameters.", sum(p.numel() for p in model.parameters() if p.requires_grad))

    n_pairs = len(cfg.stimuli.pair_corrs)
    train_cfg = TrainConfig(**OmegaConf.to_container(cfg.training, resolve=True))
    result = train(
        model, X, Y, train_cfg,
        probe=probe,
        eval_fn=make_pair_corr_eval(n_pairs, cfg.model.probe_layer),
        device=cfg.device,
    )

    save_results(
        output_dir,
        eval_steps=result.eval_steps,
        **pair_corr_arrays(result.metrics),
        loss=result.losses,
        steps_per_epoch=len(result.losses) // train_cfg.epochs,
        **stream_info(networks),
        phi_in=cfg.stimuli.phi_in,
        phi_out=resolve_phi_out(cfg.stimuli),
        pair_corrs=np.array(cfg.stimuli.pair_corrs),
        pair_strong=np.isin(np.arange(n_pairs), list(cfg.stimuli.strong_pairs)),
        items=items,
        sequence=sequence,
    )
    log.info("Done. Final loss %.4g.", result.losses[-1])


if __name__ == "__main__":
    main()
