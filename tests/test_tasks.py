"""Tests for repulsion.tasks and the Wammes task script."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

from repulsion.tasks import (
    TrainConfig,
    build_model,
    correlated_pairs,
    hidden_reps,
    superpose,
    train,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "tasks"))
import wammes2021  # noqa: E402

SPECS = [
    {"name": "net1", "hidden_sizes": [16], "init_scale": 0.1},
    {
        "name": "net2",
        "hidden_sizes": [8],
        "init_scale": 0.1,
        "fixed_projection": True,
        "fixed_projection_hidden_size": 50,
        "fixed_projection_activation": "kwta",
        "fixed_projection_kwta_frac": 0.1,
    },
]


def test_correlated_pairs_hit_target_correlation():
    rhos = np.linspace(0, 1, 5)
    for rho, (x, y) in zip(rhos, correlated_pairs(rhos, dim=32, rng=0)):
        assert np.isclose(np.linalg.norm(x), 1) and np.isclose(np.linalg.norm(y), 1)
        assert np.isclose(np.corrcoef(x, y)[0, 1], rho, atol=1e-8)


def test_superpose_limits_and_energy():
    a, b = np.eye(4)[:2]
    assert np.allclose(superpose(a, b, 1.0), a)
    assert np.allclose(superpose(a, b, 0.0), b)
    assert np.isclose(np.linalg.norm(superpose(a, b, 0.3)), 1.0)
    assert np.isclose(np.linalg.norm(superpose(a, b, 0.3, magnitude=2.0)), 2.0)
    with pytest.raises(ValueError):
        superpose(a, b, 1.5)


def test_build_model_streams_and_frozen_projection():
    model = build_model(SPECS, in_dim=12)
    x = torch.randn(5, 12)
    assert model(x).shape == (5, 12)
    reps = hidden_reps(model, x)
    assert [r.shape for r in reps] == [(5, 16), (5, 8)]
    assert all(not p.requires_grad for p in model.networks[1].projection.parameters())


@pytest.mark.parametrize("batch_size,expected_steps", [(None, 3), (1, 30), (4, 9)])
def test_train_step_counts_and_eval_schedule(batch_size, expected_steps):
    torch.manual_seed(0)
    model = build_model(SPECS, in_dim=12)
    X = np.random.default_rng(0).normal(size=(10, 12))
    cfg = TrainConfig(batch_size=batch_size, epochs=3, lr=0.1, eval_every_steps=2, log_every_epochs=0)
    res = train(model, X, X, cfg, probe=X[:2], eval_fn=lambda m, p: m(p).shape[0])
    assert len(res.losses) == expected_steps == res.final_step
    assert res.eval_steps[0] == 0 and res.eval_steps[-1] == expected_steps
    assert len(res.metrics) == len(res.eval_steps)
    assert len(set(res.eval_steps.tolist())) == len(res.eval_steps)


def test_train_step_offset_skips_duplicate_boundary_eval():
    model = build_model(SPECS, in_dim=12)
    X = np.random.default_rng(0).normal(size=(4, 12))
    cfg = TrainConfig(batch_size=None, epochs=5, lr=0.1, eval_every_steps=1, log_every_epochs=0)
    first = train(model, X, X, cfg, probe=X, eval_fn=lambda m, p: 0)
    second = train(model, X, X, cfg, probe=X, eval_fn=lambda m, p: 0, step_offset=first.final_step)
    assert first.eval_steps.tolist() == list(range(0, 6))
    assert second.eval_steps.tolist() == list(range(6, 11))


def test_wammes_build_data():
    cfg = wammes2021.StimuliConfig(n_pairs=4, dim=16, pair_repeats=3, phi_in=1.0, phi_out=0.0)
    X, Y, probe, rhos, items_x, items_y = wammes2021.build_data(cfg, np.random.default_rng(0))
    assert X.shape == Y.shape == (4 * 3 + 16, 16)
    assert probe.shape == (8, 16)
    assert np.allclose(probe[::2], items_x) and np.allclose(probe[1::2], items_y)
    # phi_in = 1 → input is the first item only; phi_out = 0 → target is the second item only
    assert np.allclose(X[:4], items_x) and np.allclose(Y[:4], items_y)


def test_wammes_end_to_end_shapes():
    torch.manual_seed(0)
    cfg = wammes2021.StimuliConfig(n_pairs=4, dim=12, pair_repeats=2)
    X, Y, probe, *_ = wammes2021.build_data(cfg, np.random.default_rng(0))
    model = build_model(SPECS, in_dim=12)
    res = train(
        model, X, Y, TrainConfig(epochs=2, lr=0.1, log_every_epochs=0),
        probe=probe, eval_fn=wammes2021.make_eval_fn(4, "hidden_0"),
    )
    corr = np.stack(res.metrics)
    assert corr.shape == (len(res.eval_steps), 2, 4, 4)
    assert np.all(np.abs(corr[np.isfinite(corr)]) <= 1 + 1e-6)
