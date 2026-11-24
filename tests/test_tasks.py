"""Tests for repulsion.tasks and the task scripts (Wammes, Schapiro)."""
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
    make_pair_corr_eval,
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


def test_train_independent_objective_sums_stream_losses():
    torch.manual_seed(0)
    model = build_model(SPECS, in_dim=12)
    X = torch.as_tensor(np.random.default_rng(0).normal(size=(4, 12)), dtype=torch.float32)
    with torch.no_grad():
        streams = sum(((net(model.prepare_input(X)) - X) ** 2).sum(-1).mean() for net in model.networks)
        summed = ((model(X) - X) ** 2).sum(-1).mean()
    cfg = dict(batch_size=None, epochs=1, lr=0.0, log_every_epochs=0)
    assert np.isclose(train(model, X, X, TrainConfig(objective="independent", **cfg)).losses[0], streams)
    assert np.isclose(train(model, X, X, TrainConfig(**cfg)).losses[0], summed)
    with pytest.raises(ValueError):
        TrainConfig(objective="joint")


def test_train_steps_per_batch_repeats_updates():
    X = np.random.default_rng(0).normal(size=(4, 12))
    models, results = [], []
    for epochs, steps in ((1, 3), (3, 1)):
        torch.manual_seed(0)
        model = build_model(SPECS, in_dim=12)
        cfg = TrainConfig(batch_size=None, epochs=epochs, lr=0.05, momentum=0.0, steps_per_batch=steps,
                          log_every_epochs=0)
        models.append(model)
        results.append(train(model, X, X, cfg))
    assert results[0].final_step == 1 and np.isclose(results[0].losses[0], results[1].losses[0])
    for a, b in zip(*(m.parameters() for m in models)):
        assert torch.allclose(a, b, atol=1e-6)
    with pytest.raises(ValueError):
        TrainConfig(steps_per_batch=0)


def test_wammes_build_data():
    cfg = wammes2021.StimuliConfig(
        n_pairs=4, dim=16, pair_repeats=3, phi_in=1.0, phi_out=0.0, separate_inputs=False
    )
    X, Y, probe, rhos, items_x, items_y = wammes2021.build_data(cfg, np.random.default_rng(0))
    assert X.shape == Y.shape == (4 * 3 + 16, 16)
    assert probe.shape == (8, 16)
    assert np.allclose(probe[::2], items_x) and np.allclose(probe[1::2], items_y)
    # phi_in = 1 → input is the first item only; phi_out = 0 → target is the second item only
    assert np.allclose(X[:4], items_x) and np.allclose(Y[:4], items_y)


def test_wammes_build_data_separate_inputs():
    cfg = wammes2021.StimuliConfig(n_pairs=4, dim=16, pair_repeats=3, phi_in=0.25)
    X, Y, probe, rhos, items_x, items_y = wammes2021.build_data(cfg, np.random.default_rng(0))
    assert X.shape == (4 * 3 + 16, 32) and Y.shape == (4 * 3 + 16, 16)
    assert np.allclose(X[:4, :16], 0.5 * items_x) and np.allclose(X[:4, 16:], np.sqrt(0.75) * items_y)
    assert np.allclose(probe[::2, :16], items_x) and np.allclose(probe[::2, 16:], 0)
    assert np.allclose(probe[1::2, :16], 0) and np.allclose(probe[1::2, 16:], items_y)


def test_sum_input_slots():
    torch.manual_seed(0)
    specs = [
        {"name": "dense", "hidden_sizes": [8]},
        {"name": "sparse", "hidden_sizes": [8], "fixed_projection": True,
         "fixed_projection_hidden_size": 50, "fixed_projection_activation": "kwta",
         "fixed_projection_kwta_frac": 0.1},
    ]
    model = build_model(specs, in_dim=6, n_input_slots=2)
    x, y = torch.randn(5, 6), torch.randn(5, 6)
    xy = torch.cat([x, y], dim=1)
    dense, sparse = model.networks
    # dense stream: concatenated slots act like their sum
    assert torch.allclose(dense.extract(xy, None, "hidden_0"),
                          dense.extract(torch.cat([x + y, torch.zeros_like(y)], dim=1), None, "hidden_0"))
    # sparse stream: kwta(Px) + kwta(Py)
    proj = sparse.projection
    assert torch.allclose(sparse.extract(xy, None, "post_projection"), proj(x) + proj(y))
    assert model(xy).shape == (5, 6)


def test_wammes_end_to_end_shapes():
    torch.manual_seed(0)
    cfg = wammes2021.StimuliConfig(n_pairs=4, dim=12, pair_repeats=2)
    X, Y, probe, *_ = wammes2021.build_data(cfg, np.random.default_rng(0))
    model = build_model(SPECS, in_dim=12, n_input_slots=2)
    res = train(
        model, X, Y, TrainConfig(epochs=2, lr=0.1, log_every_epochs=0),
        probe=probe, eval_fn=make_pair_corr_eval(4, "hidden_0"),
    )
    corr = np.stack(res.metrics)
    assert corr.shape == (len(res.eval_steps), 2, 4, 4)
    assert np.all(np.abs(corr[np.isfinite(corr)]) <= 1 + 1e-6)


def test_correlated_pairs_exact_gram():
    from repulsion.tasks import correlated_pairs_exact

    corrs = [0.5, 0.5, 0.9, 0.9]
    pairs = correlated_pairs_exact(corrs, dim=32, rng=0, baseline=0.5)
    V = np.stack([v for p in pairs for v in p])
    G = V @ V.T
    expected = np.full((8, 8), 0.5)
    np.fill_diagonal(expected, 1.0)
    for i, c in enumerate(corrs):
        expected[2 * i, 2 * i + 1] = expected[2 * i + 1, 2 * i] = c
    assert np.allclose(G, expected, atol=1e-8)
    assert np.allclose(V.mean(1), 0, atol=1e-10)


def test_schapiro_sequence_and_data():
    import schapiro2012

    cfg = schapiro2012.StimuliConfig(dim=32, n_events=400, phi_in=1.0, phi_out=None, separate_inputs=False)
    X, Y, probe, items, seq = schapiro2012.build_data(cfg, np.random.default_rng(0))
    assert schapiro2012.resolve_phi_out(cfg) == 0.0
    assert X.shape == Y.shape == (len(seq) - 1, 32)
    # phi_in = 1, phi_out = 0 → next-item prediction
    assert np.allclose(X, items[seq[:-1]]) and np.allclose(Y, items[seq[1:]])
    # strong pairs always appear as A immediately followed by B
    strong_A = {2 * p for p in cfg.strong_pairs}
    for t in np.flatnonzero(np.isin(seq[:-1], list(strong_A))):
        assert seq[t + 1] == seq[t] + 1
    assert np.array_equal(probe, items)


def test_schapiro_separate_inputs():
    import schapiro2012

    cfg = schapiro2012.StimuliConfig(dim=32, n_events=400, phi_in=0.25)
    X, Y, probe, items, seq = schapiro2012.build_data(cfg, np.random.default_rng(0))
    assert X.shape == (len(seq) - 1, 64) and Y.shape == (len(seq) - 1, 32)
    assert np.allclose(X[:, :32], 0.5 * items[seq[:-1]])
    assert np.allclose(X[:, 32:], np.sqrt(0.75) * items[seq[1:]])
    assert np.allclose(probe[:, :32], items) and np.allclose(probe[:, 32:], 0)
