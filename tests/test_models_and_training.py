"""Tests for repulsion.models and repulsion.training.

Fixtures build a small two-item-type dataset (face, object) with two subgroups
and 3 groups × 2 items each, using DIM=16 for speed.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from repulsion.dataset import build_datasets
from repulsion.models import (
    ACTIVATIONS,
    AttentionLayer,
    MultiNetwork,
    RandomProjection,
    SingleNetwork,
    build_activation,
    parse_model_spec,
)
from repulsion.models.activations import KWinnerTakesAll
from repulsion.schedule import build_training_schedule
from repulsion.stimgen import ItemGenerator
from repulsion.training import (
    TrainingConfig,
    TrainingHistory,
    build_loss_spec,
    compute_loss,
    train_schedule,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DIM = 16
N_GROUPS = 3
N_ITEMS = 2
N_ROWS = 2 * N_GROUPS * N_ITEMS  # 2 subgroups × 3 groups × 2 items = 12


def _item_type(name: str) -> dict:
    return {
        "name": name,
        "corr_between": 0.0,
        "n_groups": N_GROUPS,
        "n_items": N_ITEMS,
        "subgroups": [
            {"name": "high_sim", "corr_within": 0.8},
            {"name": "low_sim",  "corr_within": 0.0},
        ],
    }


SLOTS = {
    "input":  {"Face1": "face", "Object": "object"},
    "output": {"Face1": "face", "Object": "object", "Face2": "face"},
}

TASKS = [
    {
        "name": "autoencoding",
        "input":  {"Face1": {"manipulation": "default"},
                   "Object": {"manipulation": "default"}},
        "output": {"Face1": {"manipulation": "default"},
                   "Object": {"manipulation": "default"}},
    },
    {
        "name": "pairmate_prediction",
        "input":  {"Face1": {"manipulation": "default"}},
        "output": {"Face2": {"manipulation": "group"}},
    },
]


@pytest.fixture(scope="module")
def item_set():
    gen = ItemGenerator(
        items=[_item_type("face"), _item_type("object")],
        default_dim=DIM,
    )
    return gen.generate(np.random.default_rng(0))


@pytest.fixture(scope="module")
def collection(item_set):
    return build_datasets(item_set, SLOTS, TASKS)


@pytest.fixture(scope="module")
def simple_model(collection):
    return parse_model_spec(
        [{"name": "net", "input": ["Face1", "Object"], "output": ["Face1", "Object", "Face2"]}],
        collection,
    )


# ===========================================================================
# 1. Activations
# ===========================================================================

class TestActivations:
    def test_all_names_in_registry(self):
        assert {"sigmoid", "relu", "leaky_relu", "kwta", "identity"} == set(ACTIVATIONS)

    def test_build_activation_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            build_activation("bananas")

    def test_kwta_keeps_correct_fraction(self):
        kwta = KWinnerTakesAll(frac=0.5)
        x = torch.arange(10, dtype=torch.float).unsqueeze(0)  # [[0..9]]
        out = kwta(x)
        assert (out != 0).sum() == 5  # top 50% = 5 units

    def test_kwta_always_keeps_at_least_one(self):
        kwta = KWinnerTakesAll(frac=0.001)
        x = torch.ones(1, 8)
        out = kwta(x)
        assert (out != 0).sum() >= 1

    def test_build_leaky_relu(self):
        act = build_activation("leaky_relu", negative_slope=0.2)
        x = torch.tensor([-1.0])
        assert act(x).item() == pytest.approx(-0.2)

    def test_identity_unchanged(self):
        act = build_activation("identity")
        x = torch.randn(4, 8)
        assert torch.equal(act(x), x)


# ===========================================================================
# 2. RandomProjection
# ===========================================================================

class TestRandomProjection:
    def test_output_shape(self):
        proj = RandomProjection(input_dim=16, output_dim=64)
        x = torch.randn(5, 16)
        assert proj(x).shape == (5, 64)

    def test_weights_are_frozen(self):
        proj = RandomProjection(input_dim=16, output_dim=64)
        assert not proj.linear.weight.requires_grad

    def test_weights_unchanged_after_optimizer_step(self):
        """Frozen projection weights don't change even when a gradient flows through."""
        proj = RandomProjection(input_dim=16, output_dim=64)
        w_before = proj.linear.weight.clone()
        # Wrap in a trainable layer so the optimizer has something to update
        head = torch.nn.Linear(64, 1, bias=False)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        out = head(proj(torch.randn(2, 16))).sum()
        out.backward()
        opt.step()
        assert torch.equal(proj.linear.weight, w_before)

    def test_kwta_activation(self):
        proj = RandomProjection(16, 100, activation="kwta", frac=0.1)
        x = torch.randn(3, 16)
        out = proj(x)
        # Each row has at most 10 non-zero values
        assert int((out != 0).float().sum(dim=-1).max().item()) <= 10


# ===========================================================================
# 3. AttentionLayer
# ===========================================================================

class TestAttentionLayer:
    def test_shared_attention_output_shape(self):
        attn = AttentionLayer(input_dim=32)
        x = torch.randn(5, 32)
        assert attn(x).shape == (5, 32)

    def test_all_zero_logits_is_identity(self):
        """When all logits are 0, softmax is uniform and mean weight = 1 → no change."""
        attn = AttentionLayer(input_dim=8)
        x = torch.ones(3, 8)
        out = attn(x)
        torch.testing.assert_close(out, x)

    def test_slot_grouped_output_shape(self):
        attn = AttentionLayer(input_dim=32, slot_dims=[16, 16])
        x = torch.randn(4, 32)
        assert attn(x).shape == (4, 32)

    def test_slot_grouped_has_two_logits(self):
        attn = AttentionLayer(input_dim=32, slot_dims=[16, 16])
        assert attn.n_logits == 2

    def test_slot_dims_must_sum_to_input_dim(self):
        with pytest.raises(ValueError, match="sum"):
            AttentionLayer(input_dim=32, slot_dims=[10, 10])  # 20 ≠ 32

    def test_per_sample_requires_row_index_map(self):
        with pytest.raises(ValueError, match="row_index_to_id"):
            AttentionLayer(input_dim=8, per_sample=True, row_index_to_id=None)

    def test_per_sample_attention(self):
        row_map = {("sg", 1, 1): 0, ("sg", 1, 2): 1, ("sg", 2, 1): 2}
        attn = AttentionLayer(input_dim=8, per_sample=True, row_index_to_id=row_map)
        x = torch.randn(3, 8)
        ids = torch.tensor([0, 1, 2])
        out = attn(x, sample_ids=ids)
        assert out.shape == (3, 8)

    def test_gating_zero_makes_weights_uniform(self):
        attn = AttentionLayer(input_dim=4, gating=0.0)
        # Set logits to non-uniform values
        with torch.no_grad():
            attn.logits.copy_(torch.tensor([1.0, -1.0, 2.0, 0.5]))
        x = torch.ones(2, 4)
        out = attn(x)
        # With gating=0, softmax(0*logits)*4 = 4*(1/4) = 1.0 everywhere
        torch.testing.assert_close(out, x)

    def test_mean_weight_is_one(self):
        """Attention weights always have mean 1 regardless of logit values."""
        attn = AttentionLayer(input_dim=8)
        with torch.no_grad():
            attn.logits.copy_(torch.randn(8))
        x = torch.ones(1, 8)
        out = attn(x)
        assert out.mean().item() == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# 4. SingleNetwork / MultiNetwork
# ===========================================================================

class TestNetworkForward:
    def test_output_shape(self, simple_model, collection):
        first_task = collection["autoencoding"]
        x = torch.randn(N_ROWS, first_task.input.shape[1])
        pred = simple_model(x)
        assert pred.shape == (N_ROWS, simple_model.global_prediction_dim)

    def test_two_stream_model_output_shape(self, collection):
        model = parse_model_spec(
            [
                {"name": "net1", "input": ["Face1"], "output": ["Face1"]},
                {"name": "net2", "input": ["Face1"], "output": ["Face1"]},
            ],
            collection,
        )
        x = torch.randn(N_ROWS, 2 * DIM)
        pred = model(x)
        assert pred.shape == (N_ROWS, model.global_prediction_dim)

    def test_nan_input_slot_zeroed(self, collection):
        """Disabled input slots (NaN) are zeroed before the network."""
        model = parse_model_spec(
            [{"name": "net", "input": ["Face1", "Object"], "output": ["Face1"]}],
            collection,
        )
        x_nan = torch.full((N_ROWS, 2 * DIM), float("nan"))
        x_nan[:, :DIM] = 1.0  # Face1 active, Object is NaN
        # Should not produce NaN in output (NaN → 0)
        pred = model(x_nan)
        assert not torch.isnan(pred).any()

    def test_network_params_excludes_projection(self, collection):
        model = parse_model_spec(
            [{
                "name": "net",
                "input": ["Face1"],
                "output": ["Face1"],
                "fixed_projection": True,
                "fixed_projection_hidden_size": 32,
            }],
            collection,
        )
        net_params = set(id(p) for p in model.network_params())
        # Frozen projection weight must not appear in network_params
        for net in model.networks:
            if net.projection is not None:
                assert id(net.projection.linear.weight) not in net_params

    def test_attention_params_non_empty(self, collection):
        model = parse_model_spec(
            [{"name": "net", "input": ["Face1"], "output": ["Face1"],
              "attention_layer": True}],
            collection,
        )
        assert len(model.attention_params()) > 0

    def test_model_with_attention_slot_grouped(self, collection):
        model = parse_model_spec(
            [{"name": "net", "input": ["Face1", "Object"], "output": ["Face1"],
              "attention_layer": True, "attention_layer_slot_grouping": True}],
            collection,
        )
        # Slot-grouped: one logit per input slot (2 slots)
        for net in model.networks:
            if net.attention is not None:
                assert net.attention.n_logits == 2

    def test_model_with_fixed_projection(self, collection):
        model = parse_model_spec(
            [{
                "name": "net",
                "input": ["Face1"],
                "output": ["Face1"],
                "fixed_projection": True,
                "fixed_projection_hidden_size": 64,
                "fixed_projection_activation": "kwta",
                "fixed_projection_kwta_frac": 0.1,
            }],
            collection,
        )
        x = torch.randn(N_ROWS, 2 * DIM)
        pred = model(x)
        assert pred.shape[0] == N_ROWS


# ===========================================================================
# 5. Loss computation
# ===========================================================================

class TestComputeLoss:
    def test_mse_loss_active_slots(self, simple_model, collection):
        loss_spec = build_loss_spec(collection)
        batch_size = 4
        pred = torch.randn(batch_size, simple_model.global_prediction_dim)
        target = torch.randn(batch_size, sum(collection["autoencoding"].output_slot_dims.values()))
        weights = torch.ones(batch_size)
        loss, per_slot = compute_loss(pred, target, weights, loss_spec)
        assert loss.item() >= 0
        assert set(per_slot.keys()) == set(collection["autoencoding"].output_slot_dims.keys())

    def test_nan_slots_excluded_from_loss(self, simple_model, collection):
        """An all-NaN target slot (disabled) contributes 0 to the loss."""
        loss_spec = build_loss_spec(collection)
        # Build target with only Face1 active, rest NaN
        batch_size = 4
        task_ds = collection["autoencoding"]
        target = torch.full((batch_size, task_ds.output.shape[1]), float("nan"))
        # Fill just Face1 slot
        face1_dim = task_ds.output_slot_dims["Face1"]
        target[:, :face1_dim] = torch.randn(batch_size, face1_dim)

        pred = torch.randn(batch_size, simple_model.global_prediction_dim)
        weights = torch.ones(batch_size)
        _, per_slot = compute_loss(pred, target, weights, loss_spec)
        assert per_slot["Object"] == 0.0
        assert per_slot["Face2"] == 0.0
        assert per_slot["Face1"] > 0.0

    def test_task_weights_scale_loss(self, simple_model, collection):
        loss_spec = build_loss_spec(collection)
        batch_size = 6
        pred = torch.randn(batch_size, simple_model.global_prediction_dim)
        target = torch.randn(batch_size, collection["autoencoding"].output.shape[1])
        w1 = torch.ones(batch_size)
        w2 = torch.full((batch_size,), 2.0)
        loss1, _ = compute_loss(pred, target, w1, loss_spec)
        loss2, _ = compute_loss(pred, target, w2, loss_spec)
        assert loss2.item() == pytest.approx(loss1.item() * 2, rel=1e-4)


# ===========================================================================
# 6. Training loop
# ===========================================================================

class TestTrainSchedule:
    def test_loss_decreases_over_training(self, simple_model, collection):
        """Model loss should fall over 20 epochs of autoencoding."""
        schedule = build_training_schedule(
            {"component_type": "autoencoding", "epochs": 20, "batch_size": None},
            available_tasks=collection.task_names(),
        )
        history = train_schedule(
            simple_model, collection, schedule, TrainingConfig(lr=1e-2)
        )
        losses = [s.total_loss for s in history.phases[0].steps]
        assert losses[-1] < losses[0]

    def test_history_has_correct_phase_count(self, collection):
        model = parse_model_spec(
            [{"name": "net", "input": ["Face1", "Object"], "output": ["Face1"]}],
            collection,
        )
        schedule = build_training_schedule(
            {
                "component_type": [
                    {"component_type": "autoencoding", "epochs": 2},
                    {"component_type": "pairmate_prediction", "epochs": 1},
                ],
                "epochs": 2,
            },
            available_tasks=collection.task_names(),
        )
        history = train_schedule(model, collection, schedule, TrainingConfig())
        # 2 repeats × 2 phases = 4 phases
        assert len(history.phases) == 4

    def test_separate_attention_training(self, collection):
        model = parse_model_spec(
            [{"name": "net", "input": ["Face1", "Object"], "output": ["Face1"],
              "attention_layer": True}],
            collection,
        )
        schedule = build_training_schedule(
            {"component_type": "autoencoding", "epochs": 5},
            available_tasks=collection.task_names(),
        )
        cfg = TrainingConfig(separate_attention=True, attention_lr=1e-2, network_steps=2, attention_steps=1)
        history = train_schedule(model, collection, schedule, cfg)
        assert len(history.phases[0].steps) > 0

    def test_step_based_phase_runs_correct_number_of_steps(self, collection):
        model = parse_model_spec(
            [{"name": "net", "input": ["Face1"], "output": ["Face1"]}],
            collection,
        )
        schedule = build_training_schedule(
            {"component_type": "autoencoding", "steps": 7},
            available_tasks=collection.task_names(),
        )
        history = train_schedule(model, collection, schedule, TrainingConfig())
        assert len(history.phases[0].steps) == 7

    def test_returns_training_history(self, collection):
        model = parse_model_spec(
            [{"name": "net", "input": ["Face1"], "output": ["Face1"]}],
            collection,
        )
        schedule = build_training_schedule(
            {"component_type": "autoencoding", "epochs": 2},
            available_tasks=collection.task_names(),
        )
        history = train_schedule(model, collection, schedule, TrainingConfig())
        assert isinstance(history, TrainingHistory)
        assert all(len(p.steps) > 0 for p in history.phases)


# ===========================================================================
# 7. End-to-end: two-stream model with projection
# ===========================================================================

class TestEndToEnd:
    def test_two_stream_trains_without_error(self, collection):
        model = parse_model_spec(
            [
                {"name": "net1", "input": ["Face1", "Object"], "output": ["Face1", "Object"]},
                {
                    "name": "net2",
                    "input": ["Face1", "Object"],
                    "output": ["Face1", "Object"],
                    "fixed_projection": True,
                    "fixed_projection_hidden_size": 32,
                    "fixed_projection_activation": "kwta",
                    "fixed_projection_kwta_frac": 0.1,
                },
            ],
            collection,
        )
        schedule = build_training_schedule(
            {"component_type": "autoencoding", "epochs": 3},
            available_tasks=collection.task_names(),
        )
        history = train_schedule(model, collection, schedule, TrainingConfig(lr=1e-3))
        assert len(history.phases[0].steps) > 0
