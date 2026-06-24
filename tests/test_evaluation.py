"""Tests for repulsion.evaluation (EvalSpec, Evaluator) and evaluation-wired train_schedule."""
from __future__ import annotations

import csv
import os
import tempfile

import numpy as np
import pytest
import torch

from repulsion.dataset import build_datasets
from repulsion.evaluation import (
    EvalSpec,
    Evaluator,
    _LogSchedule,
    build_evaluator,
    parse_eval_spec,
    parse_eval_specs,
)
from repulsion.models import parse_model_spec
from repulsion.models.network import SingleNetwork
from repulsion.schedule import build_training_schedule
from repulsion.stimgen import ItemGenerator
from repulsion.training import TrainingConfig, build_loss_spec, train_schedule

# ---------------------------------------------------------------------------
# Fixtures (same small dataset as test_models_and_training)
# ---------------------------------------------------------------------------

DIM = 16
N_GROUPS = 3
N_ITEMS = 2
N_ROWS = 2 * N_GROUPS * N_ITEMS  # 12


def _item_type(name):
    return {
        "name": name,
        "corr_between": 0.0,
        "n_groups": N_GROUPS,
        "n_items": N_ITEMS,
        "subgroups": [
            {"name": "high_sim", "corr_within": 0.8},
            {"name": "low_sim", "corr_within": 0.0},
        ],
    }


SLOTS = {
    "input": {"Face1": "face", "Object": "object"},
    "output": {"Face1": "face", "Object": "object"},
}

TASKS = [
    {
        "name": "autoencoding",
        "input": {
            "Face1": {"manipulation": "default"},
            "Object": {"manipulation": "default"},
        },
        "output": {
            "Face1": {"manipulation": "default"},
            "Object": {"manipulation": "default"},
        },
    },
]


@pytest.fixture(scope="module")
def item_set():
    gen = ItemGenerator(
        items=[_item_type("face"), _item_type("object")],
        default_dim=DIM,
    )
    return gen.generate(np.random.default_rng(42))


@pytest.fixture(scope="module")
def collection(item_set):
    return build_datasets(item_set, SLOTS, TASKS)


@pytest.fixture(scope="module")
def simple_model(collection):
    return parse_model_spec(
        [{"name": "net", "input": ["Face1", "Object"], "output": ["Face1", "Object"]}],
        collection,
    )


@pytest.fixture(scope="module")
def loss_spec(collection):
    return build_loss_spec(collection)


# ===========================================================================
# 1. EvalSpec parsing
# ===========================================================================

class TestEvalSpecParsing:
    def test_parse_representation_spec(self):
        spec = parse_eval_spec({
            "name": "hidden_reps",
            "task": "autoencoding",
            "save": "representation",
            "layer": "hidden_0",
            "network": 0,
            "eval_every_steps": 10,
        })
        assert spec.save == "representation"
        assert spec.layer == "hidden_0"
        assert spec.network == 0
        assert spec.eval_every_steps == 10

    def test_parse_output_spec(self):
        spec = parse_eval_spec({
            "name": "outputs",
            "task": "autoencoding",
            "save": "output",
            "eval_every_log_steps": 2.0,
        })
        assert spec.save == "output"
        assert spec.eval_every_log_steps == 2.0

    def test_parse_loss_spec(self):
        spec = parse_eval_spec({
            "name": "losses",
            "task": "autoencoding",
            "save": "loss",
            "eval_every_steps": 5,
        })
        assert spec.save == "loss"

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            parse_eval_spec({"task": "autoencoding", "save": "output"})

    def test_representation_without_layer_or_network_is_valid(self):
        spec = parse_eval_spec({
            "name": "x", "task": "autoencoding", "save": "representation",
        })
        assert spec.layer is None
        assert spec.network is None

    def test_representation_with_specific_layer_and_network(self):
        spec = parse_eval_spec({
            "name": "x", "task": "autoencoding",
            "save": "representation", "layer": "hidden_0", "network": 0,
        })
        assert spec.layer == "hidden_0"
        assert spec.network == 0

    def test_invalid_save_raises(self):
        with pytest.raises(ValueError, match="save"):
            parse_eval_spec({"name": "x", "task": "t", "save": "banana"})

    def test_parse_list(self):
        specs = parse_eval_specs([
            {"name": "a", "task": "autoencoding", "save": "output", "eval_every_steps": 1},
            {"name": "b", "task": "autoencoding", "save": "loss", "eval_every_steps": 1},
        ])
        assert len(specs) == 2
        assert all(isinstance(s, EvalSpec) for s in specs)


# ===========================================================================
# 2. LogSchedule
# ===========================================================================

class TestLogSchedule:
    def test_step_zero_is_always_due(self):
        sched = _LogSchedule(factor=2.0)
        assert sched.due(0)

    def test_factor_must_be_greater_than_one(self):
        with pytest.raises(ValueError):
            _LogSchedule(factor=1.0)
        with pytest.raises(ValueError):
            _LogSchedule(factor=0.5)

    def test_geometric_sequence_factor_two(self):
        sched = _LogSchedule(factor=2.0)
        due_steps = []
        for step in range(20):
            if sched.due(step):
                due_steps.append(step)
                sched.advance(step)
        # Steps 0, 1, 2, 4, 8, 16 should be covered
        assert 0 in due_steps
        assert 1 in due_steps
        assert 2 in due_steps
        assert 4 in due_steps
        assert 8 in due_steps
        assert 16 in due_steps
        # Step 3 should NOT be due (skipped between 2 and 4)
        assert 3 not in due_steps

    def test_dense_factor_covers_every_early_step(self):
        """Factor 1.01 should evaluate at every step for the first ~70 steps."""
        sched = _LogSchedule(factor=1.01)
        for step in range(10):
            assert sched.due(step), f"Step {step} should be due with factor 1.01"
            sched.advance(step)

    def test_advance_prevents_immediate_re_fire(self):
        sched = _LogSchedule(factor=2.0)
        assert sched.due(0)
        sched.advance(0)
        assert not sched.due(0)


# ===========================================================================
# 3. SingleNetwork.extract
# ===========================================================================

class TestExtract:
    def test_extract_input_shape(self, simple_model, collection):
        net: SingleNetwork = simple_model.networks[0]
        x = torch.randn(N_ROWS, 2 * DIM)
        out = net.extract(x, None, "input")
        # input = NaN-zeroed concatenated slots = 2*DIM columns
        assert out.shape == (N_ROWS, 2 * DIM)

    def test_extract_output_shape(self, simple_model, collection):
        net: SingleNetwork = simple_model.networks[0]
        x = torch.randn(N_ROWS, 2 * DIM)
        out = net.extract(x, None, "output")
        # output = MLP logits = global_pred_dim (all slots)
        assert out.shape[0] == N_ROWS

    def test_extract_scattered_shape(self, simple_model, collection):
        net: SingleNetwork = simple_model.networks[0]
        x = torch.randn(N_ROWS, 2 * DIM)
        out = net.extract(x, None, "scattered")
        assert out.shape == (N_ROWS, simple_model.global_prediction_dim)

    def test_extract_hidden_0_shape(self, collection):
        model = parse_model_spec(
            [{"name": "n", "input": ["Face1"], "output": ["Face1"],
              "hidden_sizes": [64]}],
            collection,
        )
        net = model.networks[0]
        x = torch.randn(N_ROWS, 2 * DIM)
        out = net.extract(x, None, "hidden_0")
        assert out.shape == (N_ROWS, 64)

    def test_extract_post_attention_shape(self, collection):
        model = parse_model_spec(
            [{"name": "n", "input": ["Face1"], "output": ["Face1"],
              "attention_layer": True}],
            collection,
        )
        net = model.networks[0]
        x = torch.randn(N_ROWS, 2 * DIM)
        out = net.extract(x, None, "post_attention")
        assert out.shape == (N_ROWS, DIM)

    def test_extract_post_attention_missing_raises(self, simple_model):
        net: SingleNetwork = simple_model.networks[0]
        x = torch.randn(N_ROWS, 2 * DIM)
        with pytest.raises(ValueError, match="no attention layer"):
            net.extract(x, None, "post_attention")

    def test_extract_post_projection_missing_raises(self, simple_model):
        net: SingleNetwork = simple_model.networks[0]
        x = torch.randn(N_ROWS, 2 * DIM)
        with pytest.raises(ValueError, match="no projection layer"):
            net.extract(x, None, "post_projection")

    def test_extract_unknown_layer_raises(self, simple_model):
        net: SingleNetwork = simple_model.networks[0]
        x = torch.randn(N_ROWS, 2 * DIM)
        with pytest.raises(ValueError, match="Unknown layer"):
            net.extract(x, None, "banana")

    def test_extract_post_projection_shape(self, collection):
        model = parse_model_spec(
            [{"name": "n", "input": ["Face1"], "output": ["Face1"],
              "fixed_projection": True, "fixed_projection_hidden_size": 32}],
            collection,
        )
        net = model.networks[0]
        x = torch.randn(N_ROWS, 2 * DIM)
        out = net.extract(x, None, "post_projection")
        assert out.shape == (N_ROWS, 32)


# ===========================================================================
# 4. Evaluator — representation
# ===========================================================================

class TestEvaluatorRepresentation:
    def test_representation_file_written(self, simple_model, collection, loss_spec):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = parse_eval_spec({
                "name": "reps",
                "task": "autoencoding",
                "save": "representation",
                "layer": "output",
                "network": 0,
                "eval_every_steps": 1,
            })
            ev = Evaluator([spec], collection, simple_model, loss_spec, tmpdir)
            ev.maybe_run_all(0)
            ev.maybe_run_all(1)
            data = np.load(os.path.join(tmpdir, "reps_net0_output.npy"))
            steps = np.load(os.path.join(tmpdir, "reps_steps.npy"))
            assert data.shape[0] == 2         # 2 checkpoints
            assert data.shape[1] == N_ROWS    # n_samples
            assert list(steps) == [0, 1]

    def test_representation_shape_matches_layer(self, collection, loss_spec):
        model = parse_model_spec(
            [{"name": "n", "input": ["Face1"], "output": ["Face1"],
              "hidden_sizes": [32]}],
            collection,
        )
        ls = build_loss_spec(collection)
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = parse_eval_spec({
                "name": "h0",
                "task": "autoencoding",
                "save": "representation",
                "layer": "hidden_0",
                "network": 0,
                "eval_every_steps": 1,
            })
            ev = Evaluator([spec], collection, model, ls, tmpdir)
            ev.maybe_run_all(0)
            data = np.load(os.path.join(tmpdir, "h0_net0_hidden_0.npy"))
            assert data.shape == (1, N_ROWS, 32)


# ===========================================================================
# 5. Evaluator — output
# ===========================================================================

class TestEvaluatorOutput:
    def test_output_files_written_per_network_and_combined(
        self, collection, loss_spec
    ):
        model = parse_model_spec(
            [
                {"name": "n1", "input": ["Face1"], "output": ["Face1"]},
                {"name": "n2", "input": ["Object"], "output": ["Object"]},
            ],
            collection,
        )
        ls = build_loss_spec(collection)
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = parse_eval_spec({
                "name": "out",
                "task": "autoencoding",
                "save": "output",
                "eval_every_steps": 1,
            })
            ev = Evaluator([spec], collection, model, ls, tmpdir)
            ev.maybe_run_all(0)
            files = os.listdir(tmpdir)
            # Active slots for autoencoding task: Face1, Object
            assert "out_net0_Face1.npy" in files
            assert "out_net0_Object.npy" in files
            assert "out_net1_Face1.npy" in files
            assert "out_net1_Object.npy" in files
            assert "out_combined_Face1.npy" in files
            assert "out_combined_Object.npy" in files
            assert "out_steps.npy" in files

    def test_combined_output_is_sum_of_per_network(self, collection, loss_spec):
        model = parse_model_spec(
            [
                {"name": "n1", "input": ["Face1"], "output": ["Face1", "Object"]},
                {"name": "n2", "input": ["Object"], "output": ["Face1", "Object"]},
            ],
            collection,
        )
        ls = build_loss_spec(collection)
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = parse_eval_spec({
                "name": "out",
                "task": "autoencoding",
                "save": "output",
                "eval_every_steps": 1,
            })
            ev = Evaluator([spec], collection, model, ls, tmpdir)
            ev.maybe_run_all(0)
            # Check one active slot (Face1) — combined should equal sum of per-network
            net0 = np.load(os.path.join(tmpdir, "out_net0_Face1.npy"))
            net1 = np.load(os.path.join(tmpdir, "out_net1_Face1.npy"))
            combined = np.load(os.path.join(tmpdir, "out_combined_Face1.npy"))
            np.testing.assert_allclose(net0[0] + net1[0], combined[0], rtol=1e-5)


# ===========================================================================
# 6. Evaluator — loss
# ===========================================================================

class TestEvaluatorLoss:
    def test_loss_files_written(self, simple_model, collection, loss_spec):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = parse_eval_spec({
                "name": "lss",
                "task": "autoencoding",
                "save": "loss",
                "eval_every_steps": 1,
            })
            ev = Evaluator([spec], collection, simple_model, loss_spec, tmpdir)
            ev.maybe_run_all(0)
            assert os.path.exists(os.path.join(tmpdir, "lss_combined.npy"))
            assert os.path.exists(os.path.join(tmpdir, "lss_net0.npy"))
            combined = np.load(os.path.join(tmpdir, "lss_combined.npy"))
            # shape (1 checkpoint, n_slots)
            assert combined.shape[0] == 1
            assert combined.shape[1] == len(loss_spec.slots)

    def test_loss_meta_has_slot_names(self, simple_model, collection, loss_spec):
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = parse_eval_spec({
                "name": "lss",
                "task": "autoencoding",
                "save": "loss",
                "eval_every_steps": 1,
            })
            ev = Evaluator([spec], collection, simple_model, loss_spec, tmpdir)
            ev.maybe_run_all(0)
            with open(os.path.join(tmpdir, "lss_meta.json")) as f:
                meta = json.load(f)
            assert "slots" in meta
            assert set(meta["slots"]) == {"Face1", "Object"}


# ===========================================================================
# 7. Geometric-step scheduling integration
# ===========================================================================

class TestLogScheduleIntegration:
    def test_eval_every_log_steps_produces_correct_checkpoints(
        self, simple_model, collection, loss_spec
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = parse_eval_spec({
                "name": "reps",
                "task": "autoencoding",
                "save": "output",
                "eval_every_log_steps": 2.0,
            })
            ev = Evaluator([spec], collection, simple_model, loss_spec, tmpdir)
            # Simulate steps 0..16
            for step in range(17):
                ev.maybe_run_all(step)
            steps = np.load(os.path.join(tmpdir, "reps_steps.npy"))
            # With factor=2: expect 0, 1, 2, 4, 8, 16
            assert 0 in steps
            assert 1 in steps
            assert 2 in steps
            assert 4 in steps
            assert 8 in steps
            assert 16 in steps
            # 3 should NOT be present
            assert 3 not in steps


# ===========================================================================
# 8. Integration with train_schedule
# ===========================================================================

class TestTrainScheduleWithEvaluator:
    def test_training_loss_csv_written(self, simple_model, collection):
        schedule = build_training_schedule(
            {"component_type": "autoencoding", "epochs": 2},
            available_tasks=collection.task_names(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            train_schedule(
                simple_model, collection, schedule,
                TrainingConfig(lr=1e-3),
                output_dir=tmpdir,
            )
            csv_path = os.path.join(tmpdir, "training_loss.csv")
            assert os.path.exists(csv_path)
            with open(csv_path) as f:
                rows = list(csv.reader(f))
            # Header + one row per gradient step
            assert rows[0][0] == "step"
            assert len(rows) > 2  # at least header + 1 data row

    def test_training_loss_csv_has_correct_slot_columns(self, simple_model, collection):
        schedule = build_training_schedule(
            {"component_type": "autoencoding", "epochs": 1},
            available_tasks=collection.task_names(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            train_schedule(
                simple_model, collection, schedule,
                TrainingConfig(),
                output_dir=tmpdir,
            )
            with open(os.path.join(tmpdir, "training_loss.csv")) as f:
                header = next(csv.reader(f))
            assert "step" in header
            assert "total_loss" in header
            assert "Face1" in header
            assert "Object" in header

    def test_evaluator_called_during_training(self, collection, loss_spec):
        model = parse_model_spec(
            [{"name": "n", "input": ["Face1", "Object"], "output": ["Face1", "Object"]}],
            collection,
        )
        schedule = build_training_schedule(
            {"component_type": "autoencoding", "epochs": 3},
            available_tasks=collection.task_names(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ev = build_evaluator(
                [{"name": "out", "task": "autoencoding", "save": "output",
                  "eval_every_steps": 1}],
                collection, model, loss_spec, tmpdir,
            )
            train_schedule(model, collection, schedule, TrainingConfig(), evaluator=ev)
            steps = np.load(os.path.join(tmpdir, "out_steps.npy"))
            # step 0 (pre-training) + one step per gradient update
            assert len(steps) > 1
            assert steps[0] == 0  # pre-training eval always fires

    def test_evaluator_and_output_dir_together(self, collection, loss_spec):
        model = parse_model_spec(
            [{"name": "n", "input": ["Face1"], "output": ["Face1"]}],
            collection,
        )
        schedule = build_training_schedule(
            {"component_type": "autoencoding", "epochs": 2},
            available_tasks=collection.task_names(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ev = build_evaluator(
                [{"name": "reps", "task": "autoencoding", "save": "representation",
                  "layer": "output", "network": 0, "eval_every_steps": 5}],
                collection, model, loss_spec, tmpdir,
            )
            train_schedule(
                model, collection, schedule, TrainingConfig(),
                evaluator=ev, output_dir=tmpdir,
            )
            assert os.path.exists(os.path.join(tmpdir, "training_loss.csv"))
            assert os.path.exists(os.path.join(tmpdir, "reps_net0_output.npy"))
