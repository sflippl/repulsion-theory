from __future__ import annotations

import pytest

from repulsion import build_training_schedule


class TestTrainingScheduleExamples:
    def test_recursive_sequence_example(self):
        schedule = {
            "component_type": [
                {"component_type": "autoencoding", "epochs": 2},
                {"component_type": "pairmate_prediction", "epochs": 1, "weights": 0.1},
            ],
            "batch_size": 1,
            "lr": 0.1,
            "momentum": 0.9,
            "epochs": 8,
        }

        plan = build_training_schedule(
            schedule,
            available_tasks=["autoencoding", "pairmate_prediction"],
        )

        # 8 repeats × 2 child components = 16 phases
        assert len(plan.phases) == 16
        assert plan.total_epochs == 24  # 8 * (2 + 1)

        # First phase of each repeat: autoencoding for 2 epochs
        first = plan.phases[0]
        assert first.tasks == ("autoencoding",)
        assert first.weights == (1.0,)
        assert first.epochs == 2
        assert first.steps is None
        assert first.params.batch_size == 1
        assert first.params.lr == pytest.approx(0.1)
        assert first.params.momentum == pytest.approx(0.9)

        # Second phase of each repeat: pairmate with weighted loss
        second = plan.phases[1]
        assert second.tasks == ("pairmate_prediction",)
        assert second.weights == (0.1,)
        assert second.epochs == 1

    def test_simultaneous_multitask_example(self):
        schedule = {
            "component_type": ["autoencoding", "pairmate_prediction"],
            "weights": [1.0, 0.1],
            "epochs": 10,
            "batch_size": "none",
        }

        plan = build_training_schedule(
            schedule,
            available_tasks=["autoencoding", "pairmate_prediction"],
        )

        assert len(plan.phases) == 1
        phase = plan.phases[0]
        assert phase.tasks == ("autoencoding", "pairmate_prediction")
        assert phase.weights == (1.0, 0.1)
        assert phase.epochs == 10
        assert phase.params.batch_size is None


class TestOverridesAndValidation:
    def test_child_overrides_optimizer_params(self):
        schedule = {
            "component_type": [
                {
                    "component_type": "autoencoding",
                    "epochs": 1,
                    "lr": 0.01,
                },
                {
                    "component_type": "pairmate_prediction",
                    "epochs": 1,
                },
            ],
            "lr": 0.1,
            "momentum": 0.9,
            "epochs": 2,
        }

        plan = build_training_schedule(
            schedule,
            available_tasks=["autoencoding", "pairmate_prediction"],
        )

        # Phase order per repeat: autoencoding, pairmate
        assert plan.phases[0].params.lr == pytest.approx(0.01)  # override
        assert plan.phases[1].params.lr == pytest.approx(0.1)   # inherited
        assert plan.phases[0].params.momentum == pytest.approx(0.9)

    def test_unknown_task_raises(self):
        schedule = {"component_type": "ghost", "epochs": 1}
        with pytest.raises(ValueError, match="unknown task"):
            build_training_schedule(schedule, available_tasks=["autoencoding"])

    def test_atomic_requires_epochs_or_steps(self):
        schedule = {"component_type": "autoencoding"}
        with pytest.raises(ValueError, match="must define 'epochs' or 'steps'"):
            build_training_schedule(schedule)

    def test_atomic_cannot_define_both_epochs_and_steps(self):
        schedule = {"component_type": "autoencoding", "epochs": 1, "steps": 10}
        with pytest.raises(ValueError, match="only one of 'epochs' or 'steps'"):
            build_training_schedule(schedule)

    def test_multitask_weights_length_must_match(self):
        schedule = {
            "component_type": ["autoencoding", "pairmate_prediction"],
            "weights": [1.0],
            "epochs": 1,
        }
        with pytest.raises(ValueError, match="weights length"):
            build_training_schedule(schedule)

    def test_sequence_disallows_weights(self):
        schedule = {
            "component_type": [
                {"component_type": "autoencoding", "epochs": 1},
                {"component_type": "pairmate_prediction", "epochs": 1},
            ],
            "weights": [1.0, 0.1],
            "epochs": 2,
        }
        with pytest.raises(ValueError, match="not valid for a sequence"):
            build_training_schedule(schedule)

    def test_steps_based_phase_supported(self):
        schedule = {
            "component_type": "autoencoding",
            "steps": 50,
            "batch_size": 4,
        }
        plan = build_training_schedule(schedule, available_tasks=["autoencoding"])
        assert len(plan.phases) == 1
        assert plan.phases[0].steps == 50
        assert plan.phases[0].epochs is None
        assert plan.total_steps == 50
