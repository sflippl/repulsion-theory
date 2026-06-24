from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from repulsion import build_phase_dataloader, build_training_schedule
from repulsion.dataset import build_datasets
from repulsion.stimgen import ItemGenerator


def _make_item_type(name: str) -> dict:
    return {
        "name": name,
        "corr_between": 0.0,
        "n_groups": 2,
        "n_items": 2,
        "subgroups": [
            {"name": "high_sim", "corr_within": 0.8},
            {"name": "low_sim", "corr_within": 0.0},
        ],
    }


@pytest.fixture(scope="module")
def collection():
    gen = ItemGenerator(items=[_make_item_type("face"), _make_item_type("object")], default_dim=16)
    item_set = gen.generate(np.random.default_rng(0))

    slots = {
        "input": {"Face1": "face", "Object": "object"},
        "output": {"Face1": "face", "Object": "object", "Face2": "face"},
    }
    tasks = [
        {
            "name": "autoencoding",
            "input": {"Face1": {"manipulation": "default"}, "Object": {"manipulation": "default"}},
            "output": {"Face1": {"manipulation": "default"}, "Object": {"manipulation": "default"}},
        },
        {
            "name": "pairmate_prediction",
            "input": {"Face1": {"manipulation": "default"}},
            "output": {"Face2": {"manipulation": "group"}},
        },
    ]
    return build_datasets(item_set, slots, tasks)


def test_single_task_loader_emits_task_metadata(collection):
    schedule = build_training_schedule(
        {"component_type": "autoencoding", "epochs": 1, "batch_size": 3},
        available_tasks=collection.task_names(),
    )
    phase = schedule.phases[0]
    loader = build_phase_dataloader(collection, phase, shuffle=False)

    batch = next(iter(loader))
    assert set(batch.keys()) == {
        "input",
        "output",
        "task",
        "task_index",
        "task_weight",
        "row_index",
    }
    assert batch["input"].shape[0] == 3
    assert all(t == "autoencoding" for t in batch["task"])
    assert torch.all(batch["task_index"] == 0)
    assert torch.allclose(batch["task_weight"], torch.ones_like(batch["task_weight"]))


def test_multi_task_loader_has_both_tasks_and_weights(collection):
    schedule = build_training_schedule(
        {
            "component_type": ["autoencoding", "pairmate_prediction"],
            "weights": [1.0, 0.1],
            "epochs": 1,
            "batch_size": None,
        },
        available_tasks=collection.task_names(),
    )
    phase = schedule.phases[0]
    loader = build_phase_dataloader(collection, phase, shuffle=False)
    batch = next(iter(loader))

    task_names = set(batch["task"])
    assert task_names == {"autoencoding", "pairmate_prediction"}

    # Verify both weights are present in the full-batch output.
    weights = set(float(x) for x in batch["task_weight"].tolist())
    assert weights == {1.0, 0.1}


def test_full_batch_when_batch_size_none(collection):
    schedule = build_training_schedule(
        {"component_type": "pairmate_prediction", "epochs": 1, "batch_size": None},
        available_tasks=collection.task_names(),
    )
    phase = schedule.phases[0]
    loader = build_phase_dataloader(collection, phase, shuffle=False)
    batch = next(iter(loader))

    # One task dataset has 2 subgroups * 2 groups * 2 items = 8 rows.
    assert batch["input"].shape[0] == 8
    assert batch["output"].shape[0] == 8
