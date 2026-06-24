"""Tests for repulsion.dataset.

Fixture: two item types ('face', 'object') each with subgroups
['high_sim', 'low_sim'], 3 groups, 2 items → 12 rows each.
Both item types share the same row structure, making them compatible.
"""
from __future__ import annotations

import numpy as np
import pytest

from repulsion.dataset import (
    DatasetCollection,
    RowIndex,
    TaskDataset,
    build_datasets,
    parse_dataset_spec,
)
from repulsion.dataset.spec import SlotConfig, SlotDef
from repulsion.stimgen.generator import ItemGenerator


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_GROUPS = 3
N_ITEMS = 2
DIM = 32
SUBGROUPS = ["high_sim", "low_sim"]
N_ROWS = len(SUBGROUPS) * N_GROUPS * N_ITEMS  # 12


def _make_item_type(name: str) -> dict:
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


@pytest.fixture(scope="module")
def item_set():
    gen = ItemGenerator(
        items=[_make_item_type("face"), _make_item_type("object")],
        default_dim=DIM,
    )
    return gen.generate(np.random.default_rng(0))


SLOTS = {
    "input":  {"Face1": "face", "Object": "object"},
    "output": {"Face1": "face", "Object": "object", "Face2": "face"},
}

TASKS = [
    {
        "name": "autoencoding",
        "input":  {"Face1": {"manipulation": "default", "magnitude": 1.0},
                   "Object": {"manipulation": "default", "magnitude": 1.0}},
        "output": {"Face1": {"manipulation": "default", "magnitude": 1.0},
                   "Object": {"manipulation": "default", "magnitude": 1.0}},
    },
    {
        "name": "pairmate_prediction",
        "input":  {"Face1": {"manipulation": "default", "magnitude": 1.0}},
        "output": {"Face2": {"manipulation": "group"}},
    },
]


@pytest.fixture(scope="module")
def collection(item_set):
    return build_datasets(item_set, SLOTS, TASKS)


# ---------------------------------------------------------------------------
# 1. Spec parsing
# ---------------------------------------------------------------------------

class TestParseDatasetSpec:
    def test_slot_defs_are_ordered(self):
        spec = parse_dataset_spec(SLOTS, TASKS)
        assert [s.label for s in spec.input_slots] == ["Face1", "Object"]
        assert [s.label for s in spec.output_slots] == ["Face1", "Object", "Face2"]

    def test_slot_item_types(self):
        spec = parse_dataset_spec(SLOTS, TASKS)
        assert spec.input_slots[0] == SlotDef(label="Face1", item_type="face")
        assert spec.output_slots[2] == SlotDef(label="Face2", item_type="face")

    def test_task_names(self):
        spec = parse_dataset_spec(SLOTS, TASKS)
        assert [t.name for t in spec.tasks] == ["autoencoding", "pairmate_prediction"]

    def test_unmentioned_slots_disabled(self):
        """Slots not in a task's input/output dict must be None (disabled)."""
        spec = parse_dataset_spec(SLOTS, TASKS)
        pm = next(t for t in spec.tasks if t.name == "pairmate_prediction")
        # Object not in pairmate input
        assert pm.input_config["Object"] is None
        # Face1 and Object not in pairmate output
        assert pm.output_config["Face1"] is None
        assert pm.output_config["Object"] is None

    def test_active_slot_has_correct_config(self):
        spec = parse_dataset_spec(SLOTS, TASKS)
        pm = next(t for t in spec.tasks if t.name == "pairmate_prediction")
        cfg = pm.output_config["Face2"]
        assert cfg is not None
        assert cfg.manipulation == "group"
        assert cfg.magnitude == pytest.approx(1.0)

    def test_duplicate_task_name_raises(self):
        with pytest.raises(ValueError, match="Duplicate task name"):
            parse_dataset_spec(SLOTS, [
                {"name": "autoencoding"},
                {"name": "autoencoding"},
            ])

    def test_unknown_slot_reference_raises(self):
        with pytest.raises(ValueError, match="not defined in slots.input"):
            parse_dataset_spec(
                SLOTS,
                [{"name": "bad", "input": {"Ghost": {"manipulation": "default"}}}],
            )

    def test_invalid_manipulation_raises(self):
        with pytest.raises(ValueError, match="manipulation must be one of"):
            parse_dataset_spec(
                SLOTS,
                [{"name": "bad", "input": {"Face1": {"manipulation": "teleport"}}}],
            )

    def test_missing_task_name_raises(self):
        with pytest.raises(ValueError, match="'name'"):
            parse_dataset_spec(SLOTS, [{"input": {"Face1": {}}}])


# ---------------------------------------------------------------------------
# 2. Row structure
# ---------------------------------------------------------------------------

class TestRowStructure:
    def test_row_count(self, collection):
        ae = collection["autoencoding"]
        assert len(ae.rows) == N_ROWS  # 2 subgroups × 3 groups × 2 items = 12

    def test_row_order(self, collection):
        """Rows follow subgroup → group → item_id order."""
        ae = collection["autoencoding"]
        # First row: first subgroup, first group, first item
        assert ae.rows[0] == RowIndex(subgroup="high_sim", group_id=1, item_id=1)
        assert ae.rows[1] == RowIndex(subgroup="high_sim", group_id=1, item_id=2)
        assert ae.rows[2] == RowIndex(subgroup="high_sim", group_id=2, item_id=1)
        # After all high_sim rows (6), low_sim starts
        assert ae.rows[N_GROUPS * N_ITEMS] == RowIndex(subgroup="low_sim", group_id=1, item_id=1)

    def test_all_rows_unique(self, collection):
        ae = collection["autoencoding"]
        assert len(set(ae.rows)) == len(ae.rows)


# ---------------------------------------------------------------------------
# 3. Array shapes and concatenation
# ---------------------------------------------------------------------------

class TestArrayShapes:
    def test_input_shape_autoencoding(self, collection):
        ae = collection["autoencoding"]
        # Face1 (32) + Object (32) = 64
        assert ae.input.shape == (N_ROWS, 2 * DIM)

    def test_output_shape_autoencoding(self, collection):
        ae = collection["autoencoding"]
        # Face1 (32) + Object (32) + Face2 (32) = 96
        assert ae.output.shape == (N_ROWS, 3 * DIM)

    def test_slot_dims_match_dim(self, collection):
        ae = collection["autoencoding"]
        for label, d in ae.input_slot_dims.items():
            assert d == DIM
        for label, d in ae.output_slot_dims.items():
            assert d == DIM

    def test_slot_dims_allow_correct_slicing(self, collection):
        """Cumulative slot_dims can be used to slice the concatenated array."""
        ae = collection["autoencoding"]
        offset = 0
        for label, dim in ae.input_slot_dims.items():
            sliced = ae.input[:, offset:offset + dim]
            np.testing.assert_array_equal(sliced, ae.input_slot_arrays[label])
            offset += dim

    def test_pairmate_input_shape(self, collection):
        pm = collection["pairmate_prediction"]
        assert pm.input.shape == (N_ROWS, 2 * DIM)

    def test_pairmate_output_shape(self, collection):
        pm = collection["pairmate_prediction"]
        assert pm.output.shape == (N_ROWS, 3 * DIM)


# ---------------------------------------------------------------------------
# 4. Off-value (NaN) for disabled slots
# ---------------------------------------------------------------------------

class TestOffValue:
    def test_disabled_input_slot_is_nan(self, collection):
        """Object slot in pairmate input (disabled) must be all NaN."""
        pm = collection["pairmate_prediction"]
        arr = pm.input_slot_arrays["Object"]
        assert np.all(np.isnan(arr)), "Disabled slot should be all NaN"

    def test_disabled_output_slots_are_nan(self, collection):
        """Face1 and Object output slots in pairmate are disabled → all NaN."""
        pm = collection["pairmate_prediction"]
        assert np.all(np.isnan(pm.output_slot_arrays["Face1"]))
        assert np.all(np.isnan(pm.output_slot_arrays["Object"]))

    def test_active_slot_has_no_nan(self, collection):
        """Active slots must not contain NaN."""
        ae = collection["autoencoding"]
        assert not np.any(np.isnan(ae.input))
        assert not np.any(np.isnan(ae.output_slot_arrays["Face1"]))
        assert not np.any(np.isnan(ae.output_slot_arrays["Object"]))

    def test_custom_off_value(self, item_set):
        """off_value=-1 fills disabled slots with -1, not NaN."""
        coll = build_datasets(item_set, SLOTS, TASKS, off_value=-1.0)
        pm = coll["pairmate_prediction"]
        arr = pm.input_slot_arrays["Object"]
        assert np.all(arr == -1.0)
        assert not np.any(np.isnan(arr))


# ---------------------------------------------------------------------------
# 5. Default manipulation
# ---------------------------------------------------------------------------

class TestDefaultManipulation:
    def test_default_manipulation_matches_item_vector(self, item_set, collection):
        """Input Face1 in autoencoding equals the face item vector × magnitude."""
        ae = collection["autoencoding"]
        for row_idx, row in enumerate(ae.rows):
            expected = item_set.by_name(
                f"face_{row.subgroup}_{row.group_id}_{row.item_id}"
            ).vector  # magnitude=1.0
            np.testing.assert_allclose(
                ae.input_slot_arrays["Face1"][row_idx], expected
            )

    def test_magnitude_scales_vector(self, item_set):
        mag = 3.0
        coll = build_datasets(
            item_set,
            slots={"input": {"Face1": "face"}, "output": {}},
            tasks=[{"name": "t", "input": {"Face1": {"magnitude": mag}}}],
        )
        ds = coll["t"]
        for row_idx, row in enumerate(ds.rows):
            raw = item_set.by_name(f"face_{row.subgroup}_{row.group_id}_{row.item_id}").vector
            np.testing.assert_allclose(ds.input_slot_arrays["Face1"][row_idx], raw * mag)


# ---------------------------------------------------------------------------
# 6. Group manipulation
# ---------------------------------------------------------------------------

class TestGroupManipulation:
    def test_group_manipulation_equals_group_mean(self, item_set, collection):
        """Face2 output in pairmate_prediction equals the group-mean face vector."""
        pm = collection["pairmate_prediction"]
        for row_idx, row in enumerate(pm.rows):
            group_items = item_set.by_group("face", row.subgroup, row.group_id)
            expected_mean = np.mean(
                np.stack([it.vector for it in group_items]), axis=0
            )
            np.testing.assert_allclose(
                pm.output_slot_arrays["Face2"][row_idx], expected_mean,
                err_msg=f"Row {row}: group mean mismatch"
            )

    def test_group_output_constant_within_group(self, collection):
        """All item_ids in the same (subgroup, group_id) get the same group-mean vector."""
        pm = collection["pairmate_prediction"]
        for sg in SUBGROUPS:
            for g in range(1, N_GROUPS + 1):
                idxs = [
                    i for i, r in enumerate(pm.rows)
                    if r.subgroup == sg and r.group_id == g
                ]
                assert len(idxs) == N_ITEMS
                vecs = pm.output_slot_arrays["Face2"][idxs]
                # All rows in the group should be identical
                for row_vec in vecs[1:]:
                    np.testing.assert_array_equal(vecs[0], row_vec)

    def test_group_manipulation_with_magnitude(self, item_set):
        mag = 2.0
        coll = build_datasets(
            item_set,
            slots={"input": {}, "output": {"Face2": "face"}},
            tasks=[{"name": "t", "output": {"Face2": {"manipulation": "group", "magnitude": mag}}}],
        )
        ds = coll["t"]
        for row_idx, row in enumerate(ds.rows):
            group_items = item_set.by_group("face", row.subgroup, row.group_id)
            expected = np.mean(np.stack([it.vector for it in group_items]), axis=0) * mag
            np.testing.assert_allclose(ds.output_slot_arrays["Face2"][row_idx], expected)


# ---------------------------------------------------------------------------
# 7. DatasetCollection API
# ---------------------------------------------------------------------------

class TestDatasetCollection:
    def test_subscript_by_name(self, collection):
        ae = collection["autoencoding"]
        assert isinstance(ae, TaskDataset)
        assert ae.task_name == "autoencoding"

    def test_unknown_task_raises(self, collection):
        with pytest.raises(KeyError, match="'ghost'"):
            _ = collection["ghost"]

    def test_len(self, collection):
        assert len(collection) == 2

    def test_task_names(self, collection):
        assert collection.task_names() == ["autoencoding", "pairmate_prediction"]

    def test_iteration_yields_task_names(self, collection):
        assert list(collection) == ["autoencoding", "pairmate_prediction"]


# ---------------------------------------------------------------------------
# 8. Compatibility validation
# ---------------------------------------------------------------------------

class TestCompatibilityValidation:
    def test_incompatible_item_types_raises(self):
        """Slots referencing item types with different group counts must fail."""
        gen = ItemGenerator(
            items=[
                {"name": "face",   "corr": 0., "n_groups": 3, "n_items": 2},
                {"name": "object", "corr": 0., "n_groups": 4, "n_items": 2},  # different!
            ],
            default_dim=16,
        )
        item_set = gen.generate(np.random.default_rng(0))
        with pytest.raises(ValueError, match="incompatible structures"):
            build_datasets(
                item_set,
                slots={"input": {"Face1": "face", "Obj": "object"}, "output": {}},
                tasks=[{
                    "name": "bad",
                    "input": {
                        "Face1": {"manipulation": "default"},
                        "Obj":   {"manipulation": "default"},
                    },
                }],
            )

    def test_task_with_no_active_slots_raises(self, item_set):
        with pytest.raises(ValueError, match="no active slots"):
            build_datasets(
                item_set,
                slots={"input": {"Face1": "face"}, "output": {}},
                tasks=[{"name": "empty"}],  # no input or output specified
            )


# ---------------------------------------------------------------------------
# 9. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_item_set_gives_identical_datasets(self, item_set):
        c1 = build_datasets(item_set, SLOTS, TASKS)
        c2 = build_datasets(item_set, SLOTS, TASKS)
        for task_name in c1.task_names():
            np.testing.assert_array_equal(c1[task_name].input,  c2[task_name].input)
            np.testing.assert_array_equal(c1[task_name].output, c2[task_name].output)
