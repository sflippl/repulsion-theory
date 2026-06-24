"""Tests for the stimgen item generation module.

Organised into four sections:
  1. spec.py — parse_items(), inheritance, shorthand
  2. sampling.py — build_item_corr_matrix(), PSD check
  3. generator.py — end-to-end generation, naming, attribute correctness
  4. Numerical — correlation accuracy, magnitude, reproducibility, exact mode
"""
from __future__ import annotations

import numpy as np
import pytest

from repulsion.stimgen import Item, ItemGenerator, ItemSet, parse_items
from repulsion.stimgen.sampling import build_item_corr_matrix
from repulsion.stimgen.spec import SubgroupSpec


# ===========================================================================
# 1. Spec parsing
# ===========================================================================

class TestParseItems:
    def test_simple_no_subgroups_creates_default_subgroup(self):
        """A spec with no subgroups must produce exactly one 'default' subgroup."""
        items = [{"name": "face", "corr": 0.5, "n_groups": 6, "n_items": 2}]
        specs = parse_items(items, default_dim=64)
        assert len(specs) == 1
        spec = specs[0]
        assert spec.item_name == "face"
        assert len(spec.subgroups) == 1
        sg = spec.subgroups[0]
        assert sg.subgroup_name == "default"
        assert sg.n_groups == 6
        assert sg.n_items == 2
        assert sg.dim == 64

    def test_corr_shorthand_sets_both_within_and_between(self):
        """``corr: x`` must set both corr_within and corr_between to x."""
        specs = parse_items([{"name": "face", "corr": 0.3, "n_groups": 4, "n_items": 2}])
        sg = specs[0].subgroups[0]
        assert sg.corr_within == pytest.approx(0.3)
        assert sg.corr_between == pytest.approx(0.3)

    def test_separate_corr_within_corr_between(self):
        """corr_within and corr_between can be specified independently."""
        specs = parse_items([
            {"name": "face", "corr_within": 0.8, "corr_between": 0.1, "n_groups": 3, "n_items": 2}
        ])
        sg = specs[0].subgroups[0]
        assert sg.corr_within == pytest.approx(0.8)
        assert sg.corr_between == pytest.approx(0.1)

    def test_subgroup_inherits_n_groups_n_items_magnitude_dim(self):
        """Subgroups inherit all unset fields from the parent item dict."""
        items = [{
            "name": "color",
            "corr_between": 0.0,
            "n_groups": 6,
            "n_items": 2,
            "magnitude": 2.5,
            "dim": 128,
            "subgroups": [
                {"name": "high_sim", "corr_within": 0.8},
                {"name": "low_sim",  "corr_within": 0.0},
            ],
        }]
        specs = parse_items(items, default_dim=64)
        for sg in specs[0].subgroups:
            assert sg.n_groups == 6
            assert sg.n_items == 2
            assert sg.magnitude == pytest.approx(2.5)
            assert sg.dim == 128

    def test_subgroup_inherits_corr_between_from_parent(self):
        """Subgroup corr_between defaults to the item-level corr_between."""
        items = [{
            "name": "color",
            "corr_between": 0.1,
            "n_groups": 4,
            "n_items": 2,
            "subgroups": [
                {"name": "high_sim", "corr_within": 0.8},
                {"name": "low_sim",  "corr_within": 0.2},
            ],
        }]
        specs = parse_items(items)
        for sg in specs[0].subgroups:
            assert sg.corr_between == pytest.approx(0.1)

    def test_subgroup_can_override_corr_between(self):
        """Subgroup corr_between can be overridden independently."""
        items = [{
            "name": "color",
            "corr_between": 0.1,
            "n_groups": 4,
            "n_items": 2,
            "subgroups": [
                {"name": "high_sim", "corr_within": 0.8, "corr_between": 0.3},
                {"name": "low_sim",  "corr_within": 0.2},  # inherits 0.1
            ],
        }]
        specs = parse_items(items)
        assert specs[0].subgroups[0].corr_between == pytest.approx(0.3)
        assert specs[0].subgroups[1].corr_between == pytest.approx(0.1)

    def test_multiple_item_types_parsed_independently(self):
        """Multiple item types are returned in order, each fully resolved."""
        items = [
            {"name": "face",   "corr": 0.5, "n_groups": 3, "n_items": 2},
            {"name": "object", "corr": 0.0, "n_groups": 4, "n_items": 1},
        ]
        specs = parse_items(items)
        assert [s.item_name for s in specs] == ["face", "object"]
        assert specs[0].subgroups[0].corr_within == pytest.approx(0.5)
        assert specs[1].subgroups[0].n_groups == 4

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            parse_items([{"corr": 0.5, "n_groups": 3}])

    def test_missing_n_groups_raises(self):
        with pytest.raises(ValueError, match="n_groups"):
            parse_items([{"name": "face", "corr": 0.5}])

    def test_corr_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            parse_items([{"name": "face", "corr": 1.5, "n_groups": 3}])

    def test_inconsistent_subgroup_dims_raises(self):
        """Subgroups with different dims are forbidden (joint generation)."""
        items = [{
            "name": "face",
            "n_groups": 3,
            "subgroups": [
                {"name": "a", "corr": 0., "dim": 64},
                {"name": "b", "corr": 0., "dim": 128},
            ],
        }]
        with pytest.raises(ValueError, match="same 'dim'"):
            parse_items(items)

    def test_default_magnitude_is_one(self):
        specs = parse_items([{"name": "face", "corr": 0., "n_groups": 2}])
        assert specs[0].subgroups[0].magnitude == pytest.approx(1.0)

    def test_default_n_items_is_one(self):
        specs = parse_items([{"name": "face", "corr": 0., "n_groups": 2}])
        assert specs[0].subgroups[0].n_items == 1


# ===========================================================================
# 2. Correlation matrix construction
# ===========================================================================

class TestBuildItemCorrMatrix:
    def _make_spec(self, items_dicts):
        return parse_items(items_dicts)[0]

    def test_diagonal_is_one(self):
        spec = self._make_spec([{"name": "f", "corr": 0.5, "n_groups": 3, "n_items": 2}])
        C = build_item_corr_matrix(spec)
        np.testing.assert_array_equal(np.diag(C), 1.0)

    def test_symmetric(self):
        spec = self._make_spec([{"name": "f", "corr_within": 0.7, "corr_between": 0.2,
                                  "n_groups": 3, "n_items": 2}])
        C = build_item_corr_matrix(spec)
        np.testing.assert_allclose(C, C.T)

    def test_within_group_entries(self):
        """Off-diagonal entries inside the same group equal corr_within."""
        spec = self._make_spec([{"name": "f", "corr_within": 0.7, "corr_between": 0.2,
                                  "n_groups": 3, "n_items": 2}])
        C = build_item_corr_matrix(spec)
        # Group 0: rows 0, 1  (subgroup "default", group 0, items 0 and 1)
        assert C[0, 1] == pytest.approx(0.7)
        assert C[2, 3] == pytest.approx(0.7)  # group 1
        assert C[4, 5] == pytest.approx(0.7)  # group 2

    def test_between_group_entries(self):
        """Off-diagonal entries across groups equal corr_between."""
        spec = self._make_spec([{"name": "f", "corr_within": 0.7, "corr_between": 0.2,
                                  "n_groups": 3, "n_items": 2}])
        C = build_item_corr_matrix(spec)
        assert C[0, 2] == pytest.approx(0.2)
        assert C[0, 4] == pytest.approx(0.2)
        assert C[1, 3] == pytest.approx(0.2)

    def test_cross_subgroup_entries(self):
        """Items from different subgroups get the item-level corr_between."""
        items = [{
            "name": "color",
            "corr_between": 0.05,
            "n_groups": 2,
            "n_items": 1,
            "subgroups": [
                {"name": "high_sim", "corr_within": 0.8},
                {"name": "low_sim",  "corr_within": 0.0},
            ],
        }]
        spec = parse_items(items)[0]
        C = build_item_corr_matrix(spec)
        # high_sim: indices 0, 1 (2 groups × 1 item)
        # low_sim:  indices 2, 3
        # Cross-subgroup pairs: (0,2), (0,3), (1,2), (1,3)
        for i in [0, 1]:
            for j in [2, 3]:
                assert C[i, j] == pytest.approx(0.05), f"C[{i},{j}] should be 0.05"

    def test_shape_is_total_N_by_N(self):
        items = [{
            "name": "color",
            "corr_between": 0.,
            "n_groups": 3,
            "n_items": 2,
            "subgroups": [
                {"name": "high_sim", "corr_within": 0.8},
                {"name": "med_sim",  "corr_within": 0.5},
                {"name": "low_sim",  "corr_within": 0.0},
            ],
        }]
        spec = parse_items(items)[0]
        C = build_item_corr_matrix(spec)
        # 3 subgroups × 3 groups × 2 items = 18
        assert C.shape == (18, 18)

    def test_psd_check_aborts_on_invalid_params(self):
        """A matrix with all off-diagonals = -0.9 and N=3 is non-PSD."""
        # Equal-correlation N×N matrix: min eigenvalue = 1 + (N-1)*ρ
        # For ρ = -0.9, N = 3: 1 + 2*(-0.9) = -0.8 < 0 → must raise
        items = [{"name": "f", "corr": -0.9, "n_groups": 3, "n_items": 1}]
        gen = ItemGenerator(items, default_dim=32)
        with pytest.raises(ValueError, match="positive semi-definite"):
            gen.generate(np.random.default_rng(0))


# ===========================================================================
# 3. Generator — naming, attributes, ItemSet API
# ===========================================================================

class TestItemGeneratorNaming:
    def test_naming_no_subgroups(self):
        """Items are named {type}_default_{group_id}_{item_id} (1-indexed)."""
        gen = ItemGenerator(
            [{"name": "face", "corr": 0., "n_groups": 3, "n_items": 2}],
            default_dim=16,
        )
        item_set = gen.generate(np.random.default_rng(0))
        names = item_set.names()
        expected = {
            f"face_default_{g}_{i}"
            for g in range(1, 4)
            for i in range(1, 3)
        }
        assert set(names) == expected

    def test_naming_with_subgroups(self):
        """Items from subgroups are named {type}_{subgroup}_{group_id}_{item_id}."""
        gen = ItemGenerator(
            [{
                "name": "color",
                "corr_between": 0.,
                "n_groups": 2,
                "n_items": 1,
                "subgroups": [
                    {"name": "high_sim", "corr_within": 0.8},
                    {"name": "low_sim",  "corr_within": 0.0},
                ],
            }],
            default_dim=16,
        )
        item_set = gen.generate(np.random.default_rng(0))
        names = set(item_set.names())
        assert "color_high_sim_1_1" in names
        assert "color_high_sim_2_1" in names
        assert "color_low_sim_1_1" in names
        assert "color_low_sim_2_1" in names

    def test_total_item_count(self):
        gen = ItemGenerator(
            [{"name": "face", "corr": 0., "n_groups": 4, "n_items": 3}],
            default_dim=16,
        )
        item_set = gen.generate(np.random.default_rng(0))
        assert len(item_set) == 12  # 4 groups × 3 items

    def test_item_attributes_are_correct(self):
        gen = ItemGenerator(
            [{"name": "face", "corr": 0.5, "n_groups": 2, "n_items": 2, "magnitude": 3.0}],
            default_dim=32,
        )
        item_set = gen.generate(np.random.default_rng(0))
        item = item_set.by_name("face_default_2_1")
        assert item.item_type == "face"
        assert item.subgroup == "default"
        assert item.group_id == 2
        assert item.item_id == 1
        assert item.magnitude == pytest.approx(3.0)
        assert item.dim == 32
        assert item.vector.shape == (32,)

    def test_vector_shape(self):
        gen = ItemGenerator(
            [{"name": "face", "corr": 0., "n_groups": 3, "n_items": 2}],
            default_dim=48,
        )
        item_set = gen.generate(np.random.default_rng(0))
        for it in item_set:
            assert it.vector.shape == (48,)

    def test_item_set_by_type(self):
        gen = ItemGenerator(
            [
                {"name": "face",   "corr": 0., "n_groups": 3, "n_items": 2},
                {"name": "object", "corr": 0., "n_groups": 2, "n_items": 1},
            ],
            default_dim=16,
        )
        item_set = gen.generate(np.random.default_rng(0))
        assert len(item_set.by_type("face")) == 6
        assert len(item_set.by_type("object")) == 2
        assert all(it.item_type == "face" for it in item_set.by_type("face"))

    def test_item_set_by_subgroup(self):
        gen = ItemGenerator(
            [{
                "name": "color",
                "corr_between": 0.,
                "n_groups": 3,
                "n_items": 2,
                "subgroups": [
                    {"name": "high_sim", "corr_within": 0.8},
                    {"name": "low_sim",  "corr_within": 0.0},
                ],
            }],
            default_dim=16,
        )
        item_set = gen.generate(np.random.default_rng(0))
        high = item_set.by_subgroup("color", "high_sim")
        assert len(high) == 6  # 3 groups × 2 items
        assert all(it.subgroup == "high_sim" for it in high)

    def test_item_set_by_group(self):
        gen = ItemGenerator(
            [{"name": "face", "corr": 0., "n_groups": 3, "n_items": 2}],
            default_dim=16,
        )
        item_set = gen.generate(np.random.default_rng(0))
        grp2 = item_set.by_group("face", "default", 2)
        assert len(grp2) == 2
        assert all(it.group_id == 2 for it in grp2)

    def test_item_set_vectors_shape(self):
        gen = ItemGenerator(
            [{"name": "face", "corr": 0., "n_groups": 3, "n_items": 2}],
            default_dim=32,
        )
        item_set = gen.generate(np.random.default_rng(0))
        V = item_set.vectors("face")
        assert V.shape == (6, 32)

    def test_item_set_to_dict(self):
        gen = ItemGenerator(
            [{"name": "face", "corr": 0., "n_groups": 2, "n_items": 1}],
            default_dim=16,
        )
        item_set = gen.generate(np.random.default_rng(0))
        d = item_set.to_dict()
        assert set(d.keys()) == {"face_default_1_1", "face_default_2_1"}
        assert d["face_default_1_1"].shape == (16,)

    def test_item_set_mixed_dims_raises_without_filter(self):
        """vectors() without item_type= raises if dims are mixed."""
        gen = ItemGenerator(
            [
                {"name": "face",   "corr": 0., "n_groups": 2, "dim": 32},
                {"name": "object", "corr": 0., "n_groups": 2, "dim": 64},
            ],
        )
        item_set = gen.generate(np.random.default_rng(0))
        with pytest.raises(ValueError, match="mixed dims"):
            item_set.vectors()


# ===========================================================================
# 4. Numerical correctness
# ===========================================================================

def _mean_cosine(items_a: list[Item], items_b: list[Item]) -> float:
    """Mean cosine similarity between all pairs from two item lists."""
    cosines = []
    for a in items_a:
        for b in items_b:
            na = np.linalg.norm(a.vector)
            nb = np.linalg.norm(b.vector)
            cosines.append(np.dot(a.vector, b.vector) / (na * nb))
    return float(np.mean(cosines))


class TestNumericalCorrelations:
    """Numerical accuracy tests.  Use large dim and many groups so that the
    law of large numbers brings sample statistics close to theoretical values.
    Tolerances are intentionally generous (0.05) to avoid flakiness."""

    N_GROUPS = 40
    N_ITEMS = 2
    DIM = 2000

    def _make_and_generate(self, items_spec, seed=42):
        gen = ItemGenerator(items_spec, default_dim=self.DIM)
        return gen.generate(np.random.default_rng(seed))

    def test_within_group_correlation(self):
        """Mean cosine of within-group pairs ≈ corr_within."""
        target = 0.7
        item_set = self._make_and_generate([
            {"name": "face", "corr_within": target, "corr_between": 0.1,
             "n_groups": self.N_GROUPS, "n_items": 2}
        ])
        within = []
        for g in range(1, self.N_GROUPS + 1):
            pair = item_set.by_group("face", "default", g)
            v0, v1 = pair[0].vector, pair[1].vector
            within.append(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        assert abs(np.mean(within) - target) < 0.05

    def test_between_group_correlation(self):
        """Mean cosine across groups (within same subgroup) ≈ corr_between."""
        target_between = 0.2
        item_set = self._make_and_generate([
            {"name": "face", "corr_within": 0.7, "corr_between": target_between,
             "n_groups": self.N_GROUPS, "n_items": self.N_ITEMS}
        ])
        # Sample a subset of cross-group pairs (first item from each group)
        items = item_set.by_type("face")
        # Pick one representative item per group (item_id == 1)
        reps = [it for it in items if it.item_id == 1]
        cross = []
        for i, a in enumerate(reps):
            for b in reps[i + 1:]:
                cross.append(
                    np.dot(a.vector, b.vector)
                    / (np.linalg.norm(a.vector) * np.linalg.norm(b.vector))
                )
        assert abs(np.mean(cross) - target_between) < 0.05

    def test_cross_subgroup_correlation(self):
        """Pairs from different subgroups have cosine ≈ item-level corr_between."""
        parent_between = 0.1
        item_set = self._make_and_generate([
            {
                "name": "color",
                "corr_between": parent_between,
                "n_groups": 20,
                "n_items": 1,
                "subgroups": [
                    {"name": "high_sim", "corr_within": 0.8},
                    {"name": "low_sim",  "corr_within": 0.0},
                ],
            }
        ])
        high = item_set.by_subgroup("color", "high_sim")
        low  = item_set.by_subgroup("color", "low_sim")
        cos = _mean_cosine(high, low)
        assert abs(cos - parent_between) < 0.05

    def test_cross_type_independence(self):
        """Items from different item types have cosine ≈ 0 (generated independently)."""
        item_set = self._make_and_generate([
            {"name": "face",   "corr": 0., "n_groups": 20, "n_items": 1},
            {"name": "object", "corr": 0., "n_groups": 20, "n_items": 1},
        ])
        face   = item_set.by_type("face")[:10]
        object_ = item_set.by_type("object")[:10]
        cos = _mean_cosine(face, object_)
        assert abs(cos) < 0.05

    def test_magnitude_scales_norm(self):
        """Vector L2 norm ≈ magnitude (each base vector has norm ≈ 1)."""
        mag = 4.0
        gen = ItemGenerator(
            [{"name": "face", "corr": 0., "n_groups": 10, "n_items": 2, "magnitude": mag}],
            default_dim=500,
        )
        item_set = gen.generate(np.random.default_rng(7))
        norms = [np.linalg.norm(it.vector) for it in item_set]
        assert abs(np.mean(norms) - mag) < 0.3
        # Cosine similarity is invariant to magnitude
        pair = item_set.by_group("face", "default", 1)
        v0, v1 = pair[0].vector, pair[1].vector
        cos = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
        assert abs(cos) < 0.2  # corr=0, so should be near 0

    def test_reproducibility(self):
        """Same seed → identical ItemSet."""
        items = [{"name": "face", "corr": 0.5, "n_groups": 4, "n_items": 2}]
        gen = ItemGenerator(items, default_dim=32)
        set1 = gen.generate(np.random.default_rng(99))
        set2 = gen.generate(np.random.default_rng(99))
        for it1, it2 in zip(set1, set2):
            assert it1.name == it2.name
            np.testing.assert_array_equal(it1.vector, it2.vector)

    def test_different_seeds_produce_different_vectors(self):
        items = [{"name": "face", "corr": 0., "n_groups": 3, "n_items": 2}]
        gen = ItemGenerator(items, default_dim=32)
        set1 = gen.generate(np.random.default_rng(0))
        set2 = gen.generate(np.random.default_rng(1))
        assert not np.allclose(set1.items[0].vector, set2.items[0].vector)


# ===========================================================================
# 5. Exact generation mode
# ===========================================================================

class TestExactMode:
    def test_exact_within_group_correlation(self):
        """Exact mode: cosine of within-group pair == corr_within exactly."""
        corr_within = 0.6
        gen = ItemGenerator(
            [{"name": "face", "corr_within": corr_within, "corr_between": 0.1,
              "n_groups": 5, "n_items": 2}],
            default_dim=100,
            generation_mode="exact",
        )
        item_set = gen.generate()
        for g in range(1, 6):
            pair = item_set.by_group("face", "default", g)
            v0, v1 = pair[0].vector, pair[1].vector
            cos = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
            assert abs(cos - corr_within) < 1e-8, f"group {g}: cos={cos:.10f}"

    def test_exact_vector_norm_is_magnitude(self):
        """Exact mode: ‖v‖ == magnitude exactly (no sampling noise)."""
        mag = 2.5
        gen = ItemGenerator(
            [{"name": "face", "corr": 0.3, "n_groups": 4, "n_items": 2, "magnitude": mag}],
            default_dim=50,
            generation_mode="exact",
        )
        item_set = gen.generate()
        for it in item_set:
            np.testing.assert_allclose(np.linalg.norm(it.vector), mag, atol=1e-8)

    def test_exact_mode_is_deterministic(self):
        """Exact mode produces identical output regardless of rng."""
        gen = ItemGenerator(
            [{"name": "face", "corr": 0.5, "n_groups": 3, "n_items": 2}],
            default_dim=40,
            generation_mode="exact",
        )
        set1 = gen.generate()
        set2 = gen.generate(np.random.default_rng(999))
        for it1, it2 in zip(set1, set2):
            np.testing.assert_array_equal(it1.vector, it2.vector)

    def test_exact_mode_raises_when_dim_too_small(self):
        """Exact mode raises ValueError when dim < rank(C)."""
        # N=10 items (5 groups × 2), rank of C can be up to 10
        gen = ItemGenerator(
            [{"name": "face", "corr_within": 0.5, "corr_between": 0.1,
              "n_groups": 5, "n_items": 2}],
            default_dim=2,   # too small: rank is 10
            generation_mode="exact",
        )
        with pytest.raises(ValueError, match="dim >= rank"):
            gen.generate()

    def test_exact_mode_between_group_correlation(self):
        """Exact mode: cosine of between-group pair == corr_between exactly."""
        corr_between = 0.15
        gen = ItemGenerator(
            [{"name": "face", "corr_within": 0.6, "corr_between": corr_between,
              "n_groups": 4, "n_items": 1}],
            default_dim=50,
            generation_mode="exact",
        )
        item_set = gen.generate()
        items = item_set.by_type("face")
        # All pairs are between different groups (n_items=1 means no within-group pairs)
        pairs = [(items[i], items[j]) for i in range(len(items)) for j in range(i+1, len(items))]
        for a, b in pairs:
            cos = np.dot(a.vector, b.vector) / (np.linalg.norm(a.vector) * np.linalg.norm(b.vector))
            assert abs(cos - corr_between) < 1e-8


# ===========================================================================
# 6. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_n_items_one_no_within_group_pairs(self):
        """n_items=1 is valid; no within-group pairs exist."""
        gen = ItemGenerator(
            [{"name": "face", "corr": 0.5, "n_groups": 4, "n_items": 1}],
            default_dim=32,
        )
        item_set = gen.generate(np.random.default_rng(0))
        assert len(item_set) == 4

    def test_single_group(self):
        """A single group with 2 items: only within-group pairs."""
        gen = ItemGenerator(
            [{"name": "face", "corr_within": 0.8, "corr_between": 0.0,
              "n_groups": 1, "n_items": 2}],
            default_dim=32,
        )
        item_set = gen.generate(np.random.default_rng(0))
        assert len(item_set) == 2

    def test_three_example_specs_from_design_doc(self):
        """All three example specs from the design document must parse and generate."""
        # Example 1
        gen1 = ItemGenerator(
            [{"name": "face", "corr": 0.5, "n_groups": 6, "n_items": 2}],
            default_dim=32,
        )
        s1 = gen1.generate(np.random.default_rng(0))
        assert "face_default_1_1" in s1.names()

        # Example 2 (three subgroups, all corr=0)
        gen2 = ItemGenerator(
            [{
                "name": "face",
                "corr": 0.,
                "n_groups": 6,
                "n_items": 2,
                "subgroups": [
                    {"name": "high_sim"},
                    {"name": "med_sim"},
                    {"name": "low_sim"},
                ],
            }],
            default_dim=32,
        )
        s2 = gen2.generate(np.random.default_rng(0))
        assert "face_high_sim_1_1" in s2.names()
        assert "face_low_sim_6_2" in s2.names()

        # Example 3 (color, different corr_within per subgroup)
        gen3 = ItemGenerator(
            [{
                "name": "color",
                "corr_between": 0.,
                "n_groups": 6,
                "n_items": 2,
                "subgroups": [
                    {"name": "high_sim", "corr_within": 0.8},
                    {"name": "med_sim",  "corr_within": 0.5},
                    {"name": "low_sim",  "corr_within": 0.},
                ],
            }],
            default_dim=32,
        )
        s3 = gen3.generate(np.random.default_rng(0))
        assert "color_high_sim_1_1" in s3.names()
        assert "color_med_sim_3_2" in s3.names()
        assert "color_low_sim_6_2" in s3.names()


# ===========================================================================
# 6. Circular Gaussian bump stimulus type
# ===========================================================================

_CIRCULAR_SPEC = {
    "name": "orient",
    "stimulus_type": "circular",
    "n_groups": 4,
    "n_items": 2,
    "dim": 128,   # 128 / 4 = 32 dims per group
    "subgroups": [
        {"name": "close",  "distance": 30.0, "bump_width": 12.0},
        {"name": "far",    "distance": 90.0, "bump_width": 12.0},
    ],
}


class TestCircularSpec:
    def test_stimulus_type_stored(self):
        specs = parse_items([_CIRCULAR_SPEC])
        assert specs[0].stimulus_type == "circular"

    def test_subgroup_distance_stored(self):
        specs = parse_items([_CIRCULAR_SPEC])
        sgs = {sg.subgroup_name: sg for sg in specs[0].subgroups}
        assert sgs["close"].distance == pytest.approx(30.0)
        assert sgs["far"].distance == pytest.approx(90.0)

    def test_bump_width_stored(self):
        specs = parse_items([_CIRCULAR_SPEC])
        for sg in specs[0].subgroups:
            assert sg.bump_width == pytest.approx(12.0)

    def test_bump_width_default_is_12(self):
        spec = {
            "name": "x", "stimulus_type": "circular",
            "n_groups": 2, "n_items": 1, "dim": 64,
            "subgroups": [{"name": "sg", "distance": 45.0}],
        }
        specs = parse_items([spec])
        assert specs[0].subgroups[0].bump_width == pytest.approx(12.0)

    def test_bump_width_per_subgroup_override(self):
        spec = {
            "name": "x", "stimulus_type": "circular",
            "n_groups": 2, "n_items": 1, "dim": 64,
            "bump_width": 20.0,
            "subgroups": [
                {"name": "narrow", "distance": 30.0, "bump_width": 5.0},
                {"name": "wide",   "distance": 30.0},  # inherits 20.0
            ],
        }
        specs = parse_items([spec])
        sgs = {sg.subgroup_name: sg for sg in specs[0].subgroups}
        assert sgs["narrow"].bump_width == pytest.approx(5.0)
        assert sgs["wide"].bump_width == pytest.approx(20.0)

    def test_dim_not_divisible_raises(self):
        spec = {
            "name": "x", "stimulus_type": "circular",
            "n_groups": 3, "n_items": 1, "dim": 100,  # 100 % 3 != 0
            "subgroups": [{"name": "sg", "distance": 30.0}],
        }
        with pytest.raises(ValueError, match="divisible"):
            parse_items([spec])

    def test_missing_distance_raises(self):
        spec = {
            "name": "x", "stimulus_type": "circular",
            "n_groups": 2, "n_items": 1, "dim": 64,
            "subgroups": [{"name": "sg"}],  # no distance
        }
        with pytest.raises(ValueError, match="distance"):
            parse_items([spec])

    def test_no_subgroups_raises(self):
        spec = {
            "name": "x", "stimulus_type": "circular",
            "n_groups": 2, "n_items": 1, "dim": 64,
        }
        with pytest.raises(ValueError, match="subgroup"):
            parse_items([spec])


class TestCircularGeneration:
    @pytest.fixture(scope="class")
    def item_set(self):
        gen = ItemGenerator([_CIRCULAR_SPEC])
        return gen.generate(np.random.default_rng(42))

    def test_item_count(self, item_set):
        # 2 subgroups × 4 groups × 2 items = 16 items
        assert len(item_set.by_type("orient")) == 16

    def test_naming(self, item_set):
        assert "orient_close_1_1" in item_set.names()
        assert "orient_far_4_2" in item_set.names()

    def test_unit_norm(self, item_set):
        vectors = item_set.vectors("orient")
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_different_groups_orthogonal(self, item_set):
        """Items from different groups must be orthogonal (disjoint slices)."""
        g1 = item_set.by_group("orient", "close", 1)
        g2 = item_set.by_group("orient", "close", 2)
        for a in g1:
            for b in g2:
                dot = float(np.dot(a.vector, b.vector))
                assert abs(dot) < 1e-10

    def test_different_subgroups_orthogonal(self, item_set):
        """Items from different subgroups must be orthogonal (disjoint slices)."""
        close = item_set.by_group("orient", "close", 1)
        far   = item_set.by_group("orient", "far",   1)
        for a in close:
            for b in far:
                dot = float(np.dot(a.vector, b.vector))
                assert abs(dot) < 1e-10

    def test_closer_items_more_similar(self, item_set):
        """close-subgroup pairmates should be more similar than far-subgroup pairmates."""
        close_items = item_set.by_group("orient", "close", 1)
        far_items   = item_set.by_group("orient", "far",   1)
        close_sim = float(np.dot(close_items[0].vector, close_items[1].vector))
        far_sim   = float(np.dot(far_items[0].vector,   far_items[1].vector))
        assert close_sim > far_sim

    def test_reproducibility(self):
        gen = ItemGenerator([_CIRCULAR_SPEC])
        s1 = gen.generate(np.random.default_rng(7))
        s2 = gen.generate(np.random.default_rng(7))
        np.testing.assert_array_equal(
            s1.vectors("orient"), s2.vectors("orient")
        )

    def test_magnitude_zero_gives_zero_vector(self):
        spec = {
            "name": "x", "stimulus_type": "circular",
            "n_groups": 2, "n_items": 1, "dim": 64,
            "subgroups": [{"name": "silent", "distance": 30.0, "magnitude": 0.0}],
        }
        gen = ItemGenerator([spec])
        s = gen.generate(np.random.default_rng(0))
        vectors = s.vectors("x")
        assert np.all(vectors == 0.0)
