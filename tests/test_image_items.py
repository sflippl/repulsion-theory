"""Tests for the "image" stimulus type.

All tests except the last use a fake image bank (known labels, one random
feature vector per image), so they need neither the dataset nor torchvision.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from repulsion.stimgen import ItemGenerator, parse_items, relation_cosines
from repulsion.stimgen import embeddings

N_CLASSES, PER_CLASS, DIM = 12, 10, 512  # DIM = resnet18 avgpool


class FakeBank:
    def __init__(self, source):
        self.source = source
        self.labels = np.repeat(np.arange(N_CLASSES), PER_CLASS)
        self.class_names = [f"class{c}" for c in range(N_CLASSES)]
        self.features = np.random.default_rng(0).standard_normal((len(self.labels), DIM))
        self.embedded: list[int] = []

    def embed(self, indices):
        self.embedded += list(indices)
        return self.features[np.asarray(indices)]


@pytest.fixture
def fake_bank(monkeypatch):
    banks = {}

    def get_bank(source):
        return banks.setdefault(source, FakeBank(source))

    monkeypatch.setattr(embeddings, "get_bank", get_bank)
    return banks


def image_item(name="img", n_groups=3, n_items=2, subgroups=None, **kw):
    d = {"name": name, "stimulus_type": "image", "n_groups": n_groups, "n_items": n_items, **kw}
    if subgroups is not None:
        d["subgroups"] = subgroups
    return d


def generate(items, seed=0):
    return ItemGenerator(items=items).generate(np.random.default_rng(seed))


def groups(item_set, item_type, subgroup):
    out = {}
    for it in item_set.by_subgroup(item_type, subgroup):
        out.setdefault(it.group_id, []).append(it)
    return list(out.values())


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParseImageItems:
    def test_defaults(self):
        (spec,) = parse_items([image_item()])
        assert spec.stimulus_type == "image"
        assert spec.dim == 512
        assert spec.source == embeddings.ImageSource()
        (sg,) = spec.subgroups
        assert (sg.within_group, sg.across_groups) == ("same_category", "different_category")

    @pytest.mark.parametrize("model,layer,pool,dim", [
        ("resnet18", "layer3", "avg", 256),
        ("resnet18", "layer4", "flatten", 512 * 7 * 7),
        ("resnet50", "avgpool", "avg", 2048),
        ("resnet34", "fc", "avg", 1000),
    ])
    def test_dim_follows_layer(self, model, layer, pool, dim):
        (spec,) = parse_items([image_item(source={"model": model, "layer": layer, "pool": pool})])
        assert spec.dim == dim

    def test_subgroups_inherit_relations(self):
        (spec,) = parse_items([image_item(across_groups="same_category", subgroups=[
            {"name": "a", "within_group": "same_image"}, {"name": "b"}])])
        assert [(sg.within_group, sg.across_groups) for sg in spec.subgroups] == [
            ("same_image", "same_category"), ("same_category", "same_category")]

    def test_same_category_across_with_different_within_is_rejected(self):
        with pytest.raises(ValueError, match="impossible"):
            parse_items([image_item(within_group="different_category", across_groups="same_category")])

    @pytest.mark.parametrize("bad", [
        {"within_group": "same_thing"},
        {"across_groups": "same_image"},
        {"source": {"layer": "layer5"}},
        {"source": {"model": "vgg16"}},
        {"source": {"colour": "red"}},
        {"dim": 64},
        {"corr_within": 0.5},
    ])
    def test_invalid_specs_are_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_items([image_item(**bad)])

    def test_no_default_dim_needed(self):
        # train.py passes default_dim=None when the items config has none
        (spec,) = parse_items([image_item()], default_dim=None)
        assert spec.dim == 512
        with pytest.raises(ValueError, match="'dim' is required"):
            parse_items([{"name": "face", "n_groups": 2}], default_dim=None)

    def test_matching_dim_is_allowed(self):
        (spec,) = parse_items([image_item(dim=512)])
        assert spec.dim == 512


# ---------------------------------------------------------------------------
# Sampling relations
# ---------------------------------------------------------------------------

class TestImageRelations:
    @pytest.mark.parametrize("within,across", [
        ("same_image", "same_category"), ("same_image", "different_category"),
        ("same_category", "same_category"), ("same_category", "different_category"),
        ("different_category", "different_category"),
    ])
    def test_relations(self, fake_bank, within, across):
        item_set = generate([image_item(n_groups=3, n_items=3, within_group=within, across_groups=across)])
        gs = groups(item_set, "img", "default")
        assert len(gs) == 3 and all(len(g) == 3 for g in gs)
        for g in gs:
            images = [it.image_index for it in g]
            cats = [it.category for it in g]
            if within == "same_image":
                assert len(set(images)) == 1
                np.testing.assert_array_equal(g[0].vector, g[1].vector)
            elif within == "same_category":
                assert len(set(images)) == 3 and len(set(cats)) == 1
            else:
                assert len(set(images)) == 3 and len(set(cats)) == 3
        group_cats = [{it.category for it in g} for g in gs]
        if across == "same_category":
            assert len(set.union(*group_cats)) == 1
        else:
            assert all(not (a & b) for a, b in itertools.combinations(group_cats, 2))
        # every distinct item position uses its own image, except repeats within same_image groups
        all_images = [g[0].image_index if within == "same_image" else it.image_index for g in gs for it in g]
        n_unique = 3 if within == "same_image" else 9
        assert len(set(all_images)) == n_unique

    def test_no_reuse_across_subgroups_and_item_types(self, fake_bank):
        item_set = generate([
            image_item("a", n_groups=2, subgroups=[{"name": "s1"}, {"name": "s2", "within_group": "same_image"}]),
            image_item("b", n_groups=2, within_group="different_category"),
        ])
        cats_by_unit = {}
        for it in item_set:
            cats_by_unit.setdefault((it.item_type, it.subgroup, it.group_id), set()).add(it.category)
        all_cats = [c for cats in cats_by_unit.values() for c in cats]
        assert len(all_cats) == len(set(all_cats))
        images = {(it.item_type, it.subgroup, it.group_id, it.image_index) for it in item_set}
        assert len({img for *_, img in images}) == len(images)

    def test_vectors_are_the_embeddings_of_the_chosen_images(self, fake_bank):
        item_set = generate([image_item(magnitude=2.0)])
        (bank,) = fake_bank.values()
        for it in item_set:
            np.testing.assert_allclose(it.vector, 2.0 * bank.features[it.image_index])
            assert it.category == bank.class_names[bank.labels[it.image_index]]

    def test_only_needed_images_are_embedded(self, fake_bank):
        generate([image_item(n_groups=4, n_items=2, within_group="same_image")])
        (bank,) = fake_bank.values()
        assert len(set(bank.embedded)) == 4

    def test_reproducible_with_seed(self, fake_bank):
        items = [image_item(n_groups=4)]
        idx = lambda s: [it.image_index for it in generate(items, seed=s)]
        assert idx(1) == idx(1)
        assert idx(1) != idx(2)

    def test_categories_restrict_the_pool(self, fake_bank):
        allowed = ["class1", "class4", "class7"]
        item_set = generate([image_item(n_groups=3, categories=allowed)])
        assert {it.category for it in item_set} == set(allowed)

    def test_unknown_category_raises(self, fake_bank):
        with pytest.raises(ValueError, match="unknown categories"):
            generate([image_item(categories=["dragon"])])

    def test_too_few_categories_raises(self, fake_bank):
        with pytest.raises(ValueError, match="categories"):
            generate([image_item(n_groups=N_CLASSES + 1)])

    def test_too_few_images_raises(self, fake_bank):
        with pytest.raises(ValueError, match="images"):
            generate([image_item(n_groups=2, n_items=PER_CLASS, across_groups="same_category")])

    def test_mixed_with_corr_items(self, fake_bank):
        item_set = generate([image_item(), {"name": "face", "corr": 0.5, "n_groups": 3, "n_items": 2, "dim": 16}])
        assert item_set.vectors("img").shape == (6, 512)
        assert item_set.vectors("face").shape == (6, 16)
        assert all(it.category is None for it in item_set.by_type("face"))

    def test_relation_cosines(self, fake_bank):
        item_set = generate([image_item(n_groups=3, subgroups=[
            {"name": "same", "within_group": "same_image"}, {"name": "diff", "within_group": "different_category"}])])
        df = relation_cosines(item_set).set_index(["subgroup", "relation"])
        assert df.loc[("same", "within_group"), "mean_cos"] == pytest.approx(1.0)
        assert df.loc[("same", "within_group"), "n_pairs"] == 3
        assert abs(df.loc[("diff", "within_group"), "mean_cos"]) < 0.3  # independent random features
        assert ("*", "across_subgroups") in df.index


# ---------------------------------------------------------------------------
# Real embeddings (only when CIFAR-100 and torchvision are available)
# ---------------------------------------------------------------------------

def test_real_resnet_embeddings(tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    root = embeddings.data_dir(embeddings.ImageSource()) / "cifar100"
    if not (root / "cifar-100-python").exists():
        pytest.skip("CIFAR-100 not downloaded")
    # use the real dataset, but a temporary embedding cache
    real_root = str(root.parent)
    monkeypatch.setattr(embeddings, "_cache_path",
                        lambda source: tmp_path / "embeddings" / "cache.npz")
    items = [image_item(n_groups=2, n_items=2, source={"root": real_root},
                        subgroups=[{"name": "same", "within_group": "same_image"}, {"name": "cat"}])]
    a = generate(items)
    assert a.vectors("img").shape == (8, 512)
    assert (a.vectors("img") >= 0).all()  # avgpool follows a ReLU
    same = groups(a, "img", "same")[0]
    np.testing.assert_allclose(same[0].vector, same[1].vector)
    # second run reads the cache and gives identical vectors
    b = generate(items)
    np.testing.assert_allclose(a.vectors("img"), b.vectors("img"), rtol=1e-6)
