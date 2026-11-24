"""Generation for the ``"image"`` stimulus type.

Images are chosen from the class labels alone (see :func:`sample_image_indices`),
then only the chosen images are embedded (:meth:`ImageBank.embed`). Each item is
the embedding of exactly one image.

Relations (per subgroup, see :mod:`repulsion.stimgen.spec`):

=====================  ===================  =====================================
within_group           across_groups        images of each group
=====================  ===================  =====================================
same_image             same_category        one new image of the subgroup's class
same_image             different_category   one new image of a new class
same_category          same_category        n_items new images of the subgroup's class
same_category          different_category   n_items new images of a new class
different_category     different_category   one new image from each of n_items new classes
=====================  ===================  =====================================

"New" means not used anywhere else in the item set: classes and images are
drawn without replacement, shared across all image item types that use the
same dataset split.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from repulsion.stimgen import embeddings
from repulsion.stimgen.spec import ItemSpec


@dataclass
class _Pool:
    """Classes and images still unused within one ItemSet (one per dataset split)."""

    labels: np.ndarray
    used_classes: set[int] = field(default_factory=set)
    used_images: set[int] = field(default_factory=set)

    def take_classes(self, n: int, allowed: list[int], rng: np.random.Generator, where: str) -> list[int]:
        free = [c for c in allowed if c not in self.used_classes]
        if len(free) < n:
            raise ValueError(f"{where}: needs {n} more categories but only {len(free)} unused remain.")
        chosen = [int(c) for c in rng.choice(free, size=n, replace=False)]
        self.used_classes.update(chosen)
        return chosen

    def take_images(self, n: int, cls: int, rng: np.random.Generator, where: str) -> list[int]:
        free = [i for i in np.flatnonzero(self.labels == cls) if i not in self.used_images]
        if len(free) < n:
            raise ValueError(f"{where}: needs {n} more images of category {cls} but only {len(free)} remain.")
        chosen = [int(i) for i in rng.choice(free, size=n, replace=False)]
        self.used_images.update(chosen)
        return chosen


def sample_image_indices(item_spec: ItemSpec, pool: _Pool, class_names: list[str],
                         rng: np.random.Generator) -> list[int]:
    """Image index of every item, in generation order (subgroups -> groups -> items)."""
    if item_spec.categories is None:
        allowed = list(range(len(class_names)))
    else:
        unknown = sorted(set(item_spec.categories) - set(class_names))
        if unknown:
            raise ValueError(f"Item '{item_spec.item_name}': unknown categories {unknown}.")
        allowed = [class_names.index(c) for c in item_spec.categories]

    indices: list[int] = []
    for sg in item_spec.subgroups:
        where = f"Item '{sg.item_name}', subgroup '{sg.subgroup_name}'"
        shared = pool.take_classes(1, allowed, rng, where)[0] if sg.across_groups == "same_category" else None
        for _ in range(sg.n_groups):
            if sg.within_group == "different_category":
                for cls in pool.take_classes(sg.n_items, allowed, rng, where):
                    indices += pool.take_images(1, cls, rng, where)
                continue
            cls = shared if shared is not None else pool.take_classes(1, allowed, rng, where)[0]
            if sg.within_group == "same_image":
                indices += pool.take_images(1, cls, rng, where) * sg.n_items
            else:  # same_category
                indices += pool.take_images(sg.n_items, cls, rng, where)
    return indices


def generate_image_items(item_spec: ItemSpec, rng: np.random.Generator, context: dict):
    """Vectors (N, dim) and per-row metadata (category, image_index) for one item type.

    *context* is shared by all item types of one ItemSet, so that classes and
    images are not reused across item types.
    """
    bank = embeddings.get_bank(item_spec.source)
    key = (item_spec.source.dataset, item_spec.source.split, str(embeddings.data_dir(item_spec.source)))
    pool = context.setdefault(("image_pool",) + key, _Pool(bank.labels))
    indices = sample_image_indices(item_spec, pool, bank.class_names, rng)
    vectors = bank.embed(indices)
    meta = [{"category": bank.class_names[bank.labels[i]], "image_index": i} for i in indices]
    return vectors, meta
