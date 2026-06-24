"""Item generation: Item, ItemSet, and ItemGenerator.

Typical usage::

    gen = ItemGenerator(
        items=[
            {"name": "face", "corr": 0.5, "n_groups": 6, "n_items": 2},
            {"name": "color", "corr_between": 0.0, "n_groups": 6, "n_items": 2,
             "subgroups": [
                 {"name": "high_sim", "corr_within": 0.8},
                 {"name": "med_sim",  "corr_within": 0.5},
                 {"name": "low_sim",  "corr_within": 0.0},
             ]},
        ],
        default_dim=64,
    )
    item_set = gen.generate(rng=np.random.default_rng(0))

    # Access individual items
    face_1_1 = item_set.by_name("face_default_1_1")
    all_faces = item_set.by_type("face")
    high_sim_colors = item_set.by_subgroup("color", "high_sim")
    face_vectors = item_set.vectors("face")   # (N_face, dim) array
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from repulsion.stimgen.sampling import (
    build_item_corr_matrix,
    exact_from_corr_matrix,
    generate_circular_items,
    sample_from_corr_matrix,
)
from repulsion.stimgen.spec import ItemSpec, parse_items


# ---------------------------------------------------------------------------
# Item  (single stimulus with all its attributes)
# ---------------------------------------------------------------------------

@dataclass
class Item:
    """A single generated stimulus vector together with its metadata.

    Attributes:
        name:         Full identifier, e.g. ``"face_default_1_1"``.
        vector:       Generated vector of shape ``(dim,)``.
        item_type:    Item-type name from the spec, e.g. ``"face"``.
        subgroup:     Subgroup name, e.g. ``"high_sim"`` or ``"default"``.
        group_id:     1-indexed group index within the subgroup.
        item_id:      1-indexed item index within the group.
        magnitude:    The magnitude scale applied to this item's vector.
        dim:          Dimensionality of the vector.
    """

    name: str
    vector: np.ndarray
    item_type: str
    subgroup: str
    group_id: int    # 1-indexed
    item_id: int     # 1-indexed
    magnitude: float
    dim: int

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"Item(name={self.name!r}, item_type={self.item_type!r}, "
            f"subgroup={self.subgroup!r}, group_id={self.group_id}, "
            f"item_id={self.item_id}, dim={self.dim}, "
            f"magnitude={self.magnitude}, vector=<shape ({self.dim},)>)"
        )


# ---------------------------------------------------------------------------
# ItemSet  (collection returned by ItemGenerator.generate())
# ---------------------------------------------------------------------------

class ItemSet:
    """An ordered collection of :class:`Item` objects.

    Items are stored in the order they were generated: outer loop over item
    types (in spec order), then subgroups, then groups, then items within each
    group.
    """

    def __init__(self, items: list[Item]) -> None:
        self.items: list[Item] = items
        self._by_name: dict[str, Item] = {it.name: it for it in items}

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __repr__(self) -> str:  # noqa: D401
        types = sorted({it.item_type for it in self.items})
        return f"ItemSet({len(self.items)} items, types={types})"

    # --- Lookup helpers ---

    def by_name(self, name: str) -> Item:
        """Return the single item with the given full name.

        Raises:
            KeyError: If no item has that name.
        """
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"No item named {name!r}. Available: {list(self._by_name)}")

    def by_type(self, item_type: str) -> list[Item]:
        """Return all items whose ``item_type`` matches."""
        return [it for it in self.items if it.item_type == item_type]

    def by_subgroup(self, item_type: str, subgroup: str) -> list[Item]:
        """Return all items matching both ``item_type`` and ``subgroup``."""
        return [
            it for it in self.items
            if it.item_type == item_type and it.subgroup == subgroup
        ]

    def by_group(self, item_type: str, subgroup: str, group_id: int) -> list[Item]:
        """Return all items matching item_type, subgroup, and group_id."""
        return [
            it for it in self.items
            if it.item_type == item_type
            and it.subgroup == subgroup
            and it.group_id == group_id
        ]

    # --- Array helpers ---

    def vectors(self, item_type: str | None = None) -> np.ndarray:
        """Return a (N, dim) array of vectors.

        Args:
            item_type: If given, restrict to items of that type. All items of
                a single type always share the same dim. If *None*, all items
                are included — which raises ``ValueError`` when item types have
                different dims.

        Returns:
            Array of shape ``(N, dim)``.
        """
        items = self.by_type(item_type) if item_type is not None else self.items
        if not items:
            return np.empty((0, 0), dtype=np.float64)
        dims = {it.dim for it in items}
        if len(dims) > 1:
            raise ValueError(
                f"Items have mixed dims {dims}. "
                "Pass item_type= to restrict to a single item type."
            )
        return np.stack([it.vector for it in items])

    def names(self, item_type: str | None = None) -> list[str]:
        """Return item names in the same order as :meth:`vectors`."""
        items = self.by_type(item_type) if item_type is not None else self.items
        return [it.name for it in items]

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return a mapping from item name to its vector."""
        return {it.name: it.vector for it in self.items}


# ---------------------------------------------------------------------------
# ItemGenerator  (parses spec, builds matrices, samples)
# ---------------------------------------------------------------------------

class ItemGenerator:
    """Generate stimulus items with controlled within- and between-group correlations.

    Each item type is generated independently (zero correlation across types).
    Within an item type, items from the same group share higher correlation
    (``corr_within``) than items from different groups (``corr_between``).
    Multiple subgroups can have different ``corr_within``/``corr_between``
    values while sharing a parent-level cross-subgroup correlation.

    Args:
        items: List of item specification dicts. See :func:`stimgen.spec.parse_items`
            for the full schema.
        default_dim: Vector dimensionality used when ``dim`` is not given in
            the spec. Can be overridden per item type.
        generation_mode: ``"sampled"`` (stochastic, supports any PSD matrix) or
            ``"exact"`` (deterministic, requires dim >= rank(C)).
        psd_eps: Eigenvalue threshold for the mandatory PSD check. Any
            eigenvalue below ``-psd_eps`` aborts generation with a
            ``ValueError``.

    Example::

        gen = ItemGenerator(
            items=[{"name": "face", "corr": 0.5, "n_groups": 6, "n_items": 2}],
            default_dim=64,
        )
        item_set = gen.generate(rng=np.random.default_rng(42))
    """

    def __init__(
        self,
        items: list[dict],
        default_dim: int = 64,
        generation_mode: str = "sampled",
        psd_eps: float = 1e-8,
    ) -> None:
        if generation_mode not in {"sampled", "exact"}:
            raise ValueError(
                f"generation_mode must be 'sampled' or 'exact', got {generation_mode!r}."
            )
        self.item_specs: list[ItemSpec] = parse_items(items, default_dim)
        self.generation_mode = generation_mode
        self.psd_eps = psd_eps

    def generate(self, rng: np.random.Generator | None = None) -> ItemSet:
        """Generate one :class:`ItemSet`.

        Args:
            rng: NumPy random generator for reproducibility. If *None*, a fresh
                default generator is used (non-reproducible). Ignored in
                ``"exact"`` mode (output is always deterministic).

        Returns:
            :class:`ItemSet` containing one :class:`Item` per stimulus, in
            spec order: item types → subgroups → groups → items within group.

        Raises:
            ValueError: If any item type's correlation matrix is not PSD, or if
                ``generation_mode="exact"`` and dim < rank(C) for any type.
        """
        if rng is None:
            rng = np.random.default_rng()

        all_items: list[Item] = []

        for item_spec in self.item_specs:
            if item_spec.stimulus_type == "circular":
                raw_vectors = generate_circular_items(item_spec, rng)
            elif self.generation_mode == "sampled":
                C = build_item_corr_matrix(item_spec)
                raw_vectors = sample_from_corr_matrix(C, item_spec.dim, rng, self.psd_eps)
            else:
                C = build_item_corr_matrix(item_spec)
                raw_vectors = exact_from_corr_matrix(C, item_spec.dim, self.psd_eps)

            # Assemble Item objects in the same row order as the matrix
            row = 0
            for sg in item_spec.subgroups:
                for g_idx in range(sg.n_groups):
                    for i_idx in range(sg.n_items):
                        name = f"{sg.item_name}_{sg.subgroup_name}_{g_idx + 1}_{i_idx + 1}"
                        all_items.append(
                            Item(
                                name=name,
                                vector=raw_vectors[row] * sg.magnitude,
                                item_type=sg.item_name,
                                subgroup=sg.subgroup_name,
                                group_id=g_idx + 1,
                                item_id=i_idx + 1,
                                magnitude=sg.magnitude,
                                dim=sg.dim,
                            )
                        )
                        row += 1

        return ItemSet(all_items)
