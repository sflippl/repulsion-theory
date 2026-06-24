"""Specification dataclasses and YAML-dict parsing for item generation.

Two stimulus types are supported:

``"corr"`` (default)
    Correlation-based generation.  Within- and between-group correlations are
    controlled by ``corr_within`` / ``corr_between``.  Items from different
    subgroups have ``ItemSpec.corr_between``.

``"circular"``
    Gaussian bump encoding on a 360° circle.  ``dim`` must be divisible by
    ``n_groups``; each group gets its own contiguous slice of dimensions
    (``dim // n_groups`` wide) so items from different groups are orthogonal
    by construction.  Within each group, item *i* (0-indexed) is placed at
    ``θ₀ + i × distance`` degrees where ``θ₀`` is a random starting angle.
    Each dimension ``k`` in the group's slice represents the angle
    ``360° × k / (dim // n_groups)``; the bump value is
    ``exp(−circ_dist(item_angle, dim_angle)² / (2 × bump_width²))``.
    Vectors are L2-normalised after the bump is applied.
    ``distance`` and ``bump_width`` (default 12°) are specified per subgroup
    and may be inherited from the item level.

Correlation hierarchy for the ``"corr"`` type:
    - same group, same subgroup  → SubgroupSpec.corr_within
    - different group, same subgroup → SubgroupSpec.corr_between
    - different subgroup (same item type) → ItemSpec.corr_between
    - different item type → 0  (generated independently)

Naming convention:  {item_name}_{subgroup_name}_{group_id}_{item_id}
    group_id and item_id are 1-indexed integers.
    When no subgroups are given, subgroup_name is "default".

Shorthand:
    corr: x  is equivalent to  corr_within: x, corr_between: x

Inheritance:
    Subgroup fields not explicitly set are inherited from the parent item dict.
    Fields that can be inherited: n_groups, n_items, magnitude, dim,
    corr_within, corr_between, distance, bump_width.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubgroupSpec:
    """Fully-resolved specification for one subgroup within an item type."""

    item_name: str
    subgroup_name: str
    n_groups: int      # number of groups in this subgroup
    n_items: int       # items per group
    corr_within: float  # correlation between items in the same group
    corr_between: float  # correlation between items in different groups (same subgroup)
    magnitude: float   # multiplicative scale applied to generated vectors
    dim: int           # vector dimensionality
    # Circular stimulus fields (only used when ItemSpec.stimulus_type == "circular")
    distance: float = 0.0    # angular distance (degrees) between successive items in a group
    bump_width: float = 12.0  # Gaussian bump width (degrees, std dev)
    # Circular stimulus type fields (None for corr-based)
    distance: float | None = None    # angular distance (degrees) between sequential items
    bump_width: float | None = None  # Gaussian bump width (degrees, 1-sigma)


@dataclass(frozen=True)
class ItemSpec:
    """Fully-resolved specification for one item type.

    Contains one or more SubgroupSpecs. All subgroups must share the same dim
    because they are sampled jointly from a single correlation matrix
    (``stimulus_type="corr"``) or share the same dimensional layout
    (``stimulus_type="circular"``).
    """

    item_name: str
    subgroups: tuple[SubgroupSpec, ...]
    corr_between: float  # correlation between items from *different* subgroups
    dim: int
    stimulus_type: str = "corr"  # "corr" or "circular"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_corr(d: dict, parent: dict | None = None) -> tuple[float, float]:
    """Return (corr_within, corr_between) from a spec dict.

    Resolution order:
      1. ``corr`` shorthand in d  →  both within and between set to that value.
      2. Individual ``corr_within`` / ``corr_between`` keys in d.
      3. Inherit from parent (same resolution applied to parent).
      4. Default: 0.0.
    """
    if "corr" in d:
        v = float(d["corr"])
        return v, v

    if parent is not None:
        p_within, p_between = _read_corr(parent)
    else:
        p_within, p_between = 0.0, 0.0

    within = float(d.get("corr_within", p_within))
    between = float(d.get("corr_between", p_between))
    return within, between


def _check_corr(value: float, label: str) -> None:
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"{label} must be in [-1, 1], got {value!r}.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_items(items: list[dict], default_dim: int = 64) -> list[ItemSpec]:
    """Parse a list of item specification dicts into resolved ItemSpec objects.

    Args:
        items: List of dicts, each describing one item type. Required keys:
            ``name``, ``n_groups``. Optional keys: ``n_items`` (default 1),
            ``corr`` / ``corr_within`` / ``corr_between`` (default 0),
            ``magnitude`` (default 1.0), ``dim``, ``subgroups``.
        default_dim: Vector dimensionality used when ``dim`` is absent from
            both the item dict and any subgroup dict.

    Returns:
        One :class:`ItemSpec` per entry in *items*.

    Raises:
        ValueError: For missing required fields, out-of-range correlations, or
            inconsistent dims across subgroups of the same item type.
    """
    specs: list[ItemSpec] = []

    for item_dict in items:
        # --- Required fields ---
        name = item_dict.get("name")
        if not name:
            raise ValueError("Each item entry must have a 'name' field.")
        if "n_groups" not in item_dict:
            raise ValueError(f"Item '{name}' must specify 'n_groups'.")

        # --- Item-level defaults ---
        n_groups = int(item_dict["n_groups"])
        n_items = int(item_dict.get("n_items", 1))
        magnitude = float(item_dict.get("magnitude", 1.0))
        dim = int(item_dict.get("dim", default_dim))
        stimulus_type = str(item_dict.get("stimulus_type", "corr"))

        if stimulus_type not in {"corr", "circular"}:
            raise ValueError(
                f"Item '{name}': stimulus_type must be 'corr' or 'circular', "
                f"got {stimulus_type!r}."
            )

        if n_groups < 1:
            raise ValueError(f"Item '{name}': n_groups must be >= 1, got {n_groups}.")
        if n_items < 1:
            raise ValueError(f"Item '{name}': n_items must be >= 1, got {n_items}.")

        if stimulus_type == "circular":
            item_bump_width = float(item_dict.get("bump_width", 12.0))
            raw_subgroups = item_dict.get("subgroups") or []
            if not raw_subgroups:
                raise ValueError(
                    f"Item '{name}': stimulus_type='circular' requires at least one "
                    "subgroup with a 'distance' field."
                )
            n_subgroups = len(raw_subgroups)
            if dim % (n_groups * n_subgroups) != 0:
                raise ValueError(
                    f"Item '{name}': for stimulus_type='circular', dim ({dim}) must be "
                    f"evenly divisible by n_groups × n_subgroups "
                    f"({n_groups} × {n_subgroups} = {n_groups * n_subgroups})."
                )
            resolved: list[SubgroupSpec] = []
            for sg_dict in raw_subgroups:
                sg_name = sg_dict.get("name")
                if not sg_name:
                    raise ValueError(
                        f"Each subgroup of item '{name}' must have a 'name'."
                    )
                if "distance" not in sg_dict:
                    raise ValueError(
                        f"Item '{name}', subgroup '{sg_name}': "
                        "stimulus_type='circular' requires a 'distance' field (degrees)."
                    )
                distance = float(sg_dict["distance"])
                bump_width = float(sg_dict.get("bump_width", item_bump_width))
                resolved.append(
                    SubgroupSpec(
                        item_name=name,
                        subgroup_name=sg_name,
                        n_groups=int(sg_dict.get("n_groups", n_groups)),
                        n_items=int(sg_dict.get("n_items", n_items)),
                        corr_within=0.0,
                        corr_between=0.0,
                        magnitude=float(sg_dict.get("magnitude", magnitude)),
                        dim=dim,
                        distance=distance,
                        bump_width=bump_width,
                    )
                )
            subgroups: tuple[SubgroupSpec, ...] = tuple(resolved)
            specs.append(
                ItemSpec(
                    item_name=name,
                    subgroups=subgroups,
                    corr_between=0.0,
                    dim=dim,
                    stimulus_type="circular",
                )
            )
            continue

        item_corr_within, item_corr_between = _read_corr(item_dict)
        _check_corr(item_corr_within, f"item '{name}' corr_within")
        _check_corr(item_corr_between, f"item '{name}' corr_between")

        # --- Subgroups ---
        raw_subgroups = item_dict.get("subgroups") or []

        if not raw_subgroups:
            # No subgroups → single implicit "default" subgroup
            sg = SubgroupSpec(
                item_name=name,
                subgroup_name="default",
                n_groups=n_groups,
                n_items=n_items,
                corr_within=item_corr_within,
                corr_between=item_corr_between,
                magnitude=magnitude,
                dim=dim,
            )
            subgroups: tuple[SubgroupSpec, ...] = (sg,)
        else:
            resolved: list[SubgroupSpec] = []
            for sg_dict in raw_subgroups:
                sg_name = sg_dict.get("name")
                if not sg_name:
                    raise ValueError(
                        f"Each subgroup of item '{name}' must have a 'name'."
                    )

                sg_corr_within, sg_corr_between = _read_corr(sg_dict, parent=item_dict)
                _check_corr(sg_corr_within, f"item '{name}' subgroup '{sg_name}' corr_within")
                _check_corr(sg_corr_between, f"item '{name}' subgroup '{sg_name}' corr_between")

                resolved.append(
                    SubgroupSpec(
                        item_name=name,
                        subgroup_name=sg_name,
                        n_groups=int(sg_dict.get("n_groups", n_groups)),
                        n_items=int(sg_dict.get("n_items", n_items)),
                        corr_within=sg_corr_within,
                        corr_between=sg_corr_between,
                        magnitude=float(sg_dict.get("magnitude", magnitude)),
                        dim=int(sg_dict.get("dim", dim)),
                    )
                )

            # Validate dim consistency — required for joint generation
            dims = {sg.dim for sg in resolved}
            if len(dims) > 1:
                raise ValueError(
                    f"Item '{name}': all subgroups must share the same 'dim' "
                    f"because they are generated jointly. Got dims: {dims}."
                )

            subgroups = tuple(resolved)

        specs.append(
            ItemSpec(
                item_name=name,
                subgroups=subgroups,
                corr_between=item_corr_between,
                dim=subgroups[0].dim,
                stimulus_type="corr",
            )
        )

    return specs
