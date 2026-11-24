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

``"image"``
    Each item is the embedding of one image from an image dataset (default:
    CIFAR-100 through an ImageNet-pretrained ResNet-18, ``avgpool`` layer; see
    :mod:`repulsion.stimgen.embeddings`). Instead of correlations, each
    subgroup sets categorical relations:

    ``within_group`` (items of one group)
        ``same_image`` (one image, repeated), ``same_category`` (default;
        distinct images of one class) or ``different_category`` (one image
        from each of distinct classes).
    ``across_groups`` (groups of one subgroup)
        ``same_category`` (all groups share one class) or
        ``different_category`` (default; each group has its own classes).
        ``same_category`` requires ``within_group`` other than
        ``different_category``.

    ``rotation`` (degrees, default 0; requires ``within_group: same_image``)
    rotates item k of each group by (k - 1) * rotation, e.g. 10 makes the second
    item of a two-item group the same image rotated by 10 degrees.

    Classes and images are never reused across groups, subgroups or item
    types (except within a ``same_image`` group, or across groups with
    ``across_groups: same_category``). ``categories`` (class names) restricts
    the classes an item type may draw from. ``source`` picks the dataset,
    split, model, weights, layer and pooling; ``dim`` follows from the layer.
    Vectors are the raw features (no preprocessing), scaled by ``magnitude``.

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
    corr_within, corr_between, distance, bump_width, within_group,
    across_groups, rotation.
"""
from __future__ import annotations

from dataclasses import dataclass

from repulsion.stimgen.embeddings import ImageSource, layer_dim, parse_source

WITHIN_GROUP_RELATIONS = ("same_image", "same_category", "different_category")
ACROSS_GROUP_RELATIONS = ("same_category", "different_category")


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
    # Circular stimulus type fields (None otherwise)
    distance: float | None = None    # angular distance (degrees) between sequential items
    bump_width: float | None = None  # Gaussian bump width (degrees, 1-sigma)
    # Image stimulus type fields (None otherwise)
    within_group: str | None = None   # same_image | same_category | different_category
    across_groups: str | None = None  # same_category | different_category
    rotation: float = 0.0             # image items: item k of a group is rotated by (k - 1) * rotation degrees


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
    stimulus_type: str = "corr"  # "corr", "circular" or "image"
    source: ImageSource | None = None          # image type only
    categories: tuple[str, ...] | None = None  # image type only: allowed class names


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
            ``magnitude`` (default 1.0), ``dim``, ``subgroups``,
            ``stimulus_type`` (default ``"corr"``), plus the keys of that
            stimulus type (see the module docstring).
        default_dim: Vector dimensionality used when ``dim`` is absent from
            both the item dict and any subgroup dict (not used by ``"image"``,
            whose dim follows from the layer).

    Returns:
        One :class:`ItemSpec` per entry in *items*.

    Raises:
        ValueError: For missing required fields, out-of-range correlations,
            invalid relations, or inconsistent dims across subgroups of the
            same item type.
    """
    specs: list[ItemSpec] = []
    for item_dict in items:
        name = item_dict.get("name")
        if not name:
            raise ValueError("Each item entry must have a 'name' field.")
        if "n_groups" not in item_dict:
            raise ValueError(f"Item '{name}' must specify 'n_groups'.")
        stimulus_type = str(item_dict.get("stimulus_type", "corr"))
        if stimulus_type not in _PARSERS:
            raise ValueError(
                f"Item '{name}': stimulus_type must be one of {sorted(_PARSERS)}, "
                f"got {stimulus_type!r}."
            )
        common = dict(
            name=name,
            n_groups=int(item_dict["n_groups"]),
            n_items=int(item_dict.get("n_items", 1)),
            magnitude=float(item_dict.get("magnitude", 1.0)),
            dim=item_dict.get("dim", default_dim),  # None: not given (image dims follow the layer)
        )
        if common["n_groups"] < 1:
            raise ValueError(f"Item '{name}': n_groups must be >= 1, got {common['n_groups']}.")
        if common["n_items"] < 1:
            raise ValueError(f"Item '{name}': n_items must be >= 1, got {common['n_items']}.")
        if common["dim"] is not None:
            common["dim"] = int(common["dim"])
        elif stimulus_type != "image":
            raise ValueError(f"Item '{name}': 'dim' is required (no default_dim given).")
        specs.append(_PARSERS[stimulus_type](item_dict, **common))
    return specs


def _parse_circular(item_dict: dict, name: str, n_groups: int, n_items: int,
                    magnitude: float, dim: int | None) -> ItemSpec:
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
            raise ValueError(f"Each subgroup of item '{name}' must have a 'name'.")
        if "distance" not in sg_dict:
            raise ValueError(
                f"Item '{name}', subgroup '{sg_name}': "
                "stimulus_type='circular' requires a 'distance' field (degrees)."
            )
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
                distance=float(sg_dict["distance"]),
                bump_width=float(sg_dict.get("bump_width", item_bump_width)),
            )
        )
    return ItemSpec(item_name=name, subgroups=tuple(resolved), corr_between=0.0, dim=dim,
                    stimulus_type="circular")


def _parse_corr(item_dict: dict, name: str, n_groups: int, n_items: int,
                magnitude: float, dim: int | None) -> ItemSpec:
    item_corr_within, item_corr_between = _read_corr(item_dict)
    _check_corr(item_corr_within, f"item '{name}' corr_within")
    _check_corr(item_corr_between, f"item '{name}' corr_between")

    raw_subgroups = item_dict.get("subgroups") or []
    if not raw_subgroups:
        # No subgroups → single implicit "default" subgroup
        subgroups: tuple[SubgroupSpec, ...] = (
            SubgroupSpec(
                item_name=name,
                subgroup_name="default",
                n_groups=n_groups,
                n_items=n_items,
                corr_within=item_corr_within,
                corr_between=item_corr_between,
                magnitude=magnitude,
                dim=dim,
            ),
        )
    else:
        resolved: list[SubgroupSpec] = []
        for sg_dict in raw_subgroups:
            sg_name = sg_dict.get("name")
            if not sg_name:
                raise ValueError(f"Each subgroup of item '{name}' must have a 'name'.")

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

    return ItemSpec(item_name=name, subgroups=subgroups, corr_between=item_corr_between,
                    dim=subgroups[0].dim, stimulus_type="corr")


def _parse_image(item_dict: dict, name: str, n_groups: int, n_items: int,
                 magnitude: float, dim: int | None) -> ItemSpec:
    source = parse_source(item_dict.get("source"), name)
    dim = layer_dim(source)
    raw_subgroups = item_dict.get("subgroups") or [{"name": "default"}]
    for d in [item_dict, *raw_subgroups]:
        if "dim" in d and int(d["dim"]) != dim:
            raise ValueError(
                f"Item '{name}': dim {d['dim']} does not match the {dim}-d features of "
                f"{source.model} layer {source.layer!r}; leave 'dim' out for image items."
            )
        for key in ("corr", "corr_within", "corr_between"):
            if key in d:
                raise ValueError(
                    f"Item '{name}': image items set 'within_group'/'across_groups' "
                    f"instead of {key!r}."
                )

    resolved: list[SubgroupSpec] = []
    for sg_dict in raw_subgroups:
        sg_name = sg_dict.get("name")
        if not sg_name:
            raise ValueError(f"Each subgroup of item '{name}' must have a 'name'.")
        within = str(sg_dict.get("within_group", item_dict.get("within_group", "same_category")))
        across = str(sg_dict.get("across_groups", item_dict.get("across_groups", "different_category")))
        where = f"Item '{name}', subgroup '{sg_name}'"
        if within not in WITHIN_GROUP_RELATIONS:
            raise ValueError(f"{where}: within_group must be one of {WITHIN_GROUP_RELATIONS}, got {within!r}.")
        if across not in ACROSS_GROUP_RELATIONS:
            raise ValueError(f"{where}: across_groups must be one of {ACROSS_GROUP_RELATIONS}, got {across!r}.")
        rotation = float(sg_dict.get("rotation", item_dict.get("rotation", 0.0)))
        if rotation and within != "same_image":
            raise ValueError(f"{where}: rotation requires within_group='same_image', got {within!r}.")
        if within == "different_category" and across == "same_category":
            raise ValueError(
                f"{where}: across_groups='same_category' is impossible with "
                "within_group='different_category' (the groups have no single category to share)."
            )
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
                within_group=within,
                across_groups=across,
                rotation=rotation,
            )
        )
    categories = item_dict.get("categories")
    return ItemSpec(
        item_name=name,
        subgroups=tuple(resolved),
        corr_between=0.0,
        dim=dim,
        stimulus_type="image",
        source=source,
        categories=tuple(str(c) for c in categories) if categories is not None else None,
    )


# One parser per stimulus type: (item_dict, name, n_groups, n_items, magnitude, dim) -> ItemSpec
_PARSERS = {"corr": _parse_corr, "circular": _parse_circular, "image": _parse_image}
