"""Summaries of the similarity structure of a generated ItemSet."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from repulsion.stimgen.generator import ItemSet


def relation_cosines(item_set: ItemSet, item_type: str | None = None) -> pd.DataFrame:
    """Mean cosine similarity of item pairs, per item type, subgroup and relation.

    Relations: ``within_group`` (same group), ``across_groups`` (same subgroup,
    different groups) and ``across_subgroups`` (same item type, different
    subgroups; reported with subgroup ``"*"``). Useful for image items, whose
    correlations are not set directly.
    """
    types = [item_type] if item_type is not None else list(dict.fromkeys(it.item_type for it in item_set))
    rows = []
    for t in types:
        items = item_set.by_type(t)
        unit = {it.name: it.vector / np.linalg.norm(it.vector) for it in items}
        pairs: dict[tuple[str, str], list[float]] = {}
        for a, b in itertools.combinations(items, 2):
            if a.subgroup != b.subgroup:
                key = ("*", "across_subgroups")
            elif a.group_id == b.group_id:
                key = (a.subgroup, "within_group")
            else:
                key = (a.subgroup, "across_groups")
            pairs.setdefault(key, []).append(float(unit[a.name] @ unit[b.name]))
        for (subgroup, relation), cos in pairs.items():
            rows.append(dict(item_type=t, subgroup=subgroup, relation=relation,
                             mean_cos=np.mean(cos), min_cos=np.min(cos), max_cos=np.max(cos), n_pairs=len(cos)))
    return pd.DataFrame(rows)
