"""Generate the items of an items config and print their similarity structure.

Mean/min/max cosine similarity per item type, subgroup and relation
(within_group, across_groups, across_subgroups), plus the category and image
of every image item.

Usage:
    python scripts/inspect_items.py configs/items/cifar100_resnet.yaml
    python scripts/inspect_items.py configs/items/cifar100_resnet.yaml --seed 3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from repulsion.stimgen import ItemGenerator, relation_cosines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="items config, e.g. configs/items/cifar100_resnet.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    gen = ItemGenerator(items=cfg["items"], generation_mode=cfg.get("generation_mode", "sampled"))
    item_set = gen.generate(np.random.default_rng(args.seed))

    pd.set_option("display.width", 200)
    print(relation_cosines(item_set).round(3).to_string(index=False))
    images = [dict(name=it.name, category=it.category, image_index=it.image_index)
              for it in item_set if it.image_index is not None]
    if images:
        print()
        print(pd.DataFrame(images).to_string(index=False))


if __name__ == "__main__":
    main()
