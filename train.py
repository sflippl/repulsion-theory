"""train.py — Hydra-based training script for the repulsion package.

Usage
-----
# Run with defaults (outputs/ directory created automatically by Hydra):
    python train.py

# Override individual keys at the command line:
    python train.py training.lr=0.01 schedule.epochs=5

# Swap a config group:
    python train.py model=dual_stream

# Write outputs to a specific directory:
    python train.py output_dir=/tmp/my_run

# Multi-run sweep over learning rates:
    python train.py --multirun training.lr=0.001,0.01,0.1

Hydra config path: configs/  (relative to this file)
Top-level config : configs/config.yaml
"""
from __future__ import annotations

import logging
import os
import sys

# ── make the library importable when running without an editable install ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import repulsion.conf  # noqa: F401 — registers ConfigStore schemas before @hydra.main runs

log = logging.getLogger(__name__)

def sub(a, b):
    return a - b

OmegaConf.register_new_resolver("sub", sub)

def _container(cfg_node) -> dict | list:
    """Convert an OmegaConf node to a plain Python dict/list."""
    return OmegaConf.to_container(cfg_node, resolve=True, throw_on_missing=True)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    # ── resolve output directory ──────────────────────────────────────────
    if cfg.output_dir is not None:
        output_dir: str = str(cfg.output_dir)
    else:
        output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    # ── reproducibility ───────────────────────────────────────────────────
    rng = np.random.default_rng(cfg.seed)

    # ── imports (deferred so Hydra's logging is set up first) ────────────
    from repulsion.dataset import build_datasets
    from repulsion.evaluation import build_evaluator
    from repulsion.models import parse_model_spec
    from repulsion.schedule import build_training_schedule
    from repulsion.stimgen import ItemGenerator
    from repulsion.training import TrainingConfig, build_loss_spec, train_schedule

    # ── 1. Item generation ────────────────────────────────────────────────
    log.info("Generating items …")
    items_cfg: dict = _container(cfg["items"])
    gen = ItemGenerator(
        items=items_cfg["items"],
        # dim is now specified per-item, but ItemGenerator also accepts
        # a default_dim fallback; pass None if not present in cfg.
        default_dim=items_cfg.get("default_dim", None),
        generation_mode=items_cfg.get("generation_mode", "sampled"),
    )
    item_set = gen.generate(rng)
    log.info("Generated item set.")

    # ── 2. Dataset ────────────────────────────────────────────────────────
    log.info("Building datasets …")
    dataset_cfg: dict = _container(cfg.dataset)
    slots: dict = dataset_cfg["slots"]
    tasks: list = dataset_cfg["tasks"]
    model_slots: dict = dataset_cfg.get("model_slots") or {}
    collection = build_datasets(item_set, slots, tasks, model_slots=model_slots)
    log.info("Tasks: %s", collection.task_names())

    # ── 3. Model ──────────────────────────────────────────────────────────
    log.info("Building model …")
    networks_cfg: list = _container(cfg.model)["networks"]
    model = parse_model_spec(networks_cfg, collection, dataset_spec=collection.spec)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %d trainable parameters.", n_params)

    # ── 4. Training schedule ──────────────────────────────────────────────
    log.info("Building training schedule …")
    schedule_cfg: dict = _container(cfg.schedule)
    schedule = build_training_schedule(schedule_cfg, collection.task_names())
    n_phases = len(schedule.phases)
    log.info("Schedule: %d phases.", n_phases)

    # ── 5. Training config ────────────────────────────────────────────────
    training_cfg_dict: dict = _container(cfg.training)
    training_config = TrainingConfig(**training_cfg_dict)

    # ── 6. Loss spec ──────────────────────────────────────────────────────
    loss_spec = build_loss_spec(collection)

    # ── 7. Evaluator ─────────────────────────────────────────────────────
    eval_specs: list = _container(cfg.evaluation)["evaluations"]
    evaluator = (
        build_evaluator(eval_specs, collection, model, loss_spec, output_dir, cfg.device)
        if eval_specs
        else None
    )
    if evaluator:
        log.info("Evaluator: %d probe(s).", len(eval_specs))

    # ── 8. Train ──────────────────────────────────────────────────────────
    log.info("Training …")
    history = train_schedule(
        model,
        collection,
        schedule,
        training_config,
        device=cfg.device,
        evaluator=evaluator,
        output_dir=output_dir,
    )

    total_steps = sum(len(p.steps) for p in history.phases)
    final_loss = history.phases[-1].steps[-1].total_loss if total_steps else float("nan")
    log.info("Training complete. Total steps: %d. Final loss: %.6g.", total_steps, final_loss)


if __name__ == "__main__":
    main()
