"""repulsion.tasks — lightweight building blocks for standalone task scripts.

Each paradigm (e.g. ``scripts/tasks/wammes2021.py``) builds its own stimuli
and metric, and reuses the model, training loop and run helpers from here.
"""
from repulsion.tasks.models import SLOT, build_model, hidden_reps
from repulsion.tasks.run import resolve_output_dir, save_results, seed_everything
from repulsion.tasks.stimuli import correlated_pairs, superpose
from repulsion.tasks.training import TrainConfig, TrainResult, train

__all__ = [
    "SLOT",
    "TrainConfig",
    "TrainResult",
    "build_model",
    "correlated_pairs",
    "hidden_reps",
    "resolve_output_dir",
    "save_results",
    "seed_everything",
    "superpose",
    "train",
]
