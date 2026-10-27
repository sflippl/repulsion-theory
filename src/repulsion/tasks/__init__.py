"""repulsion.tasks — lightweight building blocks for standalone task scripts.

Each paradigm (e.g. ``scripts/tasks/wammes2021.py``) builds its own stimuli
and metric, and reuses the model, training loop and run helpers from here.
"""
from repulsion.tasks.metrics import (
    make_pair_corr_eval,
    make_readout_eval,
    pair_corr_arrays,
    readout_arrays,
    readout_outputs,
    stream_info,
)
from repulsion.tasks.models import SLOT, build_model, hidden_reps
from repulsion.tasks.run import resolve_output_dir, save_results, seed_everything
from repulsion.tasks.stimuli import correlated_pairs, correlated_pairs_exact, superpose, superpose_slots
from repulsion.tasks.training import TrainConfig, TrainResult, train

__all__ = [
    "SLOT",
    "TrainConfig",
    "TrainResult",
    "build_model",
    "correlated_pairs",
    "correlated_pairs_exact",
    "hidden_reps",
    "make_pair_corr_eval",
    "make_readout_eval",
    "pair_corr_arrays",
    "readout_arrays",
    "readout_outputs",
    "resolve_output_dir",
    "save_results",
    "seed_everything",
    "stream_info",
    "superpose",
    "superpose_slots",
    "train",
]
