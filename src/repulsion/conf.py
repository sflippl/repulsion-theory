"""Hydra structured-config dataclasses for the repulsion training script.

Importing this module registers all schemas with Hydra's ConfigStore so that
``config.yaml`` can reference them by name.  ``train.py`` imports this module
as a side-effect before ``@hydra.main`` runs.

ConfigStore groups registered:
    base_config          ← top-level Config schema
    items/base_items     ← ItemsConf
    dataset/base_dataset ← DatasetConf
    model/base_model     ← ModelConf
    training/base_training  ← TrainingConf
    evaluation/base_evaluation ← EvaluationConf

``schedule`` is NOT registered as a structured group because its recursive
structure cannot be described by a flat dataclass.  It is typed as ``Any``
in Config and validated at runtime by
:func:`repulsion.schedule.build_training_schedule`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

@dataclass
class ItemsConf:
    """Configuration for item generation.

    ``items`` is typed as ``List[Any]`` because each entry may contain an
    arbitrarily nested ``subgroups`` list — OmegaConf cannot represent that
    with a fixed typey 'scene_corr_between' not in 'Config'
    full_key: scene_corr_between
    object_type=Configed schema.  Validation is performed at runtime by
    :func:`repulsion.stimgen.parse_items`.
    """
    items: List[Any] = MISSING
    generation_mode: str = "sampled"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetConf:
    """Slot layout and task specifications.

    ``slots`` is ``Dict[str, Any]`` because its structure is:
    ``{input: {label: item_type, ...}, output: {label: item_type_or_dict, ...}}``

    ``tasks`` is ``List[Any]`` because each task has dynamic slot-label keys.

    ``model_slots`` is an optional mapping from network name to per-model
    input/output slot lists::

        model_slots:
          net1:
            input: [Face, Object]
            output: [Face, Object]

    Networks not listed default to all declared input/output slots in order.
    """
    slots: Dict[str, Any] = MISSING
    tasks: List[Any] = MISSING
    model_slots: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model / networks
# ---------------------------------------------------------------------------

@dataclass
class NetworkConf:
    """Specification for one :class:`~repulsion.models.SingleNetwork` stream.

    Field names match the keys consumed by
    :func:`repulsion.models.parse_model_spec` exactly.

    Input and output slot routing is no longer specified here; it is instead
    declared in the dataset config under ``model_slots``.
    """
    name: str = MISSING
    hidden_sizes: List[int] = field(default_factory=lambda: [256])
    activation: str = "identity"
    init_scale: float = 0.01
    # Attention layer
    attention_layer: bool = False
    attention_layer_slot_grouping: bool = False
    attention_layer_sample_grouping: bool = True
    attention_layer_gating: float = 1.0
    # Fixed random projection
    fixed_projection: bool = False
    fixed_projection_hidden_size: int = 1000
    fixed_projection_activation: str = "identity"
    fixed_projection_kwta_frac: float = 0.1
    fixed_projection_leaky_relu_slope: float = 0.01


@dataclass
class ModelConf:
    """List of network stream specifications."""
    networks: List[Any] = MISSING


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConf:
    """Training hyperparameters.

    Field names and defaults mirror :class:`repulsion.training.TrainingConfig`.
    """
    optimizer: str = "sgd"
    lr: float = 1e-3
    momentum: float = 0.9
    separate_attention: bool = False
    attention_optimizer: str = "sgd"
    attention_lr: float = 1e-3
    attention_momentum: float = 0.9
    network_steps: int = 1
    attention_steps: int = 1


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalSpecConf:
    """Specification for one evaluation probe.

    Field names mirror :class:`repulsion.evaluation.EvalSpec`.
    """
    name: str = MISSING
    task: str = MISSING
    save: str = "output"           # "representation" | "output" | "loss"
    layer: Optional[str] = None    # None → all hidden layers (when save="representation")
    network: Optional[int] = None  # None → all networks (when save="representation")
    eval_every_steps: Optional[int] = None
    eval_every_log_steps: Optional[float] = None


@dataclass
class EvaluationConf:
    """Collection of evaluation probes."""
    evaluations: List[Any] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Experiment Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class ExperimentHParams:
    pass

@dataclass
class Favila2016HParams(ExperimentHParams):
    scene_corr_between: float = 0.0
    scene_corr_within: float = 0.2
    face_corr: float = 0.05
    prediction_weight: float = 1.0

@dataclass
class Chanales2017Sim1HParams(ExperimentHParams):
    route_corr_between: float = 0.0
    route_corr_within: float = 0.9
    destination_corr: float = 0.0

@dataclass
class Chanales2017Sim2HParams(ExperimentHParams):
    dim_total: int = 128
    dim_shared: int = 64
    destination_corr: float = 0.0
    route_corr_between: float = 0.0

@dataclass
class Chanales2021HParams(ExperimentHParams):
    bump_width: float = 12.0
    association_weight: float = 1.0
    face_corr: float = 0.0
    prediction_test_weight: float = 1.0

# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Top-level Hydra configuration.

    ``schedule`` is typed ``Any`` because it is a recursively structured dict
    that cannot be expressed as a flat dataclass.
    """
    items: ItemsConf = MISSING
    dataset: DatasetConf = MISSING
    model: ModelConf = MISSING
    schedule: Any = MISSING
    experiment_hparams: ExperimentHParams = field(default_factory=ExperimentHParams)
    training: TrainingConf = field(default_factory=TrainingConf)
    evaluation: EvaluationConf = field(default_factory=EvaluationConf)
    seed: int = 42
    device: str = "cpu"
    # null  → use Hydra's outputs/YYYY-MM-DD/HH-MM-SS/ directory
    # string → write all outputs there
    output_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# ConfigStore registration  (runs on import)
# ---------------------------------------------------------------------------

def _register() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config",      node=Config)
    cs.store(group="items",      name="base_items",      node=ItemsConf)
    cs.store(group="dataset",    name="base_dataset",    node=DatasetConf)
    cs.store(group="model",      name="base_model",      node=ModelConf)
    cs.store(group="training",   name="base_training",   node=TrainingConf)
    cs.store(group="evaluation", name="base_evaluation", node=EvaluationConf)
    cs.store(
        group="experiment_hparams",
        name="favila2016",
        node=Favila2016HParams,
    )
    cs.store(
        group="experiment_hparams",
        name="chanales2017_sim1",
        node=Chanales2017Sim1HParams,
    )
    cs.store(
        group="experiment_hparams",
        name="chanales2017_sim2",
        node=Chanales2017Sim2HParams,
    )
    cs.store(
        group="experiment_hparams",
        name="chanales2021",
        node=Chanales2021HParams,
    )


_register()
