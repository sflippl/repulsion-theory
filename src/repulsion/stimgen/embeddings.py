"""Image embeddings from ImageNet-pretrained torchvision models.

Used by the ``"image"`` stimulus type (see :mod:`repulsion.stimgen.image_items`).
Each item is the embedding of exactly one image. Images are chosen from the
class labels alone, and only the chosen images are run through the network;
their embeddings are cached per (dataset, split, model, weights, layer, pool)
in ``<data_dir>/embeddings/`` so repeated runs do not recompute them.

Datasets are downloaded once to ``<data_dir>/<dataset>/``. ``<data_dir>`` is
``ImageSource.root`` if given, else ``$REPULSION_DATA_DIR``, else the
repository's ``data/`` directory.

torch/torchvision are imported lazily, only when images are loaded or embedded.
"""
from __future__ import annotations

import functools
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Channels of each ResNet stage; "avgpool" is the pooled penultimate layer, "fc"
# the 1000-way ImageNet logits.
_RESNET_BASIC = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512, "avgpool": 512, "fc": 1000}
_RESNET_BOTTLENECK = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048, "avgpool": 2048, "fc": 1000}
MODEL_LAYER_DIMS = {
    "resnet18": _RESNET_BASIC,
    "resnet34": _RESNET_BASIC,
    "resnet50": _RESNET_BOTTLENECK,
    "resnet101": _RESNET_BOTTLENECK,
    "resnet152": _RESNET_BOTTLENECK,
}
# Spatial size of each stage's feature map at the 224 x 224 network input.
_SPATIAL_SIZE = {"layer1": 56, "layer2": 28, "layer3": 14, "layer4": 7}
# torchvision graph node for each layer name ("avgpool" -> its flattened output).
_LAYER_NODES = {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3", "layer4": "layer4",
                "avgpool": "flatten", "fc": "fc"}
POOLS = ("avg", "flatten")
DATASETS = ("cifar100",)
SPLITS = ("train", "test")


@dataclass(frozen=True)
class ImageSource:
    """Where an image item type's vectors come from."""

    dataset: str = "cifar100"
    split: str = "train"
    model: str = "resnet18"
    weights: str = "DEFAULT"  # torchvision weights enum name, e.g. "IMAGENET1K_V1"
    layer: str = "avgpool"    # layer1..layer4, avgpool, fc
    pool: str = "avg"         # spatial layers: "avg" (global average) or "flatten"
    root: str | None = None   # data directory; see module docstring


def parse_source(d: dict | None, item_name: str) -> ImageSource:
    """Validate a ``source`` dict from an item spec (missing keys get defaults)."""
    d = dict(d or {})
    unknown = set(d) - set(ImageSource.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Item '{item_name}': unknown source keys {sorted(unknown)}.")
    source = ImageSource(**{k: (str(v) if v is not None else None) for k, v in d.items()})
    if source.dataset not in DATASETS:
        raise ValueError(f"Item '{item_name}': source.dataset must be one of {DATASETS}, got {source.dataset!r}.")
    if source.split not in SPLITS:
        raise ValueError(f"Item '{item_name}': source.split must be one of {SPLITS}, got {source.split!r}.")
    if source.model not in MODEL_LAYER_DIMS:
        raise ValueError(f"Item '{item_name}': source.model must be one of {sorted(MODEL_LAYER_DIMS)}, "
                         f"got {source.model!r}.")
    if source.layer not in _LAYER_NODES:
        raise ValueError(f"Item '{item_name}': source.layer must be one of {list(_LAYER_NODES)}, "
                         f"got {source.layer!r}.")
    if source.pool not in POOLS:
        raise ValueError(f"Item '{item_name}': source.pool must be one of {POOLS}, got {source.pool!r}.")
    return source


def layer_dim(source: ImageSource) -> int:
    """Dimensionality of the item vectors for *source* (no torch import needed)."""
    channels = MODEL_LAYER_DIMS[source.model][source.layer]
    if source.layer in _SPATIAL_SIZE and source.pool == "flatten":
        return channels * _SPATIAL_SIZE[source.layer] ** 2
    return channels


def data_dir(source: ImageSource) -> Path:
    if source.root is not None:
        return Path(source.root)
    if os.environ.get("REPULSION_DATA_DIR"):
        return Path(os.environ["REPULSION_DATA_DIR"])
    return Path(__file__).resolve().parents[3] / "data"


# ---------------------------------------------------------------------------
# Image bank: labels for sampling, embeddings on demand
# ---------------------------------------------------------------------------

class ImageBank:
    """The images and labels of one dataset split, with cached embeddings.

    Attributes:
        labels:      (N,) integer class labels.
        class_names: Class names, indexed by label.
    """

    def __init__(self, source: ImageSource, images: np.ndarray, labels: np.ndarray, class_names: list[str]):
        self.source = source
        self.images = images          # (N, H, W, 3) uint8
        self.labels = labels
        self.class_names = class_names

    def embed(self, indices) -> np.ndarray:
        """Embeddings (len(indices), dim) of the given images, computing only uncached ones."""
        indices = np.asarray(indices, dtype=np.int64)
        cache = _cache_path(self.source)
        cached = _read_cache(cache)
        missing = sorted({int(i) for i in indices} - set(cached))
        if missing:
            features = _run_model(self.source, self.images[missing])
            new = dict(zip(missing, features))
            _write_cache(cache, {**_read_cache(cache), **new})  # re-read: another run may have added entries
            cached.update(new)
        return np.stack([cached[int(i)] for i in indices]).astype(np.float64)


@functools.lru_cache(maxsize=None)
def get_bank(source: ImageSource) -> ImageBank:
    """Load (downloading if needed) the dataset split of *source*."""
    if source.dataset == "cifar100":
        from torchvision.datasets import CIFAR100

        root = data_dir(source) / "cifar100"
        try:
            ds = CIFAR100(str(root), train=source.split == "train", download=False)
        except RuntimeError:  # not downloaded yet
            ds = CIFAR100(str(root), train=source.split == "train", download=True)
        return ImageBank(source, ds.data, np.asarray(ds.targets), list(ds.classes))
    raise ValueError(f"Unknown dataset {source.dataset!r}.")


# ---------------------------------------------------------------------------
# Model and cache
# ---------------------------------------------------------------------------

def _resolved_weights(source: ImageSource):
    from torchvision.models import get_model_weights

    return get_model_weights(source.model)[source.weights]


def _cache_path(source: ImageSource) -> Path:
    weights = _resolved_weights(source).name  # e.g. DEFAULT -> IMAGENET1K_V1, so caches never mix
    pool = f"_{source.pool}" if source.layer in _SPATIAL_SIZE else ""
    name = f"{source.dataset}_{source.split}_{source.model}_{weights}_{source.layer}{pool}.npz"
    return data_dir(source) / "embeddings" / name


def _read_cache(path: Path) -> dict[int, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path) as f:
        return dict(zip(f["indices"].tolist(), f["features"]))


def _write_cache(path: Path, entries: dict[int, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(entries)
    # write-then-rename, so concurrent runs never see a half-written file
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    with os.fdopen(fd, "wb") as f:
        np.savez(f, indices=np.array(keys, dtype=np.int64), features=np.stack([entries[k] for k in keys]))
    os.chmod(tmp, 0o644)  # mkstemp creates the file readable by its owner only
    os.replace(tmp, path)


@functools.lru_cache(maxsize=None)
def _feature_extractor(model_name: str, weights_name: str, layer: str):
    import torch
    from torchvision.models import get_model, get_model_weights
    from torchvision.models.feature_extraction import create_feature_extractor

    weights = get_model_weights(model_name)[weights_name]
    model = get_model(model_name, weights=weights).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = create_feature_extractor(model, return_nodes={_LAYER_NODES[layer]: "out"}).to(device)
    return extractor, weights.transforms(), device


def _run_model(source: ImageSource, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Features (len(images), dim) of uint8 HWC images, preprocessed as for the weights."""
    import torch

    extractor, transform, device = _feature_extractor(source.model, _resolved_weights(source).name, source.layer)
    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(images[start:start + batch_size]).permute(0, 3, 1, 2)
            h = extractor(transform(x).to(device))["out"]
            if h.ndim == 4:
                h = h.mean(dim=(2, 3)) if source.pool == "avg" else h.flatten(1)
            out.append(h.cpu().numpy().astype(np.float32))
    features = np.concatenate(out)
    if features.shape[1] != layer_dim(source):
        raise RuntimeError(f"Expected {layer_dim(source)}-d features from {source}, got {features.shape[1]}.")
    return features
