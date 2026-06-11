from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .flowers import SplitPaths, create_flowers102_splits
from .pets import create_oxford_iiit_pet_splits


DATASET_PREFIXES = {
    "flowers102": "flowers",
    "oxford_iiit_pets": "pets",
    "pets": "pets",
}


def dataset_prefix(dataset_cfg: dict) -> str:
    name = str(dataset_cfg.get("name", "flowers102"))
    if name not in DATASET_PREFIXES:
        raise ValueError(f"Unsupported dataset.name={name!r}. Expected one of {sorted(DATASET_PREFIXES)}.")
    return DATASET_PREFIXES[name]


def class_map_filename(dataset_cfg: dict) -> str:
    return f"{dataset_prefix(dataset_cfg)}{int(dataset_cfg['num_classes'])}_class_map.csv"


def split_base_name(dataset_cfg: dict, seed: int) -> str:
    return f"{dataset_prefix(dataset_cfg)}{int(dataset_cfg['num_classes'])}_{int(dataset_cfg['shot'])}shot_seed{seed}"


def class_map_path(dataset_cfg: dict, output_dir: str | Path = "data/splits") -> Path:
    return Path(output_dir) / class_map_filename(dataset_cfg)


def create_fewshot_splits(
    dataset_cfg: dict,
    output_dir: str | Path,
    seed_list: Iterable[int],
    download: bool = True,
) -> list[SplitPaths]:
    name = str(dataset_cfg.get("name", "flowers102"))
    kwargs = dict(
        root=dataset_cfg["root"],
        output_dir=output_dir,
        num_classes=int(dataset_cfg["num_classes"]),
        shot=int(dataset_cfg["shot"]),
        seed_list=seed_list,
        class_indices=dataset_cfg.get("class_indices") or None,
        download=download,
    )
    if name == "flowers102":
        return create_flowers102_splits(**kwargs)
    if name in {"oxford_iiit_pets", "pets"}:
        return create_oxford_iiit_pet_splits(**kwargs)
    raise ValueError(f"Unsupported dataset.name={name!r}.")
