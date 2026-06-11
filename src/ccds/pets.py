from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import pandas as pd
from torchvision.datasets import OxfordIIITPet

from .flowers import SplitPaths
from .config import resolve_path
from .utils import ensure_dir


def create_oxford_iiit_pet_splits(
    root: str | Path,
    output_dir: str | Path,
    num_classes: int,
    shot: int,
    seed_list: Iterable[int],
    class_indices: list[int] | None = None,
    download: bool = True,
) -> list[SplitPaths]:
    """Create class-subset few-shot CSV splits for Oxford-IIIT Pets."""
    root = resolve_path(root)
    output_dir = ensure_dir(resolve_path(output_dir))

    trainval_set = OxfordIIITPet(root=str(root), split="trainval", target_types="category", download=download)
    test_set = OxfordIIITPet(root=str(root), split="test", target_types="category", download=download)

    if class_indices:
        selected_original = [int(i) for i in class_indices]
    else:
        selected_original = list(range(num_classes))

    class_records = []
    for new_label, original_idx in enumerate(selected_original):
        class_records.append(
            {
                "label": new_label,
                "class_index": original_idx,
                "class_name": _normalize_pet_class_name(trainval_set.classes[original_idx]),
            }
        )
    class_map = pd.DataFrame(class_records)
    class_map_path = output_dir / f"pets{num_classes}_class_map.csv"
    class_map.to_csv(class_map_path, index=False)

    trainval_records_all = _records_from_dataset(trainval_set, "trainval", selected_original)
    test_records = _records_from_dataset(test_set, "test", selected_original)

    paths: list[SplitPaths] = []
    for seed in seed_list:
        rng = random.Random(int(seed))
        train_records = []
        val_records = []
        for original_idx in selected_original:
            candidates = [r for r in trainval_records_all if r["class_index"] == original_idx]
            if len(candidates) < shot + 1:
                raise ValueError(f"Class {original_idx} has only {len(candidates)} trainval images, need at least shot+1={shot + 1}.")
            shuffled = candidates[:]
            rng.shuffle(shuffled)
            train_records.extend(shuffled[:shot])
            val_take = min(max(10, shot), len(shuffled) - shot)
            val_records.extend(shuffled[shot : shot + val_take])

        name = f"pets{num_classes}_{shot}shot_seed{seed}"
        train_csv = output_dir / f"{name}_train.csv"
        val_csv = output_dir / f"{name}_val.csv"
        test_csv = output_dir / f"{name}_test.csv"

        pd.DataFrame(train_records).to_csv(train_csv, index=False)
        pd.DataFrame(val_records).to_csv(val_csv, index=False)
        pd.DataFrame(test_records).to_csv(test_csv, index=False)
        paths.append(SplitPaths(train_csv, val_csv, test_csv, class_map_path))

    return paths


def _records_from_dataset(dataset: OxfordIIITPet, split_name: str, selected_original: list[int]) -> list[dict]:
    original_to_label = {original: i for i, original in enumerate(selected_original)}
    selected_set = set(selected_original)
    records = []
    for image_path, target in zip(dataset._images, dataset._labels):
        original_idx = int(target)
        if original_idx not in selected_set:
            continue
        records.append(
            {
                "image_path": str(Path(image_path).resolve()),
                "label": original_to_label[original_idx],
                "class_index": original_idx,
                "class_name": _normalize_pet_class_name(dataset.classes[original_idx]),
                "original_split": split_name,
            }
        )
    return records


def _normalize_pet_class_name(class_name: str) -> str:
    return class_name.replace("_", " ").lower()
