from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from torchvision.datasets import Flowers102

from .config import resolve_path
from .utils import ensure_dir


@dataclass(frozen=True)
class SplitPaths:
    train_csv: Path
    val_csv: Path
    test_csv: Path
    class_map_csv: Path


DEFAULT_FLOWER_CLASS_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle",
    "snapdragon", "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily",
    "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
    "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip", "lenten rose",
    "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue",
    "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion",
    "petunia", "wild pansy", "primula", "sunflower", "pelargonium",
    "bishop of llandaff", "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
    "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush", "californian poppy",
    "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus", "toad lily", "anthurium",
    "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
    "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea",
    "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily",
]


def create_flowers102_splits(
    root: str | Path,
    output_dir: str | Path,
    num_classes: int,
    shot: int,
    seed_list: Iterable[int],
    class_indices: list[int] | None = None,
    download: bool = True,
) -> list[SplitPaths]:
    """Create class-subset few-shot CSV splits for Flowers102.

    CSV columns: image_path, label, class_index, class_name, original_split.
    Labels are remapped to 0..num_classes-1 for the selected subset.
    """
    root = resolve_path(root)
    output_dir = ensure_dir(resolve_path(output_dir))

    train_set = Flowers102(root=str(root), split="train", download=download)
    val_set = Flowers102(root=str(root), split="val", download=download)
    test_set = Flowers102(root=str(root), split="test", download=download)

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
                "class_name": DEFAULT_FLOWER_CLASS_NAMES[original_idx],
            }
        )
    class_map = pd.DataFrame(class_records)
    class_map_path = output_dir / f"flowers{num_classes}_class_map.csv"
    class_map.to_csv(class_map_path, index=False)

    train_records_all = _records_from_dataset(train_set, "train", selected_original)
    val_records = _records_from_dataset(val_set, "val", selected_original)
    test_records = _records_from_dataset(test_set, "test", selected_original)

    paths: list[SplitPaths] = []
    for seed in seed_list:
        rng = random.Random(int(seed))
        shot_records = []
        for original_idx in selected_original:
            candidates = [r for r in train_records_all if r["class_index"] == original_idx]
            if len(candidates) < shot:
                raise ValueError(f"Class {original_idx} has only {len(candidates)} train images, need shot={shot}.")
            shot_records.extend(rng.sample(candidates, shot))

        name = f"flowers{num_classes}_{shot}shot_seed{seed}"
        train_csv = output_dir / f"{name}_train.csv"
        val_csv = output_dir / f"{name}_val.csv"
        test_csv = output_dir / f"{name}_test.csv"

        pd.DataFrame(shot_records).to_csv(train_csv, index=False)
        pd.DataFrame(val_records).to_csv(val_csv, index=False)
        pd.DataFrame(test_records).to_csv(test_csv, index=False)
        paths.append(SplitPaths(train_csv, val_csv, test_csv, class_map_path))

    return paths


def _records_from_dataset(dataset: Flowers102, split_name: str, selected_original: list[int]) -> list[dict]:
    original_to_label = {original: i for i, original in enumerate(selected_original)}
    selected_set = set(selected_original)
    records = []
    for image_path, target in zip(dataset._image_files, dataset._labels):
        original_idx = int(target)
        if original_idx not in selected_set:
            continue
        records.append(
            {
                "image_path": str(Path(image_path).resolve()),
                "label": original_to_label[original_idx],
                "class_index": original_idx,
                "class_name": DEFAULT_FLOWER_CLASS_NAMES[original_idx],
                "original_split": split_name,
            }
        )
    return records
