from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def build_transforms(image_size: int, train: bool, traditional_aug: bool = False):
    if train and traditional_aug:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class CsvImageDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform=None, return_sample_weight: bool = False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.return_sample_weight = return_sample_weight

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        if self.return_sample_weight:
            weight = float(row["sample_weight"]) if "sample_weight" in self.df.columns else 1.0
            return image, label, torch.tensor(weight, dtype=torch.float32)
        return image, label


def merge_real_and_generated(
    real_csv: str | Path,
    selected_csv: str | Path,
    out_csv: str | Path,
    synthetic_weight: float = 1.0,
    real_weight: float = 1.0,
) -> Path:
    real = pd.read_csv(real_csv).copy()
    gen = pd.read_csv(selected_csv).copy()

    real["source"] = real.get("source", "real")
    real["sample_weight"] = float(real_weight)
    label_meta = real[["label", "class_index", "class_name"]].drop_duplicates("label")

    if "label" not in gen.columns:
        if "target_label" in gen.columns:
            gen["label"] = gen["target_label"]
        elif "target_class" in gen.columns:
            gen = gen.merge(label_meta[["label", "class_name"]], left_on="target_class", right_on="class_name", how="left")
        else:
            raise ValueError("Selected CSV must contain label, target_label, or target_class.")

    if "class_name" not in gen.columns:
        if "target_class" in gen.columns:
            gen["class_name"] = gen["target_class"]
        else:
            gen = gen.merge(label_meta[["label", "class_name"]], on="label", how="left")

    gen = gen.merge(label_meta[["label", "class_index"]], on="label", how="left", suffixes=("", "_from_real"))
    if "class_index_from_real" in gen.columns:
        if "class_index" not in gen.columns:
            gen["class_index"] = gen["class_index_from_real"]
        else:
            gen["class_index"] = gen["class_index"].fillna(gen["class_index_from_real"])
        gen.drop(columns=["class_index_from_real"], inplace=True)

    gen_records = gen[["image_path", "label", "class_index", "class_name"]].copy()
    gen_records["original_split"] = "generated"
    gen_records["source"] = "generated"
    gen_records["sample_weight"] = gen["sample_weight"] if "sample_weight" in gen.columns else float(synthetic_weight)
    if "selection_strategy" in gen.columns:
        gen_records["selection_strategy"] = gen["selection_strategy"]

    merged = pd.concat([real, gen_records], ignore_index=True, sort=False)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return out_csv
