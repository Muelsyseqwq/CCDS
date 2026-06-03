from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

from .data import build_transforms
from .utils import ensure_dir, get_device


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image)


class PilImagePathDataset(Dataset):
    def __init__(self, paths: list[str]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        return Image.open(self.paths[index]).convert("RGB")


@torch.no_grad()
def compute_or_load_resnet50_real_candidate_features(
    train_csv: str | Path,
    scores: pd.DataFrame,
    out_npz: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    force: bool = False,
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    """Compute or load ResNet-50 penultimate features for real few-shot and candidate images."""
    out_npz = Path(out_npz)
    if out_npz.exists() and not force:
        return load_real_candidate_features_npz(out_npz)

    train = pd.read_csv(train_csv)
    real_paths = train["image_path"].astype(str).tolist()
    candidate_paths = scores["image_path"].astype(str).drop_duplicates().tolist()
    paths = real_paths + candidate_paths
    features = _extract_resnet50_features(paths, image_size=image_size, batch_size=batch_size, num_workers=num_workers)

    _write_real_candidate_features_npz(out_npz, train, candidate_paths, features)
    return load_real_candidate_features_npz(out_npz)


@torch.no_grad()
def compute_or_load_dinov2_real_candidate_features(
    train_csv: str | Path,
    scores: pd.DataFrame,
    out_npz: str | Path,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
    model_name: str = "facebook/dinov2-base",
    force: bool = False,
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    """Compute or load DINOv2 features for real few-shot and candidate images."""
    out_npz = Path(out_npz)
    if out_npz.exists() and not force:
        return load_real_candidate_features_npz(out_npz)

    train = pd.read_csv(train_csv)
    real_paths = train["image_path"].astype(str).tolist()
    candidate_paths = scores["image_path"].astype(str).drop_duplicates().tolist()
    paths = real_paths + candidate_paths
    features = _extract_dinov2_features(
        paths,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        model_name=model_name,
    )

    _write_real_candidate_features_npz(out_npz, train, candidate_paths, features)
    return load_real_candidate_features_npz(out_npz)


def _write_real_candidate_features_npz(
    out_npz: str | Path,
    train: pd.DataFrame,
    candidate_paths: list[str],
    features: np.ndarray,
) -> None:
    real_features_by_label: dict[int, list[np.ndarray]] = {}
    for idx, (_, row) in enumerate(train.iterrows()):
        label = int(row["label"])
        real_features_by_label.setdefault(label, []).append(features[idx])

    candidate_offset = len(train)
    candidate_features = features[candidate_offset:]
    payload: dict[str, np.ndarray] = {
        "candidate_paths": np.asarray(candidate_paths),
        "candidate_features": candidate_features.astype(np.float32),
    }
    for label, vectors in real_features_by_label.items():
        payload[f"real_label_{label}"] = np.stack(vectors).astype(np.float32)

    out_npz = Path(out_npz)
    ensure_dir(out_npz.parent)
    np.savez_compressed(out_npz, **payload)


def load_real_candidate_features_npz(path: str | Path) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    candidate_paths = [str(path) for path in data["candidate_paths"].tolist()]
    candidate_matrix = data["candidate_features"].astype(np.float32)
    candidate_features = {path: candidate_matrix[idx] for idx, path in enumerate(candidate_paths)}
    real_features = {
        int(key.replace("real_label_", "")): data[key].astype(np.float32)
        for key in data.files
        if key.startswith("real_label_")
    }
    return real_features, candidate_features


@torch.no_grad()
def _extract_resnet50_features(
    paths: list[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    device = get_device()
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    dataset = ImagePathDataset(paths, build_transforms(image_size, train=False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    chunks: list[np.ndarray] = []
    for images in tqdm(loader, desc="resnet50 features", leave=False):
        features = model(images.to(device))
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        chunks.append(features.detach().cpu().float().numpy())
    if not chunks:
        return np.empty((0, 2048), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


@torch.no_grad()
def _extract_dinov2_features(
    paths: list[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    model_name: str,
) -> np.ndarray:
    if not paths:
        return np.empty((0, 768), dtype=np.float32)

    from transformers import AutoImageProcessor, AutoModel

    device = get_device()
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    dataset = PilImagePathDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: batch,
    )
    chunks: list[np.ndarray] = []
    for images in tqdm(loader, desc="dinov2 features", leave=False):
        inputs = processor(images=images, return_tensors="pt", size={"height": image_size, "width": image_size})
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        features = getattr(outputs, "pooler_output", None)
        if features is None:
            features = outputs.last_hidden_state[:, 0]
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        chunks.append(features.detach().cpu().float().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)
