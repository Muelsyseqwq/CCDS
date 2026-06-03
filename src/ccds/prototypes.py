from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .utils import ensure_dir, get_device


@torch.no_grad()
def compute_real_prototypes(
    train_csv: str | Path,
    out_npz: str | Path,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 32,
) -> None:
    """Compute one normalized CLIP image prototype per class from a real few-shot train CSV."""
    device = get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()

    train = pd.read_csv(train_csv)
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}

    for start in tqdm(range(0, len(train), batch_size), desc="real prototypes"):
        batch_df = train.iloc[start : start + batch_size]
        images = [preprocess(Image.open(p).convert("RGB")) for p in batch_df["image_path"].tolist()]
        image_tensor = torch.stack(images).to(device)
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            label = int(row["label"])
            vector = features[i].detach().cpu().float().numpy()
            sums[label] = vector if label not in sums else sums[label] + vector
            counts[label] = counts.get(label, 0) + 1

    prototypes = {}
    for label, total in sums.items():
        prototype = total / max(counts[label], 1)
        norm = np.linalg.norm(prototype)
        if norm > 0:
            prototype = prototype / norm
        prototypes[f"label_{label}"] = prototype.astype(np.float32)

    out_npz = Path(out_npz)
    ensure_dir(out_npz.parent)
    np.savez_compressed(out_npz, **prototypes)


def load_prototypes_npz(path: str | Path) -> dict[int, np.ndarray]:
    data = np.load(path)
    return {int(k.replace("label_", "")): data[k] for k in data.files}
