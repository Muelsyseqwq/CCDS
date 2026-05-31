from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .prompts import build_prompts
from .utils import ensure_dir, get_device


@torch.no_grad()
def score_candidates(
    metadata_csv: str | Path,
    class_map_csv: str | Path,
    prompt_templates: list[str],
    out_csv: str | Path,
    embeddings_npz: str | Path,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    batch_size: int = 32,
) -> None:
    device = get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    class_map = pd.read_csv(class_map_csv)
    class_names = class_map["class_name"].tolist()
    text_features = []
    for class_name in class_names:
        prompts = build_prompts(class_name, prompt_templates)
        tokens = tokenizer(prompts).to(device)
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feat = feat.mean(dim=0, keepdim=True)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        text_features.append(feat)
    text_features = torch.cat(text_features, dim=0)

    metadata = pd.read_csv(metadata_csv)
    rows = []
    embedding_dict = {}
    for start in tqdm(range(0, len(metadata), batch_size), desc="CLIP scoring"):
        batch_df = metadata.iloc[start : start + batch_size]
        images = [preprocess(Image.open(p).convert("RGB")) for p in batch_df["image_path"].tolist()]
        image_tensor = torch.stack(images).to(device)
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sims = image_features @ text_features.T

        for i, (_, row) in enumerate(batch_df.iterrows()):
            target_class = row["class_name"]
            target_label = int(class_map.loc[class_map["class_name"] == target_class, "label"].iloc[0])
            sim_vec = sims[i].detach().cpu().float().numpy()
            target_score = float(sim_vec[target_label])
            confuser_scores = sim_vec.copy()
            confuser_scores[target_label] = -1e9
            confuser_label = int(confuser_scores.argmax())
            confuser_score = float(sim_vec[confuser_label])
            margin_score = target_score - confuser_score
            image_path = str(row["image_path"])
            embedding_dict[image_path] = image_features[i].detach().cpu().float().numpy()
            rows.append(
                {
                    "image_path": image_path,
                    "target_class": target_class,
                    "target_label": target_label,
                    "target_score": target_score,
                    "max_confuser_class": class_names[confuser_label],
                    "confuser_label": confuser_label,
                    "confuser_score": confuser_score,
                    "margin_score": margin_score,
                    "prompt": row.get("prompt", ""),
                    "seed": row.get("seed", ""),
                }
            )

    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    embeddings_npz = Path(embeddings_npz)
    ensure_dir(embeddings_npz.parent)
    np.savez_compressed(embeddings_npz, **{_safe_key(k): v for k, v in embedding_dict.items()})


def _safe_key(path: str) -> str:
    return path.replace(":", "__COLON__").replace("\\", "__BS__").replace("/", "__FS__").replace(".", "__DOT__")


def restore_key(key: str) -> str:
    return key.replace("__DOT__", ".").replace("__FS__", "/").replace("__BS__", "\\").replace("__COLON__", ":")


def load_embeddings_npz(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {restore_key(k): data[k] for k in data.files}
