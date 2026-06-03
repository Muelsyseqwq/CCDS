from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.config import experiment_name, load_config, resolve_path
from ccds.datasets import class_map_path, split_base_name
from ccds.metrics import classification_metrics
from ccds.prompts import build_prompts
from ccds.utils import ensure_dir, get_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an OpenCLIP zero-shot baseline on a configured split.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Run one seed only. Defaults to all seeds in config.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to evaluate.")
    parser.add_argument("--output-dir", default="results/clip_zeroshot", help="Output directory for zero-shot results.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override CLIP batch size.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing zero-shot outputs for matching dataset/seed.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    clip_cfg = cfg["clip"]
    gen_cfg = cfg["generation"]
    project_name = experiment_name(cfg)
    seeds = [args.seed] if args.seed is not None else [int(seed) for seed in cfg.get("seed_list", [0])]
    batch_size = args.batch_size or int(clip_cfg.get("batch_size", 32))

    prompt_templates = clip_cfg.get("prompts") or gen_cfg["prompts"]
    _validate_prompt_templates(prompt_templates)
    prompt_source = "clip.prompts" if clip_cfg.get("prompts") else "generation.prompts"
    print(f"Using CLIP prompt templates from {prompt_source} ({len(prompt_templates)} templates)")

    output_root = resolve_path(args.output_dir)
    class_map_csv = resolve_path(class_map_path(dataset_cfg))
    class_map = _load_class_map(class_map_csv)

    device = get_device()
    model_name = str(clip_cfg.get("model_name", "ViT-B-32"))
    pretrained = str(clip_cfg.get("pretrained", "openai"))
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    text_features = _build_text_features(model, tokenizer, class_map["class_name"].tolist(), prompt_templates, device)
    summaries = []
    for seed in seeds:
        summary = evaluate_seed(
            cfg=cfg,
            config_path=args.config,
            class_map=class_map,
            text_features=text_features,
            model=model,
            preprocess=preprocess,
            seed=int(seed),
            split=args.split,
            batch_size=batch_size,
            output_root=output_root,
            prompt_source=prompt_source,
            prompt_templates=prompt_templates,
            model_name=model_name,
            pretrained=pretrained,
            overwrite=args.overwrite,
        )
        summaries.append(summary)

    update_global_summary(output_root / "summary.csv", pd.DataFrame(summaries), overwrite=args.overwrite)
    print(f"Updated zero-shot summary: {output_root / 'summary.csv'}")


@torch.no_grad()
def evaluate_seed(
    cfg: dict,
    config_path: str,
    class_map: pd.DataFrame,
    text_features: torch.Tensor,
    model,
    preprocess,
    seed: int,
    split: str,
    batch_size: int,
    output_root: Path,
    prompt_source: str,
    prompt_templates: list[str],
    model_name: str,
    pretrained: str,
    overwrite: bool,
) -> dict:
    dataset_cfg = cfg["dataset"]
    project_name = experiment_name(cfg)
    split_base = split_base_name(dataset_cfg, seed)
    split_csv = resolve_path(f"data/splits/{split_base}_{split}.csv")
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing split CSV: {split_csv}. Run scripts/make_splits.py first.")

    out_dir = output_root / project_name / f"seed{seed}"
    predictions_csv = out_dir / "predictions.csv"
    per_class_csv = out_dir / "per_class_accuracy.csv"
    summary_csv = out_dir / "summary.csv"
    existing = [path for path in [predictions_csv, per_class_csv, summary_csv] if path.exists()]
    if existing and not overwrite:
        existing_list = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Zero-shot outputs already exist. Use --overwrite to refresh:\n{existing_list}")
    ensure_dir(out_dir)

    split_df = pd.read_csv(split_csv)
    rows = []
    y_true: list[int] = []
    y_pred: list[int] = []
    class_names = class_map["class_name"].tolist()
    label_to_class_index = dict(zip(class_map["label"].astype(int), class_map["class_index"].astype(int)))
    label_to_class_name = dict(zip(class_map["label"].astype(int), class_map["class_name"].astype(str)))
    device = text_features.device

    for start in tqdm(range(0, len(split_df), batch_size), desc=f"CLIP zero-shot {project_name} seed={seed} {split}"):
        batch_df = split_df.iloc[start : start + batch_size]
        images = [preprocess(Image.open(path).convert("RGB")) for path in batch_df["image_path"].tolist()]
        image_tensor = torch.stack(images).to(device)
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sims = image_features @ text_features.T
        top_scores, top_labels = sims.max(dim=1)
        top2_scores, _ = sims.topk(k=min(2, sims.shape[1]), dim=1)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            label = int(row["label"])
            pred_label = int(top_labels[i].item())
            sim_vec = sims[i].detach().cpu().float()
            true_score = float(sim_vec[label].item())
            top1_score = float(top_scores[i].item())
            second_score = float(top2_scores[i, 1].item()) if sims.shape[1] > 1 else float("nan")
            y_true.append(label)
            y_pred.append(pred_label)
            rows.append(
                {
                    "project_name": project_name,
                    "seed": seed,
                    "split": split,
                    "image_path": str(row["image_path"]),
                    "label": label,
                    "class_index": int(row.get("class_index", label_to_class_index[label])),
                    "class_name": str(row.get("class_name", label_to_class_name[label])),
                    "pred_label": pred_label,
                    "pred_class_name": class_names[pred_label],
                    "correct": pred_label == label,
                    "top1_score": top1_score,
                    "true_score": true_score,
                    "second_score": second_score,
                    "margin_score": top1_score - second_score if sims.shape[1] > 1 else float("nan"),
                }
            )

    metrics = classification_metrics(y_true, y_pred, int(cfg["dataset"]["num_classes"]))
    predictions = pd.DataFrame(rows)
    predictions.to_csv(predictions_csv, index=False)

    per_class = build_per_class_accuracy(class_map, metrics["confusion_matrix"])
    per_class.to_csv(per_class_csv, index=False)

    summary = {
        "project_name": project_name,
        "config_path": config_path,
        "dataset_name": str(dataset_cfg.get("name", "flowers102")),
        "num_classes": int(dataset_cfg["num_classes"]),
        "shot": int(dataset_cfg["shot"]),
        "seed": seed,
        "split": split,
        "test_size": len(split_df),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_accuracy": json.dumps(metrics["per_class_accuracy"], ensure_ascii=False),
        "confusion_matrix": json.dumps(metrics["confusion_matrix"], ensure_ascii=False),
        "model_name": model_name,
        "pretrained": pretrained,
        "prompt_source": prompt_source,
        "num_prompt_templates": len(prompt_templates),
        "predictions_csv": str(predictions_csv),
        "per_class_accuracy_csv": str(per_class_csv),
    }
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    print(
        f"{project_name} seed={seed} {split}: "
        f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, n={len(split_df)}"
    )
    return summary


@torch.no_grad()
def _build_text_features(model, tokenizer, class_names: list[str], prompt_templates: list[str], device: torch.device) -> torch.Tensor:
    text_features = []
    for class_name in class_names:
        prompts = build_prompts(class_name, prompt_templates)
        tokens = tokenizer(prompts).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.mean(dim=0, keepdim=True)
        features = features / features.norm(dim=-1, keepdim=True)
        text_features.append(features)
    return torch.cat(text_features, dim=0)


def _load_class_map(path: Path) -> pd.DataFrame:
    class_map = pd.read_csv(path).copy()
    required = {"label", "class_index", "class_name"}
    missing = required - set(class_map.columns)
    if missing:
        raise ValueError(f"Class map {path} is missing required columns: {sorted(missing)}")
    class_map["label"] = class_map["label"].astype(int)
    class_map["class_index"] = class_map["class_index"].astype(int)
    class_map.sort_values("label", inplace=True)
    expected = list(range(len(class_map)))
    actual = class_map["label"].tolist()
    if actual != expected:
        raise ValueError(f"Class map labels must be contiguous 0..N-1. Got {actual}")
    return class_map.reset_index(drop=True)


def build_per_class_accuracy(class_map: pd.DataFrame, confusion: list[list[int]]) -> pd.DataFrame:
    rows = []
    for _, row in class_map.iterrows():
        label = int(row["label"])
        class_total = int(sum(confusion[label]))
        class_correct = int(confusion[label][label])
        rows.append(
            {
                "label": label,
                "class_index": int(row["class_index"]),
                "class_name": str(row["class_name"]),
                "num_test": class_total,
                "num_correct": class_correct,
                "accuracy": class_correct / class_total if class_total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def update_global_summary(path: Path, new_rows: pd.DataFrame, overwrite: bool) -> None:
    ensure_dir(path.parent)
    if path.exists():
        old = pd.read_csv(path)
        if overwrite:
            keys = ["project_name", "seed", "split", "model_name", "pretrained"]
            old_keys = old[keys].astype(str).agg("|".join, axis=1)
            new_keys = new_rows[keys].astype(str).agg("|".join, axis=1)
            old = old.loc[~old_keys.isin(set(new_keys))]
        combined = pd.concat([old, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.drop_duplicates(subset=["project_name", "seed", "split", "model_name", "pretrained"], keep="last", inplace=True)
    combined.sort_values(["project_name", "seed", "split"], inplace=True)
    combined.to_csv(path, index=False)


def _validate_prompt_templates(prompt_templates: list[str]) -> None:
    if not isinstance(prompt_templates, list) or not prompt_templates:
        raise ValueError("CLIP prompt templates must be a non-empty list.")
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompt_templates):
        raise ValueError("Every CLIP prompt template must be a non-empty string.")


if __name__ == "__main__":
    main()
