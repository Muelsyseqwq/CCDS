from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.config import experiment_results_dir, load_config, resolve_path
from ccds.datasets import split_base_name
from ccds.utils import ensure_dir


DEFAULT_DATA_ROOT = Path("/root/gpufree-data/clip_diffusion_fewshot_ccds_lora")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a train-only image/caption manifest for SD LoRA training.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, required=True, help="Few-shot split seed. Only this seed's train CSV is used.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the manifest. Defaults to lora.output_dir or /root/gpufree-data/...",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    lora_cfg = cfg.get("lora", {})
    split_base = split_base_name(dataset_cfg, args.seed)
    train_csv = resolve_path(f"data/splits/{split_base}_train.csv")
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train split: {train_csv}. Run scripts/make_splits.py first.")

    out_dir = _output_dir(cfg, lora_cfg, args.seed, args.output_dir)
    ensure_dir(out_dir)
    manifest_csv = out_dir / "train_manifest.csv"
    metadata_json = out_dir / "metadata.jsonl"
    summary_json = out_dir / "summary.json"

    train_df = pd.read_csv(train_csv).copy()
    required = {"image_path", "label", "class_name"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Train split {train_csv} is missing required columns: {sorted(missing)}")

    caption_template = str(lora_cfg.get("caption_template") or _default_caption_template(dataset_cfg))
    records = []
    for _, row in train_df.iterrows():
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Training image not found: {image_path}")
        class_name = str(row["class_name"])
        caption = caption_template.format(class_name=class_name)
        records.append(
            {
                "image_path": str(image_path.resolve()),
                "caption": caption,
                "label": int(row["label"]),
                "class_name": class_name,
                "source_split_csv": str(train_csv),
                "split_base": split_base,
                "seed": int(args.seed),
            }
        )

    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_csv, index=False)
    with metadata_json.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps({"file_name": record["image_path"], "text": record["caption"]}, ensure_ascii=False) + "\n")

    summary = {
        "config": str(resolve_path(args.config)),
        "dataset_name": str(dataset_cfg.get("name", "flowers102")),
        "split_base": split_base,
        "seed": int(args.seed),
        "train_csv": str(train_csv),
        "num_images": len(manifest),
        "num_classes": int(manifest["label"].nunique()) if not manifest.empty else 0,
        "caption_template": caption_template,
        "manifest_csv": str(manifest_csv),
        "metadata_jsonl": str(metadata_json),
        "leakage_rule": "seed-specific train split only; no val/test images",
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote LoRA train manifest: {manifest_csv}")
    print(f"Wrote diffusers-style metadata: {metadata_json}")
    print(f"Wrote summary: {summary_json}")


def _output_dir(cfg: dict, lora_cfg: dict, seed: int, override: str | None) -> Path:
    if override:
        return resolve_path(str(override).format(seed=seed))
    configured = lora_cfg.get("output_dir")
    if configured:
        return resolve_path(str(configured).format(seed=seed))
    return DEFAULT_DATA_ROOT / str(cfg.get("project_name", "default_experiment")) / "lora" / f"seed{seed}"


def _default_caption_template(dataset_cfg: dict) -> str:
    name = str(dataset_cfg.get("name", "flowers102"))
    if name == "flowers102":
        return "a photo of a {class_name} flower"
    if name in {"oxford_iiit_pets", "pets"}:
        return "a photo of a {class_name} pet"
    return "a photo of a {class_name}"


if __name__ == "__main__":
    main()
