from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.config import (
    classifier_output_dir,
    clip_embeddings_path,
    clip_scores_path,
    experiment_name,
    generation_metadata_path,
    load_config,
    resolve_path,
    selected_csv_path,
)
from ccds.datasets import split_base_name
from ccds.prompts import build_prompts


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local smoke pipeline.")
    parser.add_argument("--config", default="configs/flowers20_5shot.yaml")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--real-as-generated", action="store_true", help="Use held-out real images as generated-candidate metadata.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--augmented-methods",
        nargs="+",
        default=["ccds"],
        choices=[
            "diffusion_random",
            "clip_topk",
            "margin_topk",
            "ccds",
            "anchored_ccds",
            "confusion_adaptive_ccds",
            "same_overlap_random",
            "replacement_aware_confusion_adaptive_ccds",
        ],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    py = sys.executable
    run([py, "scripts/make_splits.py", "--config", args.config])
    _check_splits(cfg, args.seed)

    run([py, "scripts/train_classifier.py", "--config", args.config, "--method", "real_only", "--seed", str(args.seed), "--epochs", str(args.epochs)])
    _check_classifier_outputs(cfg, "real_only", args.seed)

    if args.real_as_generated:
        _write_real_as_generated_metadata(cfg, args.seed)
        _check_metadata(cfg)
        run([py, "scripts/score_candidates.py", "--config", args.config])
        _check_scores(cfg)
        run([py, "scripts/select_candidates.py", "--config", args.config, "--strategy", "all"])
        _check_selected(cfg)
        for method in args.augmented_methods:
            run([py, "scripts/train_classifier.py", "--config", args.config, "--method", method, "--seed", str(args.seed), "--epochs", str(args.epochs)])
            _check_classifier_outputs(cfg, method, args.seed)
    elif not args.skip_generation:
        run([py, "scripts/generate_candidates.py", "--config", args.config, "--limit-per-class", "2"])
        _check_metadata(cfg)
        run([py, "scripts/score_candidates.py", "--config", args.config])
        _check_scores(cfg)
        run([py, "scripts/select_candidates.py", "--config", args.config, "--strategy", "all"])
        _check_selected(cfg)

    _check_summary(cfg, ["real_only", *([] if args.skip_generation and not args.real_as_generated else args.augmented_methods)], args.seed)
    print("\nSmoke pipeline completed successfully.")


def _split_base(cfg: dict, seed: int) -> str:
    return split_base_name(cfg["dataset"], seed)


def _check_splits(cfg: dict, seed: int) -> None:
    base = _split_base(cfg, seed)
    for split in ["train", "val", "test"]:
        path = resolve_path(f"data/splits/{base}_{split}.csv")
        if not path.exists():
            raise FileNotFoundError(f"Missing split CSV: {path}")


def _write_real_as_generated_metadata(cfg: dict, seed: int) -> None:
    dataset_cfg = cfg["dataset"]
    gen_cfg = cfg["generation"]
    base = _split_base(cfg, seed)
    source_csv = resolve_path(f"data/splits/{base}_val.csv")
    if not source_csv.exists():
        source_csv = resolve_path(f"data/splits/{base}_test.csv")
    df = pd.read_csv(source_csv)
    n_per_class = int(gen_cfg["num_candidates_per_class"])
    records = []
    for _, group in df.groupby("label"):
        class_name = str(group["class_name"].iloc[0])
        prompts = build_prompts(class_name, gen_cfg["prompts"])
        for i, (_, row) in enumerate(group.head(n_per_class).iterrows()):
            records.append(
                {
                    "image_path": row["image_path"],
                    "class_name": class_name,
                    "label": int(row["label"]),
                    "prompt": prompts[i % len(prompts)],
                    "seed": seed,
                    "model": "real_as_generated",
                    "guidance_scale": 0.0,
                    "num_steps": 0,
                }
            )
    expected = int(dataset_cfg["num_classes"]) * n_per_class
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} pseudo-generated records, got {len(records)} from {source_csv}")
    out_path = generation_metadata_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Wrote real-as-generated metadata to {out_path}")


def _check_metadata(cfg: dict) -> None:
    path = generation_metadata_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {path}")
    df = pd.read_csv(path)
    expected_cols = {"image_path", "class_name", "label", "prompt"}
    if not expected_cols.issubset(df.columns):
        raise RuntimeError(f"Metadata missing columns {expected_cols - set(df.columns)}: {path}")


def _check_scores(cfg: dict) -> None:
    score_path = clip_scores_path(cfg)
    embeddings_path = clip_embeddings_path(cfg)
    if not score_path.exists():
        raise FileNotFoundError(f"Missing CLIP score CSV: {score_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing CLIP embeddings NPZ: {embeddings_path}")
    df = pd.read_csv(score_path)
    required = {"image_path", "target_class", "target_label", "target_score", "confuser_score", "margin_score"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Score CSV missing columns {required - set(df.columns)}: {score_path}")


def _check_selected(cfg: dict) -> None:
    for strategy in cfg["selection"].get("strategies", []):
        path = selected_csv_path(cfg, strategy)
        if not path.exists():
            raise FileNotFoundError(f"Missing selected CSV: {path}")
        df = pd.read_csv(path)
        if df.empty:
            raise RuntimeError(f"Selected CSV is empty: {path}")
        max_per_class = int(cfg["selection"]["selected_per_class"])
        counts = df.groupby("target_class").size()
        if (counts > max_per_class).any():
            raise RuntimeError(f"Selected too many samples per class in {path}: {counts.to_dict()}")


def _check_classifier_outputs(cfg: dict, method: str, seed: int) -> None:
    out_dir = classifier_output_dir(cfg, method, seed)
    for name in ["model.pt", "metrics.json", "summary.csv"]:
        path = out_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing classifier artifact: {path}")


def _check_summary(cfg: dict, methods: list[str], seed: int) -> None:
    summary = resolve_path("results/classifier/all_results.csv")
    if not summary.exists():
        raise FileNotFoundError(f"Missing global classifier summary: {summary}")
    df = pd.read_csv(summary)
    name = experiment_name(cfg)
    for method in methods:
        rows = df[(df["project_name"] == name) & (df["method"] == method) & (df["seed"] == seed)]
        if rows.empty:
            raise RuntimeError(f"Missing summary row for project={name}, method={method}, seed={seed}")


if __name__ == "__main__":
    main()
