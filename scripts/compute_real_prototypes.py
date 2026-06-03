from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.config import experiment_results_dir, load_config, resolve_path
from ccds.datasets import split_base_name
from ccds.prototypes import compute_real_prototypes


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CLIP image prototypes from real few-shot training images.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, required=True, help="Few-shot split seed.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    clip_cfg = cfg["clip"]
    split_base = split_base_name(dataset_cfg, args.seed)
    train_csv = resolve_path(f"data/splits/{split_base}_train.csv")
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing split file: {train_csv}. Run scripts/make_splits.py first.")

    configured = cfg.get("selection", {}).get("prototype_npz")
    if configured:
        out_npz = resolve_path(str(configured).format(seed=args.seed))
    else:
        out_npz = experiment_results_dir(cfg) / "prototypes" / f"real_prototypes_seed{args.seed}.npz"

    compute_real_prototypes(
        train_csv=train_csv,
        out_npz=out_npz,
        model_name=str(clip_cfg.get("model_name", "ViT-B-32")),
        pretrained=str(clip_cfg.get("pretrained", "openai")),
        batch_size=int(clip_cfg.get("batch_size", 32)),
    )
    print(f"Wrote prototypes to {out_npz}")


if __name__ == "__main__":
    main()
