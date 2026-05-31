from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.config import load_config
from ccds.flowers import create_flowers102_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create few-shot split files for Flowers102.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--no-download", action="store_true", help="Do not download dataset automatically.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    paths = create_flowers102_splits(
        root=dataset_cfg["root"],
        output_dir="data/splits",
        num_classes=int(dataset_cfg["num_classes"]),
        shot=int(dataset_cfg["shot"]),
        seed_list=cfg["seed_list"],
        class_indices=dataset_cfg.get("class_indices") or None,
        download=not args.no_download,
    )
    for p in paths:
        print(f"train={p.train_csv}")
        print(f"val={p.val_csv}")
        print(f"test={p.test_csv}")
        print(f"class_map={p.class_map_csv}")


if __name__ == "__main__":
    main()
