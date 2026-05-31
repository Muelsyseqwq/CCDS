from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.clip_scoring import score_candidates
from ccds.config import clip_embeddings_path, clip_scores_path, generation_metadata_path, load_config, resolve_path
from ccds.visualize import plot_score_distributions


def main() -> None:
    parser = argparse.ArgumentParser(description="Score generated candidates with CLIP target score and margin.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--metadata", default=None, help="Override generated-candidate metadata CSV.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    clip_cfg = cfg["clip"]
    gen_cfg = cfg["generation"]
    num_classes = int(dataset["num_classes"])
    class_map_csv = resolve_path(f"data/splits/flowers{num_classes}_class_map.csv")
    out_csv = clip_scores_path(cfg)
    embeddings_npz = clip_embeddings_path(cfg)
    metadata_csv = resolve_path(args.metadata) if args.metadata else generation_metadata_path(cfg)

    score_candidates(
        metadata_csv=metadata_csv,
        class_map_csv=class_map_csv,
        prompt_templates=gen_cfg["prompts"],
        out_csv=out_csv,
        embeddings_npz=embeddings_npz,
        model_name=clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        batch_size=int(clip_cfg["batch_size"]),
    )
    plot_score_distributions(out_csv, resolve_path("figures/score_distributions"))
    print(f"Wrote scores to {out_csv}")
    print(f"Wrote embeddings to {embeddings_npz}")


if __name__ == "__main__":
    main()
