from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.clip_scoring import score_candidates
from ccds.config import clip_embeddings_path, clip_scores_path, generation_metadata_path, load_config, resolve_path
from ccds.datasets import class_map_path
from ccds.visualize import plot_score_distributions


def main() -> None:
    parser = argparse.ArgumentParser(description="Score generated candidates with CLIP target score and margin.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--metadata", default=None, help="Override generated-candidate metadata CSV.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used to format {seed} placeholders in paths.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    clip_cfg = cfg["clip"]
    gen_cfg = cfg["generation"]
    class_map_csv = resolve_path(class_map_path(dataset))
    out_csv = Path(str(clip_scores_path(cfg)).format(seed=args.seed))
    embeddings_npz = Path(str(clip_embeddings_path(cfg)).format(seed=args.seed))
    metadata_csv = resolve_path(args.metadata) if args.metadata else Path(str(generation_metadata_path(cfg)).format(seed=args.seed))

    prompt_templates = clip_cfg.get("prompts") or gen_cfg["prompts"]
    _validate_prompt_templates(prompt_templates)
    prompt_source = "clip.prompts" if clip_cfg.get("prompts") else "generation.prompts"
    print(f"Using CLIP prompt templates from {prompt_source} ({len(prompt_templates)} templates)")

    score_candidates(
        metadata_csv=metadata_csv,
        class_map_csv=class_map_csv,
        prompt_templates=prompt_templates,
        out_csv=out_csv,
        embeddings_npz=embeddings_npz,
        model_name=clip_cfg["model_name"],
        pretrained=clip_cfg["pretrained"],
        batch_size=int(clip_cfg["batch_size"]),
    )
    plot_score_distributions(out_csv, resolve_path("figures/score_distributions"))
    print(f"Wrote scores to {out_csv}")
    print(f"Wrote embeddings to {embeddings_npz}")


def _validate_prompt_templates(prompt_templates: list[str]) -> None:
    if not isinstance(prompt_templates, list) or not prompt_templates:
        raise ValueError("CLIP prompt templates must be a non-empty list.")
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompt_templates):
        raise ValueError("Every CLIP prompt template must be a non-empty string.")


if __name__ == "__main__":
    main()
