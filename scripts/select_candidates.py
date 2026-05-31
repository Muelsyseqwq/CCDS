from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.clip_scoring import load_embeddings_npz
from ccds.config import clip_embeddings_path, clip_scores_path, load_config, resolve_path, selected_csv_path
from ccds.selection import SelectionConfig, select_ccds, select_clip_topk, select_margin_topk, select_random
from ccds.visualize import make_image_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Select generated candidates by strategy.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--strategy", required=True, choices=["random", "clip_topk", "margin_topk", "ccds", "all"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    score_csv = clip_scores_path(cfg)
    if not score_csv.exists():
        raise FileNotFoundError(f"Score CSV not found: {score_csv}")

    scores = pd.read_csv(score_csv)
    sel_cfg = SelectionConfig(
        selected_per_class=int(cfg["selection"]["selected_per_class"]),
        top_m_for_diversity=int(cfg["selection"]["top_m_for_diversity"]),
    )

    strategies = cfg["selection"]["strategies"] if args.strategy == "all" else [args.strategy]
    embeddings = None
    for strategy in strategies:
        if strategy == "random":
            selected = select_random(scores, sel_cfg)
        elif strategy == "clip_topk":
            selected = select_clip_topk(scores, sel_cfg)
        elif strategy == "margin_topk":
            selected = select_margin_topk(scores, sel_cfg)
        elif strategy == "ccds":
            if embeddings is None:
                embeddings = load_embeddings_npz(clip_embeddings_path(cfg))
            selected = select_ccds(scores, embeddings, sel_cfg)
        else:
            raise ValueError(strategy)

        out_path = selected_csv_path(cfg, strategy)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(out_path, index=False)
        print(f"Wrote {len(selected)} selected samples to {out_path}")
        _visualize_selection(scores, selected, strategy)


def _visualize_selection(scores: pd.DataFrame, selected: pd.DataFrame, strategy: str) -> None:
    fig_dir = resolve_path(f"figures/selections/{strategy}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for class_name, group in selected.groupby("target_class"):
        paths = group["image_path"].head(15).tolist()
        make_image_grid(paths, fig_dir / f"{_slug(class_name)}.png", title=f"{strategy}: {class_name}")

    top = scores.sort_values("margin_score", ascending=False).head(20)["image_path"].tolist()
    bottom = scores.sort_values("margin_score", ascending=True).head(20)["image_path"].tolist()
    make_image_grid(top, fig_dir / "top_margin_overall.png", title="Top margin samples")
    make_image_grid(bottom, fig_dir / "bottom_margin_overall.png", title="Bottom margin samples")


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


if __name__ == "__main__":
    main()
