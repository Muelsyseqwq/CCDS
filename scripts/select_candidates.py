from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.clip_scoring import load_embeddings_npz
from ccds.config import clip_embeddings_path, clip_scores_path, experiment_name, experiment_results_dir, load_config, resolve_path, selected_csv_path
from ccds.datasets import split_base_name
from ccds.prototypes import load_prototypes_npz
from ccds.real_features import compute_or_load_dinov2_real_candidate_features, compute_or_load_resnet50_real_candidate_features
from ccds.selection import (
    SelectionConfig,
    _adaptive_anchor_counts,
    _adaptive_replacement_counts,
    select_anchored_ccds,
    select_ccds,
    select_cfrd_mmr,
    select_clip_topk,
    select_confusion_adaptive_ccds,
    select_margin_topk,
    select_prototype_ccds,
    select_prototype_gated_ccds,
    select_random,
    select_reliability_diverse_substitution_ccds,
    select_replacement_aware_confusion_adaptive_ccds,
    select_same_overlap_random,
)
from ccds.visualize import make_image_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Select generated candidates by strategy.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--prototype-seed", type=int, default=0, help="Few-shot split seed for prototype_ccds.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used to format {seed} placeholders in paths.")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=[
            "random",
            "clip_topk",
            "margin_topk",
            "ccds",
            "anchored_ccds",
            "cfrd_mmr",
            "confusion_adaptive_ccds",
            "same_overlap_random",
            "replacement_aware_confusion_adaptive_ccds",
            "reliability_diverse_substitution_ccds",
            "prototype_ccds",
            "prototype_gated_ccds",
            "all",
        ],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    score_csv = Path(str(clip_scores_path(cfg)).format(seed=args.seed))
    if not score_csv.exists():
        raise FileNotFoundError(f"Score CSV not found: {score_csv}")

    scores = pd.read_csv(score_csv)
    selection_cfg = cfg["selection"]
    sel_cfg = SelectionConfig(
        selected_per_class=int(selection_cfg["selected_per_class"]),
        top_m_for_diversity=int(selection_cfg["top_m_for_diversity"]),
        prototype_top_m=int(selection_cfg.get("prototype_top_m", 30)),
        anchor_count=int(selection_cfg.get("anchor_count", 7)),
        adaptive_anchor_min_count=int(selection_cfg.get("adaptive_anchor_min_count", 5)),
        adaptive_anchor_max_count=int(selection_cfg.get("adaptive_anchor_max_count", 8)),
        replacement_min_count=int(selection_cfg.get("replacement_min_count", 1)),
        replacement_max_count=int(selection_cfg.get("replacement_max_count", 3)),
        quality_delta=float(selection_cfg.get("quality_delta", 0.005)),
        quality_weight=float(selection_cfg.get("quality_weight", 0.75)),
        margin_weight=float(selection_cfg.get("margin_weight", 0.15)),
        prototype_weight=float(selection_cfg.get("prototype_weight", 0.0)),
        diversity_weight=float(selection_cfg.get("diversity_weight", 0.10)),
        unreliability_margin_weight=float(selection_cfg.get("unreliability_margin_weight", 0.45)),
        unreliability_redundancy_weight=float(selection_cfg.get("unreliability_redundancy_weight", 0.35)),
        unreliability_prototype_weight=float(selection_cfg.get("unreliability_prototype_weight", 0.20)),
        replacement_target_weight=float(selection_cfg.get("replacement_target_weight", 0.45)),
        replacement_margin_weight=float(selection_cfg.get("replacement_margin_weight", 0.20)),
        replacement_prototype_weight=float(selection_cfg.get("replacement_prototype_weight", 0.20)),
        replacement_diversity_weight=float(selection_cfg.get("replacement_diversity_weight", 0.15)),
        cfrd_clip_top_m=int(selection_cfg.get("cfrd_clip_top_m", selection_cfg.get("top_m_for_diversity", 60))),
        cfrd_real_weight=float(selection_cfg.get("cfrd_real_weight", 0.70)),
        cfrd_diversity_weight=float(selection_cfg.get("cfrd_diversity_weight", 0.30)),
    )

    strategies = cfg["selection"]["strategies"] if args.strategy == "all" else [args.strategy]
    embeddings = None
    prototypes = None
    real_features = None
    candidate_features = None
    reference_selected = None
    selected_by_strategy = {}
    for strategy in strategies:
        if strategy == "random":
            selected = select_random(scores, sel_cfg)
        elif strategy == "clip_topk":
            selected = select_clip_topk(scores, sel_cfg)
        elif strategy == "margin_topk":
            selected = select_margin_topk(scores, sel_cfg)
        elif strategy == "ccds":
            if embeddings is None:
                embeddings = load_embeddings_npz(Path(str(clip_embeddings_path(cfg)).format(seed=args.seed)))
            selected = select_ccds(scores, embeddings, sel_cfg)
        elif strategy == "anchored_ccds":
            if embeddings is None:
                embeddings = load_embeddings_npz(Path(str(clip_embeddings_path(cfg)).format(seed=args.seed)))
            selected = select_anchored_ccds(scores, embeddings, sel_cfg)
        elif strategy == "cfrd_mmr":
            if real_features is None or candidate_features is None:
                real_features, candidate_features = _load_or_compute_cfrd_features(cfg, selection_cfg, scores, args.prototype_seed)
            selected, diagnostics = select_cfrd_mmr(scores, real_features, candidate_features, sel_cfg, return_diagnostics=True)
        elif strategy == "confusion_adaptive_ccds":
            if embeddings is None:
                embeddings = load_embeddings_npz(Path(str(clip_embeddings_path(cfg)).format(seed=args.seed)))
            selected = select_confusion_adaptive_ccds(scores, embeddings, sel_cfg)
        elif strategy == "same_overlap_random":
            if reference_selected is None:
                reference_selected = _load_reference_selection(cfg, selection_cfg, selected_by_strategy)
            selected = select_same_overlap_random(scores, reference_selected, sel_cfg)
        elif strategy == "replacement_aware_confusion_adaptive_ccds":
            if embeddings is None:
                embeddings = load_embeddings_npz(Path(str(clip_embeddings_path(cfg)).format(seed=args.seed)))
            selected = select_replacement_aware_confusion_adaptive_ccds(scores, embeddings, sel_cfg)
        elif strategy == "reliability_diverse_substitution_ccds":
            if embeddings is None:
                embeddings = load_embeddings_npz(Path(str(clip_embeddings_path(cfg)).format(seed=args.seed)))
            if selection_cfg.get("use_prototypes", True) and prototypes is None:
                prototypes = _load_prototypes(cfg, selection_cfg, args.prototype_seed)
            selected, diagnostics = select_reliability_diverse_substitution_ccds(
                scores, embeddings, prototypes, sel_cfg, return_diagnostics=True
            )
        elif strategy == "prototype_ccds":
            if embeddings is None:
                embeddings = load_embeddings_npz(Path(str(clip_embeddings_path(cfg)).format(seed=args.seed)))
            if prototypes is None:
                prototypes = _load_prototypes(cfg, selection_cfg, args.prototype_seed)
            selected = select_prototype_ccds(scores, embeddings, prototypes, sel_cfg)
        elif strategy == "prototype_gated_ccds":
            if embeddings is None:
                embeddings = load_embeddings_npz(Path(str(clip_embeddings_path(cfg)).format(seed=args.seed)))
            if prototypes is None:
                prototypes = _load_prototypes(cfg, selection_cfg, args.prototype_seed)
            selected = select_prototype_gated_ccds(scores, embeddings, prototypes, sel_cfg)
        else:
            raise ValueError(strategy)

        out_path = Path(str(selected_csv_path(cfg, strategy)).format(seed=args.seed))
        out_paths = [out_path]
        if strategy == "cfrd_mmr":
            out_paths = [out_path.with_name(f"{out_path.stem}_seed{args.prototype_seed}{out_path.suffix}"), out_path]
        for path in out_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            selected.to_csv(path, index=False)
            print(f"Wrote {len(selected)} selected samples to {path}")
        selected_by_strategy[strategy] = selected
        if strategy == "confusion_adaptive_ccds":
            _write_confusion_adaptive_diagnostics(cfg, scores, selected, sel_cfg)
        elif strategy == "same_overlap_random":
            _write_replacement_diagnostics(cfg, scores, selected, sel_cfg, strategy, reference_selected)
        elif strategy == "replacement_aware_confusion_adaptive_ccds":
            _write_replacement_diagnostics(cfg, scores, selected, sel_cfg, strategy)
        elif strategy == "reliability_diverse_substitution_ccds":
            _write_rds_diagnostics(cfg, diagnostics)
        elif strategy == "cfrd_mmr":
            _write_cfrd_diagnostics(cfg, diagnostics, args.prototype_seed)
        _visualize_selection(scores, selected, strategy)


def _load_or_compute_cfrd_features(
    cfg: dict,
    selection_cfg: dict,
    scores: pd.DataFrame,
    seed: int,
):
    dataset_cfg = cfg["dataset"]
    split_base = split_base_name(dataset_cfg, seed)
    train_csv = resolve_path(f"data/splits/{split_base}_train.csv")
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing split file: {train_csv}. Run scripts/make_splits.py first.")

    backbone = str(selection_cfg.get("cfrd_feature_backbone", "resnet50")).lower()
    configured = selection_cfg.get("cfrd_features_npz")
    if configured:
        out_npz = resolve_path(str(configured).format(seed=seed))
    else:
        feature_name = "cfrd_dinov2" if backbone in {"dinov2", "facebook/dinov2-base"} else "cfrd_resnet50"
        out_npz = experiment_results_dir(cfg) / "features" / f"{feature_name}_seed{seed}.npz"

    common_kwargs = {
        "train_csv": train_csv,
        "scores": scores,
        "out_npz": out_npz,
        "image_size": int(dataset_cfg.get("image_size", 224)),
        "batch_size": int(selection_cfg.get("cfrd_feature_batch_size", cfg.get("clip", {}).get("batch_size", 32))),
        "num_workers": int(selection_cfg.get("cfrd_feature_num_workers", cfg.get("classifier", {}).get("num_workers", 4))),
        "force": bool(selection_cfg.get("cfrd_force_recompute_features", False)),
    }
    if backbone in {"resnet", "resnet50"}:
        return compute_or_load_resnet50_real_candidate_features(**common_kwargs)
    if backbone in {"dinov2", "facebook/dinov2-base"}:
        return compute_or_load_dinov2_real_candidate_features(
            **common_kwargs,
            model_name=str(selection_cfg.get("cfrd_dinov2_model_name", "facebook/dinov2-base")),
        )
    raise ValueError(f"Unsupported CFRD feature backbone: {backbone}")



def _load_reference_selection(cfg: dict, selection_cfg: dict, selected_by_strategy: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    configured = selection_cfg.get("same_overlap_reference_csv")
    reference_strategy = selection_cfg.get("same_overlap_reference_strategy", "confusion_adaptive_ccds")
    if selected_by_strategy and reference_strategy in selected_by_strategy:
        return selected_by_strategy[reference_strategy]
    if configured:
        path = resolve_path(str(configured))
    else:
        path = selected_csv_path(cfg, reference_strategy)
    if not path.exists():
        raise FileNotFoundError(f"Reference selection CSV not found: {path}")
    return pd.read_csv(path)


def _load_prototypes(cfg: dict, selection_cfg: dict, seed: int):
    configured = selection_cfg.get("prototype_npz")
    if configured:
        path = resolve_path(str(configured).format(seed=seed))
    else:
        path = experiment_results_dir(cfg) / "prototypes" / f"real_prototypes_seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Prototype NPZ not found: {path}. Run scripts/compute_real_prototypes.py first.")
    return load_prototypes_npz(path)


def _write_confusion_adaptive_diagnostics(
    cfg: dict,
    scores: pd.DataFrame,
    selected: pd.DataFrame,
    sel_cfg: SelectionConfig,
) -> None:
    anchor_counts = _adaptive_anchor_counts(scores, sel_cfg)
    rows = []
    for class_name, group in scores.groupby("target_class"):
        selected_group = selected[selected["target_class"] == class_name]
        clip_topk = group.sort_values("target_score", ascending=False).head(sel_cfg.selected_per_class)
        selected_paths = set(selected_group["image_path"])
        overlap = len(selected_paths & set(clip_topk["image_path"]))
        rows.append(
            {
                "target_class": class_name,
                "target_label": int(group["target_label"].iloc[0]),
                "topk_margin_mean": float(clip_topk["margin_score"].mean()),
                "adaptive_anchor_count": int(anchor_counts[str(class_name)]),
                "selected_count": int(len(selected_group)),
                "selected_target_score_mean": float(selected_group["target_score"].mean()),
                "selected_margin_score_mean": float(selected_group["margin_score"].mean()),
                "overlap_with_clip_topk_count": int(overlap),
                "overlap_with_clip_topk_pct": float(overlap / max(1, len(clip_topk))),
            }
        )

    out_path = resolve_path("results/research_process") / f"{experiment_name(cfg)}_ca_selection_diagnostics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote confusion-adaptive diagnostics to {out_path}")


def _write_rds_diagnostics(cfg: dict, diagnostics: pd.DataFrame) -> None:
    out_path = resolve_path("results/research_process") / f"{experiment_name(cfg)}_rds_diagnostics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(out_path, index=False)
    print(f"Wrote RDS diagnostics to {out_path}")



def _write_cfrd_diagnostics(cfg: dict, diagnostics: pd.DataFrame, seed: int) -> None:
    out_path = resolve_path("results/research_process") / f"{experiment_name(cfg)}_cfrd_diagnostics_seed{seed}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(out_path, index=False)
    print(f"Wrote CFRD diagnostics to {out_path}")



def _write_replacement_diagnostics(
    cfg: dict,
    scores: pd.DataFrame,
    selected: pd.DataFrame,
    sel_cfg: SelectionConfig,
    strategy: str,
    reference_selected: pd.DataFrame | None = None,
) -> None:
    planned_counts = None
    if strategy == "replacement_aware_confusion_adaptive_ccds":
        planned_counts = _adaptive_replacement_counts(scores, sel_cfg)

    rows = []
    for class_name, group in scores.groupby("target_class"):
        selected_group = selected[selected["target_class"] == class_name]
        clip_topk = group.sort_values("target_score", ascending=False).head(sel_cfg.selected_per_class)
        selected_paths = set(selected_group["image_path"])
        clip_topk_paths = set(clip_topk["image_path"])
        overlap = len(selected_paths & clip_topk_paths)
        planned = sel_cfg.selected_per_class - overlap
        if planned_counts is not None:
            planned = planned_counts[str(class_name)]
        elif reference_selected is not None:
            ref_paths = set(reference_selected[reference_selected["target_class"] == class_name]["image_path"])
            planned = sel_cfg.selected_per_class - len(ref_paths & clip_topk_paths)
        rows.append(
            {
                "target_class": class_name,
                "target_label": int(group["target_label"].iloc[0]),
                "selected_count": int(len(selected_group)),
                "clip_topk_count": int(len(clip_topk)),
                "planned_replacement_count": int(planned),
                "actual_non_clip_topk_count": int(len(selected_paths - clip_topk_paths)),
                "overlap_with_clip_topk_count": int(overlap),
                "overlap_with_clip_topk_pct": float(overlap / max(1, len(clip_topk))),
                "selected_target_score_mean": float(selected_group["target_score"].mean()),
                "selected_margin_score_mean": float(selected_group["margin_score"].mean()),
                "clip_topk_target_score_mean": float(clip_topk["target_score"].mean()),
                "clip_topk_margin_score_mean": float(clip_topk["margin_score"].mean()),
            }
        )

    suffix = "same_overlap_random" if strategy == "same_overlap_random" else "replacement_aware_ca"
    out_path = resolve_path("results/research_process") / f"{experiment_name(cfg)}_{suffix}_diagnostics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote replacement diagnostics to {out_path}")


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
