from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class SelectionConfig:
    selected_per_class: int = 10
    top_m_for_diversity: int = 30
    prototype_top_m: int = 30
    anchor_count: int = 7
    adaptive_anchor_min_count: int = 5
    adaptive_anchor_max_count: int = 8
    replacement_min_count: int = 1
    replacement_max_count: int = 3
    quality_delta: float = 0.005
    quality_weight: float = 0.75
    margin_weight: float = 0.15
    prototype_weight: float = 0.0
    diversity_weight: float = 0.10
    unreliability_margin_weight: float = 0.45
    unreliability_redundancy_weight: float = 0.35
    unreliability_prototype_weight: float = 0.20
    replacement_target_weight: float = 0.45
    replacement_margin_weight: float = 0.20
    replacement_prototype_weight: float = 0.20
    replacement_diversity_weight: float = 0.15
    cfrd_clip_top_m: int = 60
    cfrd_real_weight: float = 0.70
    cfrd_diversity_weight: float = 0.30
    random_state: int = 0


def select_random(scores: pd.DataFrame, cfg: SelectionConfig) -> pd.DataFrame:
    """Randomly select K generated samples per target class."""
    selected_parts = []
    for _, group in scores.groupby("target_class"):
        selected_parts.append(group.sample(min(len(group), cfg.selected_per_class), random_state=cfg.random_state))
    if not selected_parts:
        return scores.head(0).copy().reset_index(drop=True)
    return pd.concat(selected_parts, ignore_index=True)


def select_clip_topk(scores: pd.DataFrame, cfg: SelectionConfig) -> pd.DataFrame:
    """Select K samples with highest target CLIP score per class."""
    return _topk_by_column(scores, "target_score", cfg.selected_per_class)


def select_margin_topk(scores: pd.DataFrame, cfg: SelectionConfig) -> pd.DataFrame:
    """Select K samples with highest class-consistency margin per class."""
    return _topk_by_column(scores, "margin_score", cfg.selected_per_class)


def select_ccds(scores: pd.DataFrame, embeddings: dict[str, np.ndarray], cfg: SelectionConfig) -> pd.DataFrame:
    """Select samples by margin first, then enforce diversity with KMeans.

    Args:
        scores: DataFrame containing image_path, target_class, and margin_score.
        embeddings: Mapping from image_path to CLIP image embedding.
        cfg: Selection hyperparameters.
    """
    selected_parts: list[pd.DataFrame] = []
    for _, group in scores.groupby("target_class"):
        top_m = group.sort_values("margin_score", ascending=False).head(cfg.top_m_for_diversity).copy()
        if len(top_m) <= cfg.selected_per_class:
            selected_parts.append(top_m)
            continue

        matrix = np.stack([embeddings[p] for p in top_m["image_path"].tolist()])
        n_clusters = min(cfg.selected_per_class, len(top_m))
        labels = KMeans(n_clusters=n_clusters, random_state=cfg.random_state, n_init="auto").fit_predict(matrix)
        top_m["cluster"] = labels

        cluster_selected = []
        for _, cluster_df in top_m.groupby("cluster"):
            cluster_selected.append(cluster_df.sort_values("margin_score", ascending=False).head(1))
        selected_parts.append(pd.concat(cluster_selected, ignore_index=True))

    return pd.concat(selected_parts, ignore_index=True)


def select_anchored_ccds(scores: pd.DataFrame, embeddings: dict[str, np.ndarray], cfg: SelectionConfig) -> pd.DataFrame:
    """Select CLIP Top-K anchors, then fill remaining slots with quality-diversity reranking."""
    return _select_anchored_with_optional_prototypes(scores, embeddings, None, cfg)


def select_confusion_adaptive_ccds(scores: pd.DataFrame, embeddings: dict[str, np.ndarray], cfg: SelectionConfig) -> pd.DataFrame:
    """Select class-adaptive anchors based on CLIP margin difficulty, then rerank.

    Classes whose CLIP Top-K candidates have lower average class-consistency margin
    keep fewer CLIP anchors and replace more samples with margin/diversity-aware
    candidates. Easier classes retain more CLIP Top-K anchors.
    """
    adaptive_counts = _adaptive_anchor_counts(scores, cfg)
    return _select_anchored_with_optional_prototypes(scores, embeddings, None, cfg, adaptive_counts)


def select_same_overlap_random(
    scores: pd.DataFrame,
    reference_selected: pd.DataFrame,
    cfg: SelectionConfig,
) -> pd.DataFrame:
    """Match a reference selector's per-class CLIP Top-K overlap, then random-fill replacements."""
    selected_parts: list[pd.DataFrame] = []
    rng = np.random.default_rng(cfg.random_state)
    for class_name, group in scores.groupby("target_class"):
        pool = group.sort_values("target_score", ascending=False).head(cfg.top_m_for_diversity).copy()
        clip_topk = group.sort_values("target_score", ascending=False).head(cfg.selected_per_class).copy()
        reference_paths = set(reference_selected[reference_selected["target_class"] == class_name]["image_path"])
        overlap_count = len(reference_paths & set(clip_topk["image_path"]))
        keep_count = min(overlap_count, cfg.selected_per_class, len(clip_topk))
        selected = clip_topk.head(keep_count).copy()

        remaining_slots = cfg.selected_per_class - len(selected)
        if remaining_slots > 0:
            non_topk_pool = pool.loc[~pool["image_path"].isin(clip_topk["image_path"])].copy()
            if len(non_topk_pool) < remaining_slots:
                fallback = pool.loc[~pool["image_path"].isin(selected["image_path"])].copy()
                non_topk_pool = pd.concat([non_topk_pool, fallback], ignore_index=True).drop_duplicates("image_path")
            if not non_topk_pool.empty:
                take = min(remaining_slots, len(non_topk_pool))
                sampled_positions = rng.choice(len(non_topk_pool), size=take, replace=False)
                selected = pd.concat([selected, non_topk_pool.iloc[sampled_positions]], ignore_index=True)

        selected_parts.append(selected.head(cfg.selected_per_class))
    return pd.concat(selected_parts, ignore_index=True)


def select_replacement_aware_confusion_adaptive_ccds(
    scores: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    cfg: SelectionConfig,
) -> pd.DataFrame:
    """Select explicit non-CLIP-Top-K replacements using class-adaptive replacement counts."""
    replacement_counts = _adaptive_replacement_counts(scores, cfg)
    selected_parts: list[pd.DataFrame] = []
    for class_name, group in scores.groupby("target_class"):
        pool = group.sort_values("target_score", ascending=False).head(cfg.top_m_for_diversity).copy()
        clip_topk = group.sort_values("target_score", ascending=False).head(cfg.selected_per_class).copy()
        if len(pool) <= cfg.selected_per_class:
            selected_parts.append(pool)
            continue

        planned_replacements = min(replacement_counts.get(str(class_name), cfg.replacement_min_count), cfg.selected_per_class)
        anchors_to_keep = max(cfg.selected_per_class - planned_replacements, 0)
        selected = clip_topk.head(anchors_to_keep).copy()
        replacement_pool = pool.loc[~pool["image_path"].isin(clip_topk["image_path"])].copy()

        while len(selected) < cfg.selected_per_class and not replacement_pool.empty:
            scores_for_remaining = _anchored_rerank_scores(replacement_pool, selected, embeddings, None, cfg)
            best_idx = scores_for_remaining.idxmax()
            selected = pd.concat([selected, replacement_pool.loc[[best_idx]]], ignore_index=True)
            replacement_pool = replacement_pool.drop(index=best_idx)

        if len(selected) < cfg.selected_per_class:
            fallback = pool.loc[~pool["image_path"].isin(selected["image_path"])].copy()
            selected = pd.concat([selected, fallback.head(cfg.selected_per_class - len(selected))], ignore_index=True)

        selected_parts.append(selected.head(cfg.selected_per_class))
    return pd.concat(selected_parts, ignore_index=True)


def select_cfrd_mmr(
    scores: pd.DataFrame,
    real_features: dict[int, np.ndarray],
    candidate_features: dict[str, np.ndarray],
    cfg: SelectionConfig,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Select CLIP-filtered candidates by real-feature similarity and MMR diversity."""
    selected_parts: list[pd.DataFrame] = []
    diagnostic_rows: list[dict] = []
    for class_name, group in scores.groupby("target_class"):
        pool = group.sort_values("target_score", ascending=False).head(cfg.cfrd_clip_top_m).copy()
        clip_topk = group.sort_values("target_score", ascending=False).head(cfg.selected_per_class).copy()
        if len(pool) <= cfg.selected_per_class:
            selected = pool
        else:
            label = int(pool["target_label"].iloc[0])
            if label not in real_features:
                raise KeyError(f"Missing real features for label {label}")
            selected = _select_mmr_by_real_features(pool, real_features[label], candidate_features, cfg)
        selected_parts.append(selected)

        if return_diagnostics:
            label = int(group["target_label"].iloc[0])
            if label not in real_features:
                raise KeyError(f"Missing real features for label {label}")
            diagnostic_rows.extend(
                _cfrd_diagnostic_rows(
                    class_name=str(class_name),
                    group=group,
                    pool=pool,
                    selected=selected,
                    clip_topk=clip_topk,
                    real_features=real_features[label],
                    candidate_features=candidate_features,
                )
            )

    if not selected_parts:
        selected_df = scores.head(0).copy().reset_index(drop=True)
    else:
        selected_df = pd.concat(selected_parts, ignore_index=True)
    if return_diagnostics:
        return selected_df, pd.DataFrame(diagnostic_rows)
    return selected_df



def select_reliability_diverse_substitution_ccds(
    scores: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    prototypes: dict[int, np.ndarray] | None,
    cfg: SelectionConfig,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Replace only the least reliable CLIP Top-K samples with quality-gated diverse candidates.

    RDS-CCDS starts from CLIP Top-K, diagnoses low-value Top-K samples by
    low margin, within-Top-K redundancy, and optional real-prototype mismatch,
    then substitutes a small class-adaptive number of non-Top-K candidates.
    """
    replacement_counts = _adaptive_replacement_counts(scores, cfg)
    selected_parts: list[pd.DataFrame] = []
    diagnostic_rows: list[dict] = []
    for class_name, group in scores.groupby("target_class"):
        pool = group.sort_values("target_score", ascending=False).head(cfg.top_m_for_diversity).copy()
        clip_topk = group.sort_values("target_score", ascending=False).head(cfg.selected_per_class).copy()
        if len(pool) <= cfg.selected_per_class:
            selected_parts.append(pool)
            continue

        planned_replacements = min(
            int(replacement_counts.get(str(class_name), cfg.replacement_min_count)),
            cfg.selected_per_class,
            len(clip_topk),
        )
        unreliability = _rds_unreliability(clip_topk, embeddings, prototypes, cfg)
        remove_paths = set(unreliability.sort_values(ascending=False).head(planned_replacements).index)
        selected = clip_topk.loc[~clip_topk["image_path"].isin(remove_paths)].copy()

        quality_floor = float(clip_topk["target_score"].min()) - float(cfg.quality_delta)
        replacement_pool = pool.loc[~pool["image_path"].isin(clip_topk["image_path"])].copy()
        quality_pool = replacement_pool.loc[replacement_pool["target_score"] >= quality_floor].copy()
        if quality_pool.empty:
            quality_pool = replacement_pool.copy()

        while len(selected) < cfg.selected_per_class and not quality_pool.empty:
            candidate_scores = _rds_replacement_scores(quality_pool, selected, embeddings, prototypes, cfg)
            best_idx = candidate_scores.idxmax()
            selected = pd.concat([selected, quality_pool.loc[[best_idx]]], ignore_index=True)
            quality_pool = quality_pool.drop(index=best_idx)

        if len(selected) < cfg.selected_per_class:
            fallback = pool.loc[~pool["image_path"].isin(selected["image_path"])].copy()
            selected = pd.concat([selected, fallback.head(cfg.selected_per_class - len(selected))], ignore_index=True)

        selected = selected.head(cfg.selected_per_class)
        selected_parts.append(selected)
        if return_diagnostics:
            diagnostic_rows.extend(
                _rds_diagnostic_rows(
                    class_name=str(class_name),
                    group=group,
                    selected=selected,
                    clip_topk=clip_topk,
                    embeddings=embeddings,
                    prototypes=prototypes,
                    cfg=cfg,
                    planned_replacements=planned_replacements,
                    quality_floor=quality_floor,
                    unreliability=unreliability,
                )
            )

    selected_df = pd.concat(selected_parts, ignore_index=True)
    if return_diagnostics:
        return selected_df, pd.DataFrame(diagnostic_rows)
    return selected_df



def select_prototype_ccds(
    scores: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    prototypes: dict[int, np.ndarray],
    cfg: SelectionConfig,
) -> pd.DataFrame:
    """Select CLIP Top-K anchors, then rerank with quality, margin, prototype alignment, and diversity."""
    return _select_anchored_with_optional_prototypes(scores, embeddings, prototypes, cfg)


def select_prototype_gated_ccds(
    scores: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    prototypes: dict[int, np.ndarray],
    cfg: SelectionConfig,
) -> pd.DataFrame:
    """Select anchors, gate non-anchor candidates by prototype alignment, then rerank."""
    selected_parts: list[pd.DataFrame] = []
    for _, group in scores.groupby("target_class"):
        pool = group.sort_values("target_score", ascending=False).head(cfg.top_m_for_diversity).copy()
        if len(pool) <= cfg.selected_per_class:
            selected_parts.append(pool)
            continue

        anchor_count = min(cfg.anchor_count, cfg.selected_per_class, len(pool))
        selected = pool.sort_values("target_score", ascending=False).head(anchor_count).copy()
        remaining = pool.loc[~pool["image_path"].isin(selected["image_path"])].copy()
        if not remaining.empty:
            prototype_scores = _prototype_alignment(remaining, embeddings, prototypes)
            keep = min(cfg.prototype_top_m, len(remaining))
            remaining = remaining.assign(_prototype_gate_score=prototype_scores).sort_values(
                "_prototype_gate_score", ascending=False
            ).head(keep)

        while len(selected) < cfg.selected_per_class and not remaining.empty:
            scores_for_remaining = _anchored_rerank_scores(remaining, selected, embeddings, None, cfg)
            best_idx = scores_for_remaining.idxmax()
            selected = pd.concat([selected, remaining.loc[[best_idx]].drop(columns=["_prototype_gate_score"], errors="ignore")], ignore_index=True)
            remaining = remaining.drop(index=best_idx)

        selected_parts.append(selected)

    return pd.concat(selected_parts, ignore_index=True)


def _select_mmr_by_real_features(
    pool: pd.DataFrame,
    real_features: np.ndarray,
    candidate_features: dict[str, np.ndarray],
    cfg: SelectionConfig,
) -> pd.DataFrame:
    remaining = pool.copy()
    selected = pool.head(0).copy()
    real_matrix = _normalize_feature_matrix(real_features)
    real_scores = _real_similarity(remaining, real_matrix, candidate_features)

    while len(selected) < cfg.selected_per_class and not remaining.empty:
        if selected.empty:
            mmr_scores = real_scores.loc[remaining.index]
        else:
            redundancy = _feature_redundancy(remaining, selected, candidate_features)
            mmr_scores = cfg.cfrd_real_weight * real_scores.loc[remaining.index] - cfg.cfrd_diversity_weight * redundancy
        best_idx = mmr_scores.idxmax()
        selected = pd.concat([selected, remaining.loc[[best_idx]]], ignore_index=True)
        remaining = remaining.drop(index=best_idx)

    return selected.head(cfg.selected_per_class)



def _cfrd_diagnostic_rows(
    class_name: str,
    group: pd.DataFrame,
    pool: pd.DataFrame,
    selected: pd.DataFrame,
    clip_topk: pd.DataFrame,
    real_features: np.ndarray,
    candidate_features: dict[str, np.ndarray],
) -> list[dict]:
    real_matrix = _normalize_feature_matrix(real_features)
    selected_paths = set(selected["image_path"])
    clip_topk_paths = set(clip_topk["image_path"])
    pool_paths = set(pool["image_path"])
    selected_real_sim = _real_similarity(selected, real_matrix, candidate_features)
    clip_topk_real_sim = _real_similarity(clip_topk, real_matrix, candidate_features)
    selected_redundancy = _feature_redundancy_or_zero(selected, candidate_features)
    rows = [
        {
            "diagnostic_type": "class_summary",
            "target_class": class_name,
            "target_label": int(group["target_label"].iloc[0]),
            "clip_filter_top_m": int(len(pool)),
            "selected_count": int(len(selected)),
            "overlap_with_clip_topk_count": int(len(selected_paths & clip_topk_paths)),
            "overlap_with_clip_topk_pct": float(len(selected_paths & clip_topk_paths) / max(1, len(clip_topk))),
            "selected_outside_clip_filter_count": int(len(selected_paths - pool_paths)),
            "selected_real_sim_mean": float(selected_real_sim.mean()) if not selected_real_sim.empty else np.nan,
            "clip_topk_real_sim_mean": float(clip_topk_real_sim.mean()) if not clip_topk_real_sim.empty else np.nan,
            "selected_redundancy_mean": float(selected_redundancy.mean()) if not selected_redundancy.empty else np.nan,
            "selected_target_score_mean": float(selected["target_score"].mean()) if not selected.empty else np.nan,
            "clip_topk_target_score_mean": float(clip_topk["target_score"].mean()) if not clip_topk.empty else np.nan,
            "selected_margin_score_mean": float(selected["margin_score"].mean()) if not selected.empty else np.nan,
            "clip_topk_margin_score_mean": float(clip_topk["margin_score"].mean()) if not clip_topk.empty else np.nan,
            "image_path": "",
            "target_score": np.nan,
            "margin_score": np.nan,
            "real_sim": np.nan,
            "redundancy": np.nan,
            "is_clip_topk": np.nan,
            "is_selected": np.nan,
        }
    ]
    for _, row in selected.iterrows():
        image_path = row["image_path"]
        rows.append(
            {
                "diagnostic_type": "selected_sample",
                "target_class": class_name,
                "target_label": int(row["target_label"]),
                "clip_filter_top_m": int(len(pool)),
                "selected_count": np.nan,
                "overlap_with_clip_topk_count": np.nan,
                "overlap_with_clip_topk_pct": np.nan,
                "selected_outside_clip_filter_count": np.nan,
                "selected_real_sim_mean": np.nan,
                "clip_topk_real_sim_mean": np.nan,
                "selected_redundancy_mean": np.nan,
                "selected_target_score_mean": np.nan,
                "clip_topk_target_score_mean": np.nan,
                "selected_margin_score_mean": np.nan,
                "clip_topk_margin_score_mean": np.nan,
                "image_path": image_path,
                "target_score": float(row["target_score"]),
                "margin_score": float(row["margin_score"]),
                "real_sim": float(selected_real_sim.get(row.name, np.nan)),
                "redundancy": float(selected_redundancy.get(row.name, np.nan)),
                "is_clip_topk": bool(image_path in clip_topk_paths),
                "is_selected": True,
            }
        )
    return rows



def _select_anchored_with_optional_prototypes(
    scores: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    prototypes: dict[int, np.ndarray] | None,
    cfg: SelectionConfig,
    adaptive_anchor_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    selected_parts: list[pd.DataFrame] = []
    for class_name, group in scores.groupby("target_class"):
        pool = group.sort_values("target_score", ascending=False).head(cfg.top_m_for_diversity).copy()
        if len(pool) <= cfg.selected_per_class:
            selected_parts.append(pool)
            continue

        configured_anchor_count = cfg.anchor_count
        if adaptive_anchor_counts is not None:
            configured_anchor_count = adaptive_anchor_counts.get(str(class_name), cfg.anchor_count)
        anchor_count = min(configured_anchor_count, cfg.selected_per_class, len(pool))
        selected = pool.sort_values("target_score", ascending=False).head(anchor_count).copy()
        remaining = pool.loc[~pool["image_path"].isin(selected["image_path"])].copy()

        while len(selected) < cfg.selected_per_class and not remaining.empty:
            scores_for_remaining = _anchored_rerank_scores(remaining, selected, embeddings, prototypes, cfg)
            best_idx = scores_for_remaining.idxmax()
            selected = pd.concat([selected, remaining.loc[[best_idx]]], ignore_index=True)
            remaining = remaining.drop(index=best_idx)

        selected_parts.append(selected)

    return pd.concat(selected_parts, ignore_index=True)


def _adaptive_anchor_counts(scores: pd.DataFrame, cfg: SelectionConfig) -> dict[str, int]:
    """Map each class to an anchor count using CLIP Top-K margin difficulty."""
    min_count = int(cfg.adaptive_anchor_min_count)
    max_count = int(cfg.adaptive_anchor_max_count)
    if min_count > max_count:
        raise ValueError("adaptive_anchor_min_count must be <= adaptive_anchor_max_count")
    if min_count == max_count:
        return {str(class_name): min_count for class_name in scores["target_class"].unique()}

    margins: dict[str, float] = {}
    for class_name, group in scores.groupby("target_class"):
        topk = group.sort_values("target_score", ascending=False).head(cfg.selected_per_class)
        margins[str(class_name)] = float(topk["margin_score"].mean())

    values = pd.Series(margins, dtype=float)
    if np.isclose(values.max(), values.min()):
        fallback = int(np.clip(cfg.anchor_count, min_count, max_count))
        return {class_name: fallback for class_name in values.index}

    normalized = _minmax_normalize(values)
    span = max_count - min_count
    counts: dict[str, int] = {}
    for class_name, value in normalized.items():
        # Low margin means hard/confusable class -> fewer anchors, more replacements.
        count = int(round(min_count + float(value) * span))
        counts[str(class_name)] = int(np.clip(count, min_count, max_count))
    return counts


def _adaptive_replacement_counts(scores: pd.DataFrame, cfg: SelectionConfig) -> dict[str, int]:
    """Map low-margin classes to more explicit non-CLIP-Top-K replacements."""
    min_count = int(cfg.replacement_min_count)
    max_count = int(cfg.replacement_max_count)
    if min_count > max_count:
        raise ValueError("replacement_min_count must be <= replacement_max_count")
    if min_count == max_count:
        return {str(class_name): min_count for class_name in scores["target_class"].unique()}

    margins: dict[str, float] = {}
    for class_name, group in scores.groupby("target_class"):
        topk = group.sort_values("target_score", ascending=False).head(cfg.selected_per_class)
        margins[str(class_name)] = float(topk["margin_score"].mean())

    values = pd.Series(margins, dtype=float)
    if np.isclose(values.max(), values.min()):
        fallback = int(np.clip(cfg.replacement_min_count, min_count, max_count))
        return {class_name: fallback for class_name in values.index}

    normalized = _minmax_normalize(values)
    difficulty = 1.0 - normalized
    span = max_count - min_count
    counts: dict[str, int] = {}
    for class_name, value in difficulty.items():
        count = int(round(min_count + float(value) * span))
        counts[str(class_name)] = int(np.clip(count, min_count, max_count))
    return counts


def _rds_unreliability(
    clip_topk: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    prototypes: dict[int, np.ndarray] | None,
    cfg: SelectionConfig,
) -> pd.Series:
    low_margin = 1.0 - _minmax_normalize(clip_topk["margin_score"])
    redundancy = _topk_redundancy(clip_topk, embeddings)
    score = cfg.unreliability_margin_weight * low_margin + cfg.unreliability_redundancy_weight * redundancy
    if prototypes is not None and cfg.unreliability_prototype_weight > 0:
        low_prototype = 1.0 - _prototype_alignment(clip_topk, embeddings, prototypes)
        score = score + cfg.unreliability_prototype_weight * low_prototype
    score.index = clip_topk["image_path"]
    return score



def _rds_replacement_scores(
    remaining: pd.DataFrame,
    selected: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    prototypes: dict[int, np.ndarray] | None,
    cfg: SelectionConfig,
) -> pd.Series:
    quality = _minmax_normalize(remaining["target_score"])
    margin = _minmax_normalize(remaining["margin_score"])
    diversity = _diversity_gain(remaining, selected, embeddings)
    score = (
        cfg.replacement_target_weight * quality
        + cfg.replacement_margin_weight * margin
        + cfg.replacement_diversity_weight * diversity
    )
    if prototypes is not None and cfg.replacement_prototype_weight > 0:
        prototype = _prototype_alignment(remaining, embeddings, prototypes)
        score = score + cfg.replacement_prototype_weight * prototype
    return score



def _topk_redundancy(topk: pd.DataFrame, embeddings: dict[str, np.ndarray]) -> pd.Series:
    paths = topk["image_path"].tolist()
    matrix = _normalized_matrix(paths, embeddings)
    similarities = matrix @ matrix.T
    if len(paths) > 1:
        np.fill_diagonal(similarities, -np.inf)
        redundancy = similarities.max(axis=1)
    else:
        redundancy = np.zeros(len(paths), dtype=np.float32)
    return _minmax_normalize(pd.Series(redundancy, index=topk.index, dtype=float))



def _rds_diagnostic_rows(
    class_name: str,
    group: pd.DataFrame,
    selected: pd.DataFrame,
    clip_topk: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    prototypes: dict[int, np.ndarray] | None,
    cfg: SelectionConfig,
    planned_replacements: int,
    quality_floor: float,
    unreliability: pd.Series,
) -> list[dict]:
    selected_paths = set(selected["image_path"])
    clip_topk_paths = set(clip_topk["image_path"])
    removed = clip_topk.loc[~clip_topk["image_path"].isin(selected_paths)].copy()
    added = selected.loc[~selected["image_path"].isin(clip_topk_paths)].copy()
    selected_proto = _prototype_alignment(selected, embeddings, prototypes) if prototypes is not None else pd.Series(0.0, index=selected.index)
    topk_proto = _prototype_alignment(clip_topk, embeddings, prototypes) if prototypes is not None else pd.Series(0.0, index=clip_topk.index)
    rows = [
        {
            "diagnostic_type": "class_summary",
            "target_class": class_name,
            "target_label": int(group["target_label"].iloc[0]),
            "planned_replacement_count": int(planned_replacements),
            "actual_replacement_count": int(len(added)),
            "quality_floor": float(quality_floor),
            "selected_count": int(len(selected)),
            "overlap_with_clip_topk_count": int(len(selected_paths & clip_topk_paths)),
            "overlap_with_clip_topk_pct": float(len(selected_paths & clip_topk_paths) / max(1, len(clip_topk))),
            "selected_target_score_mean": float(selected["target_score"].mean()),
            "selected_margin_score_mean": float(selected["margin_score"].mean()),
            "selected_prototype_score_mean": float(selected_proto.mean()) if prototypes is not None else np.nan,
            "clip_topk_target_score_mean": float(clip_topk["target_score"].mean()),
            "clip_topk_margin_score_mean": float(clip_topk["margin_score"].mean()),
            "clip_topk_prototype_score_mean": float(topk_proto.mean()) if prototypes is not None else np.nan,
            "image_path": "",
            "target_score": np.nan,
            "margin_score": np.nan,
            "prototype_score": np.nan,
            "unreliability_score": np.nan,
        }
    ]
    for _, row in removed.iterrows():
        rows.append(_rds_sample_diagnostic("removed", class_name, row, embeddings, prototypes, unreliability.get(row["image_path"], np.nan)))
    for _, row in added.iterrows():
        rows.append(_rds_sample_diagnostic("added", class_name, row, embeddings, prototypes, np.nan))
    return rows



def _rds_sample_diagnostic(
    kind: str,
    class_name: str,
    row: pd.Series,
    embeddings: dict[str, np.ndarray],
    prototypes: dict[int, np.ndarray] | None,
    unreliability_score: float,
) -> dict:
    prototype_score = np.nan
    if prototypes is not None:
        label = int(row["target_label"])
        prototype_score = float(_normalize_vector(embeddings[row["image_path"]]) @ _normalize_vector(prototypes[label]))
    return {
        "diagnostic_type": kind,
        "target_class": class_name,
        "target_label": int(row["target_label"]),
        "planned_replacement_count": np.nan,
        "actual_replacement_count": np.nan,
        "quality_floor": np.nan,
        "selected_count": np.nan,
        "overlap_with_clip_topk_count": np.nan,
        "overlap_with_clip_topk_pct": np.nan,
        "selected_target_score_mean": np.nan,
        "selected_margin_score_mean": np.nan,
        "selected_prototype_score_mean": np.nan,
        "clip_topk_target_score_mean": np.nan,
        "clip_topk_margin_score_mean": np.nan,
        "clip_topk_prototype_score_mean": np.nan,
        "image_path": row["image_path"],
        "target_score": float(row["target_score"]),
        "margin_score": float(row["margin_score"]),
        "prototype_score": prototype_score,
        "unreliability_score": float(unreliability_score) if not pd.isna(unreliability_score) else np.nan,
    }



def _anchored_rerank_scores(
    remaining: pd.DataFrame,
    selected: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    prototypes: dict[int, np.ndarray] | None,
    cfg: SelectionConfig,
) -> pd.Series:
    quality = _minmax_normalize(remaining["target_score"])
    margin = _minmax_normalize(remaining["margin_score"])
    diversity = _diversity_gain(remaining, selected, embeddings)
    score = cfg.quality_weight * quality + cfg.margin_weight * margin + cfg.diversity_weight * diversity
    if prototypes is not None and cfg.prototype_weight > 0:
        prototype = _prototype_alignment(remaining, embeddings, prototypes)
        score = score + cfg.prototype_weight * prototype
    return score


def _prototype_alignment(
    remaining: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    prototypes: dict[int, np.ndarray],
) -> pd.Series:
    scores: dict[int, float] = {}
    for idx, row in remaining.iterrows():
        label = int(row["target_label"])
        vector = _normalize_vector(embeddings[row["image_path"]])
        prototype = _normalize_vector(prototypes[label])
        scores[idx] = float(vector @ prototype)
    return _minmax_normalize(pd.Series(scores, dtype=float))


def _diversity_gain(
    remaining: pd.DataFrame,
    selected: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> pd.Series:
    selected_matrix = _normalized_matrix(selected["image_path"].tolist(), embeddings)
    gains: dict[int, float] = {}
    for idx, image_path in remaining["image_path"].items():
        vector = _normalize_vector(embeddings[image_path])
        similarities = selected_matrix @ vector
        max_similarity = float(similarities.max())
        gains[idx] = 1.0 - max_similarity
    return _minmax_normalize(pd.Series(gains, dtype=float))


def _normalized_matrix(paths: list[str], embeddings: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([_normalize_vector(embeddings[p]) for p in paths])


def _normalize_feature_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _real_similarity(
    candidates: pd.DataFrame,
    real_matrix: np.ndarray,
    candidate_features: dict[str, np.ndarray],
) -> pd.Series:
    scores: dict[int, float] = {}
    for idx, image_path in candidates["image_path"].items():
        vector = _normalize_vector(candidate_features[image_path])
        scores[idx] = float((real_matrix @ vector).max())
    return pd.Series(scores, dtype=float)


def _feature_redundancy(
    remaining: pd.DataFrame,
    selected: pd.DataFrame,
    candidate_features: dict[str, np.ndarray],
) -> pd.Series:
    selected_matrix = _normalize_feature_matrix(np.stack([candidate_features[p] for p in selected["image_path"].tolist()]))
    scores: dict[int, float] = {}
    for idx, image_path in remaining["image_path"].items():
        vector = _normalize_vector(candidate_features[image_path])
        scores[idx] = float((selected_matrix @ vector).max())
    return pd.Series(scores, dtype=float)


def _feature_redundancy_or_zero(
    selected: pd.DataFrame,
    candidate_features: dict[str, np.ndarray],
) -> pd.Series:
    paths = selected["image_path"].tolist()
    if len(paths) <= 1:
        return pd.Series(0.0, index=selected.index, dtype=float)
    matrix = _normalize_feature_matrix(np.stack([candidate_features[p] for p in paths]))
    similarities = matrix @ matrix.T
    np.fill_diagonal(similarities, -np.inf)
    return pd.Series(similarities.max(axis=1), index=selected.index, dtype=float)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _minmax_normalize(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    min_value = values.min()
    max_value = values.max()
    if np.isclose(max_value, min_value):
        return pd.Series(0.0, index=values.index)
    return (values - min_value) / (max_value - min_value)


def _topk_by_column(scores: pd.DataFrame, column: str, k: int) -> pd.DataFrame:
    return (
        scores.sort_values(column, ascending=False)
        .groupby("target_class", group_keys=False)
        .head(k)
        .reset_index(drop=True)
    )
