from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class SelectionConfig:
    selected_per_class: int = 10
    top_m_for_diversity: int = 30
    random_state: int = 0


def select_random(scores: pd.DataFrame, cfg: SelectionConfig) -> pd.DataFrame:
    """Randomly select K generated samples per target class."""
    return (
        scores.groupby("target_class", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), cfg.selected_per_class), random_state=cfg.random_state))
        .reset_index(drop=True)
    )


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


def _topk_by_column(scores: pd.DataFrame, column: str, k: int) -> pd.DataFrame:
    return (
        scores.sort_values(column, ascending=False)
        .groupby("target_class", group_keys=False)
        .head(k)
        .reset_index(drop=True)
    )
