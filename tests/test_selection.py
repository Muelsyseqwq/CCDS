import numpy as np
import pandas as pd
import pytest

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
    select_replacement_aware_confusion_adaptive_ccds,
    select_same_overlap_random,
)


def _scores():
    rows = []
    for label, class_name in enumerate(["a", "b"]):
        for i in range(4):
            rows.append(
                {
                    "image_path": f"/{class_name}_{i}.png",
                    "target_class": class_name,
                    "target_label": label,
                    "target_score": float(i),
                    "margin_score": float(4 - i),
                }
            )
    return pd.DataFrame(rows)


def test_topk_selectors_limit_each_class():
    scores = _scores()
    cfg = SelectionConfig(selected_per_class=2, top_m_for_diversity=3)

    assert select_random(scores, cfg).groupby("target_class").size().max() == 2
    assert select_clip_topk(scores, cfg).groupby("target_class").size().max() == 2
    assert select_margin_topk(scores, cfg).groupby("target_class").size().max() == 2


def test_ccds_limits_each_class():
    scores = _scores()
    embeddings = {path: np.array([idx, idx + 1], dtype=np.float32) for idx, path in enumerate(scores["image_path"])}
    selected = select_ccds(scores, embeddings, SelectionConfig(selected_per_class=2, top_m_for_diversity=3))

    counts = selected.groupby("target_class").size()
    assert counts.max() <= 2
    assert set(counts.index) == {"a", "b"}


def test_anchored_ccds_keeps_clip_topk_anchors_and_limits_each_class():
    scores = _scores()
    embeddings = {path: np.array([idx, idx + 1], dtype=np.float32) for idx, path in enumerate(scores["image_path"])}
    cfg = SelectionConfig(selected_per_class=3, top_m_for_diversity=4, anchor_count=2)

    selected = select_anchored_ccds(scores, embeddings, cfg)

    counts = selected.groupby("target_class").size()
    assert counts.max() <= 3
    assert set(counts.index) == {"a", "b"}
    for class_name, group in scores.groupby("target_class"):
        expected_anchors = set(group.sort_values("target_score", ascending=False).head(2)["image_path"])
        selected_paths = set(selected[selected["target_class"] == class_name]["image_path"])
        assert expected_anchors <= selected_paths


def test_confusion_adaptive_ccds_uses_fewer_anchors_for_low_margin_class():
    rows = []
    for label, class_name in enumerate(["easy", "hard"]):
        for i in range(5):
            margin = float(i + 10) if class_name == "easy" else float(4 - i)
            rows.append(
                {
                    "image_path": f"/{class_name}_{i}.png",
                    "target_class": class_name,
                    "target_label": label,
                    "target_score": float(i),
                    "margin_score": margin,
                }
            )
    scores = pd.DataFrame(rows)
    embeddings = {path: np.array([idx, idx + 1], dtype=np.float32) for idx, path in enumerate(scores["image_path"])}
    cfg = SelectionConfig(
        selected_per_class=4,
        top_m_for_diversity=5,
        adaptive_anchor_min_count=1,
        adaptive_anchor_max_count=3,
        quality_weight=0.0,
        margin_weight=1.0,
        diversity_weight=0.0,
    )

    selected = select_confusion_adaptive_ccds(scores, embeddings, cfg)

    counts = selected.groupby("target_class").size()
    assert counts.max() <= 4
    easy_selected = set(selected[selected["target_class"] == "easy"]["image_path"])
    hard_selected = set(selected[selected["target_class"] == "hard"]["image_path"])
    easy_top3 = set(scores[scores["target_class"] == "easy"].sort_values("target_score", ascending=False).head(3)["image_path"])
    hard_top3 = set(scores[scores["target_class"] == "hard"].sort_values("target_score", ascending=False).head(3)["image_path"])
    assert easy_top3 <= easy_selected
    assert not hard_top3 <= hard_selected


def test_adaptive_anchor_counts_use_fixed_value_when_min_equals_max():
    scores = _scores()
    cfg = SelectionConfig(selected_per_class=3, anchor_count=2, adaptive_anchor_min_count=1, adaptive_anchor_max_count=1)

    counts = _adaptive_anchor_counts(scores, cfg)

    assert counts == {"a": 1, "b": 1}


def test_adaptive_anchor_counts_fall_back_to_anchor_count_when_margins_tie():
    scores = _scores().assign(margin_score=1.0)
    cfg = SelectionConfig(selected_per_class=3, anchor_count=2, adaptive_anchor_min_count=1, adaptive_anchor_max_count=3)

    counts = _adaptive_anchor_counts(scores, cfg)

    assert counts == {"a": 2, "b": 2}


def test_adaptive_anchor_counts_reject_invalid_range():
    scores = _scores()
    cfg = SelectionConfig(adaptive_anchor_min_count=3, adaptive_anchor_max_count=1)

    with pytest.raises(ValueError, match="adaptive_anchor_min_count"):
        _adaptive_anchor_counts(scores, cfg)


def test_same_overlap_random_matches_reference_overlap_and_is_deterministic():
    scores = _scores()
    reference = select_clip_topk(scores, SelectionConfig(selected_per_class=2)).copy()
    reference = reference[reference["image_path"] != "/a_3.png"]
    reference = pd.concat([reference, scores[scores["image_path"] == "/a_1.png"]], ignore_index=True)
    cfg = SelectionConfig(selected_per_class=2, top_m_for_diversity=4, random_state=7)

    selected_a = select_same_overlap_random(scores, reference, cfg)
    selected_b = select_same_overlap_random(scores, reference, cfg)

    assert selected_a["image_path"].tolist() == selected_b["image_path"].tolist()
    for class_name, group in scores.groupby("target_class"):
        clip_topk = set(group.sort_values("target_score", ascending=False).head(2)["image_path"])
        ref_overlap = len(set(reference[reference["target_class"] == class_name]["image_path"]) & clip_topk)
        selected_overlap = len(set(selected_a[selected_a["target_class"] == class_name]["image_path"]) & clip_topk)
        assert selected_overlap == ref_overlap


def test_cfrd_mmr_filters_by_clip_then_prefers_real_similarity_and_diversity():
    scores = pd.DataFrame(
        [
            {"image_path": "/a0.png", "target_class": "a", "target_label": 0, "target_score": 0.9, "margin_score": 0.0},
            {"image_path": "/a1.png", "target_class": "a", "target_label": 0, "target_score": 0.8, "margin_score": 0.0},
            {"image_path": "/a2.png", "target_class": "a", "target_label": 0, "target_score": 0.7, "margin_score": 0.0},
            {"image_path": "/a3.png", "target_class": "a", "target_label": 0, "target_score": 0.1, "margin_score": 0.0},
        ]
    )
    real_features = {0: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}
    candidate_features = {
        "/a0.png": np.array([1.0, 0.0], dtype=np.float32),
        "/a1.png": np.array([0.99, 0.01], dtype=np.float32),
        "/a2.png": np.array([0.0, 1.0], dtype=np.float32),
        "/a3.png": np.array([0.0, 1.0], dtype=np.float32),
    }
    cfg = SelectionConfig(selected_per_class=2, cfrd_clip_top_m=3, cfrd_real_weight=0.7, cfrd_diversity_weight=0.3)

    selected = select_cfrd_mmr(scores, real_features, candidate_features, cfg)

    assert selected["image_path"].tolist() == ["/a0.png", "/a2.png"]
    assert "/a3.png" not in set(selected["image_path"])



def test_replacement_aware_ca_uses_more_replacements_for_low_margin_class():
    rows = []
    for label, class_name in enumerate(["easy", "hard"]):
        for i in range(8):
            margin = float(i + 10) if class_name == "easy" else float(7 - i)
            rows.append(
                {
                    "image_path": f"/{class_name}_{i}.png",
                    "target_class": class_name,
                    "target_label": label,
                    "target_score": float(i),
                    "margin_score": margin,
                }
            )
    scores = pd.DataFrame(rows)
    embeddings = {path: np.array([idx, idx + 1], dtype=np.float32) for idx, path in enumerate(scores["image_path"])}
    cfg = SelectionConfig(
        selected_per_class=4,
        top_m_for_diversity=8,
        replacement_min_count=1,
        replacement_max_count=3,
        quality_weight=0.0,
        margin_weight=1.0,
        diversity_weight=0.0,
    )

    counts = _adaptive_replacement_counts(scores, cfg)
    selected = select_replacement_aware_confusion_adaptive_ccds(scores, embeddings, cfg)

    assert counts["hard"] > counts["easy"]
    for class_name, group in scores.groupby("target_class"):
        clip_topk = set(group.sort_values("target_score", ascending=False).head(4)["image_path"])
        selected_paths = set(selected[selected["target_class"] == class_name]["image_path"])
        non_topk_count = len(selected_paths - clip_topk)
        assert non_topk_count == counts[class_name]


def test_prototype_ccds_keeps_anchors_and_uses_prototypes():
    scores = _scores()
    embeddings = {path: np.array([idx, idx + 1], dtype=np.float32) for idx, path in enumerate(scores["image_path"])}
    prototypes = {0: np.array([1.0, 1.0], dtype=np.float32), 1: np.array([1.0, 1.0], dtype=np.float32)}
    cfg = SelectionConfig(
        selected_per_class=3,
        top_m_for_diversity=4,
        anchor_count=2,
        quality_weight=0.6,
        margin_weight=0.1,
        prototype_weight=0.2,
        diversity_weight=0.1,
    )

    selected = select_prototype_ccds(scores, embeddings, prototypes, cfg)

    counts = selected.groupby("target_class").size()
    assert counts.max() <= 3
    assert set(counts.index) == {"a", "b"}
    for class_name, group in scores.groupby("target_class"):
        expected_anchors = set(group.sort_values("target_score", ascending=False).head(2)["image_path"])
        selected_paths = set(selected[selected["target_class"] == class_name]["image_path"])
        assert expected_anchors <= selected_paths


def test_prototype_gated_ccds_keeps_anchors_and_limits_candidates():
    scores = _scores()
    embeddings = {path: np.array([idx, idx + 1], dtype=np.float32) for idx, path in enumerate(scores["image_path"])}
    prototypes = {0: np.array([1.0, 1.0], dtype=np.float32), 1: np.array([1.0, 1.0], dtype=np.float32)}
    cfg = SelectionConfig(
        selected_per_class=3,
        top_m_for_diversity=4,
        prototype_top_m=1,
        anchor_count=2,
        quality_weight=0.6,
        margin_weight=0.2,
        diversity_weight=0.2,
    )

    selected = select_prototype_gated_ccds(scores, embeddings, prototypes, cfg)

    counts = selected.groupby("target_class").size()
    assert counts.max() <= 3
    assert set(counts.index) == {"a", "b"}
    for class_name, group in scores.groupby("target_class"):
        expected_anchors = set(group.sort_values("target_score", ascending=False).head(2)["image_path"])
        selected_paths = set(selected[selected["target_class"] == class_name]["image_path"])
        assert expected_anchors <= selected_paths
