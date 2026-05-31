import numpy as np
import pandas as pd

from ccds.selection import SelectionConfig, select_ccds, select_clip_topk, select_margin_topk, select_random


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
