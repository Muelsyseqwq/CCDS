import pandas as pd

from ccds.data import merge_real_and_generated


def test_merge_real_and_generated_preserves_class_metadata(tmp_path):
    real_csv = tmp_path / "real.csv"
    selected_csv = tmp_path / "selected.csv"
    out_csv = tmp_path / "merged.csv"

    pd.DataFrame(
        [
            {"image_path": "/real_a.png", "label": 0, "class_index": 10, "class_name": "alpha", "original_split": "train"},
            {"image_path": "/real_b.png", "label": 1, "class_index": 11, "class_name": "beta", "original_split": "train"},
        ]
    ).to_csv(real_csv, index=False)
    pd.DataFrame(
        [
            {"image_path": "/gen_a.png", "target_label": 0, "target_class": "alpha", "margin_score": 0.5},
            {"image_path": "/gen_b.png", "target_label": 1, "target_class": "beta", "margin_score": 0.4},
        ]
    ).to_csv(selected_csv, index=False)

    merge_real_and_generated(real_csv, selected_csv, out_csv)
    merged = pd.read_csv(out_csv)
    generated = merged[merged["source"] == "generated"].sort_values("label")

    assert generated["class_name"].tolist() == ["alpha", "beta"]
    assert generated["class_index"].tolist() == [10, 11]
    assert generated["original_split"].tolist() == ["generated", "generated"]
    assert set(merged["source"]) == {"real", "generated"}
