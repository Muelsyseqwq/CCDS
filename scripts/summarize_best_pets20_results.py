from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.config import load_config, resolve_path  # noqa: E402
from ccds.utils import ensure_dir  # noqa: E402


DEFAULT_CONFIG = "configs/sweeps/pets20_160_sp80_core_realft.yaml"
DEFAULT_CLASSIFIER_ROOT = "/root/gpufree-data/clip_diffusion_fewshot_ccds_results/classifier"

BASELINE_SPECS = [
    {
        "setting": "Pets20 fair RealFT real_only",
        "project_name": "ccds_pets20_5shot_30ep_realft",
        "method": "real_only",
        "notes": "30+5 RealFT fair baseline; real images only.",
    },
    {
        "setting": "Pets20 fair RealFT traditional_aug",
        "project_name": "ccds_pets20_5shot_30ep_realft",
        "method": "traditional_aug",
        "notes": "30+5 RealFT fair baseline with traditional augmentation.",
    },
    {
        "setting": "Pets20 fair RealFT diffusion_random",
        "project_name": "ccds_pets20_5shot_30ep_realft",
        "method": "diffusion_random",
        "notes": "30+5 RealFT, 10 selected/class random synthetic baseline.",
    },
    {
        "setting": "Pets20 fair RealFT clip_topk",
        "project_name": "ccds_pets20_5shot_30ep_realft",
        "method": "clip_topk",
        "notes": "Strong 30+5 RealFT baseline, 10 selected/class.",
    },
    {
        "setting": "Pets20 fair RealFT margin_topk",
        "project_name": "ccds_pets20_5shot_30ep_realft",
        "method": "margin_topk",
        "notes": "30+5 RealFT baseline, 10 selected/class.",
    },
    {
        "setting": "Pets20 fair RealFT ccds",
        "project_name": "ccds_pets20_5shot_30ep_realft",
        "method": "ccds",
        "notes": "30+5 RealFT baseline, 10 selected/class.",
    },
    {
        "setting": "Pets20 fair RealFT anchored_ccds",
        "project_name": "ccds_pets20_5shot_30ep_realft",
        "method": "anchored_ccds",
        "notes": "30+5 RealFT baseline, 10 selected/class.",
    },
    {
        "setting": "Pets20 D7 anchored + RealFT",
        "project_name": "ccds_pets20_d7_anchor6_top60_w701515_realft",
        "method": "anchored_ccds",
        "notes": "Tuned anchored CCDS result before large-synthetic sweep.",
    },
    {
        "setting": "Pets20 160/sp60 CCDS + RealFT",
        "project_name": "ccds_pets20_160_sp60_core_realft",
        "method": "ccds",
        "notes": "Large synthetic ablation; 60 selected/class.",
    },
    {
        "setting": "Pets20 160/sp80 margin_topk + RealFT (best)",
        "project_name": "ccds_pets20_160_sp80_core_realft",
        "method": "margin_topk",
        "notes": "Current best 3-seed mean; 80 selected/class.",
    },
    {
        "setting": "Pets20 160/sp100 margin_topk + RealFT",
        "project_name": "ccds_pets20_160_sp100_core_realft",
        "method": "margin_topk",
        "notes": "Large synthetic ablation; 100 selected/class.",
    },
    {
        "setting": "Pets20 160/sp100 anchored_ccds + RealFT",
        "project_name": "ccds_pets20_160_sp100_core_realft",
        "method": "anchored_ccds",
        "notes": "Large synthetic ablation; 100 selected/class.",
    },
    {
        "setting": "Pets20 160/sp80 DINOv2 CFRD-MMR + RealFT",
        "project_name": "ccds_pets20_160_sp80_cfrd_mmr_dinov2_realft",
        "method": "cfrd_mmr",
        "notes": "Single-seed diagnostic/ablation currently available.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the best Pets20 sp80 margin_topk + RealFT result.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Best-result config path.")
    parser.add_argument("--classifier-root", default=DEFAULT_CLASSIFIER_ROOT, help="External classifier results root.")
    parser.add_argument("--output-dir", default="results/research_process", help="Project-relative output directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    classifier_root = Path(args.classifier_root)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    all_results = pd.read_csv(classifier_root / "all_results.csv")

    best_per_seed = _best_per_seed(all_results, cfg, classifier_root)
    best_summary = _summarize_group(best_per_seed).assign(
        setting="Pets20 160/sp80 margin_topk + RealFT (best)",
        candidates_per_class=int(cfg["generation"]["num_candidates_per_class"]),
        config_path=args.config,
    )
    best_summary = best_summary[_best_summary_columns()]

    vs_baselines = _vs_baselines(all_results, best_summary.iloc[0])
    tables_md = _tables_markdown(best_summary, best_per_seed, vs_baselines)

    best_summary_path = output_dir / "pets20_160_sp80_core_realft_best_summary.csv"
    per_seed_path = output_dir / "pets20_160_sp80_core_realft_per_seed.csv"
    vs_path = output_dir / "pets20_160_sp80_core_realft_vs_baselines.csv"
    md_path = output_dir / "pets20_160_sp80_core_realft_tables.md"

    best_summary.to_csv(best_summary_path, index=False)
    best_per_seed.to_csv(per_seed_path, index=False)
    vs_baselines.to_csv(vs_path, index=False)
    md_path.write_text(tables_md, encoding="utf-8")

    print(f"Wrote {best_summary_path}")
    print(f"Wrote {per_seed_path}")
    print(f"Wrote {vs_path}")
    print(f"Wrote {md_path}")
    print("\nCurrent best:")
    print(best_summary.to_string(index=False))


def _best_per_seed(all_results: pd.DataFrame, cfg: dict, classifier_root: Path) -> pd.DataFrame:
    project = str(cfg["project_name"])
    method = "margin_topk"
    seeds = [int(s) for s in cfg.get("seed_list", [0, 1, 2])]
    subset = all_results[(all_results["project_name"] == project) & (all_results["method"] == method)].copy()
    subset["seed"] = subset["seed"].astype(int)
    subset = subset[subset["seed"].isin(seeds)].sort_values("seed")
    if sorted(subset["seed"].tolist()) != seeds:
        raise ValueError(f"Expected seeds {seeds} for {project}/{method}, got {sorted(subset['seed'].tolist())}")
    subset["summary_csv"] = subset["seed"].map(
        lambda seed: str(classifier_root / project / method / f"seed{seed}" / "summary.csv")
    )
    columns = [
        "project_name",
        "method",
        "seed",
        "accuracy",
        "macro_f1",
        "best_val_accuracy",
        "epochs",
        "real_finetune_epochs",
        "real_finetune_lr",
        "train_size",
        "real_finetune_size",
        "test_size",
        "selected_per_class",
        "config_path",
        "summary_csv",
    ]
    return subset[[c for c in columns if c in subset.columns]]


def _summarize_group(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.copy()
    for column in ["accuracy", "macro_f1", "best_val_accuracy"]:
        numeric[column] = pd.to_numeric(numeric[column])
    first = numeric.iloc[0]
    return pd.DataFrame(
        [
            {
                "project_name": first["project_name"],
                "method": first["method"],
                "num_seeds": int(numeric["seed"].nunique()),
                "num_classes": int(first.get("num_classes", 20)) if "num_classes" in first else 20,
                "shot": int(first.get("shot", 5)) if "shot" in first else 5,
                "selected_per_class": int(float(first.get("selected_per_class", 0))),
                "epochs": int(float(first.get("epochs", 0))),
                "real_finetune_epochs": int(float(first.get("real_finetune_epochs", 0))),
                "real_finetune_lr": float(first.get("real_finetune_lr", 0.0)),
                "accuracy_mean": numeric["accuracy"].mean(),
                "accuracy_std": numeric["accuracy"].std(ddof=1),
                "macro_f1_mean": numeric["macro_f1"].mean(),
                "macro_f1_std": numeric["macro_f1"].std(ddof=1),
                "best_val_accuracy_mean": numeric["best_val_accuracy"].mean(),
                "train_size": int(float(first.get("train_size", 0))),
                "real_finetune_size": int(float(first.get("real_finetune_size", 0))) if "real_finetune_size" in first else 0,
                "test_size": int(float(first.get("test_size", 0))),
            }
        ]
    )


def _vs_baselines(all_results: pd.DataFrame, best: pd.Series) -> pd.DataFrame:
    rows = []
    for spec in BASELINE_SPECS:
        subset = all_results[(all_results["project_name"] == spec["project_name"]) & (all_results["method"] == spec["method"])].copy()
        if subset.empty:
            rows.append(spec | {"num_seeds": 0, "notes": spec["notes"] + " Missing in all_results."})
            continue
        summary = _summarize_group(subset).iloc[0].to_dict()
        summary.update(spec)
        summary["candidates_per_class"] = _infer_candidates_per_class(spec["project_name"])
        summary["delta_accuracy_vs_best"] = summary["accuracy_mean"] - float(best["accuracy_mean"])
        summary["delta_macro_f1_vs_best"] = summary["macro_f1_mean"] - float(best["macro_f1_mean"])
        rows.append(summary)
    df = pd.DataFrame(rows)
    columns = [
        "setting",
        "project_name",
        "method",
        "num_seeds",
        "candidates_per_class",
        "selected_per_class",
        "epochs",
        "real_finetune_epochs",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "best_val_accuracy_mean",
        "delta_accuracy_vs_best",
        "delta_macro_f1_vs_best",
        "train_size",
        "test_size",
        "notes",
    ]
    return df[[c for c in columns if c in df.columns]]


def _infer_candidates_per_class(project_name: str) -> int | str:
    if "160" in project_name:
        return 160
    if "5shot_30ep" in project_name or "fair" in project_name or "d7" in project_name:
        return 80
    return "unknown"


def _best_summary_columns() -> list[str]:
    return [
        "setting",
        "project_name",
        "method",
        "num_seeds",
        "num_classes",
        "shot",
        "candidates_per_class",
        "selected_per_class",
        "epochs",
        "real_finetune_epochs",
        "real_finetune_lr",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "best_val_accuracy_mean",
        "train_size",
        "real_finetune_size",
        "test_size",
        "config_path",
    ]


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    display = df.copy()
    display = display.fillna("")
    columns = list(display.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.10f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)



def _tables_markdown(best_summary: pd.DataFrame, per_seed: pd.DataFrame, vs_baselines: pd.DataFrame) -> str:
    best = best_summary.iloc[0]
    lines = [
        "# Pets20 160/sp80 margin_topk + RealFT tables",
        "",
        "## Best summary",
        "",
        f"- Accuracy mean: {best['accuracy_mean']:.10f}",
        f"- Macro F1 mean: {best['macro_f1_mean']:.10f}",
        f"- Seeds: {int(best['num_seeds'])}",
        "",
        "## Per-seed results",
        "",
        _markdown_table(per_seed),
        "",
        "## Baseline comparison",
        "",
        _markdown_table(vs_baselines),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
