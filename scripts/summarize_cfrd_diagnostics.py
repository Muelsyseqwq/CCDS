from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.config import experiment_name, load_config, resolve_path


SUMMARY_COLUMNS = [
    "target_class",
    "target_label",
    "overlap_with_clip_topk_pct",
    "selected_real_sim_mean",
    "clip_topk_real_sim_mean",
    "real_sim_delta",
    "selected_redundancy_mean",
    "selected_target_score_mean",
    "clip_topk_target_score_mean",
    "target_score_delta",
    "selected_margin_score_mean",
    "clip_topk_margin_score_mean",
    "margin_score_delta",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CFRD-MMR diagnostics.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, required=True, help="Few-shot seed used for CFRD selection diagnostics.")
    parser.add_argument("--diagnostics-csv", default=None, help="Override diagnostics CSV path.")
    parser.add_argument("--output", default=None, help="Optional output CSV path for summary rows.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.diagnostics_csv:
        diagnostics_csv = resolve_path(args.diagnostics_csv)
    else:
        diagnostics_csv = resolve_path("results/research_process") / f"{experiment_name(cfg)}_cfrd_diagnostics_seed{args.seed}.csv"
    if not diagnostics_csv.exists():
        raise FileNotFoundError(f"Missing CFRD diagnostics CSV: {diagnostics_csv}")

    diagnostics = pd.read_csv(diagnostics_csv)
    summary = diagnostics[diagnostics["diagnostic_type"] == "class_summary"].copy()
    if summary.empty:
        raise ValueError(f"No class_summary rows found in {diagnostics_csv}")

    summary["real_sim_delta"] = summary["selected_real_sim_mean"] - summary["clip_topk_real_sim_mean"]
    summary["target_score_delta"] = summary["selected_target_score_mean"] - summary["clip_topk_target_score_mean"]
    summary["margin_score_delta"] = summary["selected_margin_score_mean"] - summary["clip_topk_margin_score_mean"]
    summary = summary[SUMMARY_COLUMNS].sort_values("real_sim_delta", ascending=False)

    aggregate = {
        "classes": int(len(summary)),
        "overlap_mean": float(summary["overlap_with_clip_topk_pct"].mean()),
        "overlap_min": float(summary["overlap_with_clip_topk_pct"].min()),
        "overlap_max": float(summary["overlap_with_clip_topk_pct"].max()),
        "real_sim_delta_mean": float(summary["real_sim_delta"].mean()),
        "redundancy_mean": float(summary["selected_redundancy_mean"].mean()),
        "target_score_delta_mean": float(summary["target_score_delta"].mean()),
        "margin_score_delta_mean": float(summary["margin_score_delta"].mean()),
    }

    print("=== CFRD diagnostics summary ===")
    print(f"diagnostics_csv: {diagnostics_csv}")
    for key, value in aggregate.items():
        print(f"{key}: {value}")
    print("\nTop classes by real_sim_delta:")
    print(summary.head(10).to_string(index=False))
    print("\nBottom classes by real_sim_delta:")
    print(summary.tail(10).to_string(index=False))

    if args.output:
        out_path = resolve_path(args.output)
    else:
        out_path = diagnostics_csv.with_name(diagnostics_csv.stem + "_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\nWrote summary CSV to {out_path}")


if __name__ == "__main__":
    main()
