from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.config import resolve_path
from ccds.visualize import plot_results_bar, plot_score_distributions, summarize_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot result figures.")
    parser.add_argument("--results", default="results/classifier/all_results.csv")
    parser.add_argument("--scores", default="results/clip_scores.csv")
    args = parser.parse_args()

    results = resolve_path(args.results)
    if results.exists():
        summary_path = resolve_path("results/classifier/summary_by_method.csv")
        summarize_results(results, summary_path)
        plot_results_bar(results, resolve_path("figures/main_results_accuracy.png"), metric="accuracy")
        plot_results_bar(results, resolve_path("figures/main_results_macro_f1.png"), metric="macro_f1")
        print("Wrote results/classifier/summary_by_method.csv")
        print("Wrote figures/main_results_accuracy.png")
        print("Wrote figures/main_results_macro_f1.png")
    scores = resolve_path(args.scores)
    if scores.exists():
        plot_score_distributions(scores, resolve_path("figures/score_distributions"))
        print("Wrote score distribution figures")


if __name__ == "__main__":
    main()
