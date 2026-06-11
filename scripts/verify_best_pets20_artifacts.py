from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.config import (  # noqa: E402
    clip_embeddings_path,
    clip_scores_path,
    generation_metadata_path,
    load_config,
    resolve_path,
    selected_csv_path,
)
from ccds.datasets import split_base_name  # noqa: E402
from ccds.utils import ensure_dir  # noqa: E402


DEFAULT_CONFIG = "configs/sweeps/pets20_160_sp80_core_realft.yaml"
DEFAULT_CLASSIFIER_ROOT = "/root/gpufree-data/clip_diffusion_fewshot_ccds_results/classifier"
EXPECTED_ACCURACY_MEAN = 0.9093050647820965
EXPECTED_MACRO_F1_MEAN = 0.9086824557438357
TOLERANCE = 1e-8


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify artifacts for the best Pets20 sp80 margin_topk + RealFT result.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Best-result config path.")
    parser.add_argument("--classifier-root", default=DEFAULT_CLASSIFIER_ROOT, help="External classifier results root.")
    parser.add_argument("--write-report", action="store_true", help="Write a verification report under results/research_process.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    verifier = BestResultVerifier(cfg=cfg, config_path=Path(args.config), classifier_root=Path(args.classifier_root))
    report_lines = verifier.run()

    text = "\n".join(report_lines)
    print(text)
    if args.write_report:
        out_path = resolve_path("results/research_process/pets20_160_sp80_core_realft_verification.txt")
        ensure_dir(out_path.parent)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote verification report: {out_path}")

    if verifier.failures:
        raise SystemExit(1)


class BestResultVerifier:
    def __init__(self, cfg: dict, config_path: Path, classifier_root: Path) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.classifier_root = classifier_root
        self.project_name = str(cfg["project_name"])
        self.method = "margin_topk"
        self.seeds = [int(s) for s in cfg.get("seed_list", [0, 1, 2])]
        self.selected_per_class = int(cfg["selection"]["selected_per_class"])
        self.num_classes = int(cfg["dataset"]["num_classes"])
        self.candidates_per_class = int(cfg["generation"]["num_candidates_per_class"])
        self.expected_metadata_rows = self.num_classes * self.candidates_per_class
        self.expected_selected_rows = self.num_classes * self.selected_per_class
        self.expected_real_rows = self.num_classes * int(cfg["dataset"]["shot"])
        self.expected_merged_rows = self.expected_real_rows + self.expected_selected_rows
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.lines: list[str] = []

    def run(self) -> list[str]:
        self.lines.append("# Best Pets20 result artifact verification")
        self.lines.append("")
        self.lines.append(f"Project: {self.project_name}")
        self.lines.append(f"Method: {self.method}")
        self.lines.append(f"Config: {self.config_path}")
        self.lines.append(f"Classifier root: {self.classifier_root}")
        self.lines.append("")

        self._check_core_artifacts()
        per_seed = self._check_seed_artifacts()
        self._check_all_results(per_seed)

        self.lines.append("")
        if self.failures:
            self.lines.append("Verification: FAIL")
            self.lines.extend(f"FAIL: {msg}" for msg in self.failures)
        else:
            self.lines.append("Verification: PASS")
        if self.warnings:
            self.lines.append("")
            self.lines.append("Warnings:")
            self.lines.extend(f"WARN: {msg}" for msg in self.warnings)
        return self.lines

    def _check_core_artifacts(self) -> None:
        metadata = generation_metadata_path(self.cfg)
        scores = clip_scores_path(self.cfg)
        embeddings = clip_embeddings_path(self.cfg)
        selected = selected_csv_path(self.cfg, self.method)

        self._require_file(resolve_path(self.config_path), "config")
        self._require_csv_rows(metadata, self.expected_metadata_rows, "generation metadata")
        self._require_csv_rows(scores, self.expected_metadata_rows, "CLIP scores")
        self._require_file(embeddings, "CLIP embeddings")
        selected_df = self._require_csv_rows(selected, self.expected_selected_rows, "selected margin_topk")
        if selected_df is not None:
            self._check_selected_counts(selected_df)

    def _check_seed_artifacts(self) -> pd.DataFrame:
        rows = []
        for seed in self.seeds:
            split_base = split_base_name(self.cfg["dataset"], seed)
            for split in ["train", "val", "test"]:
                self._require_file(resolve_path(f"data/splits/{split_base}_{split}.csv"), f"split seed{seed} {split}")

            merged = resolve_path(f"results/merged_train/{split_base}_{self.project_name}_{self.method}.csv")
            merged_df = self._require_csv_rows(merged, self.expected_merged_rows, f"merged train seed{seed}")
            if merged_df is not None and "source" in merged_df.columns:
                counts = merged_df["source"].value_counts().to_dict()
                if counts.get("real", 0) != self.expected_real_rows or counts.get("generated", 0) != self.expected_selected_rows:
                    self.failures.append(
                        f"merged train seed{seed} source counts expected real={self.expected_real_rows}, "
                        f"generated={self.expected_selected_rows}, got {counts}"
                    )

            run_dir = self.classifier_root / self.project_name / self.method / f"seed{seed}"
            summary = run_dir / "summary.csv"
            metrics = run_dir / "metrics.json"
            model = run_dir / "model.pt"
            self._require_file(metrics, f"metrics seed{seed}")
            self._require_file(model, f"model checkpoint seed{seed}")
            summary_df = self._require_csv_rows(summary, 1, f"summary seed{seed}")
            if summary_df is None:
                continue
            row = summary_df.iloc[0].to_dict()
            self._check_summary_row(row, seed, summary)
            rows.append(row | {"summary_csv": str(summary)})

        per_seed = pd.DataFrame(rows)
        if len(per_seed) == len(self.seeds):
            self._check_means(per_seed)
        return per_seed

    def _check_all_results(self, per_seed: pd.DataFrame) -> None:
        all_results = self.classifier_root / "all_results.csv"
        all_df = self._require_csv_rows(all_results, None, "classifier all_results")
        if all_df is None:
            return
        subset = all_df[(all_df["project_name"] == self.project_name) & (all_df["method"] == self.method)]
        seeds = sorted(int(s) for s in subset["seed"].tolist()) if not subset.empty else []
        if seeds != self.seeds:
            self.failures.append(f"all_results expected seeds {self.seeds}, got {seeds}")
        if len(subset) == len(self.seeds):
            self._check_means(subset, source="all_results")

    def _check_selected_counts(self, selected_df: pd.DataFrame) -> None:
        if "target_class" not in selected_df.columns:
            self.failures.append("selected CSV missing target_class column")
            return
        counts = selected_df.groupby("target_class").size()
        if len(counts) != self.num_classes:
            self.failures.append(f"selected class count expected {self.num_classes}, got {len(counts)}")
        if counts.min() != self.selected_per_class or counts.max() != self.selected_per_class:
            self.failures.append(
                f"selected per-class count expected exactly {self.selected_per_class}, "
                f"got min={counts.min()}, max={counts.max()}"
            )

    def _check_summary_row(self, row: dict, seed: int, path: Path) -> None:
        expected = {
            "project_name": self.project_name,
            "method": self.method,
            "seed": seed,
            "selected_per_class": self.selected_per_class,
            "epochs": int(self.cfg["classifier"]["epochs"]),
            "real_finetune_epochs": int(self.cfg["classifier"].get("real_finetune_epochs", 0)),
            "train_size": self.expected_merged_rows,
        }
        for key, expected_value in expected.items():
            actual = row.get(key)
            try:
                actual_value = int(float(actual)) if key != "project_name" and key != "method" else actual
            except (TypeError, ValueError):
                actual_value = actual
            if actual_value != expected_value:
                self.failures.append(f"{path} expected {key}={expected_value}, got {actual}")

    def _check_means(self, df: pd.DataFrame, source: str = "seed summaries") -> None:
        acc_mean = float(pd.to_numeric(df["accuracy"]).mean())
        f1_mean = float(pd.to_numeric(df["macro_f1"]).mean())
        self.lines.append(f"{source} accuracy_mean={acc_mean:.10f}, macro_f1_mean={f1_mean:.10f}")
        if abs(acc_mean - EXPECTED_ACCURACY_MEAN) > TOLERANCE:
            self.failures.append(f"{source} accuracy mean expected {EXPECTED_ACCURACY_MEAN}, got {acc_mean}")
        if abs(f1_mean - EXPECTED_MACRO_F1_MEAN) > TOLERANCE:
            self.failures.append(f"{source} macro F1 mean expected {EXPECTED_MACRO_F1_MEAN}, got {f1_mean}")

    def _require_file(self, path: Path, label: str) -> bool:
        path = Path(path)
        if not path.exists():
            self.failures.append(f"missing {label}: {path}")
            return False
        self.lines.append(f"OK {label}: {path}")
        return True

    def _require_csv_rows(self, path: Path, expected_rows: int | None, label: str) -> pd.DataFrame | None:
        path = Path(path)
        if not self._require_file(path, label):
            return None
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - report path-specific IO/parsing errors.
            self.failures.append(f"failed reading {label}: {path}: {exc}")
            return None
        self.lines.append(f"OK {label} rows={len(df)}")
        if expected_rows is not None and len(df) != expected_rows:
            self.failures.append(f"{label} expected {expected_rows} rows, got {len(df)}: {path}")
        return df


if __name__ == "__main__":
    main()
