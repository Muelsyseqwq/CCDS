from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    """Return the repository root inferred from this file location."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


def experiment_name(cfg: dict[str, Any]) -> str:
    """Return the experiment identifier used to isolate artifacts."""
    return str(cfg.get("project_name") or "default_experiment")


def experiment_results_dir(cfg: dict[str, Any]) -> Path:
    """Return the result directory for one experiment."""
    return resolve_path(Path("results") / experiment_name(cfg))


def generation_metadata_path(cfg: dict[str, Any]) -> Path:
    """Return the generated-candidate metadata CSV path."""
    configured = cfg.get("generation", {}).get("metadata_csv")
    if configured:
        return resolve_path(configured)
    return experiment_results_dir(cfg) / "generation_metadata.csv"


def clip_scores_path(cfg: dict[str, Any]) -> Path:
    """Return the CLIP candidate score CSV path."""
    configured = cfg.get("clip", {}).get("score_csv")
    if configured:
        return resolve_path(configured)
    return experiment_results_dir(cfg) / "clip_scores.csv"


def clip_embeddings_path(cfg: dict[str, Any]) -> Path:
    """Return the CLIP image embeddings NPZ path."""
    configured = cfg.get("clip", {}).get("embeddings_npz")
    if configured:
        return resolve_path(configured)
    return experiment_results_dir(cfg) / "clip_image_embeddings.npz"


def selected_csv_path(cfg: dict[str, Any], strategy: str) -> Path:
    """Return the selected-candidates CSV path for a selection strategy."""
    configured = cfg.get("selection", {}).get("output_dir")
    if configured:
        return resolve_path(configured) / f"selected_{strategy}.csv"
    return experiment_results_dir(cfg) / "selected" / f"selected_{strategy}.csv"


def classifier_output_dir(cfg: dict[str, Any], method: str, seed: int) -> Path:
    """Return the classifier artifact directory for one experiment/method/seed."""
    configured = cfg.get("classifier", {}).get("output_dir", "results/classifier")
    return resolve_path(configured) / experiment_name(cfg) / method / f"seed{seed}"
