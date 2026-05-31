from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw

from .utils import ensure_dir


def plot_score_distributions(score_csv: str | Path, out_dir: str | Path) -> None:
    df = pd.read_csv(score_csv)
    out_dir = ensure_dir(out_dir)
    for column in ["target_score", "confuser_score", "margin_score"]:
        if column not in df.columns:
            continue
        plt.figure(figsize=(7, 4))
        sns.histplot(df[column], bins=40, kde=True)
        plt.title(column)
        plt.tight_layout()
        plt.savefig(out_dir / f"{column}_distribution.png", dpi=200)
        plt.close()


def make_image_grid(image_paths: list[str], out_path: str | Path, title: str = "", thumb_size: int = 160, cols: int = 5) -> None:
    if not image_paths:
        return
    imgs = [Image.open(p).convert("RGB").resize((thumb_size, thumb_size)) for p in image_paths]
    rows = (len(imgs) + cols - 1) // cols
    title_h = 30 if title else 0
    canvas = Image.new("RGB", (cols * thumb_size, rows * thumb_size + title_h), "white")
    if title:
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), title, fill="black")
    for i, img in enumerate(imgs):
        x = (i % cols) * thumb_size
        y = (i // cols) * thumb_size + title_h
        canvas.paste(img, (x, y))
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    canvas.save(out_path)


def summarize_results(results_csv: str | Path, out_csv: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    if "project_name" not in df.columns:
        df["project_name"] = "default_experiment"
    summary = (
        df.groupby(["project_name", "method"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            num_seeds=("seed", "nunique"),
        )
        .fillna({"accuracy_std": 0.0, "macro_f1_std": 0.0})
    )
    if out_csv is not None:
        out_csv = Path(out_csv)
        ensure_dir(out_csv.parent)
        summary.to_csv(out_csv, index=False)
    return summary


def plot_results_bar(results_csv: str | Path, out_path: str | Path, metric: str = "accuracy") -> None:
    summary = summarize_results(results_csv)
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary.columns:
        return
    summary = summary.copy()
    summary["label"] = summary["project_name"] + "\n" + summary["method"]
    plt.figure(figsize=(max(8, 0.8 * len(summary)), 4))
    ax = sns.barplot(data=summary, x="label", y=mean_col, errorbar=None)
    ax.errorbar(x=range(len(summary)), y=summary[mean_col], yerr=summary[std_col], fmt="none", c="black", capsize=3)
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=200)
    plt.close()
