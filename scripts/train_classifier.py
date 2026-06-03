from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ccds.config import classifier_output_dir, experiment_name, load_config, resolve_path, selected_csv_path
from ccds.data import CsvImageDataset, build_transforms, merge_real_and_generated
from ccds.datasets import split_base_name
from ccds.metrics import classification_metrics
from ccds.utils import ensure_dir, get_device, seed_everything, write_json


METHOD_TO_SELECTION_STRATEGY = {
    "diffusion_random": "random",
    "clip_topk": "clip_topk",
    "margin_topk": "margin_topk",
    "ccds": "ccds",
    "anchored_ccds": "anchored_ccds",
    "cfrd_mmr": "cfrd_mmr",
    "confusion_adaptive_ccds": "confusion_adaptive_ccds",
    "same_overlap_random": "same_overlap_random",
    "replacement_aware_confusion_adaptive_ccds": "replacement_aware_confusion_adaptive_ccds",
    "reliability_diverse_substitution_ccds": "reliability_diverse_substitution_ccds",
    "prototype_ccds": "prototype_ccds",
    "prototype_gated_ccds": "prototype_gated_ccds",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a few-shot classifier for one data setting.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "real_only",
            "traditional_aug",
            "diffusion_random",
            "clip_topk",
            "margin_topk",
            "ccds",
            "anchored_ccds",
            "cfrd_mmr",
            "confusion_adaptive_ccds",
            "same_overlap_random",
            "replacement_aware_confusion_adaptive_ccds",
            "reliability_diverse_substitution_ccds",
            "prototype_ccds",
            "prototype_gated_ccds",
        ],
    )
    parser.add_argument("--seed", type=int, default=None, help="Run one seed only. Defaults to all seeds in config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for quick tests.")
    parser.add_argument("--config-path-for-summary", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = [args.seed] if args.seed is not None else cfg["seed_list"]
    all_metrics = []
    for seed in seeds:
        metrics = train_one_setting(cfg, args.method, int(seed), args.epochs, args.config_path_for_summary or args.config)
        all_metrics.append(metrics)

    summary_path = resolve_path("results/classifier/all_results.csv")
    ensure_dir(summary_path.parent)
    old = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    new = pd.DataFrame(all_metrics)
    combined = pd.concat([old, new], ignore_index=True)
    combined.drop_duplicates(subset=["project_name", "method", "seed"], keep="last", inplace=True)
    combined.to_csv(summary_path, index=False)
    print(f"Updated summary: {summary_path}")


def train_one_setting(cfg: dict, method: str, seed: int, epochs_override: int | None = None, config_path: str = "") -> dict:
    seed_everything(seed)
    device = get_device()
    dataset_cfg = cfg["dataset"]
    clf_cfg = cfg["classifier"]
    num_classes = int(dataset_cfg["num_classes"])
    image_size = int(dataset_cfg["image_size"])
    shot = int(dataset_cfg["shot"])

    split_base = split_base_name(dataset_cfg, seed)
    real_train_csv = resolve_path(f"data/splits/{split_base}_train.csv")
    train_csv = real_train_csv
    val_csv = resolve_path(f"data/splits/{split_base}_val.csv")
    test_csv = resolve_path(f"data/splits/{split_base}_test.csv")
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing split file: {train_csv}. Run scripts/make_splits.py first.")

    if method in METHOD_TO_SELECTION_STRATEGY:
        strategy = METHOD_TO_SELECTION_STRATEGY[method]
        selected_csv = Path(str(selected_csv_path(cfg, strategy)).format(seed=seed))
        if strategy == "cfrd_mmr":
            seed_specific_csv = selected_csv.with_name(f"{selected_csv.stem}_seed{seed}{selected_csv.suffix}")
            if seed_specific_csv.exists():
                selected_csv = seed_specific_csv
        if not selected_csv.exists():
            raise FileNotFoundError(f"Missing selected CSV: {selected_csv}. Run scripts/select_candidates.py first.")
        merged_csv = resolve_path(f"results/merged_train/{split_base}_{experiment_name(cfg)}_{method}.csv")
        train_csv = merge_real_and_generated(
            train_csv,
            selected_csv,
            merged_csv,
            synthetic_weight=float(clf_cfg.get("synthetic_loss_weight", 1.0)),
            real_weight=float(clf_cfg.get("real_loss_weight", 1.0)),
        )

    traditional_aug = method == "traditional_aug"
    use_sample_weights = method in METHOD_TO_SELECTION_STRATEGY
    train_ds = CsvImageDataset(train_csv, build_transforms(image_size, train=True, traditional_aug=traditional_aug), return_sample_weight=use_sample_weights)
    real_finetune_ds = CsvImageDataset(real_train_csv, build_transforms(image_size, train=True), return_sample_weight=False)
    val_ds = CsvImageDataset(val_csv, build_transforms(image_size, train=False))
    test_ds = CsvImageDataset(test_csv, build_transforms(image_size, train=False))

    batch_size = int(clf_cfg["batch_size"])
    num_workers = int(clf_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    real_finetune_loader = DataLoader(real_finetune_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(num_classes, freeze_backbone=bool(clf_cfg["freeze_backbone"])).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(clf_cfg["lr"]), weight_decay=float(clf_cfg["weight_decay"]))
    criterion = nn.CrossEntropyLoss(reduction="none" if use_sample_weights else "mean")
    epochs = epochs_override or int(clf_cfg["epochs"])

    best_val_acc = -1.0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, num_classes, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix" and k != "per_class_accuracy"}})
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"[{method} seed={seed}] epoch={epoch}/{epochs} loss={train_loss:.4f} val_acc={val_metrics['accuracy']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    real_finetune_epochs = int(clf_cfg.get("real_finetune_epochs", 0))
    if real_finetune_epochs > 0:
        finetune_lr = float(clf_cfg.get("real_finetune_lr", clf_cfg["lr"]))
        finetune_optimizer = torch.optim.AdamW(params, lr=finetune_lr, weight_decay=float(clf_cfg["weight_decay"]))
        finetune_criterion = nn.CrossEntropyLoss()
        for epoch in range(1, real_finetune_epochs + 1):
            train_loss = train_epoch(model, real_finetune_loader, finetune_criterion, finetune_optimizer, device)
            val_metrics = evaluate(model, val_loader, num_classes, device)
            history.append(
                {
                    "epoch": epochs + epoch,
                    "stage": "real_finetune",
                    "train_loss": train_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix" and k != "per_class_accuracy"},
                }
            )
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(
                f"[{method} seed={seed}] real_finetune epoch={epoch}/{real_finetune_epochs} "
                f"loss={train_loss:.4f} val_acc={val_metrics['accuracy']:.4f}"
            )
        if best_state is not None:
            model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, num_classes, device)

    out_dir = ensure_dir(classifier_output_dir(cfg, method, seed))
    torch.save(model.state_dict(), out_dir / "model.pt")
    write_json({"history": history, "test": test_metrics}, out_dir / "metrics.json")

    flat = {
        "project_name": experiment_name(cfg),
        "config_path": config_path,
        "num_classes": num_classes,
        "shot": shot,
        "selected_per_class": int(cfg.get("selection", {}).get("selected_per_class", 0)),
        "method": method,
        "seed": seed,
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "best_val_accuracy": best_val_acc,
        "epochs": epochs,
        "real_finetune_epochs": int(clf_cfg.get("real_finetune_epochs", 0)),
        "real_finetune_lr": float(clf_cfg.get("real_finetune_lr", 0.0)),
        "train_size": len(train_ds),
        "real_finetune_size": len(real_finetune_ds),
        "test_size": len(test_ds),
        "real_loss_weight": float(clf_cfg.get("real_loss_weight", 1.0)),
        "synthetic_loss_weight": float(clf_cfg.get("synthetic_loss_weight", 1.0)),
    }
    pd.DataFrame([flat]).to_csv(out_dir / "summary.csv", index=False)
    print(f"Saved metrics to {out_dir}")
    return flat


def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for batch in tqdm(loader, desc="train", leave=False):
        if len(batch) == 3:
            images, labels, sample_weights = batch
            sample_weights = sample_weights.to(device)
        else:
            images, labels = batch
            sample_weights = None
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss_values = criterion(logits, labels)
        if sample_weights is not None:
            loss = (loss_values * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
        else:
            loss = loss_values
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total += images.size(0)
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, num_classes: int, device) -> dict:
    model.eval()
    y_true = []
    y_pred = []
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())
    return classification_metrics(y_true, y_pred, num_classes)


if __name__ == "__main__":
    main()
