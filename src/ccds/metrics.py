from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_accuracy": per_class.tolist(),
        "confusion_matrix": cm.tolist(),
    }
