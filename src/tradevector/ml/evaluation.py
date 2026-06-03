"""ML evaluation metrics."""

from typing import Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> dict:
    metrics = {
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] >= 2:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1], multi_class="ovr"))
        except ValueError:
            metrics["auc"] = 0.0
    return metrics


def compare_to_baseline(
    model_metrics: dict,
    baseline_metrics: dict,
) -> dict:
    comparison = {}
    for key in model_metrics:
        if key in baseline_metrics and baseline_metrics[key] != 0:
            comparison[key] = float(model_metrics[key] / baseline_metrics[key])
    return comparison
