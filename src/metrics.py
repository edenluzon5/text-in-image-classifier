"""Classification metrics with bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np

POS = "alcohol"


def binary_metrics(y_true, y_pred, pos: str = POS) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == pos) & (y_pred == pos)).sum())
    fp = int(((y_true != pos) & (y_pred == pos)).sum())
    fn = int(((y_true == pos) & (y_pred != pos)).sum())
    tn = int(((y_true != pos) & (y_pred != pos)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else 0.0,
    }


def bootstrap_ci(y_true, y_pred, n: int = 2000, seed: int = 0, pos: str = POS) -> dict:
    """95% percentile bootstrap intervals for precision, recall, F1, accuracy."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in ("precision", "recall", "f1", "accuracy")}
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        m = binary_metrics(y_true[idx], y_pred[idx], pos)
        for k in samples:
            samples[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in samples.items()}


def precision_at_prevalence(recall: float, fpr: float, prevalence: float) -> float:
    """Precision if alcohol made up ``prevalence`` of the images.

    Recall and the false-positive rate do not depend on how common alcohol is.
    Precision does: this dataset is half alcohol, a real ad stream is not.

    precision = p * recall / (p * recall + (1 - p) * fpr)
    """
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError(f"prevalence must be between 0 and 1, got {prevalence}")
    numerator = prevalence * recall
    denominator = numerator + (1.0 - prevalence) * fpr
    return numerator / denominator if denominator else 0.0
