"""
evaluate.py
-----------
Evaluation utilities: metrics, classification reports, confusion matrices,
and error analysis.

Evaluation is always against Macro F1 as the primary metric.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.constants import LABEL_NAMES, TRAIN_TEXT_COL
from src.utils import get_logger, save_text

logger = get_logger(__name__)


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Macro F1 Score — the primary competition metric."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute a full metrics dict including:
    - macro_f1, accuracy
    - per-class precision, recall, f1

    Returns a JSON-serialisable dict.
    """
    macro_f1 = compute_macro_f1(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    # Per-class F1 scores
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        if i < len(per_class_f1):
            per_class[name] = round(float(per_class_f1[i]), 4)

    metrics = {
        "macro_f1": round(float(macro_f1), 4),
        "accuracy": round(float(acc), 4),
        "per_class_f1": per_class,
    }
    return metrics


# ── Classification report ─────────────────────────────────────────────────────

def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """
    Generate sklearn classification report as a formatted string.
    """
    if target_names is None:
        # Use only the label names that appear in y_true or y_pred
        all_labels = sorted(set(y_true) | set(y_pred))
        target_names = [LABEL_NAMES[i] for i in all_labels if i < len(LABEL_NAMES)]

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )
    return report


# ── Confusion matrix ──────────────────────────────────────────────────────────

def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Return a confusion matrix as a labeled DataFrame.
    Rows = True labels, Columns = Predicted labels.
    """
    labels = sorted(set(y_true) | set(y_pred))
    label_names = [LABEL_NAMES[i] if i < len(LABEL_NAMES) else str(i) for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=label_names, columns=label_names)
    df_cm.index.name = "True \\ Pred"
    return df_cm


# ── Error analysis ────────────────────────────────────────────────────────────

def error_analysis(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: Optional[np.ndarray],
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Identify and return the most-confused validation examples.

    Returns a DataFrame with columns:
    - text (truncated)
    - true_label, true_name
    - pred_label, pred_name
    - max_confidence (if proba available)
    - correct (bool)
    """
    df = pd.DataFrame({
        "text": [t[:300] for t in texts],  # truncate for readability
        "true_label": y_true,
        "true_name": [LABEL_NAMES[i] if i < len(LABEL_NAMES) else str(i) for i in y_true],
        "pred_label": y_pred,
        "pred_name": [LABEL_NAMES[i] if i < len(LABEL_NAMES) else str(i) for i in y_pred],
        "correct": y_true == y_pred,
    })

    if proba is not None:
        df["max_confidence"] = proba.max(axis=1)

    # Focus on misclassified examples, sorted by confidence (wrong answers with high confidence first)
    errors = df[~df["correct"]].copy()

    if proba is not None and "max_confidence" in errors.columns:
        errors = errors.sort_values("max_confidence", ascending=False)

    return errors.head(top_n).reset_index(drop=True)


# ── Fold-level diagnostics ────────────────────────────────────────────────────

def log_fold_metrics(
    fold: int,
    train_f1: float,
    val_f1: float,
    logger: logging.Logger,
) -> None:
    """Log per-fold train/val F1 and flag potential overfitting."""
    gap = train_f1 - val_f1
    flag = " [!] HIGH OVERFIT GAP" if gap > 0.10 else ""
    logger.info(
        f"  Fold {fold+1}: train_f1={train_f1:.4f}  val_f1={val_f1:.4f}"
        f"  gap={gap:+.4f}{flag}"
    )


def summarise_cv_results(fold_metrics: List[Dict]) -> Dict:
    """
    Compute mean ± std of CV metrics across folds.
    Logs a warning if std(val_f1) is high.
    """
    val_f1s = [m["val_macro_f1"] for m in fold_metrics]
    train_f1s = [m["train_macro_f1"] for m in fold_metrics]

    mean_val = float(np.mean(val_f1s))
    std_val = float(np.std(val_f1s))
    mean_train = float(np.mean(train_f1s))

    if std_val > 0.05:
        logger.warning(
            f"High CV variance detected: std(val_f1)={std_val:.4f}. "
            "Model may be unstable or data may be heterogeneous."
        )

    summary = {
        "mean_val_macro_f1": round(mean_val, 4),
        "std_val_macro_f1": round(std_val, 4),
        "mean_train_macro_f1": round(mean_train, 4),
        "mean_overfit_gap": round(mean_train - mean_val, 4),
        "fold_val_f1s": [round(f, 4) for f in val_f1s],
        "fold_train_f1s": [round(f, 4) for f in train_f1s],
    }
    return summary
