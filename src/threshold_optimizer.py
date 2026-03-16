"""
threshold_optimizer.py
-----------------------
Post-processing: optimize per-class decision thresholds on OOF probabilities
to maximize Macro F1 score.

In standard argmax prediction, all classes compete equally.
By scaling class probabilities with learned per-class weights (thresholds),
we can boost recall on minority classes (DeepSeek, Claude) at the cost of
small precision drops on majority classes.

Usage
-----
    from src.threshold_optimizer import optimize_thresholds, apply_thresholds

    # After CV with a model that has predict_proba:
    thresholds = optimize_thresholds(oof_proba, y_true)
    preds = apply_thresholds(test_proba, thresholds)
"""

import itertools
import numpy as np
from sklearn.metrics import f1_score
from src.utils import get_logger

logger = get_logger(__name__)


def apply_thresholds(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Scale each class probability by its threshold weight and return argmax.

    Parameters
    ----------
    proba      : (n_samples, n_classes) predicted probabilities
    thresholds : (n_classes,) per-class scale factors

    Returns
    -------
    preds : (n_samples,) integer class labels
    """
    scaled = proba * thresholds  # broadcast over samples
    return np.argmax(scaled, axis=1)


def optimize_thresholds(
    proba: np.ndarray,
    y_true: np.ndarray,
    n_grid: int = 11,
    lo: float = 0.5,
    hi: float = 3.0,
) -> np.ndarray:
    """
    Grid-search per-class scale factors to maximise Macro F1 on OOF data.

    Strategy: iterate classes in order of their baseline F1 (worst first).
    For each class, sweep its scale factor while holding others fixed.
    Repeat two passes to capture interactions.

    Parameters
    ----------
    proba   : (n_samples, n_classes) OOF predicted probabilities
    y_true  : (n_samples,) true integer labels
    n_grid  : number of scale values to try per class per sweep
    lo, hi  : range of scale factor values

    Returns
    -------
    thresholds : (n_classes,) best scale factors found
    """
    n_classes  = proba.shape[1]
    thresholds = np.ones(n_classes)
    grid       = np.linspace(lo, hi, n_grid)

    # Baseline F1
    baseline_preds = np.argmax(proba, axis=1)
    per_class_f1   = f1_score(y_true, baseline_preds, average=None, zero_division=0)
    baseline_macro = f1_score(y_true, baseline_preds, average="macro", zero_division=0)
    logger.info(f"  Threshold opt: baseline macro F1 = {baseline_macro:.4f}")
    logger.info(f"  Per-class F1 before: {[round(x,3) for x in per_class_f1]}")

    # Sweep in order of worst class first (most to gain)
    class_order = np.argsort(per_class_f1)

    best_macro = baseline_macro

    for _pass in range(2):  # two passes to capture interactions
        for cls in class_order:
            best_val = thresholds[cls]
            for val in grid:
                thresholds[cls] = val
                preds = apply_thresholds(proba, thresholds)
                macro = f1_score(y_true, preds, average="macro", zero_division=0)
                if macro > best_macro:
                    best_macro = macro
                    best_val   = val
            thresholds[cls] = best_val

    final_preds    = apply_thresholds(proba, thresholds)
    final_macro    = f1_score(y_true, final_preds, average="macro", zero_division=0)
    final_per_cls  = f1_score(y_true, final_preds, average=None, zero_division=0)
    gain           = final_macro - baseline_macro

    logger.info(f"  Threshold opt: tuned  macro F1 = {final_macro:.4f}  (gain={gain:+.4f})")
    logger.info(f"  Per-class F1 after:  {[round(x,3) for x in final_per_cls]}")
    logger.info(f"  Thresholds: {[round(t,3) for t in thresholds]}")

    return thresholds
