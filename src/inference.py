"""
inference.py
------------
Load a trained pipeline and generate predictions on new text.

Predictions are always integer labels in [0, 5].
"""

from pathlib import Path
from typing import List, Union

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from src.constants import MAX_LABEL, MIN_LABEL
from src.utils import get_logger

logger = get_logger(__name__)


def load_pipeline(path: str) -> Pipeline:
    """
    Load a serialised sklearn Pipeline from disk.

    Parameters
    ----------
    path : str
        Path to the .joblib file saved during training.

    Returns
    -------
    sklearn.pipeline.Pipeline (fitted)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    logger.info(f"Loading pipeline from: {path}")
    pipeline = joblib.load(path)
    logger.info(f"  Pipeline steps: {[name for name, _ in pipeline.steps]}")
    return pipeline


def predict(
    pipeline: Pipeline,
    X_text: List[str],
    thresholds: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """
    Run inference and return integer label predictions.

    Parameters
    ----------
    pipeline   : fitted sklearn Pipeline
    X_text     : list of raw text strings
    thresholds : optional (n_classes,) per-class scale factors from threshold_optimizer.
                 If provided AND the pipeline has predict_proba, thresholds are applied.

    Returns
    -------
    np.ndarray of int, shape (n_samples,), values in [0, 5]
    """
    if not X_text:
        raise ValueError("predict() received empty input list.")

    logger.info(f"Running inference on {len(X_text)} samples...")

    # Apply per-class thresholds when available (improves minority-class recall)
    if thresholds is not None and hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_text)
            from src.threshold_optimizer import apply_thresholds
            preds = apply_thresholds(proba, np.array(thresholds)).astype(int)
            logger.info("  Applied per-class thresholds to predictions.")
        except Exception as e:
            logger.warning(f"  Threshold application failed ({e}), falling back to predict()")
            preds = pipeline.predict(X_text).astype(int)
    else:
        preds = pipeline.predict(X_text).astype(int)

    # Sanity check: labels must be in valid range
    out_of_range = preds[(preds < MIN_LABEL) | (preds > MAX_LABEL)]
    if len(out_of_range) > 0:
        raise ValueError(
            f"Inference produced {len(out_of_range)} labels outside "
            f"[{MIN_LABEL}, {MAX_LABEL}]: {np.unique(out_of_range).tolist()}"
        )

    unique, counts = np.unique(preds, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    logger.info(f"  Prediction distribution: {dist}")

    return preds


def predict_proba(
    pipeline: Pipeline,
    X_text: List[str],
) -> Union[np.ndarray, None]:
    """
    Return class probabilities if the pipeline supports predict_proba.
    Returns None if the model does not support it (e.g., LinearSVC).
    """
    if hasattr(pipeline, "predict_proba"):
        try:
            return pipeline.predict_proba(X_text)
        except Exception as e:
            logger.warning(f"predict_proba failed: {e}")
    return None
