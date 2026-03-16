"""
train.py
--------
Training logic: cross-validation loop, model comparison, learning curves,
and final fit.

Anti-leakage guarantee:
  The full feature pipeline (Preprocessor + FeatureUnion TF-IDF) is rebuilt
  and fitted INSIDE each CV fold. No vectorizer state bleeds across folds.

Outputs per model:
  - CV metrics per fold
  - OOF predictions for full-dataset error analysis
  - Learning curve data (computed after CV)
  - Overfitting plot (train vs val per fold)
  - Learning curve plot
  - Model comparison table
  - Final trained pipeline (on all training data)
"""

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import pandas as pd

from src.constants import RANDOM_SEED
from src.evaluate import (
    compute_macro_f1,
    compute_metrics,
    error_analysis,
    generate_classification_report,
    generate_confusion_matrix,
    log_fold_metrics,
    summarise_cv_results,
)
from src.features import build_feature_union
from src.models import get_model
from src.preprocess import Preprocessor
from src.utils import ensure_dir, get_logger, save_json, save_text

logger = get_logger(__name__)


# ── Pipeline factory ──────────────────────────────────────────────────────────

def build_pipeline(model_name: str, config: Dict) -> Pipeline:
    """
    Build a full sklearn Pipeline:
      Preprocessor → FeatureUnion(TF-IDF) → Classifier

    Unfitted. Must be fitted inside each CV fold to prevent leakage.
    """
    preprocessor = Preprocessor.from_config(config)
    feature_union = build_feature_union(config)
    classifier = get_model(model_name, config)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("features", feature_union),
        ("classifier", classifier),
    ])


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cross_validation(
    X: List[str],
    y: np.ndarray,
    model_name: str,
    config: Dict,
    experiment_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None,
) -> Dict:
    """
    Run StratifiedKFold CV for a single model.

    Returns results dict with per-fold metrics, OOF predictions,
    and aggregate summary.
    """
    val_cfg = config.get("validation", {})
    n_splits = val_cfg.get("n_splits", 5)
    random_state = val_cfg.get("random_state", RANDOM_SEED)
    shuffle = val_cfg.get("shuffle", True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    X_arr = np.array(X, dtype=object)
    oof_preds = np.zeros(len(y), dtype=int)
    oof_proba = None

    fold_metrics = []
    best_fold_pipeline = None
    best_fold_val_f1   = -1.0

    logger.info(f"\n{'='*65}")
    logger.info(f"  MODEL: {model_name}")
    logger.info(f"{'='*65}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y)):
        fold_start = time.time()

        X_train_fold = X_arr[train_idx].tolist()
        X_val_fold   = X_arr[val_idx].tolist()
        y_train_fold = y[train_idx]
        y_val_fold   = y[val_idx]

        # Build and fit a FRESH pipeline — critical anti-leakage step
        pipeline = build_pipeline(model_name, config)
        pipeline.fit(X_train_fold, y_train_fold)

        # Train metrics (overfitting diagnostic)
        train_preds = pipeline.predict(X_train_fold)
        train_f1    = compute_macro_f1(y_train_fold, train_preds)

        # Validation metrics
        val_preds = pipeline.predict(X_val_fold)
        val_f1    = compute_macro_f1(y_val_fold, val_preds)

        oof_preds[val_idx] = val_preds

        # Track best fold (for save_best_fold option)
        if val_f1 > best_fold_val_f1:
            best_fold_val_f1   = val_f1
            best_fold_pipeline = copy.deepcopy(pipeline)

        # Collect probabilities if available
        if hasattr(pipeline, "predict_proba"):
            try:
                fold_proba = pipeline.predict_proba(X_val_fold)
                if oof_proba is None:
                    oof_proba = np.zeros((len(y), fold_proba.shape[1]))
                oof_proba[val_idx] = fold_proba
            except Exception:
                pass

        fold_time = time.time() - fold_start
        best_mark = " *** NEW BEST FOLD ***" if val_f1 == best_fold_val_f1 else ""
        logger.info(
            f"  Fold {fold+1}/{n_splits} | "
            f"train={train_f1:.4f}  val={val_f1:.4f}  "
            f"time={fold_time:.0f}s{best_mark}"
        )

        fold_metrics.append({
            "fold": fold + 1,
            "train_macro_f1": round(float(train_f1), 4),
            "val_macro_f1":   round(float(val_f1), 4),
        })

    # ── Fold summary ──────────────────────────────────────────────────────────
    val_f1s = [m["val_macro_f1"] for m in fold_metrics]
    logger.info(f"\n  {model_name} fold results: {[round(v,4) for v in val_f1s]}")
    logger.info(f"  Best fold F1   : {best_fold_val_f1:.4f}")

    # Aggregate
    summary    = summarise_cv_results(fold_metrics)
    oof_report = generate_classification_report(y, oof_preds)
    oof_cm     = generate_confusion_matrix(y, oof_preds)
    oof_metrics = compute_metrics(y, oof_preds)

    logger.info(
        f"  {model_name} FINAL => "
        f"mean={summary['mean_val_macro_f1']:.4f}  "
        f"std={summary['std_val_macro_f1']:.4f}  "
        f"oof={oof_metrics['macro_f1']:.4f}  "
        f"overfit_gap={summary['mean_overfit_gap']:.4f}"
    )
    logger.info(f"{'='*65}\n")

    results = {
        "model_name":                model_name,
        "fold_metrics":              fold_metrics,
        "summary":                   summary,
        "oof_metrics":               oof_metrics,
        "oof_classification_report": oof_report,
        "oof_proba":                 oof_proba.tolist() if oof_proba is not None else None,
        "best_fold_val_f1":          round(float(best_fold_val_f1), 4),
    }

    # ── Per-model artifact saving ─────────────────────────────────────────────
    if experiment_dir is not None:
        model_dir = experiment_dir / model_name
        ensure_dir(model_dir)
        save_json(results, model_dir / "cv_results.json")
        save_text(oof_report, model_dir / "classification_report.txt")
        oof_cm.to_csv(model_dir / "confusion_matrix.csv")
        logger.info(f"  Saved CV artifacts to: {model_dir}")

        # Error analysis (requires probabilities)
        if oof_proba is not None:
            top_n = config.get("analysis", {}).get("top_n_errors", 50)
            err_df = error_analysis(X, y, oof_preds, oof_proba, top_n=top_n)
            err_df.to_csv(model_dir / "error_analysis.csv", index=False)

        # Save best fold model (trained on 80% data — better generalization)
        if best_fold_pipeline is not None:
            joblib.dump(best_fold_pipeline, model_dir / "best_fold_model.joblib")
            logger.info(
                f"  Best fold model saved (val_f1={best_fold_val_f1:.4f}): "
                f"{model_dir / 'best_fold_model.joblib'}"
            )

    # ── Overfitting plot ──────────────────────────────────────────────────────
    if plots_dir is not None:
        from src.plots import plot_overfitting, plot_confusion_matrix
        plot_overfitting(
            model_name, fold_metrics,
            save_path=str(plots_dir / f"overfitting_{model_name}.png"),
        )
        plot_confusion_matrix(
            oof_cm, model_name,
            save_path=str(plots_dir / f"confusion_matrix_{model_name}.png"),
        )

    return results


# ── Learning curve ────────────────────────────────────────────────────────────

def run_learning_curve(
    X: List[str],
    y: np.ndarray,
    model_name: str,
    config: Dict,
    plots_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """
    Compute and plot the learning curve for one model.
    Uses sklearn.model_selection.learning_curve with 3-fold CV.

    Returns a dict of curve data, or None if disabled in config.
    """
    lc_cfg = config.get("learning_curve", {})
    if not lc_cfg.get("enabled", True):
        return None

    logger.info(f"  Computing learning curve for: {model_name}")
    pipeline = build_pipeline(model_name, config)

    try:
        from src.plots import compute_learning_curve, plot_learning_curve
        ts, tm, ts_std, vm, vs_std = compute_learning_curve(pipeline, X, y, config)

        lc_data = {
            "train_sizes": ts.tolist(),
            "train_mean":  tm.tolist(),
            "train_std":   ts_std.tolist(),
            "val_mean":    vm.tolist(),
            "val_std":     vs_std.tolist(),
        }

        if plots_dir is not None:
            plot_learning_curve(
                model_name, ts, tm, ts_std, vm, vs_std,
                save_path=str(plots_dir / f"learning_curve_{model_name}.png"),
            )

        return lc_data

    except Exception as e:
        logger.warning(f"  Learning curve failed for {model_name}: {e}")
        return None


# ── Model comparison ──────────────────────────────────────────────────────────

def run_model_comparison(
    X: List[str],
    y: np.ndarray,
    config: Dict,
    experiment_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, str, Dict]:
    """
    Run CV + learning curves for all configured models.

    Returns (comparison_df, best_model_name, all_cv_results).
    """
    model_cfg  = config.get("models", {})
    run_models = model_cfg.get("run_models", ["logistic_regression"])

    rows         = []
    all_results  = {}
    all_lc_data  = {}

    logger.info("=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)

    for model_name in run_models:
        logger.info(f"\n[Model: {model_name}]")

        # ── Cross-validation ──────────────────────────────────────────────────
        results = run_cross_validation(
            X, y, model_name, config,
            experiment_dir=experiment_dir,
            plots_dir=plots_dir,
        )
        all_results[model_name] = results
        summary    = results["summary"]
        oof_metrics = results["oof_metrics"]

        # ── Running leaderboard ───────────────────────────────────────────────
        current_f1 = summary["mean_val_macro_f1"]
        current_best = max((r["mean_val_macro_f1"] for r in rows), default=-1)
        if current_f1 > current_best:
            logger.info(f"  >>> NEW BEST: {model_name} = {current_f1:.4f}")
        else:
            logger.info(f"  (best so far: {current_best:.4f}, this: {current_f1:.4f})")

        rows.append({
            "model":                model_name,
            "mean_val_macro_f1":    summary["mean_val_macro_f1"],
            "std_val_macro_f1":     summary["std_val_macro_f1"],
            "mean_train_macro_f1":  summary["mean_train_macro_f1"],
            "mean_overfit_gap":     summary["mean_overfit_gap"],
            "oof_macro_f1":         oof_metrics["macro_f1"],
            "oof_accuracy":         oof_metrics["accuracy"],
        })

        # ── Learning curve ────────────────────────────────────────────────────
        lc_data = run_learning_curve(X, y, model_name, config, plots_dir=plots_dir)
        if lc_data:
            all_lc_data[model_name] = lc_data

    # ── Sort and rank ─────────────────────────────────────────────────────────
    comparison_df = (
        pd.DataFrame(rows)
        .sort_values("mean_val_macro_f1", ascending=False)
        .reset_index(drop=True)
    )

    best_model_name = comparison_df.iloc[0]["model"]
    logger.info("\n" + "#"*65)
    logger.info(f"  BEST MODEL : {best_model_name}")
    logger.info(f"  CV F1      : {comparison_df.iloc[0]['mean_val_macro_f1']:.4f}")
    logger.info(f"  OOF F1     : {comparison_df.iloc[0]['oof_macro_f1']:.4f}")
    logger.info("#"*65 + "\n")
    logger.info("Full comparison:")
    logger.info("\n" + comparison_df.to_string(index=False))

    # ── Per-class threshold optimisation on OOF proba of best model ───────────
    best_oof_proba = all_results[best_model_name].get("oof_proba")
    if best_oof_proba is not None:
        logger.info(f"\nOptimizing per-class thresholds for: {best_model_name}")
        from src.threshold_optimizer import optimize_thresholds
        oof_proba_arr = np.array(best_oof_proba)
        thresholds = optimize_thresholds(oof_proba_arr, y)
        all_results["_thresholds"] = thresholds.tolist()
        all_results["_threshold_model"] = best_model_name
        # Also optimise pair-specific DS/Grok threshold
        from src.threshold_optimizer import optimize_ds_grok_threshold
        pair_thr = optimize_ds_grok_threshold(oof_proba_arr, y)
        all_results["_ds_grok_pair_threshold"] = pair_thr
        if experiment_dir is not None:
            save_json(
                {
                    "model": best_model_name,
                    "thresholds": thresholds.tolist(),
                    "ds_grok_pair_threshold": pair_thr,
                },
                experiment_dir / "thresholds.json",
            )
    else:
        logger.info(f"\n  Skipping threshold optimization ({best_model_name} has no predict_proba)")

    # ── Summary plots ─────────────────────────────────────────────────────────
    if plots_dir is not None:
        from src.plots import (
            plot_model_comparison,
            plot_all_overfitting,
            plot_all_learning_curves,
        )

        plot_model_comparison(
            comparison_df,
            save_path=str(plots_dir / "model_comparison.png"),
        )
        plot_all_overfitting(
            {k: v for k, v in all_results.items() if not k.startswith("_")},
            save_path=str(plots_dir / "overfitting_all_models.png"),
        )
        if all_lc_data:
            plot_all_learning_curves(
                all_lc_data,
                save_path=str(plots_dir / "learning_curves_all_models.png"),
            )

    return comparison_df, best_model_name, all_results


# ── Final training ────────────────────────────────────────────────────────────

def train_final_model(
    X: List[str],
    y: np.ndarray,
    model_name: str,
    config: Dict,
    save_path: Optional[str] = None,
    best_fold_pipeline: Optional[Pipeline] = None,
) -> Pipeline:
    """
    Fit a full pipeline on ALL training data.
    This is the model used for test predictions.
    """
    use_best_fold = config.get("training", {}).get("use_best_fold_model", False)

    logger.info("=" * 60)
    logger.info(f"FINAL TRAINING — model: {model_name}")
    logger.info(f"  Training samples: {len(X)}")

    if use_best_fold and best_fold_pipeline is not None:
        pipeline = best_fold_pipeline
        logger.info("  Using best CV fold model (NOT retrained on all data)")
        elapsed = 0.0
    else:
        start    = time.time()
        pipeline = build_pipeline(model_name, config)
        pipeline.fit(X, y)
        elapsed  = time.time() - start

    logger.info(f"  Final training completed in {elapsed:.1f}s")

    if save_path is not None:
        ensure_dir(Path(save_path).parent)
        joblib.dump(pipeline, save_path)
        logger.info(f"  Pipeline saved to: {save_path}")

        # ── Feature importance (training-time only, no inference overhead) ────
        fi_path = str(Path(save_path).with_suffix("")) + "_feature_importance.csv"
        _save_feature_importance(pipeline, fi_path)

    return pipeline


# ── Feature importance helper ─────────────────────────────────────────────────

def _save_feature_importance(pipeline: Pipeline, save_path: str, top_n: int = 50) -> None:
    """
    Extract top-N TF-IDF features per class from the trained LR/Ridge classifier
    and write them to a CSV file.  Training-time only — zero inference overhead.
    """
    from src.constants import LABEL_NAMES
    try:
        classifier = pipeline.named_steps.get("classifier")
        feature_union = pipeline.named_steps.get("features")

        # Unwrap TwoStageClassifier — use its 6-class base
        if hasattr(classifier, "base_clf_"):
            classifier = classifier.base_clf_

        if classifier is None or not hasattr(classifier, "coef_"):
            logger.info("  Feature importance: classifier has no coef_, skipping.")
            return

        # Build feature name list from the fitted FeatureUnion transformers
        feature_names = []
        for tname, transformer in feature_union.transformer_list:
            try:
                names = transformer.get_feature_names_out()
                feature_names.extend([f"{tname}__{n}" for n in names])
            except Exception:
                pass

        coef = classifier.coef_       # (n_classes, n_features)
        classes = getattr(classifier, "classes_", list(range(coef.shape[0])))

        rows = []
        for i, cls in enumerate(classes):
            if i >= coef.shape[0]:
                break
            cls_coef = coef[i]
            top_idx = np.argsort(cls_coef)[::-1][:top_n]
            cls_name = LABEL_NAMES[cls] if cls < len(LABEL_NAMES) else str(cls)
            for rank, idx in enumerate(top_idx):
                fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                rows.append({
                    "class_id":   int(cls),
                    "class_name": cls_name,
                    "rank":       rank + 1,
                    "feature":    fname,
                    "coefficient": round(float(cls_coef[idx]), 5),
                })

        pd.DataFrame(rows).to_csv(save_path, index=False)
        logger.info(f"  Feature importance saved to: {save_path}")
    except Exception as e:
        logger.warning(f"  Feature importance export failed: {e}")
