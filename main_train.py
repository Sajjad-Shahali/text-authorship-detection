"""
main_train.py
-------------
Entry point for the full training pipeline.

Usage:
    python main_train.py --config configs/config.yaml

Pipeline steps:
  1. Load config
  2. Set up experiment directory and logging
  3. Load and validate training data
  4. Run cross-validation and model comparison
  5. Compute learning curves (train score vs. val score vs. training size)
  6. Save overfitting plots and learning curve plots to artifacts/plots/
  7. Train final model on all data
  8. Save pipeline, metrics, and artifacts
"""

import argparse
import time
from pathlib import Path

import joblib

from src.constants import BEST_MODEL_FILE, CONFIG_SNAPSHOT_FILE, MODEL_COMPARISON_FILE, RUN_LOG_FILE
from src.data import get_texts_and_labels, load_train
from src.train import run_model_comparison, run_cross_validation, train_final_model
from src.utils import (
    ensure_dir,
    get_experiment_dir,
    get_logger,
    load_config,
    log_system_info,
    resolve_paths,
    save_config_snapshot,
    save_json,
    save_text,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MALTO Text Authorship Detection — Training Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file (default: configs/config.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_start = time.time()

    # ── Load config ────────────────────────────────────────────────────────────
    config = load_config(args.config)
    config = resolve_paths(config)

    paths          = config["paths"]
    training_cfg   = config.get("training", {})
    experiment_cfg = config.get("experiment", {})
    model_cfg      = config.get("models", {})

    # ── Set up experiment directory ────────────────────────────────────────────
    exp_dir = None
    if experiment_cfg.get("enabled", True):
        prefix  = experiment_cfg.get("name_prefix", "run")
        exp_dir = get_experiment_dir(paths["experiments_dir"], prefix=prefix)

    log_file = str(exp_dir / RUN_LOG_FILE) if exp_dir else str(
        Path(paths.get("logs_dir", "artifacts/logs")) / "run.log"
    )

    logger = get_logger("main_train", log_file=log_file)
    log_system_info(logger)
    logger.info(f"Config loaded from : {args.config}")
    logger.info(f"Experiment dir     : {exp_dir}")

    if exp_dir:
        save_config_snapshot(config, exp_dir / CONFIG_SNAPSHOT_FILE)

    # ── Ensure artifact directories exist ─────────────────────────────────────
    plots_dir = ensure_dir(paths.get("plots_dir", "artifacts/plots"))
    if exp_dir:
        # Also write plots into the experiment folder for a complete snapshot
        exp_plots_dir = ensure_dir(exp_dir / "plots")
    else:
        exp_plots_dir = plots_dir

    for dir_key in ["models_dir", "metrics_dir", "submissions_dir", "analysis_dir"]:
        ensure_dir(paths[dir_key])

    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Loading training data...")
    train_df = load_train(paths["train_file"])
    X, y     = get_texts_and_labels(train_df)
    logger.info(f"Dataset: {len(X)} samples, {len(set(y))} classes")

    # ── Model comparison + learning curves ────────────────────────────────────
    best_model_name = model_cfg.get("best_model", "logistic_regression")

    if training_cfg.get("run_cv", True) and training_cfg.get("run_model_comparison", True):
        logger.info("\nStarting model comparison (CV + learning curves)...")
        comparison_df, cv_best_name, all_cv_results = run_model_comparison(
            X, y, config,
            experiment_dir=exp_dir,
            plots_dir=exp_plots_dir,
        )

        # Save comparison table
        comp_path = str(Path(paths["metrics_dir"]) / MODEL_COMPARISON_FILE)
        comparison_df.to_csv(comp_path, index=False)
        logger.info(f"Model comparison saved to: {comp_path}")

        if exp_dir:
            comparison_df.to_csv(exp_dir / MODEL_COMPARISON_FILE, index=False)

        logger.info(f"CV best model: {cv_best_name} | Config best model: {best_model_name}")
        logger.info(f"Plots saved to: {exp_plots_dir}")

        # Copy thresholds to artifacts root so main_infer.py can find them
        if "_thresholds" in all_cv_results:
            thresh_artifact = {
                "model": all_cv_results.get("_threshold_model", best_model_name),
                "thresholds": all_cv_results["_thresholds"],
            }
            save_json(thresh_artifact, str(Path(paths["artifacts_dir"]) / "thresholds.json"))
            logger.info("  Per-class thresholds saved to: artifacts/thresholds.json")

    elif training_cfg.get("run_cv", True):
        # Single model CV only (no comparison)
        logger.info(f"\nRunning CV for model: {best_model_name}")
        cv_results = run_cross_validation(
            X, y, best_model_name, config,
            experiment_dir=exp_dir,
            plots_dir=exp_plots_dir,
        )
        cv_path = str(Path(paths["metrics_dir"]) / "cv_results.json")
        save_json(cv_results, cv_path)
        save_text(
            cv_results["oof_classification_report"],
            str(Path(paths["metrics_dir"]) / "classification_report.txt"),
        )

    # ── Final training ─────────────────────────────────────────────────────────
    model_save_path = paths["best_model_file"]
    logger.info(f"\nTraining final model: {best_model_name}")

    # Pass best fold pipeline if available (use_best_fold_model in config)
    _best_fold_pipe = None
    if training_cfg.get("use_best_fold_model", False) and "all_cv_results" in dir():
        best_cv = all_cv_results.get(best_model_name, {})
        _best_fold_path = (
            exp_dir / best_model_name / "best_fold_model.joblib"
            if exp_dir else None
        )
        if _best_fold_path and _best_fold_path.exists():
            import joblib as _jl
            _best_fold_pipe = _jl.load(_best_fold_path)
            logger.info(f"  Loaded best fold pipeline from: {_best_fold_path}")

    pipeline = train_final_model(
        X, y, best_model_name, config,
        save_path=model_save_path,
        best_fold_pipeline=_best_fold_pipe,
    )

    if exp_dir:
        joblib.dump(pipeline, exp_dir / BEST_MODEL_FILE)
        logger.info(f"Final model also saved to: {exp_dir / BEST_MODEL_FILE}")

    # ── Summary ────────────────────────────────────────────────────────────────
    total_time = time.time() - run_start
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Total time     : {total_time:.1f}s")
    logger.info(f"  Best model     : {best_model_name}")
    logger.info(f"  Model artifact : {model_save_path}")
    logger.info(f"  Plots dir      : {exp_plots_dir}")
    if exp_dir:
        logger.info(f"  Experiment dir : {exp_dir}")
    logger.info("=" * 60)
    logger.info("Next step: python main_infer.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
