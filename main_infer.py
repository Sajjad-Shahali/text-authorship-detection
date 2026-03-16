"""
main_infer.py
-------------
Entry point for inference on the test set.

Usage:
    # Use the default best model saved by main_train.py:
    python main_infer.py --config configs/config.yaml

    # Use a specific model file:
    python main_infer.py --model artifacts/models/best_model.joblib

    # Use a model from a specific experiment run:
    python main_infer.py --model artifacts/experiments/2026-03-16_161401_run/best_model.joblib

    # List all available saved models across experiments:
    python main_infer.py --list-models

Saves predictions to artifacts/submissions/submission_<timestamp>.csv
and also updates artifacts/submissions/submission_latest.csv.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data import get_test_texts, load_sample_submission, load_test
from src.inference import load_pipeline, predict
from src.submission import make_submission, save_submission, validate_submission
from src.utils import (
    ensure_dir,
    get_logger,
    load_config,
    log_system_info,
    resolve_paths,
)


def list_available_models(artifacts_dir: str = "artifacts") -> None:
    """Print all .joblib model files found under the artifacts directory."""
    root = Path(artifacts_dir)
    models = sorted(root.glob("**/*.joblib"))
    if not models:
        print("No .joblib model files found under", root.resolve())
        return
    print(f"\nAvailable models under {root.resolve()}:")
    print("-" * 70)
    for m in models:
        size_mb = m.stat().st_size / 1e6
        print(f"  {m}  ({size_mb:.1f} MB)")
    print()
    print("Usage:  python main_infer.py --model <path>")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="MALTO Text Authorship Detection — Inference Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Path to a specific .joblib model file to use for inference. "
            "Overrides paths.best_model_file from config. "
            "Example: --model artifacts/experiments/2026-03-16_161401_run/best_model.joblib"
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available .joblib model files and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── List models shortcut ───────────────────────────────────────────────────
    if args.list_models:
        list_available_models()
        return

    run_start = time.time()

    # ── Load config ────────────────────────────────────────────────────────────
    config = load_config(args.config)
    config = resolve_paths(config)
    paths = config["paths"]

    logger = get_logger("main_infer")
    log_system_info(logger)
    logger.info(f"Config loaded from: {args.config}")

    # ── Ensure output directories ──────────────────────────────────────────────
    ensure_dir(paths["submissions_dir"])

    # ── Load model ─────────────────────────────────────────────────────────────
    model_path = args.model or paths["best_model_file"]
    pipeline = load_pipeline(model_path)

    # ── Load per-class thresholds (if available from threshold optimization) ───
    from src.utils import load_json
    thresholds = None
    ds_grok_pair_threshold = None
    threshold_path = Path(paths.get("artifacts_dir", "artifacts")) / "thresholds.json"
    if threshold_path.exists():
        try:
            thresh_data = load_json(str(threshold_path))
            thresholds = thresh_data.get("thresholds")
            ds_grok_pair_threshold = thresh_data.get("ds_grok_pair_threshold")
            threshold_model = thresh_data.get("model", "unknown")
            # Warn if threshold file was computed for a different model
            if args.model and threshold_model not in str(args.model):
                logger.warning(
                    f"  [THRESHOLD MISMATCH WARNING] Thresholds were computed for "
                    f"'{threshold_model}' but you are using a custom model path "
                    f"'{args.model}'. These thresholds may not match. "
                    f"Applying anyway — verify results."
                )
            if ds_grok_pair_threshold is not None:
                logger.info(
                    f"  DS/Grok pair threshold: {ds_grok_pair_threshold:.3f}"
                )
            logger.info(f"Loaded per-class thresholds from: {threshold_path}")
            logger.info(f"  Thresholds: {[round(t,3) for t in thresholds]}")
        except Exception as e:
            logger.warning(f"  Could not load thresholds: {e}")

    # ── Load test data ─────────────────────────────────────────────────────────
    test_df = load_test(paths["test_file"])
    X_test = get_test_texts(test_df)
    logger.info(f"Test samples: {len(X_test)}")

    # ── Predict ────────────────────────────────────────────────────────────────
    preds = predict(
        pipeline, X_test,
        thresholds=thresholds,
        ds_grok_pair_threshold=ds_grok_pair_threshold,
    )
    logger.info(f"Generated {len(preds)} predictions.")

    # ── Build submission ───────────────────────────────────────────────────────
    submission_df = make_submission(preds)

    # Validate against sample submission if available
    sample_path = paths.get("sample_submission_file")
    sample_df = load_sample_submission(sample_path) if sample_path else None
    validate_submission(submission_df, sample_df)

    # ── Save ───────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(Path(paths["submissions_dir"]) / f"submission_{timestamp}.csv")
    save_submission(submission_df, out_path, also_save_latest=True)

    # Also copy submission into the experiment folder for full run reproducibility
    exp_dir_file = Path(paths.get("artifacts_dir", "artifacts")) / "latest_exp_dir.txt"
    if exp_dir_file.exists():
        try:
            exp_dir = Path(exp_dir_file.read_text().strip())
            if exp_dir.exists():
                import shutil
                exp_sub = exp_dir / f"submission_{timestamp}.csv"
                shutil.copy2(out_path, exp_sub)
                logger.info(f"  Submission also saved to experiment: {exp_sub}")
        except Exception as e:
            logger.warning(f"  Could not copy submission to experiment dir: {e}")

    total_time = time.time() - run_start
    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETE")
    logger.info(f"  Predictions     : {len(preds)}")
    logger.info(f"  Output file     : {out_path}")
    logger.info(f"  Total time      : {total_time:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
