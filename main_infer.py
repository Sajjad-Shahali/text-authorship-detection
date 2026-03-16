"""
main_infer.py
-------------
Entry point for inference on the test set.

Usage:
    python main_infer.py --config configs/config.yaml

Loads the trained pipeline, runs it on test.csv,
and saves predictions to artifacts/submissions/.
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
        help="Override model path (default: paths.best_model_file from config)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
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
    threshold_path = Path(paths.get("artifacts_dir", "artifacts")) / "thresholds.json"
    if threshold_path.exists():
        try:
            thresh_data = load_json(str(threshold_path))
            thresholds = thresh_data.get("thresholds")
            logger.info(f"Loaded per-class thresholds from: {threshold_path}")
            logger.info(f"  Thresholds: {[round(t,3) for t in thresholds]}")
        except Exception as e:
            logger.warning(f"  Could not load thresholds: {e}")

    # ── Load test data ─────────────────────────────────────────────────────────
    test_df = load_test(paths["test_file"])
    X_test = get_test_texts(test_df)
    logger.info(f"Test samples: {len(X_test)}")

    # ── Predict ────────────────────────────────────────────────────────────────
    preds = predict(pipeline, X_test, thresholds=thresholds)
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

    total_time = time.time() - run_start
    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETE")
    logger.info(f"  Predictions     : {len(preds)}")
    logger.info(f"  Output file     : {out_path}")
    logger.info(f"  Total time      : {total_time:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
