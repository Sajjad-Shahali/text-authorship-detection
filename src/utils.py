"""
utils.py
--------
Shared utilities: logging, config loading, path management,
artifact saving, and system diagnostics.
"""

import json
import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes to stdout and optionally to a file.
    Calling this multiple times with the same name returns the same logger
    (handlers are not duplicated).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — force UTF-8 on Windows to avoid cp1252 encode errors
    import io
    stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") \
        if hasattr(sys.stdout, "buffer") else sys.stdout
    ch = logging.StreamHandler(stdout_utf8)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        ensure_dir(Path(log_file).parent)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file and return as nested dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config_snapshot(config: Dict[str, Any], dest_path: str) -> None:
    """Save a copy of the current config for experiment reproducibility."""
    ensure_dir(Path(dest_path).parent)
    with open(dest_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    If running in Kaggle mode, override data/artifact paths.
    Returns the config with paths adjusted.
    """
    env = config.get("environment", {})
    if env.get("mode") == "kaggle":
        kaggle_input = env.get("kaggle_input_dir", "/kaggle/input/malto-hackathon")
        kaggle_work = env.get("kaggle_working_dir", "/kaggle/working")
        config["paths"]["data_dir"] = kaggle_input
        config["paths"]["train_file"] = f"{kaggle_input}/train.csv"
        config["paths"]["test_file"] = f"{kaggle_input}/test.csv"
        config["paths"]["sample_submission_file"] = f"{kaggle_input}/sample_submission.csv"
        config["paths"]["artifacts_dir"] = kaggle_work
        config["paths"]["models_dir"] = f"{kaggle_work}/models"
        config["paths"]["metrics_dir"] = f"{kaggle_work}/metrics"
        config["paths"]["submissions_dir"] = f"{kaggle_work}/submissions"
        config["paths"]["analysis_dir"] = f"{kaggle_work}/analysis"
        config["paths"]["experiments_dir"] = f"{kaggle_work}/experiments"
        config["paths"]["logs_dir"] = f"{kaggle_work}/logs"
        config["paths"]["best_model_file"] = f"{kaggle_work}/models/best_model.joblib"
    return config


# ── File system ───────────────────────────────────────────────────────────────

def ensure_dir(path) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_experiment_dir(base_dir: str, prefix: str = "run") -> Path:
    """
    Create and return a timestamped experiment directory.
    Example: artifacts/experiments/2026-03-16_143022_run/
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    exp_name = f"{timestamp}_{prefix}"
    exp_dir = Path(base_dir) / exp_name
    ensure_dir(exp_dir)
    return exp_dir


# ── Serialisation ─────────────────────────────────────────────────────────────

def save_json(data: Any, path: str) -> None:
    """Save dict/list as pretty-printed JSON."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, path: str) -> None:
    """Write a text string to file."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ── Diagnostics ───────────────────────────────────────────────────────────────

def log_system_info(logger: logging.Logger) -> None:
    """Log Python and key library versions for reproducibility."""
    import sklearn
    import numpy as np
    import pandas as pd

    logger.info("=" * 60)
    logger.info("SYSTEM INFO")
    logger.info(f"  Python      : {sys.version.split()[0]}")
    logger.info(f"  Platform    : {platform.platform()}")
    logger.info(f"  NumPy       : {np.__version__}")
    logger.info(f"  Pandas      : {pd.__version__}")
    logger.info(f"  scikit-learn: {sklearn.__version__}")
    logger.info("=" * 60)
