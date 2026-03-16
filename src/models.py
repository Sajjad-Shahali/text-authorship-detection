"""
models.py
---------
Model registry and factory.

All classifiers are sklearn-compatible and work with sparse TF-IDF matrices.
Regularisation is applied by default to prevent overfitting.

Available models
----------------
Linear / margin-based (best for high-dim sparse text):
  logistic_regression         — LR, uniform class weights
  logistic_regression_balanced— LR, class_weight='balanced' (helps imbalanced data)
  linear_svc                  — LinearSVC, uniform weights
  linear_svc_balanced         — LinearSVC, class_weight='balanced'
  calibrated_svc              — LinearSVC wrapped in CalibratedClassifierCV (gives proba)
  sgd_log                     — SGDClassifier log-loss (fast LR equiv, balanced weights)
  sgd_hinge                   — SGDClassifier hinge-loss (fast SVM equiv, balanced weights)
  ridge_classifier            — RidgeClassifier (fast, good baseline)
  passive_aggressive          — PassiveAggressiveClassifier

Probabilistic / generative:
  complement_nb               — ComplementNB (good for imbalanced text classes)
"""

from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from src.utils import get_logger


class SeedAveragingClassifier(BaseEstimator, ClassifierMixin):
    """
    Run the same base_estimator N times with different random seeds,
    then average their predicted probabilities (soft bagging).

    Reduces variance on minority classes like DeepSeek without changing
    the underlying model family. Works well when the base estimator is
    non-deterministic (LR with lbfgs is deterministic, so use SGD-based
    models or vary other params).
    """

    def __init__(self, base_estimator, seeds=(42, 123, 456, 789, 2024)):
        self.base_estimator = base_estimator
        self.seeds = seeds

    def fit(self, X, y):
        self.estimators_ = []
        self.classes_ = np.unique(y)
        for s in self.seeds:
            est = clone(self.base_estimator)
            try:
                est.set_params(random_state=s)
            except ValueError:
                pass  # estimator doesn't have random_state
            est.fit(X, y)
            self.estimators_.append(est)
        return self

    def predict_proba(self, X):
        probas = []
        for est in self.estimators_:
            if hasattr(est, "predict_proba"):
                probas.append(est.predict_proba(X))
            else:
                # Fallback: one-hot from predict
                p = est.predict(X)
                oh = np.zeros((len(p), len(self.classes_)))
                for i, c in enumerate(p):
                    oh[i, c] = 1.0
                probas.append(oh)
        return np.mean(probas, axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

logger = get_logger(__name__)

AVAILABLE_MODELS = [
    "logistic_regression",
    "logistic_regression_balanced",
    "linear_svc",
    "linear_svc_balanced",
    "calibrated_svc",
    "sgd_log",
    "sgd_hinge",
    "ridge_classifier",
    "passive_aggressive",
    "complement_nb",
    "ensemble_top3",      # VotingClassifier hard: sgd_hinge + ridge + lgr_balanced
    "ensemble_soft",      # VotingClassifier soft: calibrated models (has predict_proba)
    "lr_seed_avg",        # LR balanced, averaged over 5 random seeds (stable minority recall)
]


def get_model(name: str, config: Dict) -> BaseEstimator:
    """
    Factory — return an unfitted sklearn estimator by name.

    Parameters
    ----------
    name   : one of AVAILABLE_MODELS
    config : full project config dict (models section is used)

    Returns
    -------
    Unfitted sklearn estimator
    """
    model_cfg = config.get("models", {})
    seed = config.get("training", {}).get("random_state", 42)

    # ── Logistic Regression ──────────────────────────────────────────────────
    if name == "logistic_regression":
        cfg = model_cfg.get("logistic_regression", {})
        return LogisticRegression(
            C=cfg.get("C", 1.0),
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight=cfg.get("class_weight", None),
            random_state=seed,
        )

    if name == "logistic_regression_balanced":
        cfg = model_cfg.get("logistic_regression_balanced", {})
        return LogisticRegression(
            C=cfg.get("C", 1.0),
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight="balanced",
            random_state=seed,
        )

    # ── LinearSVC ────────────────────────────────────────────────────────────
    if name == "linear_svc":
        cfg = model_cfg.get("linear_svc", {})
        return LinearSVC(
            C=cfg.get("C", 0.1),
            max_iter=cfg.get("max_iter", 2000),
            class_weight=cfg.get("class_weight", None),
            random_state=seed,
        )

    if name == "linear_svc_balanced":
        cfg = model_cfg.get("linear_svc_balanced", {})
        return LinearSVC(
            C=cfg.get("C", 0.1),
            max_iter=cfg.get("max_iter", 2000),
            class_weight="balanced",
            random_state=seed,
        )

    # ── CalibratedClassifierCV (LinearSVC + Platt scaling) ───────────────────
    if name == "calibrated_svc":
        cfg = model_cfg.get("calibrated_svc", {})
        base = LinearSVC(
            C=cfg.get("C", 0.1),
            max_iter=cfg.get("max_iter", 2000),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=seed,
        )
        return CalibratedClassifierCV(base, cv=3, method="sigmoid")

    # ── SGD Classifier ────────────────────────────────────────────────────────
    if name == "sgd_log":
        cfg = model_cfg.get("sgd_log", {})
        return SGDClassifier(
            loss="log_loss",
            alpha=cfg.get("alpha", 1e-4),
            max_iter=cfg.get("max_iter", 100),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    if name == "sgd_hinge":
        cfg = model_cfg.get("sgd_hinge", {})
        return SGDClassifier(
            loss="hinge",
            alpha=cfg.get("alpha", 1e-4),
            max_iter=cfg.get("max_iter", 100),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    # ── Ridge Classifier ──────────────────────────────────────────────────────
    if name == "ridge_classifier":
        cfg = model_cfg.get("ridge_classifier", {})
        return RidgeClassifier(
            alpha=cfg.get("alpha", 1.0),
            class_weight=cfg.get("class_weight", "balanced"),
        )

    # ── Passive Aggressive (via SGD — PAC deprecated in sklearn 1.8) ──────────
    if name == "passive_aggressive":
        cfg = model_cfg.get("passive_aggressive", {})
        # Equivalent to PassiveAggressiveClassifier(C=...) per sklearn 1.8 migration guide
        return SGDClassifier(
            loss="hinge",
            penalty=None,
            learning_rate="pa1",
            eta0=cfg.get("C", 0.1),
            max_iter=cfg.get("max_iter", 1000),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=seed,
            n_jobs=-1,
        )

    # ── ComplementNB ──────────────────────────────────────────────────────────
    if name == "complement_nb":
        cfg = model_cfg.get("complement_nb", {})
        return ComplementNB(alpha=cfg.get("alpha", 0.1))

    # ── Ensemble: Top-3 hard voting ───────────────────────────────────────────
    if name == "ensemble_top3":
        # Combines sgd_hinge + ridge_classifier + logistic_regression_balanced
        # Hard voting (majority) — works even when models lack predict_proba
        cfg_sgd = model_cfg.get("sgd_hinge", {})
        cfg_ridge = model_cfg.get("ridge_classifier", {})
        cfg_lr = model_cfg.get("logistic_regression_balanced", {})

        estimators = [
            ("sgd_hinge", SGDClassifier(
                loss="hinge", alpha=cfg_sgd.get("alpha", 1e-4),
                max_iter=cfg_sgd.get("max_iter", 100),
                class_weight="balanced", random_state=seed, n_jobs=-1,
            )),
            ("ridge", RidgeClassifier(
                alpha=cfg_ridge.get("alpha", 1.0),
                class_weight="balanced",
            )),
            ("lr_balanced", LogisticRegression(
                C=cfg_lr.get("C", 1.0),
                max_iter=cfg_lr.get("max_iter", 1000),
                solver=cfg_lr.get("solver", "lbfgs"),
                class_weight="balanced",
                random_state=seed,
            )),
        ]
        return VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1)

    # ── Ensemble: Soft voting with calibrated models (has predict_proba) ─────────
    if name == "ensemble_soft":
        # Soft voting uses predicted probability averages — better for minority classes.
        # SGD and Ridge don't have predict_proba natively, so wrap in CalibratedClassifierCV.
        cfg_sgd   = model_cfg.get("sgd_hinge", {})
        cfg_ridge = model_cfg.get("ridge_classifier", {})
        cfg_lr    = model_cfg.get("logistic_regression_balanced", {})
        cfg_svc   = model_cfg.get("calibrated_svc", {})

        cal_sgd = CalibratedClassifierCV(
            SGDClassifier(
                loss="hinge", alpha=cfg_sgd.get("alpha", 1e-4),
                max_iter=cfg_sgd.get("max_iter", 100),
                class_weight="balanced", random_state=seed, n_jobs=-1,
            ),
            cv=3, method="isotonic",
        )
        cal_ridge = CalibratedClassifierCV(
            RidgeClassifier(
                alpha=cfg_ridge.get("alpha", 1.0),
                class_weight="balanced",
            ),
            cv=3, method="isotonic",
        )
        lr = LogisticRegression(
            C=cfg_lr.get("C", 1.0),
            max_iter=cfg_lr.get("max_iter", 1000),
            solver=cfg_lr.get("solver", "lbfgs"),
            class_weight="balanced",
            random_state=seed,
        )
        # CalibratedClassifierCV on LinearSVC — isotonic handles multiclass better
        cal_svc = CalibratedClassifierCV(
            LinearSVC(
                C=cfg_svc.get("C", 0.1),
                max_iter=cfg_svc.get("max_iter", 2000),
                class_weight="balanced",
                random_state=seed,
            ),
            cv=3, method="isotonic",
        )
        estimators = [
            ("cal_sgd",   cal_sgd),
            ("cal_ridge", cal_ridge),
            ("lr",        lr),
            ("cal_svc",   cal_svc),
        ]
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)

    # ── Seed-averaging LR (5 seeds, averaged probabilities) ──────────────────────
    if name == "lr_seed_avg":
        cfg = model_cfg.get("logistic_regression_balanced", {})
        base_lr = LogisticRegression(
            C=cfg.get("C", 0.5),
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight="balanced",
            random_state=seed,
        )
        return SeedAveragingClassifier(
            base_estimator=base_lr,
            seeds=(42, 123, 456, 789, 2024),
        )

    raise ValueError(f"Unknown model '{name}'. Available: {AVAILABLE_MODELS}")


def get_all_models(config: Dict) -> Dict[str, BaseEstimator]:
    """Return dict of all models listed in config.models.run_models."""
    run_models = config.get("models", {}).get("run_models", AVAILABLE_MODELS)
    return {name: get_model(name, config) for name in run_models}
