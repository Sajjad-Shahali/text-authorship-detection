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

# Approximate class distribution for the MALTO dataset (2 400 training samples)
# Used to compute custom class weights with a DeepSeek-boost multiplier.
_KNOWN_CLASS_DIST = {0: 1520, 1: 80, 2: 160, 3: 80, 4: 240, 5: 320}


def _compute_deepseek_boost_weights(boost_factor: float = 2.0) -> dict:
    """
    Class weights = balanced + extra boost for DeepSeek (class 1).
    Balanced weight for class i  =  n_total / (n_classes * n_i).
    DeepSeek weight is then multiplied by boost_factor (default 2x).
    """
    total = sum(_KNOWN_CLASS_DIST.values())
    n_classes = len(_KNOWN_CLASS_DIST)
    weights = {
        cls: total / (n_classes * count)
        for cls, count in _KNOWN_CLASS_DIST.items()
    }
    weights[1] = weights[1] * boost_factor  # DeepSeek extra boost
    return weights


class TwoStageClassifier(ClassifierMixin, BaseEstimator):
    """
    Two-stage classifier that sharpens the DeepSeek / Grok boundary.

    Stage 1  — full 6-class classifier trained on all data.
    Stage 2  — binary DeepSeek-vs-Grok classifier trained only on those
               two classes with a higher-resolution LR decision boundary.

    Parameters
    ----------
    base_classifier    : sklearn estimator for stage-1 (default: LR balanced C=0.5)
    binary_classifier  : sklearn estimator for stage-2 (default: LR C=1.0, no balance)
    margin_trigger_gap : float or None.
        If set, stage-2 is only invoked when
        |P(DeepSeek) - P(Grok)| < margin_trigger_gap (the base is uncertain).
        None = always invoke stage-2 for any DS/Grok prediction.
        Recommended: 0.4 — prevents over-reclassification of clear Grok samples.
    binary_ds_threshold : float (default 0.5).
        Minimum P(DS | binary) required to predict DeepSeek.
        Values > 0.5 make the binary stage more conservative about predicting DS.
    """

    _estimator_type = "classifier"   # explicit so VotingClassifier accepts it

    DEEPSEEK = 1
    GROK = 2

    def __init__(
        self,
        base_classifier=None,
        binary_classifier=None,
        margin_trigger_gap=None,
        binary_ds_threshold=0.5,
        top2_trigger=False,
    ):
        self.base_classifier = base_classifier
        self.binary_classifier = binary_classifier
        self.margin_trigger_gap = margin_trigger_gap
        self.binary_ds_threshold = binary_ds_threshold
        self.top2_trigger = top2_trigger

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # ── Stage 1: 6-class base ────────────────────────────────────────────
        if self.base_classifier is not None:
            self.base_clf_ = clone(self.base_classifier)
        else:
            self.base_clf_ = LogisticRegression(
                C=0.5, max_iter=1000, solver="lbfgs",
                class_weight="balanced", random_state=42,
            )
        self.base_clf_.fit(X, y)

        # ── Stage 2: binary DeepSeek vs Grok ────────────────────────────────
        mask = (y == self.DEEPSEEK) | (y == self.GROK)
        if mask.sum() < 4:
            logger.warning("TwoStageClassifier: <4 DS/Grok samples — binary stage disabled")
            self.binary_clf_ = None
            return self

        X_bin = X[mask]
        y_bin = y[mask]

        if self.binary_classifier is not None:
            self.binary_clf_ = clone(self.binary_classifier)
        else:
            # Natural class weights (no balancing): reflects 1:2 DS:Grok ratio,
            # so the binary is Grok-biased when uncertain — avoids false DS predictions.
            self.binary_clf_ = LogisticRegression(
                C=1.0, max_iter=1000, solver="lbfgs",
                class_weight=None, random_state=42,
            )
        self.binary_clf_.fit(X_bin, y_bin)
        return self

    def _compute_trigger_mask(self, X, preds, base_proba=None):
        """Return boolean mask of samples to refine with stage-2.

        When multiple trigger conditions are set (top2_trigger + margin_trigger_gap),
        ALL conditions must be true (AND logic) — this makes the trigger maximally
        selective and avoids over-correction.
        """
        base_ambig = (preds == self.DEEPSEEK) | (preds == self.GROK)
        if base_proba is None:
            return base_ambig

        conditions = [base_ambig]

        if self.top2_trigger:
            # Only refine when DS and Grok are BOTH in the top-2 predicted classes.
            sorted_idx = np.argsort(base_proba, axis=1)[:, -2:]  # top-2 class indices
            top2_are_ds_grok = np.all(
                np.isin(sorted_idx, [self.DEEPSEEK, self.GROK]), axis=1
            )
            conditions.append(top2_are_ds_grok)

        if self.margin_trigger_gap is not None:
            # Only refine when |P(DS) - P(Grok)| < gap (base model is genuinely uncertain)
            margin = np.abs(base_proba[:, self.DEEPSEEK] - base_proba[:, self.GROK])
            conditions.append(margin < self.margin_trigger_gap)

        # All conditions must hold — AND them together
        result = conditions[0]
        for cond in conditions[1:]:
            result = result & cond
        return result

    def _apply_binary(self, X_ambig, bin_proba=None):
        """Run binary stage and return class predictions respecting ds_threshold."""
        if bin_proba is None:
            if hasattr(self.binary_clf_, "predict_proba"):
                bin_proba = self.binary_clf_.predict_proba(X_ambig)
            else:
                return self.binary_clf_.predict(X_ambig)

        bin_classes = self.binary_clf_.classes_
        if self.DEEPSEEK not in bin_classes or self.GROK not in bin_classes:
            return self.binary_clf_.predict(X_ambig)

        ds_pos = int(np.where(bin_classes == self.DEEPSEEK)[0][0])
        # Only predict DS when binary confidence >= threshold; else Grok
        return np.where(bin_proba[:, ds_pos] >= self.binary_ds_threshold, self.DEEPSEEK, self.GROK)

    def predict(self, X):
        base_proba = None
        need_proba = self.margin_trigger_gap is not None or self.top2_trigger
        if hasattr(self.base_clf_, "predict_proba") and need_proba:
            base_proba = self.base_clf_.predict_proba(X)
            preds = np.argmax(base_proba, axis=1).copy()
        else:
            preds = self.base_clf_.predict(X).copy()

        if self.binary_clf_ is None:
            return preds

        mask = self._compute_trigger_mask(X, preds, base_proba)
        if mask.any():
            preds[mask] = self._apply_binary(X[mask])
        return preds

    def predict_proba(self, X):
        if not hasattr(self.base_clf_, "predict_proba"):
            raise AttributeError("Base classifier has no predict_proba")

        proba = self.base_clf_.predict_proba(X).copy()  # (n, 6)

        if self.binary_clf_ is None or not hasattr(self.binary_clf_, "predict_proba"):
            return proba

        preds = np.argmax(proba, axis=1)
        mask = self._compute_trigger_mask(X, preds, proba)

        if mask.any():
            bin_proba = self.binary_clf_.predict_proba(X[mask])
            bin_classes = self.binary_clf_.classes_
            ds_pos = int(np.where(bin_classes == self.DEEPSEEK)[0][0]) if self.DEEPSEEK in bin_classes else None
            gk_pos = int(np.where(bin_classes == self.GROK)[0][0]) if self.GROK in bin_classes else None

            if ds_pos is not None and gk_pos is not None:
                indices = np.where(mask)[0]
                for i, idx in enumerate(indices):
                    total = proba[idx, self.DEEPSEEK] + proba[idx, self.GROK]
                    if total > 0:
                        proba[idx, self.DEEPSEEK] = total * bin_proba[i, ds_pos]
                        proba[idx, self.GROK]     = total * bin_proba[i, gk_pos]
        return proba


class SeedAveragingClassifier(ClassifierMixin, BaseEstimator):
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
    "ensemble_top3",           # VotingClassifier hard: sgd_hinge + ridge + lgr_balanced
    "ensemble_soft",           # VotingClassifier soft: calibrated models (has predict_proba)
    "lr_seed_avg",             # LR balanced, averaged over 5 random seeds (stable minority recall)
    "lr_deepseek_boost",       # LR with DeepSeek class weight 2x beyond balanced
    "two_stage_lr",            # Two-stage: 6-class base + binary DeepSeek/Grok sub-classifier
    "two_stage_conservative",  # Two-stage with margin trigger + conservative DS threshold
    "two_stage_top2",          # Two-stage: top-2 trigger (DS+Grok must be top-2)
    "two_stage_top2_conservative",  # top2 trigger + natural binary weights (less Grok->DS)
    "two_stage_combined",      # top2 AND margin<0.30 (AND logic, most selective trigger)
    "ensemble_v2",             # Soft vote: two_stage_top2 + ensemble_soft
    "ensemble_two_stage",      # Soft vote: ensemble_soft + two_stage_conservative + lr_deepseek_boost
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

    # ── DeepSeek-boosted LR (2x weight on top of balanced) ───────────────────────
    if name == "lr_deepseek_boost":
        cfg = model_cfg.get("logistic_regression_balanced", {})
        boost = model_cfg.get("deepseek_boost", 2.0)
        cw = _compute_deepseek_boost_weights(boost)
        return LogisticRegression(
            C=cfg.get("C", 0.5),
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight=cw,
            random_state=seed,
        )

    # ── Two-stage: 6-class base + binary DeepSeek/Grok specialist ─────────────
    if name == "two_stage_lr":
        cfg = model_cfg.get("logistic_regression_balanced", {})
        base_lr = LogisticRegression(
            C=cfg.get("C", 0.5),
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight="balanced",
            random_state=seed,
        )
        binary_lr = LogisticRegression(
            C=min(cfg.get("C", 0.5) * 4, 4.0),   # less regularisation for 2-class
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight="balanced",
            random_state=seed,
        )
        return TwoStageClassifier(
            base_classifier=base_lr,
            binary_classifier=binary_lr,
        )

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

    # ── Two-stage conservative (margin trigger + natural binary weights) ─────────
    if name == "two_stage_conservative":
        cfg = model_cfg.get("logistic_regression_balanced", {})
        base_lr = LogisticRegression(
            C=cfg.get("C", 0.5),
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight="balanced",
            random_state=seed,
        )
        # Binary: natural class weights (Grok-biased 2:1) + more regularisation
        binary_lr = LogisticRegression(
            C=0.8,
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight=None,   # natural 1:2 DS:Grok ratio — avoids false DS predictions
            random_state=seed,
        )
        return TwoStageClassifier(
            base_classifier=base_lr,
            binary_classifier=binary_lr,
            margin_trigger_gap=model_cfg.get("two_stage_margin_gap", 0.40),
            binary_ds_threshold=model_cfg.get("two_stage_ds_threshold", 0.52),
        )

    # ── Ensemble: soft vote of (ensemble_soft + two_stage_conservative + lr_deepseek_boost)
    if name == "ensemble_two_stage":
        # Each component brings a different view of the DS/Grok boundary:
        #   ensemble_soft        — best overall calibration, best Grok recall
        #   two_stage_conservative— margin-triggered DS specialist
        #   lr_deepseek_boost    — nudges DS probabilities up across the board
        cfg_lr  = model_cfg.get("logistic_regression_balanced", {})
        cfg_sgd = model_cfg.get("sgd_hinge", {})
        cfg_rg  = model_cfg.get("ridge_classifier", {})
        cfg_svc = model_cfg.get("calibrated_svc", {})
        boost   = model_cfg.get("deepseek_boost", 2.0)
        gap     = model_cfg.get("two_stage_margin_gap", 0.40)
        ds_thr  = model_cfg.get("two_stage_ds_threshold", 0.52)

        # Rebuild ensemble_soft inline
        cal_sgd = CalibratedClassifierCV(
            SGDClassifier(
                loss="hinge", alpha=cfg_sgd.get("alpha", 1e-4),
                max_iter=cfg_sgd.get("max_iter", 100),
                class_weight="balanced", random_state=seed, n_jobs=-1,
            ),
            cv=3, method="isotonic",
        )
        cal_ridge = CalibratedClassifierCV(
            RidgeClassifier(alpha=cfg_rg.get("alpha", 1.0), class_weight="balanced"),
            cv=3, method="isotonic",
        )
        lr_soft = LogisticRegression(
            C=cfg_lr.get("C", 0.5), max_iter=cfg_lr.get("max_iter", 1000),
            solver=cfg_lr.get("solver", "lbfgs"), class_weight="balanced",
            random_state=seed,
        )
        cal_svc = CalibratedClassifierCV(
            LinearSVC(C=cfg_svc.get("C", 0.1), max_iter=cfg_svc.get("max_iter", 2000),
                      class_weight="balanced", random_state=seed),
            cv=3, method="isotonic",
        )
        soft_ensemble = VotingClassifier(
            [("cal_sgd", cal_sgd), ("cal_ridge", cal_ridge),
             ("lr", lr_soft), ("cal_svc", cal_svc)],
            voting="soft", n_jobs=1,
        )

        # Two-stage conservative
        two_stage_cons = TwoStageClassifier(
            base_classifier=LogisticRegression(
                C=cfg_lr.get("C", 0.5), max_iter=cfg_lr.get("max_iter", 1000),
                solver=cfg_lr.get("solver", "lbfgs"), class_weight="balanced",
                random_state=seed,
            ),
            binary_classifier=LogisticRegression(
                C=0.8, max_iter=cfg_lr.get("max_iter", 1000),
                solver=cfg_lr.get("solver", "lbfgs"), class_weight=None,
                random_state=seed,
            ),
            margin_trigger_gap=gap,
            binary_ds_threshold=ds_thr,
        )

        # LR DeepSeek boost
        lr_ds_boost = LogisticRegression(
            C=cfg_lr.get("C", 0.5), max_iter=cfg_lr.get("max_iter", 1000),
            solver=cfg_lr.get("solver", "lbfgs"),
            class_weight=_compute_deepseek_boost_weights(boost),
            random_state=seed,
        )

        return VotingClassifier(
            [
                ("ensemble_soft",       soft_ensemble),
                ("two_stage_cons",      two_stage_cons),
                ("lr_deepseek_boost",   lr_ds_boost),
            ],
            voting="soft",
            n_jobs=1,
        )

    # ── Two-stage: top-2 trigger (only fires when DS+Grok are genuinely the top 2) ──
    if name == "two_stage_top2":
        cfg = model_cfg.get("logistic_regression_balanced", {})
        base_lr = LogisticRegression(
            C=cfg.get("C", 0.5),
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight="balanced",
            random_state=seed,
        )
        # Binary: balanced weights, C=1.5 (more flexible than conservative's C=0.8)
        # but uses top2_trigger so it's much more selective about when to fire
        binary_lr = LogisticRegression(
            C=1.5,
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight="balanced",
            random_state=seed,
        )
        return TwoStageClassifier(
            base_classifier=base_lr,
            binary_classifier=binary_lr,
            top2_trigger=True,
            binary_ds_threshold=model_cfg.get("two_stage_ds_threshold", 0.50),
        )

    # ── Two-stage top2 conservative (natural binary weights, C=0.8) ───────────
    if name == "two_stage_top2_conservative":
        # Same top2 trigger as two_stage_top2 but with natural 1:2 DS:Grok weights
        # in the binary stage. Goal: keep DS recall gains while reducing the 14
        # new Grok->DS errors introduced by the balanced-weight binary.
        cfg = model_cfg.get("logistic_regression_balanced", {})
        base_lr = LogisticRegression(
            C=cfg.get("C", 0.5),
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight="balanced",
            random_state=seed,
        )
        binary_lr = LogisticRegression(
            C=0.8,
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight=None,  # natural 1:2 DS:Grok ratio — Grok-biased when uncertain
            random_state=seed,
        )
        return TwoStageClassifier(
            base_classifier=base_lr,
            binary_classifier=binary_lr,
            top2_trigger=True,
            binary_ds_threshold=0.52,  # slightly conservative DS threshold
        )

    # ── Two-stage combined trigger (top2 AND margin) — GPT suggestion ─────────
    if name == "two_stage_combined":
        # GPT recommendation: fire ONLY when top-2 are DS+Grok AND the model is
        # genuinely uncertain (margin < 0.30). The AND logic is the most selective
        # trigger possible — almost zero risk of Grok false positives.
        cfg = model_cfg.get("logistic_regression_balanced", {})
        base_lr = LogisticRegression(
            C=cfg.get("C", 0.5),
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight="balanced",
            random_state=seed,
        )
        binary_lr = LogisticRegression(
            C=1.5,
            max_iter=cfg.get("max_iter", 1000),
            solver=cfg.get("solver", "lbfgs"),
            class_weight="balanced",
            random_state=seed,
        )
        return TwoStageClassifier(
            base_classifier=base_lr,
            binary_classifier=binary_lr,
            top2_trigger=True,
            margin_trigger_gap=0.30,   # AND: only when |P(DS)-P(Grok)| < 0.30
            binary_ds_threshold=0.50,
        )

    # ── Ensemble v2: soft vote of two_stage_top2 + ensemble_soft ─────────────
    if name == "ensemble_v2":
        # Combines the complementary strengths of both approaches:
        #   two_stage_top2: better DS recall (0.70 vs 0.61)
        #   ensemble_soft:  better Grok recall (0.93 vs 0.91) + precision
        # Soft averaging should improve both classes simultaneously.
        cfg_lr  = model_cfg.get("logistic_regression_balanced", {})
        cfg_sgd = model_cfg.get("sgd_hinge", {})
        cfg_rg  = model_cfg.get("ridge_classifier", {})
        cfg_svc = model_cfg.get("calibrated_svc", {})

        # Rebuild ensemble_soft inline
        cal_sgd = CalibratedClassifierCV(
            SGDClassifier(
                loss="hinge", alpha=cfg_sgd.get("alpha", 1e-4),
                max_iter=cfg_sgd.get("max_iter", 100),
                class_weight="balanced", random_state=seed, n_jobs=-1,
            ),
            cv=3, method="isotonic",
        )
        cal_ridge = CalibratedClassifierCV(
            RidgeClassifier(alpha=cfg_rg.get("alpha", 1.0), class_weight="balanced"),
            cv=3, method="isotonic",
        )
        lr_soft = LogisticRegression(
            C=cfg_lr.get("C", 0.5), max_iter=cfg_lr.get("max_iter", 1000),
            solver=cfg_lr.get("solver", "lbfgs"), class_weight="balanced",
            random_state=seed,
        )
        cal_svc = CalibratedClassifierCV(
            LinearSVC(C=cfg_svc.get("C", 0.1), max_iter=cfg_svc.get("max_iter", 2000),
                      class_weight="balanced", random_state=seed),
            cv=3, method="isotonic",
        )
        soft_ens = VotingClassifier(
            [("cal_sgd", cal_sgd), ("cal_ridge", cal_ridge),
             ("lr", lr_soft), ("cal_svc", cal_svc)],
            voting="soft", n_jobs=1,
        )

        # Rebuild two_stage_top2 inline
        base_lr = LogisticRegression(
            C=cfg_lr.get("C", 0.5), max_iter=cfg_lr.get("max_iter", 1000),
            solver=cfg_lr.get("solver", "lbfgs"), class_weight="balanced",
            random_state=seed,
        )
        binary_lr = LogisticRegression(
            C=1.5, max_iter=cfg_lr.get("max_iter", 1000),
            solver=cfg_lr.get("solver", "lbfgs"), class_weight="balanced",
            random_state=seed,
        )
        top2_stage = TwoStageClassifier(
            base_classifier=base_lr,
            binary_classifier=binary_lr,
            top2_trigger=True,
            binary_ds_threshold=0.50,
        )

        return VotingClassifier(
            [("top2_stage", top2_stage), ("soft_ens", soft_ens)],
            voting="soft",
            n_jobs=1,
        )

    raise ValueError(f"Unknown model '{name}'. Available: {AVAILABLE_MODELS}")


def get_all_models(config: Dict) -> Dict[str, BaseEstimator]:
    """Return dict of all models listed in config.models.run_models."""
    run_models = config.get("models", {}).get("run_models", AVAILABLE_MODELS)
    return {name: get_model(name, config) for name in run_models}
