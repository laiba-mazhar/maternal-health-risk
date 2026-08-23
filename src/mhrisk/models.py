"""Model zoo and the evaluation protocol.

The protocol matters as much as the models. Two mistakes are easy to make here
and both inflate results:

1. **Threshold leakage.** The referral thresholds are tuned parameters. Tuning
   them on the same rows used to score the model is leakage, so every fold tunes
   its operating point on *inner* cross-validated probabilities from the training
   portion only, then applies that frozen point to the held-out fold.
2. **No floor.** Reporting 84% accuracy means nothing without knowing that
   always-predict-low-risk scores 37% and eleven documented clinical cut-offs
   score more. Both baselines are evaluated in the same loop as the models,
   under the same metrics, so the comparison is like-for-like.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import guidelines as G
from . import metrics as M

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_XGBOOST = False


# ---------------------------------------------------------------------------
# Guideline baseline as a scikit-learn estimator
# ---------------------------------------------------------------------------
class GuidelineClassifier(BaseEstimator, ClassifierMixin):
    """The published clinical thresholds, wrapped so it can share the CV loop.

    Fits nothing -- ``fit`` exists only to satisfy the estimator contract. Its
    ``predict_proba`` is a one-hot of the rule decision, which means threshold
    tuning has no effect on it. That is faithful: a guideline is a fixed rule,
    and pretending otherwise would flatter it.
    """

    def __init__(self, bs_interpretation: str = G.BS_INTERPRETATION):
        self.bs_interpretation = bs_interpretation

    def fit(self, X, y=None):
        self.rules_ = G.build_rules(self.bs_interpretation)
        self.classes_ = np.arange(len(C.LABELS))
        return self

    def predict(self, X):
        rules = getattr(self, "rules_", None) or G.build_rules(self.bs_interpretation)
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=C.FEATURES)
        pred, _ = G.guideline_predict(X, rules)
        return pred

    def predict_proba(self, X):
        pred = self.predict(X)
        proba = np.zeros((len(pred), len(C.LABELS)))
        proba[np.arange(len(pred)), pred] = 1.0
        return proba


# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    factory: Callable[[], Any]
    tunable_threshold: bool = True   # False for fixed-rule baselines
    notes: str = ""
    tags: list[str] = field(default_factory=list)


def build_models(seed: int = C.RANDOM_SEED, include_slow: bool = True) -> list[ModelSpec]:
    """The candidate set.

    ``class_weight="balanced"`` throughout: the high-risk class is the smallest
    and the one we least want to miss, so leaving the loss unweighted would push
    every model in exactly the wrong direction. XGBoost has no ``class_weight``,
    so it gets per-sample weights at fit time instead (see ``_fit``).
    """
    specs = [
        ModelSpec(
            "majority_baseline",
            lambda: DummyClassifier(strategy="most_frequent"),
            tunable_threshold=False,
            notes="Always predicts the most common class. The floor any model must clear.",
            tags=["baseline"],
        ),
        ModelSpec(
            "guideline_baseline",
            GuidelineClassifier,
            tunable_threshold=False,
            notes="Published obstetric cut-offs only; no training.",
            tags=["baseline", "clinical"],
        ),
        ModelSpec(
            "logistic_regression",
            lambda: Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=5000, class_weight="balanced", C=1.0, random_state=seed)),
            ]),
            notes="Linear, inspectable coefficients; the interpretable reference model.",
            tags=["linear"],
        ),
        ModelSpec(
            "random_forest",
            lambda: RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
            notes="Handles the non-monotone interactions between BP and glucose.",
            tags=["tree"],
        ),
    ]

    if HAS_XGBOOST:
        specs.append(ModelSpec(
            "xgboost",
            lambda: XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.06,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                objective="multi:softprob", num_class=len(C.LABELS),
                tree_method="hist", random_state=seed, n_jobs=-1,
                eval_metric="mlogloss"),
            notes="Gradient boosting; the usual strongest performer on tabular vitals.",
            tags=["tree", "boosting"],
        ))

    if include_slow:
        specs.append(ModelSpec(
            "mlp",
            lambda: Pipeline([
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=1200,
                    early_stopping=True, n_iter_no_change=25, random_state=seed)),
            ]),
            notes="Small dense network; included to show extra capacity does not help at n~1000.",
            tags=["neural"],
        ))

    return specs


def _sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights, for estimators without ``class_weight``."""
    classes, counts = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([w[v] for v in y])


def _fit(spec: ModelSpec, model, X, y):
    """Fit, routing class balancing to whichever mechanism the estimator has."""
    if "boosting" in spec.tags:
        model.fit(X, y, sample_weight=_sample_weights(y))
    else:
        model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------
def cross_validate(
    X: pd.DataFrame,
    y: np.ndarray,
    specs: list[ModelSpec] | None = None,
    n_splits: int = C.CV_FOLDS,
    n_repeats: int = C.CV_REPEATS,
    seed: int = C.RANDOM_SEED,
    inner_splits: int = 3,
    verbose: bool = True,
) -> tuple[list[M.FoldResult], pd.DataFrame]:
    """Repeated stratified CV with per-fold, leakage-free threshold tuning.

    Returns every fold result plus a tidy frame of the chosen operating points,
    so threshold stability across folds can be inspected -- a threshold that
    swings wildly fold to fold is not a threshold you deploy.
    """
    specs = specs or build_models(seed)
    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    results: list[M.FoldResult] = []
    points: list[dict] = []

    for fold, (tr, te) in enumerate(outer.split(X, y)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        for spec in specs:
            model = _fit(spec, spec.factory(), X_tr, y_tr)

            if spec.tunable_threshold:
                # Inner CV on the training portion only: the thresholds never
                # see a row that is about to be used for scoring.
                inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    inner_proba = cross_val_predict(
                        spec.factory(), X_tr, y_tr, cv=inner, method="predict_proba",
                        **({"params": {"sample_weight": _sample_weights(y_tr)}}
                           if "boosting" in spec.tags else {}),
                    )
                point, _ = M.tune_operating_point(y_tr, inner_proba)
            else:
                point = M.DEFAULT_OPERATING_POINT

            proba = model.predict_proba(X_te)
            y_pred = M.decide(proba, point)
            results.append(M.FoldResult(spec.name, fold, point, M.safety_metrics(y_te, y_pred)))
            points.append({"model": spec.name, "fold": fold,
                           "t_high": point.t_high, "t_escalate": point.t_escalate})

        if verbose and (fold + 1) % n_splits == 0:
            print(f"  ... completed repeat {(fold + 1) // n_splits} of {n_repeats}")

    return results, pd.DataFrame(points)


def select_best(agg: pd.DataFrame, exclude_baselines: bool = True) -> str:
    """Pick the model to ship.

    Selection is on high-risk recall, tie-broken on expected cost -- explicitly
    *not* on accuracy. This is the whole safety-first argument expressed as three
    lines of code, and it can and does choose a model with lower accuracy than
    the alternatives.
    """
    candidates = agg.index.tolist()
    if exclude_baselines:
        candidates = [m for m in candidates if not m.endswith("_baseline")]
    if not candidates:
        candidates = agg.index.tolist()

    ranked = sorted(
        candidates,
        key=lambda m: (-agg.loc[m, ("high_risk_recall", "mean")],
                       agg.loc[m, ("expected_cost", "mean")]),
    )
    return ranked[0]


def fit_final(
    spec: ModelSpec, X: pd.DataFrame, y: np.ndarray, seed: int = C.RANDOM_SEED,
    inner_splits: int = 5,
) -> tuple[Any, M.OperatingPoint]:
    """Refit on all data and freeze one operating point for deployment.

    The point comes from cross-validated probabilities over the full dataset, not
    from the refit model's own in-sample predictions, which would be optimistic.
    """
    if spec.tunable_threshold:
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = cross_val_predict(
                spec.factory(), X, y, cv=inner, method="predict_proba",
                **({"params": {"sample_weight": _sample_weights(y)}}
                   if "boosting" in spec.tags else {}),
            )
        point, _ = M.tune_operating_point(y, proba)
    else:
        point = M.DEFAULT_OPERATING_POINT

    model = _fit(spec, spec.factory(), X, y)
    return model, point


def get_spec(name: str, specs: list[ModelSpec] | None = None) -> ModelSpec:
    specs = specs or build_models()
    for s in specs:
        if s.name == name:
            return s
    raise KeyError(f"No model named {name!r}. Available: {[s.name for s in specs]}")
