"""Feature attribution.

Two audiences, two granularities.

*Global* (``global_importance``) answers "which vitals drive this model at all?"
-- a validity check. If a maternal-risk model leans on body temperature more
than on blood pressure, something is wrong regardless of its accuracy.

*Local* (``explain_instance``) answers "why this mother?" and is what the
front-end shows. An unexplained referral is one a health worker can neither act
on nor sensibly overrule.

SHAP is used where it is exact and cheap (tree ensembles) and the fallback is
labelled rather than hidden: an attribution method silently swapped underneath a
plot is worse than no plot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from . import config as C

try:
    import shap
    HAS_SHAP = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_SHAP = False


def _final_estimator(model):
    return model.steps[-1][1] if isinstance(model, Pipeline) else model


def _is_tree_ensemble(model) -> bool:
    est = _final_estimator(model)
    name = type(est).__name__
    return name in {"RandomForestClassifier", "ExtraTreesClassifier",
                    "GradientBoostingClassifier", "XGBClassifier",
                    "LGBMClassifier", "DecisionTreeClassifier"}


def _is_linear(model) -> bool:
    return type(_final_estimator(model)).__name__ in {
        "LogisticRegression", "RidgeClassifier", "SGDClassifier"}


def _transform(model, X: pd.DataFrame) -> np.ndarray:
    """Push features through everything but the final estimator.

    SHAP for a linear model has to be computed in the space the coefficients
    live in, which for a scaled pipeline is not the raw feature space.
    """
    if isinstance(model, Pipeline):
        return np.asarray(model[:-1].transform(X))
    return np.asarray(X)


@dataclass
class Attribution:
    """One feature's contribution, in the units a reader can act on."""

    feature: str
    value: float
    contribution: float      # signed: positive pushes toward the explained class
    direction: str           # "increases risk" | "decreases risk"

    @property
    def magnitude(self) -> float:
        return abs(self.contribution)


@dataclass
class LocalExplanation:
    predicted_label: str
    explained_class: str
    method: str
    attributions: list[Attribution]
    base_value: float | None = None

    def top(self, k: int = 3) -> list[Attribution]:
        return sorted(self.attributions, key=lambda a: -a.magnitude)[:k]

    def drivers(self, k: int = 3) -> list[str]:
        """Feature names pushing *toward* the explained class, strongest first."""
        pushing = [a for a in self.attributions if a.contribution > 0]
        return [a.feature for a in sorted(pushing, key=lambda a: -a.magnitude)[:k]]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "feature": a.feature, "value": a.value,
            "contribution": round(a.contribution, 4), "direction": a.direction,
        } for a in sorted(self.attributions, key=lambda a: -a.magnitude)])


# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------
def global_importance(
    model, X: pd.DataFrame, y: np.ndarray | None = None, seed: int = C.RANDOM_SEED
) -> tuple[pd.DataFrame, str]:
    """Mean absolute attribution per feature, per risk class.

    Returns the table and the method actually used, so figures can be labelled
    honestly.
    """
    fallback_reason = "shap not installed"

    if HAS_SHAP and _is_tree_ensemble(model):
        try:
            values = _tree_shap_values(model, X)          # (n, features, classes)
            return _shap_table(values, X), "TreeSHAP (exact)"
        except Exception as exc:  # noqa: BLE001 - fall back rather than fail the run
            fallback_reason = f"TreeSHAP unavailable ({type(exc).__name__})"

    elif HAS_SHAP and _is_linear(model):
        try:
            values = _linear_shap_values(model, X)
            return _shap_table(values, X), "LinearSHAP (exact)"
        except Exception as exc:  # noqa: BLE001
            fallback_reason = f"LinearSHAP unavailable ({type(exc).__name__})"

    elif HAS_SHAP:
        fallback_reason = "no exact SHAP explainer for this estimator"

    if y is None:
        raise ValueError(
            f"Permutation importance needs y (falling back because: {fallback_reason})"
        )
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=seed, scoring="balanced_accuracy"
    )
    table = pd.DataFrame(
        {"importance_mean": result.importances_mean, "importance_std": result.importances_std},
        index=list(X.columns),
    ).sort_values("importance_mean", ascending=False)
    return table, f"permutation importance ({fallback_reason})"


def _shap_table(values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature per class, plus a cross-class mean."""
    table = pd.DataFrame(
        np.abs(values).mean(axis=0),
        index=list(X.columns),
        columns=[f"|SHAP| {l}" for l in C.LABELS[: values.shape[2]]],
    )
    table["|SHAP| mean"] = table.mean(axis=1)
    return table.sort_values("|SHAP| mean", ascending=False)


def _linear_shap_values(
    model, X: pd.DataFrame, background: pd.DataFrame | None = None
) -> np.ndarray:
    """Exact SHAP for a linear model, as ``(n, features, classes)``.

    For a linear model SHAP values are closed-form, so there is no reason to
    settle for permutation importance on the project's selected model just
    because it is not a tree.

    ``background`` must be supplied when explaining a small number of rows.
    A linear SHAP value is the coefficient times the feature's deviation from
    its background mean, so using the explained rows as their own background
    makes every deviation zero and every attribution zero -- silently, with no
    error. Passing the training distribution is what makes the numbers mean
    anything.
    """
    est = _final_estimator(model)
    Xt = _transform(model, X)
    bg = _transform(model, background) if background is not None else Xt
    explainer = shap.LinearExplainer(est, bg)
    raw = explainer.shap_values(Xt)

    if isinstance(raw, list):
        return np.stack(raw, axis=-1)
    raw = np.asarray(raw)
    if raw.ndim == 3:
        return raw
    return raw[:, :, None]


def _xgboost_shap_values(est, X: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP via XGBoost's own ``pred_contribs``.

    Needed because shap 0.49 cannot read the vector-valued ``base_score`` that
    XGBoost >= 3.0 writes for multi-class models -- ``shap.TreeExplainer`` dies
    with a ValueError parsing it (see docs/KNOWN_ISSUES.md). XGBoost computes the
    same exact TreeSHAP values internally, so we ask it directly instead of
    silently dropping to an approximate method.

    ``pred_contribs`` returns ``(n, classes, features + 1)`` for multi-class,
    where the trailing column is the bias term; drop it and move axes to the
    ``(n, features, classes)`` convention used everywhere else here.
    """
    import xgboost as xgb

    booster = est.get_booster()
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    contribs = np.asarray(contribs)
    if contribs.ndim == 3:                       # (n, classes, features + 1)
        return np.transpose(contribs[:, :, :-1], (0, 2, 1))
    return contribs[:, :-1, None]                # binary: (n, features + 1)


def _tree_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values as ``(n_samples, n_features, n_classes)``.

    shap returns several different shapes depending on the estimator and version,
    so normalise once here rather than at every call site.
    """
    est = _final_estimator(model)

    if type(est).__name__ == "XGBClassifier":
        return _xgboost_shap_values(est, X)

    explainer = shap.TreeExplainer(est)
    raw = explainer.shap_values(X)

    if isinstance(raw, list):                     # list of (n, features) per class
        return np.stack(raw, axis=-1)
    raw = np.asarray(raw)
    if raw.ndim == 3:                             # already (n, features, classes)
        return raw
    if raw.ndim == 2:                             # binary/regression -> single column
        return raw[:, :, None]
    raise ValueError(f"Unexpected SHAP shape {raw.shape}")


# ---------------------------------------------------------------------------
# Local explanation
# ---------------------------------------------------------------------------
def explain_instance(
    model,
    row: pd.Series | pd.DataFrame,
    background: pd.DataFrame,
    predicted_class: int | None = None,
) -> LocalExplanation:
    """Attribute one prediction to the six vitals.

    Explains the *predicted* class by default, which is the question a health
    worker asks ("why is she flagged?"), not the model-average behaviour.
    """
    frame = row.to_frame().T if isinstance(row, pd.Series) else row
    frame = frame[list(background.columns)].astype(float)

    proba = model.predict_proba(frame)[0]
    if predicted_class is None:
        predicted_class = int(np.argmax(proba))

    contributions, method, base = _local_contributions(
        model, frame, background, predicted_class
    )

    attributions = [
        Attribution(
            feature=col,
            value=float(frame.iloc[0][col]),
            contribution=float(contrib),
            direction="increases risk" if contrib > 0 else "decreases risk",
        )
        for col, contrib in zip(frame.columns, contributions)
    ]

    return LocalExplanation(
        predicted_label=C.INT_TO_LABEL[int(np.argmax(proba))],
        explained_class=C.INT_TO_LABEL[predicted_class],
        method=method,
        attributions=attributions,
        base_value=base,
    )


def _local_contributions(
    model, frame: pd.DataFrame, background: pd.DataFrame, cls: int
) -> tuple[np.ndarray, str, float | None]:
    n_features = frame.shape[1]

    if HAS_SHAP and _is_tree_ensemble(model):
        try:
            values = _tree_shap_values(model, frame)
            idx = min(cls, values.shape[2] - 1)
            return values[0, :, idx], "TreeSHAP", None
        except Exception:  # noqa: BLE001
            pass

    if HAS_SHAP and _is_linear(model):
        try:
            values = _linear_shap_values(model, frame, background=background)
            idx = min(cls, values.shape[2] - 1)
            return values[0, :, idx], "LinearSHAP", None
        except Exception:  # noqa: BLE001
            pass

    if HAS_SHAP:
        try:
            # Permutation SHAP over predict_proba. Six features and one instance,
            # so an exact-ish estimate is affordable here even though it would
            # not be for a global pass.
            sample = background.sample(
                min(len(background), 100), random_state=C.RANDOM_SEED
            )
            explainer = shap.Explainer(
                lambda d: model.predict_proba(pd.DataFrame(d, columns=background.columns)),
                shap.maskers.Independent(sample, max_samples=100),
                algorithm="permutation",
            )
            expl = explainer(frame, max_evals=2 * n_features + 1, silent=True)
            vals = np.asarray(expl.values)
            if vals.ndim == 3:
                return vals[0, :, min(cls, vals.shape[2] - 1)], "PermutationSHAP", None
            return vals[0], "PermutationSHAP", None
        except Exception:  # noqa: BLE001
            pass

    # Last resort: a one-at-a-time occlusion score against the background median.
    # Crude, but transparent and always available -- and labelled as such so it
    # never masquerades as SHAP in a figure caption.
    baseline = background.median().to_frame().T[list(frame.columns)].astype(float)
    base_p = float(model.predict_proba(baseline)[0][cls])
    contribs = np.zeros(n_features)
    for i, col in enumerate(frame.columns):
        probe = baseline.copy()
        probe.iloc[0, i] = frame.iloc[0][col]
        contribs[i] = float(model.predict_proba(probe)[0][cls]) - base_p
    return contribs, "occlusion vs median (SHAP unavailable)", base_p


def importance_agreement(shap_table: pd.DataFrame, perm_table: pd.DataFrame) -> pd.DataFrame:
    """Rank-compare two attribution methods on the same model.

    A validity check on the explanations themselves: if SHAP and permutation
    importance disagree on the ordering, neither ranking should be presented as
    "the" feature importance.
    """
    a = shap_table.iloc[:, -1].rank(ascending=False)
    b = perm_table.iloc[:, 0].rank(ascending=False)
    joined = pd.DataFrame({"shap_rank": a, "permutation_rank": b}).dropna()
    joined["rank_gap"] = (joined["shap_rank"] - joined["permutation_rank"]).abs()
    return joined.sort_values("shap_rank")
