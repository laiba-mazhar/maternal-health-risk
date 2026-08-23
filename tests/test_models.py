"""Model zoo, selection rule, and the evaluation protocol."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mhrisk import config as C
from mhrisk import metrics as M
from mhrisk import models as Mo


# ---------------------------------------------------------------------------
# guideline estimator
# ---------------------------------------------------------------------------
def test_guideline_classifier_follows_the_estimator_api(xy):
    X, y = xy
    clf = Mo.GuidelineClassifier().fit(X, y)
    pred = clf.predict(X)
    proba = clf.predict_proba(X)
    assert pred.shape == (len(X),)
    assert proba.shape == (len(X), len(C.LABELS))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_guideline_probabilities_are_one_hot(xy):
    """A fixed rule has no uncertainty to express, and pretending otherwise
    would let threshold tuning flatter it."""
    X, _ = xy
    proba = Mo.GuidelineClassifier().fit(X).predict_proba(X)
    assert set(np.unique(proba)) <= {0.0, 1.0}


def test_guideline_predictions_match_the_rule_module(xy):
    from mhrisk import guidelines as G
    X, _ = xy
    direct, _ = G.guideline_predict(X)
    assert (Mo.GuidelineClassifier().fit(X).predict(X) == direct).all()


# ---------------------------------------------------------------------------
# zoo composition
# ---------------------------------------------------------------------------
def test_zoo_includes_both_baselines():
    names = {s.name for s in Mo.build_models()}
    assert {"majority_baseline", "guideline_baseline"} <= names


def test_baselines_are_not_threshold_tunable():
    for spec in Mo.build_models():
        if spec.name.endswith("_baseline"):
            assert not spec.tunable_threshold, f"{spec.name} must stay fixed"


def test_every_spec_is_documented():
    for spec in Mo.build_models():
        assert spec.notes.strip(), f"{spec.name} has no notes"


def test_specs_build_independently():
    """The factory must return a fresh estimator each call, or folds share state."""
    spec = Mo.get_spec("random_forest")
    assert spec.factory() is not spec.factory()


def test_get_spec_rejects_unknown_names():
    with pytest.raises(KeyError):
        Mo.get_spec("magic_model")


# ---------------------------------------------------------------------------
# class balancing
# ---------------------------------------------------------------------------
def test_sample_weights_are_inverse_frequency():
    y = np.array([0] * 90 + [2] * 10)
    w = Mo._sample_weights(y)
    # The rare class must carry the larger weight, in inverse proportion.
    assert w[y == 2][0] > w[y == 0][0]
    assert w[y == 2][0] / w[y == 0][0] == pytest.approx(9.0)


def test_weights_are_uniform_when_classes_are_balanced():
    w = Mo._sample_weights(np.array([0, 1, 2] * 10))
    assert np.allclose(w, 1.0)


# ---------------------------------------------------------------------------
# selection rule
# ---------------------------------------------------------------------------
def _agg(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).T
    frame.columns = pd.MultiIndex.from_tuples([(c, "mean") for c in frame.columns])
    return frame


def test_selection_prefers_recall_over_accuracy():
    """The whole safety-first argument, asserted."""
    agg = _agg({
        "accurate_model": {"high_risk_recall": 0.60, "expected_cost": 1.0, "accuracy": 0.92},
        "safe_model": {"high_risk_recall": 0.88, "expected_cost": 1.4, "accuracy": 0.79},
    })
    assert Mo.select_best(agg) == "safe_model"


def test_selection_breaks_ties_on_cost():
    agg = _agg({
        "a": {"high_risk_recall": 0.80, "expected_cost": 2.0},
        "b": {"high_risk_recall": 0.80, "expected_cost": 1.1},
    })
    assert Mo.select_best(agg) == "b"


def test_selection_excludes_baselines_by_default():
    agg = _agg({
        "guideline_baseline": {"high_risk_recall": 0.99, "expected_cost": 0.5},
        "random_forest": {"high_risk_recall": 0.70, "expected_cost": 1.2},
    })
    assert Mo.select_best(agg) == "random_forest"
    assert Mo.select_best(agg, exclude_baselines=False) == "guideline_baseline"


def test_selection_falls_back_when_only_baselines_exist():
    agg = _agg({"guideline_baseline": {"high_risk_recall": 0.9, "expected_cost": 0.5}})
    assert Mo.select_best(agg) == "guideline_baseline"


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_cross_validate_shapes(xy):
    X, y = xy
    specs = [Mo.get_spec("majority_baseline"), Mo.get_spec("logistic_regression")]
    results, points = Mo.cross_validate(
        X, y, specs=specs, n_splits=3, n_repeats=1, verbose=False)
    assert len(results) == 3 * len(specs)
    assert len(points) == len(results)
    assert set(points["model"]) == {s.name for s in specs}


@pytest.mark.slow
def test_fixed_rule_models_keep_the_default_operating_point(xy):
    X, y = xy
    results, _ = Mo.cross_validate(
        X, y, specs=[Mo.get_spec("guideline_baseline")],
        n_splits=3, n_repeats=1, verbose=False)
    for r in results:
        assert r.operating_point == M.DEFAULT_OPERATING_POINT


@pytest.mark.slow
def test_tuned_models_move_off_the_default(xy):
    """If tuning never changes anything, the tuning code is not running."""
    X, y = xy
    results, _ = Mo.cross_validate(
        X, y, specs=[Mo.get_spec("logistic_regression")],
        n_splits=3, n_repeats=1, verbose=False)
    assert any(r.operating_point != M.DEFAULT_OPERATING_POINT for r in results)


@pytest.mark.slow
def test_fit_final_returns_a_usable_model(xy):
    X, y = xy
    model, point = Mo.fit_final(Mo.get_spec("logistic_regression"), X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), len(C.LABELS))
    assert 0.0 < point.t_high <= 1.0
    assert point.t_escalate <= point.t_high


def test_every_model_beats_the_majority_baseline_on_recall(xy):
    """A sanity floor: any model that cannot beat always-predict-low on
    high-risk recall is not worth reporting."""
    X, y = xy
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=C.RANDOM_SEED)

    for name in ("logistic_regression", "random_forest"):
        spec = Mo.get_spec(name)
        model = Mo._fit(spec, spec.factory(), X_tr, y_tr)
        proba = model.predict_proba(X_te)
        point, _ = M.tune_operating_point(y_tr, Mo._fit(
            spec, spec.factory(), X_tr, y_tr).predict_proba(X_tr))
        recall = M.safety_metrics(y_te, M.decide(proba, point))["high_risk_recall"]
        assert recall > 0.2, f"{name} barely detects high-risk cases"
