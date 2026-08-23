"""Attribution: shape correctness, honest method labelling, and sanity."""
from __future__ import annotations

import numpy as np
import pytest

from mhrisk import config as C
from mhrisk import explain as E
from mhrisk import models as Mo


@pytest.fixture(scope="module")
def fitted(xy):
    X, y = xy
    out = {}
    for name in ("random_forest", "logistic_regression"):
        spec = Mo.get_spec(name)
        out[name] = Mo._fit(spec, spec.factory(), X, y)
    if Mo.HAS_XGBOOST:
        spec = Mo.get_spec("xgboost")
        out["xgboost"] = Mo._fit(spec, spec.factory(), X, y)
    return out


# ---------------------------------------------------------------------------
# global
# ---------------------------------------------------------------------------
def test_global_importance_covers_every_feature(fitted, xy):
    X, y = xy
    for name, model in fitted.items():
        table, method = E.global_importance(model, X, y)
        assert set(table.index) == set(C.FEATURES), f"{name} missing features"
        assert method.strip()


def test_global_importance_names_its_method(fitted, xy):
    """A figure caption must be able to say which method produced it."""
    X, y = xy
    _, tree_method = E.global_importance(fitted["random_forest"], X, y)
    _, linear_method = E.global_importance(fitted["logistic_regression"], X, y)
    assert tree_method == "TreeSHAP (exact)"
    # Linear models have closed-form SHAP values; settling for permutation
    # importance on the project's selected model would be a needless downgrade.
    assert linear_method == "LinearSHAP (exact)"


def test_linear_shap_shape_is_normalised(fitted, xy):
    X, _ = xy
    values = E._linear_shap_values(fitted["logistic_regression"], X.head(25))
    assert values.shape == (25, len(C.FEATURES), len(C.LABELS))


def test_linear_shap_respects_the_pipeline_scaler(fitted, xy):
    """SHAP for a linear model must be computed in the space its coefficients
    live in -- the scaled space, not the raw feature space."""
    X, _ = xy
    model = fitted["logistic_regression"]
    transformed = E._transform(model, X.head(10))
    assert transformed.shape == (10, len(C.FEATURES))
    # StandardScaler output is centred; raw vitals are emphatically not.
    assert abs(transformed.mean()) < abs(X.head(10).to_numpy().mean())


def test_permutation_fallback_still_available(fitted, xy, monkeypatch):
    """With shap gone, global importance degrades but keeps working."""
    X, y = xy
    monkeypatch.setattr(E, "HAS_SHAP", False)
    table, method = E.global_importance(fitted["logistic_regression"], X, y)
    assert "permutation" in method
    assert set(table.index) == set(C.FEATURES)


@pytest.mark.skipif(not Mo.HAS_XGBOOST, reason="xgboost not installed")
def test_xgboost_gets_exact_treeshap(fitted, xy):
    """XGBoost >= 3.0 breaks shap's TreeExplainer; we route via pred_contribs
    instead of silently dropping to an approximation."""
    X, y = xy
    _, method = E.global_importance(fitted["xgboost"], X, y)
    assert method == "TreeSHAP (exact)"


@pytest.mark.skipif(not Mo.HAS_XGBOOST, reason="xgboost not installed")
def test_xgboost_shap_shape_is_normalised(fitted, xy):
    X, _ = xy
    values = E._tree_shap_values(fitted["xgboost"], X.head(25))
    assert values.shape == (25, len(C.FEATURES), len(C.LABELS))


def test_tree_shap_shape_is_normalised(fitted, xy):
    X, _ = xy
    values = E._tree_shap_values(fitted["random_forest"], X.head(25))
    assert values.shape == (25, len(C.FEATURES), len(C.LABELS))


def test_blood_sugar_is_a_leading_driver(fitted, xy):
    """A validity check on the model, not the code: glucose and blood pressure
    should dominate a maternal-risk model. If body temperature led, something
    would be wrong regardless of accuracy."""
    X, y = xy
    table, _ = E.global_importance(fitted["random_forest"], X, y)
    top_two = set(table.index[:2])
    assert "BS" in top_two
    assert top_two & {"SystolicBP", "DiastolicBP"}, "no blood-pressure term in the top two"


# ---------------------------------------------------------------------------
# local
# ---------------------------------------------------------------------------
def test_local_explanation_is_complete(fitted, xy):
    X, _ = xy
    for name, model in fitted.items():
        expl = E.explain_instance(model, X.iloc[3], X)
        assert {a.feature for a in expl.attributions} == set(C.FEATURES), name
        assert expl.predicted_label in C.LABELS
        assert expl.method.strip()


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest"])
def test_local_attributions_are_not_all_zero(fitted, xy, name):
    """Regression test for a silent failure.

    ``LinearExplainer`` given the explained row as its own background returns
    all-zero attributions -- no error, no warning, just an explanation that says
    nothing, and a risk message that drops its 'what stood out' line. The
    background distribution has to come from the training data.
    """
    X, _ = xy
    expl = E.explain_instance(fitted[name], X.iloc[9], X)
    magnitudes = [abs(a.contribution) for a in expl.attributions]
    assert max(magnitudes) > 0, f"{name} produced an empty explanation"
    assert expl.drivers(3), f"{name} named no drivers"


def test_linear_shap_collapses_without_a_background(fitted, xy):
    """Prove the failure mode the test above guards against is real."""
    X, _ = xy
    row = X.iloc[[9]]
    degenerate = E._linear_shap_values(fitted["logistic_regression"], row)
    proper = E._linear_shap_values(fitted["logistic_regression"], row, background=X)
    assert np.allclose(degenerate, 0.0)
    assert not np.allclose(proper, 0.0)


def test_drivers_are_only_risk_increasing(fitted, xy):
    X, _ = xy
    expl = E.explain_instance(fitted["random_forest"], X.iloc[7], X)
    by_name = {a.feature: a for a in expl.attributions}
    for driver in expl.drivers(3):
        assert by_name[driver].contribution > 0, f"{driver} pushes the other way"


def test_drivers_are_ordered_by_magnitude(fitted, xy):
    X, _ = xy
    expl = E.explain_instance(fitted["random_forest"], X.iloc[11], X)
    mags = [abs(a.contribution) for a in expl.top(len(C.FEATURES))]
    assert mags == sorted(mags, reverse=True)


def test_explaining_a_chosen_class_is_respected(fitted, xy):
    X, _ = xy
    expl = E.explain_instance(fitted["random_forest"], X.iloc[0], X, predicted_class=2)
    assert expl.explained_class == "high risk"


def test_direction_labels_match_the_sign(fitted, xy):
    X, _ = xy
    expl = E.explain_instance(fitted["random_forest"], X.iloc[2], X)
    for a in expl.attributions:
        expected = "increases risk" if a.contribution > 0 else "decreases risk"
        assert a.direction == expected


def test_local_frame_is_sorted_and_labelled(fitted, xy):
    X, _ = xy
    frame = E.explain_instance(fitted["random_forest"], X.iloc[4], X).to_frame()
    assert list(frame.columns) == ["feature", "value", "contribution", "direction"]
    assert len(frame) == len(C.FEATURES)


def test_occlusion_fallback_still_explains(fitted, xy, monkeypatch):
    """With shap unavailable the tool must still say why -- degraded, but
    labelled as degraded rather than passed off as SHAP."""
    X, _ = xy
    monkeypatch.setattr(E, "HAS_SHAP", False)
    expl = E.explain_instance(fitted["logistic_regression"], X.iloc[1], X)
    assert "occlusion" in expl.method
    assert len(expl.attributions) == len(C.FEATURES)


# ---------------------------------------------------------------------------
# cross-checking
# ---------------------------------------------------------------------------
def test_importance_agreement_reports_rank_gaps(fitted, xy):
    X, y = xy
    shap_table, _ = E.global_importance(fitted["random_forest"], X, y)
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(fitted["random_forest"], X, y, n_repeats=3,
                                  random_state=0, scoring="balanced_accuracy")
    import pandas as pd
    perm_table = pd.DataFrame({"importance_mean": perm.importances_mean},
                              index=list(X.columns))
    joined = E.importance_agreement(shap_table, perm_table)
    assert len(joined) == len(C.FEATURES)
    assert (joined["rank_gap"] >= 0).all()
