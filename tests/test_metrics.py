"""The safety metrics, checked against cases worked by hand.

These are the numbers the whole project is judged on, so they are tested against
constructed examples with known answers rather than against themselves.
"""
from __future__ import annotations

import numpy as np
import pytest

from mhrisk import config as C
from mhrisk import metrics as M

HIGH, MID, LOW = M.HIGH, M.MID, M.LOW


def _proba(rows):
    return np.asarray(rows, dtype=float)


# ---------------------------------------------------------------------------
# critical miss
# ---------------------------------------------------------------------------
def test_critical_miss_counts_only_high_to_low():
    y_true = np.array([HIGH, HIGH, HIGH, HIGH])
    y_pred = np.array([LOW, MID, HIGH, LOW])
    # Two of four high-risk mothers were sent home.
    assert M.critical_miss_rate(y_true, y_pred) == pytest.approx(0.5)


def test_mid_prediction_is_not_a_critical_miss():
    """Predicting mid still escalates her; it must not be counted as a miss."""
    assert M.critical_miss_rate(np.array([HIGH]), np.array([MID])) == 0.0


def test_critical_miss_is_nan_without_high_risk_cases():
    assert np.isnan(M.critical_miss_rate(np.array([LOW, MID]), np.array([LOW, MID])))


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------
def test_expected_cost_matches_hand_calculation():
    y_true = np.array([HIGH, LOW])
    y_pred = np.array([LOW, HIGH])
    # COST_MATRIX[high][low] = 25, COST_MATRIX[low][high] = 2 -> mean 13.5
    expected = (C.COST_MATRIX[HIGH][LOW] + C.COST_MATRIX[LOW][HIGH]) / 2
    assert M.expected_cost(y_true, y_pred) == pytest.approx(expected)


def test_perfect_prediction_costs_nothing():
    y = np.array([LOW, MID, HIGH])
    assert M.expected_cost(y, y) == 0.0


def test_missing_a_high_risk_case_costs_more_than_over_referring():
    """The asymmetry is the point; assert it rather than trusting the table."""
    miss = M.expected_cost(np.array([HIGH]), np.array([LOW]))
    over = M.expected_cost(np.array([LOW]), np.array([HIGH]))
    assert miss > over * 5


# ---------------------------------------------------------------------------
# decision rule
# ---------------------------------------------------------------------------
def test_decide_escalates_below_argmax():
    """A 40% chance of high risk should escalate even when low risk leads."""
    proba = _proba([[0.45, 0.15, 0.40]])
    assert np.argmax(proba[0]) == LOW
    assert M.decide(proba, M.OperatingPoint(t_high=0.35, t_escalate=0.30))[0] == HIGH


def test_decide_respects_thresholds_monotonically():
    proba = _proba([[0.20, 0.30, 0.50]])
    strict = M.decide(proba, M.OperatingPoint(t_high=0.90, t_escalate=0.90))[0]
    loose = M.decide(proba, M.OperatingPoint(t_high=0.40, t_escalate=0.40))[0]
    assert loose >= strict


def test_decide_all_low_when_thresholds_unreachable():
    proba = _proba([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]])
    assert (M.decide(proba, M.OperatingPoint(0.99, 0.99)) == LOW).all()


def test_referral_rate_counts_everything_above_low():
    assert M.referral_rate(np.array([LOW, MID, HIGH, LOW])) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# threshold tuning
# ---------------------------------------------------------------------------
def test_tuning_respects_the_referral_budget():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    proba = rng.dirichlet([1, 1, 1], 400)

    point, sweep = M.tune_operating_point(y, proba, max_referral_rate=0.30)
    achieved = M.referral_rate(M.decide(proba, point))
    assert achieved <= 0.30 + 1e-9, "tuning breached its own referral budget"
    assert not sweep.empty


def test_tuning_never_returns_an_unreachable_mid_band():
    """t_escalate above t_high would make the mid band impossible."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 200)
    proba = rng.dirichlet([2, 2, 2], 200)
    point, sweep = M.tune_operating_point(y, proba)
    assert point.t_escalate <= point.t_high
    assert (sweep["t_escalate"] <= sweep["t_high"]).all()


def test_tuning_prefers_recall_over_accuracy():
    """Given a choice, the tuner must take the higher-recall point."""
    # Half the cohort is high risk with a moderate probability signal.
    y = np.array([HIGH] * 50 + [LOW] * 50)
    proba = _proba([[0.55, 0.05, 0.40]] * 50 + [[0.95, 0.03, 0.02]] * 50)
    point, _ = M.tune_operating_point(y, proba, max_referral_rate=0.60)
    recall = M.safety_metrics(y, M.decide(proba, point))["high_risk_recall"]
    assert recall > 0.9, "tuner settled for a low-recall operating point"


# ---------------------------------------------------------------------------
# matched referral comparison
# ---------------------------------------------------------------------------
def test_matched_referral_hits_its_target():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 500)
    proba = rng.dirichlet([1.5, 1.5, 1.5], 500)
    out = M.recall_at_matched_referral(y, proba, target_referral_rate=0.40, tolerance=0.03)
    assert abs(out["matched_referral_rate"] - 0.40) <= 0.05


def test_matched_referral_reports_what_it_achieved():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 100)
    # Degenerate probabilities: an exact match is impossible, so the function
    # must report the gap rather than pretend it matched.
    proba = _proba([[1.0, 0.0, 0.0]] * 100)
    out = M.recall_at_matched_referral(y, proba, target_referral_rate=0.50)
    assert out["matched_referral_rate"] == 0.0
    assert out["target_referral_rate"] == 0.50


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def test_safety_metrics_leads_with_safety():
    assert M.METRIC_ORDER[0] == "high_risk_recall"
    assert M.METRIC_ORDER[1] == "critical_miss_rate"
    m = M.safety_metrics(np.array([LOW, MID, HIGH]), np.array([LOW, MID, HIGH]))
    assert set(M.METRIC_ORDER) <= set(m)


def test_metric_directions_are_declared():
    assert set(M.HIGHER_IS_BETTER) == set(M.METRIC_ORDER)
    assert M.HIGHER_IS_BETTER["high_risk_recall"] is True
    assert M.HIGHER_IS_BETTER["critical_miss_rate"] is False
    assert M.HIGHER_IS_BETTER["expected_cost"] is False


def test_always_predicting_low_scores_terribly_on_safety():
    """The metric set must expose the degenerate model that accuracy rewards."""
    y = np.array([LOW] * 60 + [MID] * 25 + [HIGH] * 15)
    pred = np.full_like(y, LOW)
    m = M.safety_metrics(y, pred)
    assert m["accuracy"] > 0.55           # looks respectable
    assert m["high_risk_recall"] == 0.0   # and is useless
    assert m["critical_miss_rate"] == 1.0


def test_aggregate_and_format(clean_df):
    results = [
        M.FoldResult("m", 0, M.DEFAULT_OPERATING_POINT,
                     M.safety_metrics(np.array([LOW, HIGH]), np.array([LOW, HIGH]))),
        M.FoldResult("m", 1, M.DEFAULT_OPERATING_POINT,
                     M.safety_metrics(np.array([LOW, HIGH]), np.array([LOW, MID]))),
    ]
    agg = M.aggregate(results)
    table = M.format_table(agg)
    assert "high_risk_recall" in table.columns
    assert "+/-" in table.loc["m", "high_risk_recall"]


def test_confusion_frame_is_labelled(clean_df):
    cm = M.confusion_frame(np.array([LOW, HIGH]), np.array([LOW, HIGH]))
    assert list(cm.index) == [f"true:{l}" for l in C.LABELS]
    assert list(cm.columns) == [f"pred:{l}" for l in C.LABELS]
