"""Guideline rules and the calibration study.

Each threshold is tested at its boundary, because an off-by-one on a clinical
cut-off is exactly the kind of error that survives code review and then
mis-triages someone.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mhrisk import config as C
from mhrisk import guidelines as G


def _mother(**overrides) -> pd.DataFrame:
    """A healthy baseline row, with named vitals overridden."""
    base = {"Age": 25, "SystolicBP": 110, "DiastolicBP": 70,
            "BS": 7.0, "BodyTemp": 98.4, "HeartRate": 75}
    base.update(overrides)
    return pd.DataFrame([base])[C.FEATURES]


def _band(**overrides) -> str:
    pred, _ = G.guideline_predict(_mother(**overrides))
    return C.INT_TO_LABEL[int(pred[0])]


# ---------------------------------------------------------------------------
# escalation logic
# ---------------------------------------------------------------------------
def test_healthy_mother_is_low_risk():
    assert _band() == "low risk"


def test_one_moderate_feature_gives_mid_risk():
    assert _band(SystolicBP=145) == "mid risk"


def test_two_moderate_features_escalate_to_high():
    assert _band(SystolicBP=145, Age=37) == "high risk"


def test_any_single_severe_feature_gives_high_risk():
    assert _band(SystolicBP=165) == "high risk"
    assert _band(BS=12.0) == "high risk"
    assert _band(BodyTemp=102.5) == "high risk"


# ---------------------------------------------------------------------------
# threshold boundaries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sbp,expected", [(139, "low risk"), (140, "mid risk")])
def test_systolic_hypertension_boundary(sbp, expected):
    assert _band(SystolicBP=sbp) == expected


@pytest.mark.parametrize("sbp,expected", [(159, "mid risk"), (160, "high risk")])
def test_severe_systolic_boundary(sbp, expected):
    assert _band(SystolicBP=sbp) == expected


@pytest.mark.parametrize("dbp,expected", [(89, "low risk"), (90, "mid risk")])
def test_diastolic_boundary(dbp, expected):
    assert _band(DiastolicBP=dbp) == expected


@pytest.mark.parametrize("temp,expected", [(100.3, "low risk"), (100.4, "mid risk")])
def test_fever_boundary_is_38C(temp, expected):
    """100.4 F is 38.0 C, the WHO fever threshold."""
    assert _band(BodyTemp=temp) == expected


@pytest.mark.parametrize("age,expected", [(34, "low risk"), (35, "mid risk")])
def test_advanced_maternal_age_boundary(age, expected):
    assert _band(Age=age) == expected


@pytest.mark.parametrize("age,expected", [(18, "low risk"), (17, "mid risk")])
def test_adolescent_boundary(age, expected):
    assert _band(Age=age) == expected


# ---------------------------------------------------------------------------
# blood-sugar interpretation
# ---------------------------------------------------------------------------
def test_ogtt_and_fasting_cutoffs_differ():
    ogtt = {r.name: r for r in G.build_rules("ogtt_2h")}
    fasting = {r.name: r for r in G.build_rules("fasting")}
    assert ogtt["diabetes_in_pregnancy"].lower == 11.1
    assert fasting["diabetes_in_pregnancy"].lower == 7.0


def test_bs_interpretation_changes_the_verdict():
    """A glucose of 8.0 mmol/L is unremarkable post-load and diabetic fasting."""
    row = _mother(BS=8.0)
    ogtt, _ = G.guideline_predict(row, G.build_rules("ogtt_2h"))
    fast, _ = G.guideline_predict(row, G.build_rules("fasting"))
    assert C.INT_TO_LABEL[int(ogtt[0])] == "low risk"
    assert C.INT_TO_LABEL[int(fast[0])] == "high risk"


def test_bad_interpretation_rejected():
    with pytest.raises(ValueError):
        G.build_rules("guesswork")


def test_sensitivity_table_covers_both_readings(clean_df):
    table = G.sensitivity_to_bs_interpretation(clean_df)
    assert set(table["bs_interpretation"]) == {"ogtt_2h", "fasting"}
    # The choice must visibly move the result, or the analysis is pointless.
    assert table["flagged_high_risk_%"].nunique() == 2


# ---------------------------------------------------------------------------
# rule bookkeeping
# ---------------------------------------------------------------------------
def test_every_rule_cites_a_source():
    for rule in G.THRESHOLDS:
        assert rule.source.strip(), f"{rule.name} has no cited source"
        assert rule.description.strip()
        assert rule.severity in {"moderate", "severe"}


def test_every_rule_is_bounded():
    for rule in G.THRESHOLDS:
        assert rule.lower is not None or rule.upper is not None


def test_rules_reference_real_columns():
    for rule in G.THRESHOLDS:
        assert rule.column in C.FEATURES


def test_flag_frame_aligns_with_input(clean_df):
    flags = G.flag_frame(clean_df)
    assert len(flags) == len(clean_df)
    assert flags.index.equals(clean_df.index)
    assert flags.dtypes.eq(bool).all()


def test_explain_flags_reports_what_fired():
    row = _mother(SystolicBP=165, Age=38).iloc[0]
    fired = {f["rule"] for f in G.explain_flags(row)}
    assert "severe_systolic_hypertension" in fired
    assert "advanced_maternal_age" in fired
    assert "gestational_hypertension" not in fired  # 165 is severe, not moderate


def test_explain_flags_is_empty_for_a_healthy_mother():
    assert G.explain_flags(_mother().iloc[0]) == []


# ---------------------------------------------------------------------------
# calibration report
# ---------------------------------------------------------------------------
def test_calibration_report_is_coherent(clean_df):
    rep = G.calibration_report(clean_df)
    assert rep.n == len(clean_df)
    assert 0.0 <= rep.agreement <= 1.0
    assert -1.0 <= rep.kappa_quadratic <= 1.0
    assert rep.confusion.values.sum() == len(clean_df)
    assert "agreement" in rep.to_text()


def test_calibration_flags_silent_rules(clean_df):
    """A criterion whose cut-off the data cannot reach must be called out."""
    rep = G.calibration_report(clean_df)
    # HeartRate maxes out at 90 in this dataset; tachycardia needs >100.
    assert "tachycardia" in rep.silent_rules
    assert any("never fire" in note for note in rep.notes)
