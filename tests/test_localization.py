"""Message-safety tests.

The three design commitments in ``localization`` are only real if a later edit
that breaks them fails the build. That is what these tests are for -- adding
"this is dangerous" to make the high band feel more urgent should turn the
suite red, not ship.
"""
from __future__ import annotations

import pytest

from mhrisk import config as C
from mhrisk import localization as L


# ---------------------------------------------------------------------------
# the lint gate
# ---------------------------------------------------------------------------
def test_templates_pass_the_safety_lint():
    findings = L.lint_templates()
    assert not findings, "template safety lint failed:\n" + "\n".join(str(f) for f in findings)


def test_lint_catches_catastrophising(monkeypatch):
    """Prove the lint has teeth rather than trusting it to."""
    broken = dict(L.TEMPLATES)
    high = L.TEMPLATES["high risk"]
    broken["high risk"] = L.Template(
        band=high.band,
        opening={"ur": "یہ خطرناک ہے۔", "en": "This is dangerous."},
        action=high.action, timeframe=high.timeframe,
        reassurance=high.reassurance, family=high.family,
    )
    monkeypatch.setattr(L, "TEMPLATES", broken)
    rules = {f.rule for f in L.lint_templates()}
    assert "catastrophising" in rules


def test_lint_catches_a_missing_action(monkeypatch):
    broken = dict(L.TEMPLATES)
    mid = L.TEMPLATES["mid risk"]
    broken["mid risk"] = L.Template(
        band=mid.band, opening=mid.opening,
        action={"ur": "انتظار کریں۔", "en": "Please wait."},
        timeframe=mid.timeframe, reassurance=mid.reassurance, family=None,
    )
    monkeypatch.setattr(L, "TEMPLATES", broken)
    rules = {f.rule for f in L.lint_templates()}
    assert "no-actionable-step" in rules


def test_lint_catches_a_diagnosis_claim(monkeypatch):
    broken = dict(L.TEMPLATES)
    mid = L.TEMPLATES["mid risk"]
    broken["mid risk"] = L.Template(
        band=mid.band,
        opening={"ur": "آپ کو ذیابیطس ہے۔", "en": "You have gestational diabetes."},
        action=mid.action, timeframe=mid.timeframe,
        reassurance=mid.reassurance, family=None,
    )
    monkeypatch.setattr(L, "TEMPLATES", broken)
    rules = {f.rule for f in L.lint_templates()}
    assert "diagnosis-claim" in rules


# ---------------------------------------------------------------------------
# rendered content
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("band", C.LABELS)
@pytest.mark.parametrize("lang", ["ur", "en"])
def test_every_band_renders_in_both_languages(band, lang):
    msg = L.render(band, drivers=["SystolicBP"], values={"SystolicBP": 150},
                   language=lang)
    assert msg.text.strip()
    assert msg.band == band
    assert msg.band_name.strip()
    assert L.DISCLAIMER[lang] in msg.text


@pytest.mark.parametrize("band", C.LABELS)
def test_urdu_and_english_carry_the_same_structure(band):
    both = L.render_both(band, ["BS"], {"BS": 13.0})
    # Same number of sentence blocks in each language: a missing line in one
    # language is a silent information gap for whoever reads that language.
    assert len(both["ur"].lines) == len(both["en"].lines)
    assert both["ur"].drivers_used == both["en"].drivers_used


def test_high_risk_addresses_the_family():
    msg = L.render("high risk", ["SystolicBP"], {"SystolicBP": 165}, language="ur")
    assert L.TEMPLATES["high risk"].family["ur"] in msg.text


def test_low_risk_withholds_driver_phrases():
    """Telling a mother she is fine, then listing what 'stood out', plants
    worry the result does not justify."""
    msg = L.render("low risk", ["SystolicBP", "BS"],
                   {"SystolicBP": 150, "BS": 13.0}, language="en")
    assert msg.drivers_used == []
    assert "stood out" not in msg.text


def test_mid_and_high_do_mention_drivers():
    for band in ("mid risk", "high risk"):
        msg = L.render(band, ["SystolicBP"], {"SystolicBP": 150}, language="en")
        assert msg.drivers_used == ["SystolicBP"]


def test_driver_count_is_capped():
    msg = L.render("high risk", C.FEATURES, {f: 200 for f in C.FEATURES},
                   language="en", max_drivers=2)
    assert len(msg.drivers_used) == 2


def test_direction_wording_follows_the_value():
    high = L.render("mid risk", ["SystolicBP"], {"SystolicBP": 160}, language="en")
    low = L.render("mid risk", ["SystolicBP"], {"SystolicBP": 85}, language="en")
    assert "higher than usual" in high.text
    assert "lower than usual" in low.text


def test_unknown_drivers_are_ignored_not_crashed():
    msg = L.render("high risk", ["NotAVital", "BS"], {"BS": 13.0}, language="en")
    assert msg.drivers_used == ["BS"]


# ---------------------------------------------------------------------------
# coverage and bookkeeping
# ---------------------------------------------------------------------------
def test_every_feature_has_driver_phrases_both_ways():
    for feature in C.FEATURES:
        assert feature in L.DRIVER_PHRASES, f"no phrase for {feature}"
        for direction in ("high", "low"):
            for lang in ("ur", "en"):
                assert L.DRIVER_PHRASES[feature][direction][lang].strip()


def test_every_feature_and_band_has_a_display_name():
    for feature in C.FEATURES:
        assert L.FEATURE_NAMES[feature]["ur"].strip()
    for band in C.LABELS:
        assert L.BAND_NAMES[band]["ur"].strip()


def test_vital_formatting_respects_measurable_precision():
    """Float steppers drift; a glucose reading must never display as 14.5216
    next to a clinical threshold."""
    assert L.format_vital("BS", 14.521600000000001) == "14.5"
    assert L.format_vital("BodyTemp", 98.60000000000002) == "98.6"
    assert L.format_vital("SystolicBP", 148.0) == "148"
    assert L.format_vital("Age", 34.4) == "34"
    assert L.format_vital("HeartRate", 88) == "88"


def test_round_vitals_matches_the_displayed_value():
    """The scored value and the shown value have to be the same number."""
    raw = {"Age": 34.4, "SystolicBP": 148.0, "DiastolicBP": 96.0,
           "BS": 14.5216, "BodyTemp": 99.10000000000001, "HeartRate": 88}
    rounded = L.round_vitals(raw)
    assert rounded["BS"] == 14.5
    assert rounded["BodyTemp"] == 99.1
    assert rounded["Age"] == 34
    for feature, value in rounded.items():
        assert L.format_vital(feature, value) == L.format_vital(feature, raw[feature])


def test_round_vitals_covers_every_feature():
    assert set(L.VITAL_PRECISION) == set(C.FEATURES)


def test_review_status_is_honest():
    """Templates must not claim clinical sign-off they have not had."""
    statuses = set(L.review_summary().values())
    assert statuses == {"UNREVIEWED"}
    assert set(L.unreviewed_bands()) == set(C.LABELS)


def test_unknown_band_and_language_rejected():
    with pytest.raises(KeyError):
        L.render("catastrophic risk")
    with pytest.raises(ValueError):
        L.render("low risk", language="fr")
