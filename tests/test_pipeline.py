"""End-to-end smoke test.

Runs the real pipeline at minimum size and checks that the artifacts a reader
would rely on actually appear, are internally consistent, and carry their
provenance. Marked slow; run with ``pytest -m slow``.
"""
from __future__ import annotations

import json

import pytest

from mhrisk import config as C
from mhrisk import pipeline

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("artifacts")
    return pipeline.run(source="bundled", n_repeats=1, n_splits=3,
                        outdir=outdir, include_slow=False, verbose=False), outdir


def test_run_completes_and_selects_a_model(run):
    result, _ = run
    assert result.best_model
    assert not result.best_model.endswith("_baseline")
    assert not result.results_table.empty


def test_baselines_are_in_the_results(run):
    """The comparison is worthless without the floor and the clinical rule."""
    result, _ = run
    assert {"majority_baseline", "guideline_baseline"} <= set(result.results_table.index)


def test_expected_artifacts_exist(run):
    _, outdir = run
    expected = {
        "fold_results.csv", "operating_points.csv", "results_table.csv",
        "results_table.md", "matched_referral_comparison.csv",
        "feature_importance.csv", "bs_interpretation_sensitivity.csv",
        "calibration_confusion.csv", "guideline_rule_activity.csv",
        "calibration_report.txt", "sample_messages.md", "run_metadata.json",
        "model.joblib",
    }
    missing = expected - {p.name for p in outdir.iterdir()}
    assert not missing, f"missing artifacts: {sorted(missing)}"


def test_synthetic_provenance_propagates_everywhere(run):
    """A synthetic run must be impossible to mistake for a real one, in every
    artifact a reader might open on its own."""
    result, outdir = run
    assert result.dataset_info.is_synthetic

    meta = json.loads((outdir / "run_metadata.json").read_text())
    assert meta["data_is_synthetic"] is True
    assert "SYNTHETIC" in meta["warning"]

    for name in ("results_table.md", "sample_messages.md", "calibration_report.txt"):
        assert "SYNTHETIC" in (outdir / name).read_text(encoding="utf-8"), name


def test_metadata_records_what_it_takes_to_reproduce(run):
    _, outdir = run
    meta = json.loads((outdir / "run_metadata.json").read_text())
    for key in ("data_sha256", "cv", "bs_interpretation", "selected_model",
                "final_operating_point", "attribution_method", "python"):
        assert key in meta, f"metadata missing {key}"
    assert meta["cv"]["seed"] == C.RANDOM_SEED


def test_saved_bundle_reloads_and_predicts(run):
    _, outdir = run
    bundle = pipeline.load_bundle(outdir / "model.joblib")
    assert bundle["features"] == C.FEATURES
    assert bundle["labels"] == C.LABELS
    assert 0.0 < bundle["operating_point"].t_high <= 1.0

    import pandas as pd
    row = pd.DataFrame([{"Age": 34, "SystolicBP": 150, "DiastolicBP": 95,
                         "BS": 13.0, "BodyTemp": 99.0, "HeartRate": 85}])[C.FEATURES]
    proba = bundle["model"].predict_proba(row)
    assert proba.shape == (1, len(C.LABELS))


def test_load_bundle_fails_clearly_when_untrained(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/train.py"):
        pipeline.load_bundle(tmp_path / "nope.joblib")


def test_sample_messages_cover_bands_and_are_bilingual(run):
    result, _ = run
    assert result.sample_messages
    for m in result.sample_messages:
        assert m["predicted_band"] in C.LABELS
        assert m["message_ur"].strip() and m["message_en"].strip()
        assert m["message_ur"] != m["message_en"]
        assert m["template_review"] == "UNREVIEWED"


def test_matched_comparison_equalises_referral_load(run):
    """The point of this table is a like-for-like comparison; if the referral
    rates diverge wildly the comparison is not matched at all."""
    result, _ = run
    matched = result.matched_comparison
    target = matched["referral_rate"].iloc[0]
    tuned = matched[matched["t_high"].notna()]
    assert not tuned.empty
    assert (tuned["referral_rate"] - target).abs().max() < 0.15


def test_calibration_is_reported_with_both_glucose_readings(run):
    result, _ = run
    assert set(result.bs_sensitivity["bs_interpretation"]) == {"ogtt_2h", "fasting"}
    assert result.calibration.n == result.metadata["data_rows_clean"]
