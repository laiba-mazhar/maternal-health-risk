"""Data contract, determinism, and the cleaning behaviour we rely on."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mhrisk import config as C
from mhrisk import data as D


def test_schema_and_labels(raw_df):
    assert list(raw_df.columns) == C.FEATURES + [C.TARGET]
    assert set(raw_df[C.TARGET]) <= set(C.LABELS)
    assert len(raw_df) == D.REFERENCE_MARGINALS["n_rows"]


def test_generator_is_deterministic():
    """Two runs must be byte-identical, or no result is reproducible."""
    a = D.generate_synthetic(seed=7)
    b = D.generate_synthetic(seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_generator_seed_actually_matters():
    a = D.generate_synthetic(seed=7)
    b = D.generate_synthetic(seed=8)
    assert not a.equals(b)


def test_generator_respects_published_ranges():
    df = D.generate_synthetic()
    for col, (lo, hi) in D.REFERENCE_MARGINALS["ranges"].items():
        assert df[col].min() >= lo, f"{col} below published minimum"
        assert df[col].max() <= hi, f"{col} above published maximum"


def test_generator_injects_the_awkward_bits():
    """The stand-in must not be cleaner than the real thing."""
    df = D.generate_synthetic()
    assert (df["HeartRate"] < 40).any(), "no implausible heart rates injected"
    assert df.duplicated().any(), "no duplicate rows injected"
    # All three classes present and none vanishingly small.
    counts = df[C.TARGET].value_counts()
    assert set(counts.index) == set(C.LABELS)
    assert counts.min() > 100


def test_clean_removes_duplicates_and_repairs_outliers(raw_df):
    cleaned, report = D.clean(raw_df)
    assert report.duplicates_dropped > 0
    assert not cleaned.duplicated().any()
    # Every repaired value now sits inside the plausible band.
    for col, (lo, hi) in D.PLAUSIBLE.items():
        assert cleaned[col].between(lo, hi).all(), f"{col} still implausible"


def test_clean_does_not_drop_rows_for_outliers(raw_df):
    """Outliers are repaired, not deleted -- a bad heart rate should not cost
    the five other usable vitals on that row."""
    cleaned, report = D.clean(raw_df)
    assert report.n_out == report.n_in - report.duplicates_dropped


def test_clean_normalises_label_case_and_spacing():
    df = pd.DataFrame({
        **{f: [1.0, 2.0] for f in C.FEATURES},
        C.TARGET: ["  HIGH   RISK ", "Low Risk"],
    })
    cleaned, _ = D.clean(df)
    assert set(cleaned[C.TARGET]) == {"high risk", "low risk"}


def test_clean_rejects_unknown_labels():
    df = pd.DataFrame({**{f: [1.0] for f in C.FEATURES}, C.TARGET: ["extreme risk"]})
    with pytest.raises(ValueError, match="Unrecognised risk labels"):
        D.clean(df)


def test_split_xy_encoding_is_ordinal(clean_df):
    X, y = D.split_xy(clean_df)
    assert list(X.columns) == C.FEATURES
    assert set(np.unique(y)) <= {0, 1, 2}
    # Ordering must be low < mid < high; the cost matrix depends on it.
    assert C.LABEL_TO_INT["low risk"] < C.LABEL_TO_INT["mid risk"] < C.LABEL_TO_INT["high risk"]


def test_synthetic_is_always_flagged():
    _, info = D.load_dataset("bundled")
    assert info.is_synthetic
    assert "SYNTHETIC" in info.banner()


def test_real_source_fails_loudly_when_absent(monkeypatch):
    """Asking for real data must never quietly return synthetic data."""
    monkeypatch.setattr(D, "_find_real_csv", lambda: None)
    with pytest.raises(FileNotFoundError):
        D.load_dataset("real")


def test_bad_source_rejected():
    with pytest.raises(ValueError):
        D.load_dataset("whatever")
