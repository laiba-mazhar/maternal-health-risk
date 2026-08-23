from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mhrisk import data as D  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    df, _ = D.load_dataset("bundled")
    return df


@pytest.fixture(scope="session")
def clean_df(raw_df):
    cleaned, _ = D.clean(raw_df)
    return cleaned


@pytest.fixture(scope="session")
def xy(clean_df):
    return D.split_xy(clean_df)
