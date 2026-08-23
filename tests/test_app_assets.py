"""Contracts the interface depends on.

The Streamlit app cannot be imported in a test (its module body renders a page),
so what is testable is the set of facts it relies on. These are all regressions
of failures that actually happened during development, each of which was silent:
the page still rendered, just wrong.
"""
from __future__ import annotations

import base64
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mhrisk import config as C
from mhrisk import localization as L

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
MOTIF = ROOT / "app" / "assets" / "motif.svg"


# ---------------------------------------------------------------------------
# background motif
# ---------------------------------------------------------------------------
def test_motif_exists():
    assert MOTIF.exists(), "the app inlines this file at startup"


def test_motif_starts_with_the_svg_element():
    """Regression: a leading XML comment is valid XML but makes the file fail to
    decode when used as a CSS background image -- silently, with no console
    error and no visible artwork. Documentation belongs in the sibling README."""
    text = MOTIF.read_text(encoding="utf-8").lstrip()
    assert text.startswith("<svg"), (
        "motif.svg must begin with <svg>; a leading comment breaks data-URI decoding"
    )


def test_motif_is_parseable_and_sized():
    root = ET.fromstring(MOTIF.read_text(encoding="utf-8"))
    assert root.tag.endswith("svg")
    # Intrinsic dimensions, so the image has a size when used as a CSS background.
    assert root.get("viewBox")
    assert root.get("width") and root.get("height")


def test_motif_round_trips_through_base64():
    """Exactly what the app does at startup."""
    svg = MOTIF.read_text(encoding="utf-8").lstrip()
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    decoded = base64.b64decode(uri.split("base64,", 1)[1]).decode()
    assert decoded.startswith("<svg")
    assert decoded == svg


def test_motif_is_non_figurative():
    """The artwork is deliberately abstract -- see app/assets/README.md. Guard
    against a later edit dropping in an embedded photograph."""
    text = MOTIF.read_text(encoding="utf-8")
    assert "<image" not in text, "no raster image should be embedded in the motif"
    assert "data:image" not in text


def test_motif_avoids_the_risk_band_colours():
    """Green, amber and red mean exactly one thing on this screen. Decoration
    must not borrow them, or colour stops being a reliable signal."""
    text = MOTIF.read_text(encoding="utf-8").lower()
    for banned in ("#17a06a", "#d9963c", "#c4443e", "red", "green", "orange"):
        assert banned not in text, f"motif uses a reserved risk colour: {banned}"


def test_motif_has_accessible_text():
    root = ET.fromstring(MOTIF.read_text(encoding="utf-8"))
    tags = {child.tag.split("}")[-1] for child in root}
    assert "title" in tags and "desc" in tags


def test_assets_readme_documents_the_choice():
    readme = (ROOT / "app" / "assets" / "README.md").read_text(encoding="utf-8")
    assert "non-figurative" in readme.lower()
    assert "<svg" in readme, "the leading-comment trap must stay documented"


# ---------------------------------------------------------------------------
# app source contracts
# ---------------------------------------------------------------------------
def test_app_exists_and_compiles():
    import py_compile
    py_compile.compile(str(APP), doraise=True)


def test_cached_loaders_take_a_cache_key():
    """Regression: a zero-argument ``st.cache_data`` loader serves its first
    result forever, so an edited asset or a retrained model is never picked up.
    Every cached loader that reads a file must take a stamp argument."""
    src = APP.read_text(encoding="utf-8")
    for name in ("def motif_data_uri", "def load_model"):
        match = re.search(rf"{name}\((.*?)\)", src)
        assert match, f"{name} not found in the app"
        assert "_stamp_key" in match.group(1), f"{name} must be keyed on file state"


def test_app_verifies_the_bundle_feature_list():
    """A bundle is a pickle from a possibly older version of this code; loading
    one whose columns no longer align would produce confident nonsense."""
    src = APP.read_text(encoding="utf-8")
    assert 'bundle.get("features"' in src


def test_app_does_not_depend_on_theme_config_for_the_primary_action():
    """Streamlit resolves .streamlit/config.toml against the working directory,
    so launching from elsewhere falls back to a red primary button -- the colour
    that means 'high risk' everywhere else on the screen."""
    src = APP.read_text(encoding="utf-8")
    assert "primaryFormSubmit" in src, "submit button colour must be set in CSS"


def test_app_handles_scoring_failure_without_a_traceback():
    src = APP.read_text(encoding="utf-8")
    assert "Could not score these measurements" in src
    assert "The reasons for this result could not be computed" in src


def test_app_renders_urdu_before_english():
    """Urdu is the primary output, not a translation appended after English."""
    src = APP.read_text(encoding="utf-8")
    assert src.index('class="msg-ur"') < src.index('class="msg-en"')


def test_app_always_shows_provenance_and_disclaimer():
    src = APP.read_text(encoding="utf-8")
    assert "data_is_synthetic" in src
    assert "NOT_A_DEVICE" in src
    assert "unreviewed_bands" in src


@pytest.mark.parametrize("band", C.LABELS)
def test_every_band_has_interface_colours_and_names(band):
    """The app indexes both of these by band; a missing entry is a KeyError at
    exactly the moment a result is being shown."""
    src = APP.read_text(encoding="utf-8")
    assert f'"{band}"' in src
    assert L.BAND_NAMES[band]["ur"].strip()
    assert L.BAND_NAMES[band]["en"].strip()


def test_streamlit_theme_config_present():
    """Belt and braces: correct when launched from the project root, while the
    CSS above covers launching from anywhere else."""
    cfg = ROOT / ".streamlit" / "config.toml"
    assert cfg.exists()
    assert "primaryColor" in cfg.read_text(encoding="utf-8")
