"""Bilingual risk-screening interface for a Lady Health Worker.

    python -m streamlit run app/streamlit_app.py

Design constraints that come from the setting, not from Streamlit:

* **Urdu first, and in a real Urdu face.** The Urdu message is the primary
  output, set above the English one in Nastaliq/Naskh rather than in whatever
  fallback the browser picks for Arabic script. A message rendered in an ugly
  fallback font reads as untrustworthy before it is even understood.
* **The number is never the headline.** A probability of 0.68 means nothing at
  the point of care. The band, the reason, and the recommended action lead.
* **Nothing is shown without its reason.** A band always appears with the
  drivers behind it and any guideline criteria that fired, so a health worker
  can disagree with it on informed grounds.
* **The disclaimer is not dismissible.** Prototype status and not-a-diagnosis
  render on every result, not in an About tab.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mhrisk import config as C           # noqa: E402
from mhrisk import data as D             # noqa: E402
from mhrisk import explain as E          # noqa: E402
from mhrisk import guidelines as G       # noqa: E402
from mhrisk import localization as L     # noqa: E402
from mhrisk import metrics as M          # noqa: E402
from mhrisk import pipeline              # noqa: E402

st.set_page_config(
    page_title="Maternal Risk Screening",
    page_icon="\U0001fa7a",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Palette. One accent per band, used for the hero, the driver bars, and the
# severity chips, so colour means exactly one thing across the whole screen.
# ---------------------------------------------------------------------------
BAND = {
    "low risk":  {"ink": "#0f6b47", "bg": "#e9f6ef", "edge": "#b7e0ca", "dot": "#17a06a"},
    "mid risk":  {"ink": "#8a5a0b", "bg": "#fdf5e6", "edge": "#f0dcae", "dot": "#d9963c"},
    "high risk": {"ink": "#9c2f2b", "bg": "#fdedec", "edge": "#f3c9c6", "dot": "#c4443e"},
}

ASSETS = ROOT / "app" / "assets"


def _stamp(path: Path) -> tuple[float, int]:
    """Cache key that changes when a file on disk changes.

    Streamlit caches on the argument list, so a zero-argument loader keeps
    serving its first result forever -- including after the model is retrained
    or an asset is edited. Passing the file's mtime and size in makes the cache
    invalidate when the thing it caches actually changes.
    """
    try:
        s = path.stat()
        return (s.st_mtime, s.st_size)
    except OSError:
        return (0.0, 0)


@st.cache_data(show_spinner=False)
def motif_data_uri(_stamp_key: tuple[float, int]) -> str:
    """The background motif, inlined as a data URI.

    Inlined rather than served so the interface renders identically offline --
    the deployment setting this is written for cannot assume a network.
    """
    svg = (ASSETS / "motif.svg").read_text(encoding="utf-8").lstrip()
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&family=Noto+Naskh+Arabic:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

:root {
  --ink:      #16202a;
  --ink-soft: #5a6b78;
  --line:     #e3e8ec;
  --surface:  #ffffff;
  --sunken:   #f6f8fa;
  --accent:   #1f6f8b;
}

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

/* ---------------- background motif ----------------
   Fixed, low-opacity, and behind everything. Content cards below are opaque,
   so no body text is ever set over the artwork: in a screening tool
   legibility is a safety property, not a style choice. */
[data-testid="stAppViewContainer"]::before {
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:
    radial-gradient(1100px 620px at 88% -12%, rgba(31,111,139,.085), transparent 68%),
    radial-gradient(880px 520px at -8% 104%, rgba(217,150,60,.065), transparent 66%);
}
/* The artwork gets its own layer so it can be held at watermark strength
   without also fading the wash above it. */
[data-testid="stAppViewContainer"]::after {
  content:""; position:fixed; z-index:0; pointer-events:none;
  right:-120px; bottom:-96px; width:520px; height:520px;
  background:url("__MOTIF__") no-repeat center/contain;
  opacity:.14;
}
[data-testid="stAppViewContainer"] > .main { position:relative; z-index:1; }

/* Section labels and driver bars sit directly on the page, so give the
   content column a faint frosted backing rather than trusting the motif to
   stay out of the way. */
.block-container { position:relative; z-index:1; }

/* Streamlit chrome we do not want in a point-of-care screen. */
#MainMenu, footer, header [data-testid="stStatusWidget"] { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 780px; }

/* ---------------- masthead ---------------- */
.masthead { display:flex; align-items:center; gap:14px; margin-bottom:6px; }
.mark {
  width:42px; height:42px; border-radius:12px; flex:none;
  background:linear-gradient(145deg,#1f6f8b,#2b8ca8);
  display:flex; align-items:center; justify-content:center;
  font-size:21px; box-shadow:0 2px 8px rgba(31,111,139,.28);
}
.masthead h1 { font-size:1.42rem; font-weight:700; margin:0; color:var(--ink); letter-spacing:-.02em; }
.masthead .ur-title {
  font-family:'Noto Nastaliq Urdu', serif; font-size:1.02rem; color:var(--ink-soft);
  direction:rtl; line-height:2.1; margin-top:2px;
}

/* ---------------- notices ---------------- */
.notice {
  display:flex; gap:10px; align-items:flex-start;
  border:1px solid var(--line); border-left-width:4px;
  border-radius:8px; padding:11px 13px; margin:9px 0;
  font-size:.845rem; line-height:1.5; background:var(--sunken);
}
.notice .ico { font-size:1rem; line-height:1.3; flex:none; }
.notice.warn   { border-left-color:#c4443e; background:#fdedec; color:#7d2622; }
.notice.info   { border-left-color:#1f6f8b; background:#eef6f9; color:#134f64; }
.notice.review { border-left-color:#d9963c; background:#fdf5e6; color:#7a4f0c; }
.notice b { font-weight:650; }

/* ---------------- section labels ---------------- */
.sect {
  display:flex; align-items:baseline; gap:10px;
  margin:26px 0 6px; padding-bottom:7px; border-bottom:1px solid var(--line);
}
.sect .en { font-size:.74rem; font-weight:700; letter-spacing:.09em;
            text-transform:uppercase; color:var(--ink-soft); }
.sect .ur { font-family:'Noto Naskh Arabic', serif; font-size:.9rem;
            color:var(--ink-soft); direction:rtl; margin-left:auto; }

/* ---------------- result hero ---------------- */
.hero { border-radius:14px; padding:20px 22px; margin:6px 0 4px; border:1px solid; }
.hero .ur-band {
  font-family:'Noto Nastaliq Urdu', serif; font-size:1.95rem; font-weight:600;
  direction:rtl; text-align:right; line-height:2.25; margin:0;
}
.hero .en-band { font-size:1rem; font-weight:600; opacity:.82; margin-top:2px; }
.hero .when {
  display:inline-block; margin-top:12px; padding:5px 12px; border-radius:999px;
  background:rgba(255,255,255,.72); font-size:.8rem; font-weight:600;
}

/* ---------------- scored-vitals chips ----------------
   The result panel restates the exact numbers it scored. The form above is an
   input buffer; this is the record of what was actually assessed. If the two
   ever disagree -- a stale widget, a mis-tapped stepper -- the discrepancy is
   visible on the same screen as the risk band rather than silently trusted. */
.vitals { display:flex; flex-wrap:wrap; gap:7px; margin:12px 0 2px; }
.chip {
  display:inline-flex; align-items:baseline; gap:5px;
  border:1px solid var(--line); background:var(--surface);
  border-radius:8px; padding:5px 10px; font-size:.79rem; color:var(--ink-soft);
}
.chip b { font-variant-numeric:tabular-nums; font-weight:650; color:var(--ink); font-size:.86rem; }
.chip.flag { border-color:#f0c9c6; background:#fdf1f0; }
.chip.flag b { color:#9c2f2b; }
.vitals-note { font-size:.74rem; color:#93a2ad; margin:5px 0 0; }

/* ---------------- messages ---------------- */
.msg-ur {
  font-family:'Noto Naskh Arabic', serif; direction:rtl; text-align:right;
  font-size:1.12rem; line-height:2.15; color:var(--ink);
  background:var(--surface); border:1px solid var(--line);
  border-radius:12px; padding:18px 20px; margin:14px 0 10px;
}
.msg-en {
  font-size:.93rem; line-height:1.72; color:var(--ink-soft);
  padding:2px 4px 0; margin-bottom:4px;
}

/* ---------------- driver bars ---------------- */
.driver { display:flex; align-items:center; gap:10px; margin:7px 0; font-size:.85rem; }
.driver .name { width:170px; flex:none; color:var(--ink); }
.driver .name .ur {
  font-family:'Noto Naskh Arabic', serif; direction:rtl;
  color:var(--ink-soft); font-size:.82rem; margin-right:5px;
}
.driver .track { flex:1; height:9px; border-radius:99px; background:var(--sunken);
                 position:relative; overflow:hidden; }
.driver .fill  { height:100%; border-radius:99px; }
.driver .val   { width:56px; flex:none; text-align:right;
                 font-variant-numeric:tabular-nums; color:var(--ink-soft); }

/* ---------------- criteria chips ---------------- */
.crit { border:1px solid var(--line); border-radius:10px; padding:11px 13px; margin:8px 0;
        background:var(--surface); }
.crit .head { display:flex; align-items:center; gap:8px; font-size:.87rem; font-weight:600;
              color:var(--ink); }
.crit .tag { font-size:.64rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase;
             padding:2px 7px; border-radius:5px; flex:none; }
.crit .tag.severe   { background:#fdedec; color:#9c2f2b; }
.crit .tag.moderate { background:#fdf5e6; color:#8a5a0b; }
.crit .src { font-size:.74rem; color:var(--ink-soft); margin-top:5px; line-height:1.45; }
.crit .obs { font-variant-numeric:tabular-nums; font-weight:600; color:var(--ink); }
.crit-none { font-size:.87rem; color:var(--ink-soft); padding:11px 13px;
             border:1px dashed var(--line); border-radius:10px; background:var(--sunken); }

/* ---------------- probability strip ---------------- */
.pstrip { display:flex; height:26px; border-radius:7px; overflow:hidden;
          border:1px solid var(--line); margin:8px 0 4px; }
.pstrip div { display:flex; align-items:center; justify-content:center;
              font-size:.72rem; font-weight:650; color:#fff; min-width:0; }
.plegend { display:flex; gap:16px; font-size:.75rem; color:var(--ink-soft); }
.plegend span { display:flex; align-items:center; gap:5px; }
.plegend i { width:8px; height:8px; border-radius:2px; display:inline-block; }

/* ---------------- footer ---------------- */
.foot { margin-top:26px; padding-top:14px; border-top:1px solid var(--line);
        font-size:.76rem; color:var(--ink-soft); line-height:1.65; }
.foot .ur { font-family:'Noto Naskh Arabic', serif; direction:rtl; }
.meta { font-size:.72rem; color:#93a2ad; margin-top:6px; }

/* ---------------- controls ----------------
   Coloured here rather than left to the theme. Streamlit resolves
   .streamlit/config.toml relative to the *working directory*, so launching the
   app from anywhere other than the project root silently falls back to the
   default palette -- which paints the primary action red, the one colour that
   means "high risk" everywhere else on this screen. */
div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
  background:var(--accent) !important; border-color:var(--accent) !important;
  color:#fff !important; font-weight:600; border-radius:9px; padding:.55rem 1rem;
}
div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover {
  background:#195b73 !important; border-color:#195b73 !important;
}
div[data-testid="stButton"] button {
  border:1px solid var(--line) !important; background:var(--surface) !important;
  color:var(--ink) !important; font-weight:500; font-size:.86rem; border-radius:9px;
}
div[data-testid="stButton"] button:hover {
  border-color:var(--accent) !important; color:var(--accent) !important;
  background:#f2f8fa !important;
}
div[data-testid="stButton"] button:focus:not(:active) {
  border-color:var(--accent) !important; color:var(--accent) !important;
  box-shadow:none !important;
}

/* Inputs: tighter labels, and unit hints that do not shout. */
div[data-testid="stNumberInput"] label p { font-size:.83rem !important; font-weight:500; }
div[data-testid="stForm"] { border:1px solid var(--line); border-radius:14px;
                            padding:18px 20px 6px; background:var(--surface); }

/* Long code and URLs must never widen their container. A notice that scrolls
   sideways on a phone is a notice that does not get read. */
.notice code, .foot code {
  white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word;
  font-size:.8em; background:rgba(0,0,0,.05); padding:1px 4px; border-radius:4px;
}

/* ---------------- phones ----------------
   The intended user is a health worker with an entry-level Android handset, so
   this is the primary layout rather than a courtesy. */
@media (max-width: 640px) {
  .block-container { padding-left:1rem; padding-right:1rem; padding-top:1.4rem; }
  .masthead h1 { font-size:1.18rem; }
  .masthead .ur-title { font-size:.9rem; }
  .mark { width:36px; height:36px; font-size:18px; border-radius:10px; }
  .hero { padding:16px 16px; }
  .hero .ur-band { font-size:1.55rem; line-height:2.1; }
  .msg-ur { font-size:1.05rem; line-height:2.05; padding:15px 16px; }
  /* Give the bar room: the label drops above it instead of stealing width. */
  .driver { flex-wrap:wrap; row-gap:3px; margin:11px 0; }
  .driver .name { width:auto; flex:1 1 100%; font-weight:600; font-size:.82rem; }
  .driver .track { flex:1 1 auto; }
  .driver .val { width:48px; }
  .sect { margin-top:20px; }
  .plegend { flex-wrap:wrap; row-gap:4px; }
  [data-testid="stAppViewContainer"]::after {
    width:300px; height:300px; right:-70px; bottom:-56px; opacity:.10;
  }
}
</style>
"""
MOTIF_PATH = ASSETS / "motif.svg"
st.markdown(
    CSS.replace("__MOTIF__", motif_data_uri(_stamp(MOTIF_PATH))),
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
MODEL_PATH = C.ARTIFACTS_DIR / "model.joblib"


@st.cache_resource(show_spinner="Loading model ...")
def load_model(_stamp_key: tuple[float, int]):
    """Load the trained bundle, and check it matches this code.

    A bundle is a pickle written by a possibly older version of the project, so
    the feature list is verified rather than trusted. Loading a model whose
    columns no longer line up would otherwise produce confident nonsense instead
    of an error.
    """
    try:
        bundle = pipeline.load_bundle(MODEL_PATH)
    except FileNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - a corrupt bundle must not show a traceback
        return None, (f"The saved model could not be read ({type(exc).__name__}: {exc}). "
                      f"Retrain to rebuild it.")

    if list(bundle.get("features", [])) != list(C.FEATURES):
        return None, (f"The saved model expects features {bundle.get('features')}, "
                      f"but this code uses {C.FEATURES}. Retrain to resolve.")
    return bundle, None


@st.cache_data(show_spinner=False)
def load_background() -> pd.DataFrame:
    raw, _ = D.load_dataset("auto")
    clean, _ = D.clean(raw)
    X, _ = D.split_xy(clean)
    return X


bundle, load_error = load_model(_stamp(MODEL_PATH))

st.markdown(
    """<div class="masthead">
      <div class="mark">\U0001fa7a</div>
      <div>
        <h1>Maternal risk pre-screening</h1>
        <div class="ur-title">زچگی سے متعلق خطرے کی ابتدائی جانچ</div>
      </div>
    </div>""",
    unsafe_allow_html=True,
)

if load_error:
    st.markdown(
        f'<div class="notice warn"><span class="ico">⚠️</span><div>'
        f'<b>No trained model found.</b><br>{load_error}</div></div>',
        unsafe_allow_html=True)
    st.code("python scripts/train.py", language="bash")
    st.stop()

meta = bundle["metadata"]

# Provenance. A user of a health tool is entitled to know, without digging,
# whether the thing was trained on real data.
if meta.get("data_is_synthetic"):
    st.markdown(
        '<div class="notice warn"><span class="ico">⚠️</span><div>'
        '<b>Trained on synthetic data.</b> This is a software demonstration and its '
        'output carries no clinical meaning. Run <code>python scripts/download_data.py</code> '
        'and retrain before drawing any conclusion from it.</div></div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="notice info"><span class="ico">ℹ️</span><div>'
        f'Model <b>{meta.get("selected_model")}</b> trained on the real dataset '
        f'({meta.get("data_rows_clean")} records).</div></div>',
        unsafe_allow_html=True)

unreviewed = L.unreviewed_bands()
if unreviewed:
    st.markdown(
        f'<div class="notice review"><span class="ico">\U0001f4dd</span><div>'
        f'Urdu wording for <b>{", ".join(unreviewed)}</b> has not been signed off by a '
        f'native-speaker reviewer or a clinician. Treat the phrasing as provisional.'
        f'</div></div>',
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Example cases
# ---------------------------------------------------------------------------
# Present so the tool can be demonstrated without inventing plausible vitals on
# the spot, and so a reviewer can reach each band deliberately.
DEFAULTS = {"age": 27, "sbp": 118, "dbp": 78, "bs": 8.0, "temp": 98.6, "hr": 78}
PRESETS = {
    "Routine": {"age": 24, "sbp": 112, "dbp": 72, "bs": 6.8, "temp": 98.2, "hr": 74},
    "Raised BP": {"age": 31, "sbp": 146, "dbp": 94, "bs": 7.6, "temp": 98.4, "hr": 82},
    "High glucose": {"age": 37, "sbp": 152, "dbp": 96, "bs": 14.5, "temp": 99.1, "hr": 88},
}

# Short form keys, and the model's feature names. One mapping, used by the
# deep-link handler and the submit handler alike, so the two cannot drift.
KEY_TO_FEATURE = {
    "age": "Age", "sbp": "SystolicBP", "dbp": "DiastolicBP",
    "bs": "BS", "temp": "BodyTemp", "hr": "HeartRate",
}
PRESET_SLUGS = {label.lower().replace(" ", "-"): label for label in PRESETS}

# Pending form values live under a "v_" prefix, deliberately NOT as widget keys.
#
# Binding the inputs to widget keys and populating them through session_state
# looked equivalent and was not: Streamlit paints a keyed number_input at its
# `min_value` and applies the stored value in a later delta, so a preset or a
# deep link could render as "Age 10, BP 60/30" for a beat -- and a screenshot,
# or a fast reader, catches exactly that. Passing `value=` explicitly makes the
# server render the right number in the first frame, with no client-side
# reconciliation step to lose a race against.
STATE_PREFIX = "v_"


def _slot(key: str) -> str:
    return f"{STATE_PREFIX}{key}"


def set_pending(values: dict) -> None:
    """Stage a set of vitals for the form to render."""
    for key, value in values.items():
        st.session_state[_slot(key)] = value


def vitals_from(values: dict) -> dict[str, float]:
    """Short form keys -> feature-named vitals, snapped to measurable precision."""
    return L.round_vitals({f: values[k] for k, f in KEY_TO_FEATURE.items()})


for key, value in DEFAULTS.items():
    st.session_state.setdefault(_slot(key), value)

# ---------------------------------------------------------------------------
# Deep links:  ?case=high-glucose  ·  ?case=high-glucose&show=1
# ---------------------------------------------------------------------------
# A case can be reached by URL rather than only by clicking. This is what makes
# the tool demonstrable in documentation and reviewable without a click-through,
# and it is how the screenshots in the README are captured reproducibly.
# Applied once per session, before the widgets are built, so it sets their
# initial value rather than fighting their state afterwards.
if not st.session_state.get("_deeplink_applied"):
    st.session_state["_deeplink_applied"] = True
    requested = str(st.query_params.get("case", "")).strip().lower()
    preset_label = PRESET_SLUGS.get(requested)
    if preset_label:
        set_pending(PRESETS[preset_label])
        if str(st.query_params.get("show", "")).lower() in {"1", "true", "yes"}:
            st.session_state["result"] = vitals_from(PRESETS[preset_label])

st.markdown('<div class="sect"><span class="en">Example cases</span>'
            '<span class="ur">نمونہ کیس</span></div>', unsafe_allow_html=True)
cols = st.columns(len(PRESETS))
for col, (label, values) in zip(cols, PRESETS.items()):
    if col.button(label, use_container_width=True):
        set_pending(values)
        st.session_state.pop("result", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.markdown('<div class="sect"><span class="en">Measurements</span>'
            '<span class="ur">علامات درج کریں</span></div>', unsafe_allow_html=True)

entered: dict[str, float] = {}
with st.form("vitals"):
    c1, c2 = st.columns(2)
    with c1:
        entered["age"] = st.number_input(
            "Age · عمر  (years)", 10, 60,
            value=int(st.session_state[_slot("age")]), step=1)
        entered["sbp"] = st.number_input(
            "Systolic BP · خون کا دباؤ اوپر  (mmHg)", 60, 220,
            value=int(st.session_state[_slot("sbp")]), step=1)
        entered["dbp"] = st.number_input(
            "Diastolic BP · خون کا دباؤ نیچے  (mmHg)", 30, 140,
            value=int(st.session_state[_slot("dbp")]), step=1)
    with c2:
        entered["bs"] = st.number_input(
            "Blood glucose · خون میں شوگر  (mmol/L)", 2.0, 30.0,
            value=float(st.session_state[_slot("bs")]), step=0.1, format="%.1f")
        entered["temp"] = st.number_input(
            "Body temperature · درجۂ حرارت  (°F)", 94.0, 107.0,
            value=float(st.session_state[_slot("temp")]), step=0.1, format="%.1f")
        entered["hr"] = st.number_input(
            "Heart rate · دل کی دھڑکن  (bpm)", 40, 200,
            value=int(st.session_state[_slot("hr")]), step=1)
    submitted = st.form_submit_button("Check  ·  جانچ کریں", type="primary",
                                      use_container_width=True)

if submitted:
    # Keep what was typed, so a rerun re-renders the same form, and snap to
    # measurable precision before scoring so the value shown back and the value
    # scored are the same number.
    set_pending(entered)
    st.session_state["result"] = vitals_from(entered)

if "result" not in st.session_state:
    st.markdown('<div class="foot">Enter measurements and press <b>Check</b>. '
                'Nothing is sent anywhere — scoring happens on this machine.</div>',
                unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
row = pd.Series(st.session_state["result"])[C.FEATURES]
frame = row.to_frame().T.astype(float)
model, point = bundle["model"], bundle["operating_point"]

# The band and the explanation fail independently, and they are not equally
# important: a health worker can act on a band without an attribution chart, but
# an attribution chart without a band is useless. So scoring failing is fatal to
# the result, while explanation failing degrades it and says so.
try:
    proba = model.predict_proba(frame)[0]
    band_idx = int(M.decide(proba.reshape(1, -1), point)[0])
    band = C.INT_TO_LABEL[band_idx]
except Exception as exc:  # noqa: BLE001 - never show a traceback on a health screen
    st.markdown(
        f'<div class="notice warn"><span class="ico">⚠️</span><div>'
        f'<b>Could not score these measurements.</b> Nothing has been assessed — do '
        f'not read anything into this screen. Refer as you normally would and '
        f'report this fault.<br><code>{type(exc).__name__}: {exc}</code>'
        f'</div></div>',
        unsafe_allow_html=True)
    st.stop()

try:
    expl = E.explain_instance(model, row, load_background(), predicted_class=band_idx)
    drivers = expl.drivers(3)
    explain_error = None
except Exception as exc:  # noqa: BLE001
    expl, drivers, explain_error = None, [], f"{type(exc).__name__}: {exc}"

messages = L.render_both(band, drivers, row.to_dict())
palette = BAND[band]
template = L.TEMPLATES[band]

st.markdown('<div class="sect"><span class="en">Result</span>'
            '<span class="ur">نتیجہ</span></div>', unsafe_allow_html=True)

st.markdown(
    f"""<div class="hero" style="background:{palette['bg']};border-color:{palette['edge']}">
      <div class="ur-band" style="color:{palette['ink']}">{L.BAND_NAMES[band]['ur']}</div>
      <div class="en-band" style="color:{palette['ink']}">{L.BAND_NAMES[band]['en']}</div>
      <div class="when" style="color:{palette['ink']}">\U0001f552 {template.timeframe['en']}</div>
    </div>""",
    unsafe_allow_html=True,
)

# Restate what was scored. The band is only as trustworthy as the numbers behind
# it, and a health worker should not have to scroll back up and cross-check the
# form to know which reading produced this screen.
flags = G.explain_flags(row)
flagged_columns = {f["column"] for f in flags}


def _chip(label: str, value: str, columns: tuple[str, ...]) -> str:
    cls = "chip flag" if flagged_columns & set(columns) else "chip"
    return f'<span class="{cls}">{label} <b>{value}</b></span>'


st.markdown(
    '<div class="vitals">'
    + _chip("Age", L.format_vital("Age", row["Age"]), ("Age",))
    + _chip("BP", f'{L.format_vital("SystolicBP", row["SystolicBP"])}/'
                  f'{L.format_vital("DiastolicBP", row["DiastolicBP"])}',
            ("SystolicBP", "DiastolicBP"))
    + _chip("Glucose", L.format_vital("BS", row["BS"]), ("BS",))
    + _chip("Temp", L.format_vital("BodyTemp", row["BodyTemp"]), ("BodyTemp",))
    + _chip("Pulse", L.format_vital("HeartRate", row["HeartRate"]), ("HeartRate",))
    + "</div>"
    '<div class="vitals-note">These are the measurements this result was '
    'calculated from. If they differ from the form above, re-enter and check again.'
    "</div>",
    unsafe_allow_html=True,
)

# Urdu first: this is the message that gets spoken aloud.
st.markdown(f'<div class="msg-ur">{messages["ur"].text}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="msg-en">{messages["en"].text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Why
# ---------------------------------------------------------------------------
st.markdown('<div class="sect"><span class="en">Why this result</span>'
            '<span class="ur">یہ نتیجہ کیوں</span></div>', unsafe_allow_html=True)

if explain_error:
    st.markdown(
        f'<div class="notice review"><span class="ico">\U0001f4dd</span><div>'
        f'<b>The reasons for this result could not be computed.</b> The risk band '
        f'above and the guideline criteria below are unaffected. '
        f'<br><code>{explain_error}</code></div></div>',
        unsafe_allow_html=True)
else:
    attributions = sorted(expl.attributions, key=lambda a: -abs(a.contribution))
    widest = max((abs(a.contribution) for a in attributions), default=0.0) or 1.0

    bars = []
    for a in attributions:
        pct = min(100.0, 100.0 * abs(a.contribution) / widest)
        colour = palette["dot"] if a.contribution > 0 else "#9fb0bc"
        bars.append(
            f'<div class="driver">'
            f'<div class="name">{a.feature}'
            f'<span class="ur">{L.FEATURE_NAMES[a.feature]["ur"]}</span></div>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%;'
            f'background:{colour}"></div></div>'
            f'<div class="val">{L.format_vital(a.feature, a.value)}</div>'
            f'</div>'
        )
    st.markdown("".join(bars), unsafe_allow_html=True)
    st.caption(
        f"Bar length is each vital's contribution toward **{band}** "
        f"({expl.method}); grey bars push the other way."
    )


# ---------------------------------------------------------------------------
# Guideline criteria
# ---------------------------------------------------------------------------
st.markdown('<div class="sect"><span class="en">Guideline criteria met</span>'
            '<span class="ur">طبی اصول</span></div>', unsafe_allow_html=True)

if flags:  # computed above, alongside the scored-vitals chips
    for f in flags:
        st.markdown(
            f'<div class="crit">'
            f'<div class="head"><span class="tag {f["severity"]}">{f["severity"]}</span>'
            f'{f["description"]}</div>'
            f'<div class="src">observed <span class="obs">{f["column"]} = '
            f'{L.format_vital(f["column"], f["observed"])}</span> · {f["source"]}</div>'
            f'</div>',
            unsafe_allow_html=True)
else:
    st.markdown('<div class="crit-none">No documented threshold was crossed. '
                'The band above comes from the model alone.</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model detail
# ---------------------------------------------------------------------------
with st.expander("Model probabilities and decision thresholds"):
    segments = "".join(
        f'<div style="width:{p * 100:.2f}%;background:{BAND[l]["dot"]}">'
        f'{f"{p:.0%}" if p > 0.08 else ""}</div>'
        for l, p in zip(C.LABELS, proba)
    )
    legend = "".join(
        f'<span><i style="background:{BAND[l]["dot"]}"></i>{l} {p:.1%}</span>'
        for l, p in zip(C.LABELS, proba)
    )
    st.markdown(f'<div class="pstrip">{segments}</div>'
                f'<div class="plegend">{legend}</div>', unsafe_allow_html=True)
    st.caption(
        f"Thresholds: {point.describe()}. The band is assigned by these thresholds, "
        f"not by the highest probability — a moderate chance of high risk should "
        f"escalate even when low risk holds the plurality."
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    f"""<div class="foot">
      {L.NOT_A_DEVICE['en']} <span class="ur">{L.NOT_A_DEVICE['ur']}</span><br>
      {L.DISCLAIMER['en']}
      <div class="meta">
        Blood glucose interpreted as <code>{G.BS_INTERPRETATION}</code> (see docs/UNITS.md) ·
        model <code>{meta.get('selected_model')}</code> ·
        trained {meta.get('generated_utc', 'unknown')} on {meta.get('data_source', 'unknown')} data ·
        Urdu templates {messages['ur'].review}
      </div>
    </div>""",
    unsafe_allow_html=True,
)
