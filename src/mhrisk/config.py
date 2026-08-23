"""Central configuration: paths, schema, label ordering, and safety constants."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BUNDLED_DIR = DATA_DIR / "bundled"
ARTIFACTS_DIR = ROOT / "artifacts"
PAPER_DIR = ROOT / "paper"

BUNDLED_CSV = BUNDLED_DIR / "maternal_health_risk_SYNTHETIC.csv"
RAW_CSV = RAW_DIR / "Maternal Health Risk Data Set.csv"

# ---------------------------------------------------------------- schema
# Column names follow the UCI "Maternal Health Risk Data Set" (id 863) exactly,
# so a real download drops straight in with no renaming.
FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET = "RiskLevel"

# Units, stated explicitly because the source file does not carry them and the
# BS column in particular is routinely misread (see docs/UNITS.md).
UNITS = {
    "Age": "years",
    "SystolicBP": "mmHg",
    "DiastolicBP": "mmHg",
    "BS": "mmol/L (glucose; sampling condition undocumented upstream)",
    "BodyTemp": "degrees Fahrenheit",
    "HeartRate": "beats per minute (resting)",
}

# ---------------------------------------------------------------- labels
# Ordered low -> high. Order matters: the cost matrix and the ordinal-distance
# metrics below index into it.
LABELS = ["low risk", "mid risk", "high risk"]
LABEL_TO_INT = {name: i for i, name in enumerate(LABELS)}
INT_TO_LABEL = {i: name for name, i in LABEL_TO_INT.items()}
POSITIVE_LABEL = "high risk"
POSITIVE_INDEX = LABEL_TO_INT[POSITIVE_LABEL]

# ---------------------------------------------------------------- safety
# Asymmetric misclassification costs. The project's whole premise is that these
# errors are not interchangeable: sending a genuinely high-risk mother home is
# categorically worse than over-referring a low-risk one. Rows are truth,
# columns are prediction, indexed by LABELS order.
#
#                    pred low  pred mid  pred high
COST_MATRIX = [
    [0.0, 1.0, 2.0],    # truth low   -> over-referral: real but bounded cost
    [4.0, 0.0, 1.0],    # truth mid   -> under-call is worse than over-call
    [25.0, 8.0, 0.0],   # truth high  -> missing this is the failure we optimise against
]

# A prediction of "low risk" for a truly high-risk mother. Reported separately
# from ordinary recall because it is the single error that can kill someone.
CRITICAL_MISS_FROM = POSITIVE_LABEL
CRITICAL_MISS_TO = "low risk"

# Operating-point constraint used when tuning the decision threshold: we will
# accept a larger referral load in exchange for high-risk recall, but not an
# unbounded one, or the tool degenerates into "refer everybody" and gets ignored.
MAX_REFERRAL_RATE = 0.55

RANDOM_SEED = 20260822
CV_FOLDS = 5
CV_REPEATS = 4
