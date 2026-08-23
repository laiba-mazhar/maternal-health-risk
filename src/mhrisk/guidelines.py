"""Guideline-based risk assessment, and the calibration study built on it.

Two jobs.

**A transparent clinical baseline.** ``guideline_predict`` assigns a risk band
using published obstetric thresholds only -- no training, no fitted parameters.
Every ML result in this project is reported against it, because a learned model
that cannot beat a handful of documented cut-offs has not earned its complexity.

**A calibration audit.** ``calibration_report`` asks the question the dataset
cannot answer for itself: do the labels in a South-Asian community-clinic
dataset actually line up with the thresholds a Pakistani clinic would apply?
Where they diverge is where a deployed model would need local recalibration.

Thresholds and their sources are in ``THRESHOLDS`` below. They are recorded as
data, not buried in ``if`` statements, so a clinician can review them without
reading Python and a reviewer can see exactly what was assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from . import config as C

# ---------------------------------------------------------------------------
# Unit ambiguity in the BS column -- the central measurement problem here
# ---------------------------------------------------------------------------
# The source file gives blood sugar in mmol/L but never states the sampling
# condition, and the two plausible readings imply completely different
# thresholds:
#
#   Fasting plasma glucose (WHO 2013):  5.1-6.9 = GDM,  >=7.0 = diabetes in pregnancy
#   2-hour 75g OGTT (WHO 2013):         8.5-11.0 = GDM, >=11.1 = diabetes in pregnancy
#
# The observed range is roughly 6-19 mmol/L. Read as fasting values, essentially
# every row in the dataset would be diabetic, which is not credible for a
# community-screening cohort. Read as post-load values the distribution is
# clinically sensible. This project therefore assumes the OGTT reading and says
# so out loud; ``BS_INTERPRETATION`` switches it so the sensitivity of every
# downstream result to that assumption can be measured rather than argued about.
BS_INTERPRETATION = "ogtt_2h"   # "ogtt_2h" | "fasting"

_BS_CUTOFFS = {
    "ogtt_2h": {"moderate": 8.5, "severe": 11.1},
    "fasting": {"moderate": 5.1, "severe": 7.0},
}


@dataclass(frozen=True)
class Rule:
    """One guideline criterion, evaluable against a dataframe of vitals."""

    name: str
    column: str
    severity: str          # "moderate" | "severe"
    description: str
    source: str
    lower: float | None = None    # fires when column >= lower
    upper: float | None = None    # fires when column <= upper

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        col = df[self.column]
        if self.lower is not None and self.upper is not None:
            return (col >= self.lower) & (col <= self.upper)
        if self.lower is not None:
            return col >= self.lower
        if self.upper is not None:
            return col <= self.upper
        raise ValueError(f"Rule {self.name} has no bound")


def build_rules(bs_interpretation: str = BS_INTERPRETATION) -> list[Rule]:
    """Assemble the rule set for a given blood-sugar interpretation."""
    if bs_interpretation not in _BS_CUTOFFS:
        raise ValueError(f"bs_interpretation must be one of {sorted(_BS_CUTOFFS)}")
    bs = _BS_CUTOFFS[bs_interpretation]

    isshp = "ISSHP 2018 / WHO recommendations on hypertensive disorders of pregnancy"
    who_hip = f"WHO 2013 hyperglycaemia in pregnancy ({bs_interpretation})"

    return [
        # --- severe features: any one alone warrants urgent referral ---
        Rule("severe_systolic_hypertension", "SystolicBP", "severe",
             "Systolic BP >= 160 mmHg (severe hypertension)", isshp, lower=160),
        Rule("severe_diastolic_hypertension", "DiastolicBP", "severe",
             "Diastolic BP >= 110 mmHg (severe hypertension)", isshp, lower=110),
        Rule("diabetes_in_pregnancy", "BS", "severe",
             f"Blood glucose >= {bs['severe']} mmol/L (diabetes in pregnancy range)",
             who_hip, lower=bs["severe"]),
        Rule("high_fever", "BodyTemp", "severe",
             "Body temperature >= 102.0 F / 38.9 C (high fever)",
             "WHO integrated management of pregnancy and childbirth", lower=102.0),

        # --- moderate features: risk accumulates across them ---
        Rule("gestational_hypertension", "SystolicBP", "moderate",
             "Systolic BP 140-159 mmHg", isshp, lower=140, upper=159),
        Rule("gestational_hypertension_diastolic", "DiastolicBP", "moderate",
             "Diastolic BP 90-109 mmHg", isshp, lower=90, upper=109),
        Rule("gestational_diabetes", "BS", "moderate",
             f"Blood glucose {bs['moderate']}-{bs['severe'] - 0.1} mmol/L (GDM range)",
             who_hip, lower=bs["moderate"], upper=bs["severe"] - 0.1),
        Rule("fever", "BodyTemp", "moderate",
             "Body temperature 100.4-101.9 F (fever, >= 38 C)",
             "WHO integrated management of pregnancy and childbirth",
             lower=100.4, upper=101.9),
        Rule("tachycardia", "HeartRate", "moderate",
             "Resting heart rate > 100 bpm", "Standard obstetric observation charts",
             lower=101),
        Rule("advanced_maternal_age", "Age", "moderate",
             "Age >= 35 years at delivery", "WHO / obstetric age-risk consensus",
             lower=35),
        Rule("adolescent_pregnancy", "Age", "moderate",
             "Age < 18 years (adolescent pregnancy)",
             "WHO adolescent pregnancy guidance", upper=17),
    ]


THRESHOLDS = build_rules()


# ---------------------------------------------------------------------------
# Guideline scoring
# ---------------------------------------------------------------------------
def flag_frame(df: pd.DataFrame, rules: list[Rule] | None = None) -> pd.DataFrame:
    """One boolean column per rule, aligned to ``df``."""
    rules = rules or THRESHOLDS
    return pd.DataFrame({r.name: r.evaluate(df) for r in rules}, index=df.index)


def guideline_predict(
    df: pd.DataFrame, rules: list[Rule] | None = None
) -> tuple[np.ndarray, pd.DataFrame]:
    """Assign a risk band from guideline criteria alone.

    Escalation logic, deliberately simple enough to defend in a viva:

    * any **severe** feature                      -> high risk
    * two or more **moderate** features           -> high risk
    * exactly one moderate feature                -> mid risk
    * none                                        -> low risk

    Returns integer-encoded predictions plus the flag frame that produced them,
    so any single decision can be traced back to the rules that fired.
    """
    rules = rules or THRESHOLDS
    flags = flag_frame(df, rules)

    severe_names = [r.name for r in rules if r.severity == "severe"]
    moderate_names = [r.name for r in rules if r.severity == "moderate"]

    any_severe = flags[severe_names].any(axis=1)
    n_moderate = flags[moderate_names].sum(axis=1)

    pred = np.full(len(df), C.LABEL_TO_INT["low risk"], dtype=int)
    pred[n_moderate == 1] = C.LABEL_TO_INT["mid risk"]
    pred[n_moderate >= 2] = C.LABEL_TO_INT["high risk"]
    pred[any_severe.to_numpy()] = C.LABEL_TO_INT["high risk"]
    return pred, flags


# ---------------------------------------------------------------------------
# Calibration study
# ---------------------------------------------------------------------------
@dataclass
class CalibrationReport:
    """How far the dataset's labels sit from published guideline thresholds."""

    bs_interpretation: str
    n: int
    agreement: float
    kappa_linear: float
    kappa_quadratic: float
    confusion: pd.DataFrame
    rule_activity: pd.DataFrame
    silent_rules: list[str]
    label_disagreement: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"Guideline-vs-dataset calibration  (BS read as {self.bs_interpretation}, n={self.n})",
            f"  exact agreement      : {self.agreement:.1%}",
            f"  Cohen kappa (linear) : {self.kappa_linear:.3f}",
            f"  Cohen kappa (quadr.) : {self.kappa_quadratic:.3f}",
            "",
            "Confusion (rows = dataset label, cols = guideline band):",
            self.confusion.to_string(),
            "",
            "Rule activity (how often each criterion fires, by dataset label):",
            self.rule_activity.to_string(),
        ]
        if self.silent_rules:
            lines += ["", "Rules that never fire on this dataset:"]
            lines += [f"  - {r}" for r in self.silent_rules]
        if self.notes:
            lines += ["", "Findings:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


def calibration_report(
    df: pd.DataFrame, bs_interpretation: str = BS_INTERPRETATION
) -> CalibrationReport:
    """Compare dataset labels against guideline bands and describe the gap."""
    rules = build_rules(bs_interpretation)
    y_true = df[C.TARGET].map(C.LABEL_TO_INT).to_numpy()
    y_rule, flags = guideline_predict(df, rules)

    cm = confusion_matrix(y_true, y_rule, labels=range(len(C.LABELS)))
    confusion = pd.DataFrame(
        cm,
        index=[f"dataset:{l}" for l in C.LABELS],
        columns=[f"guideline:{l}" for l in C.LABELS],
    )

    # How often does each rule fire, overall and within each dataset label?
    activity = {"overall": flags.mean()}
    for label in C.LABELS:
        activity[label] = flags[df[C.TARGET].to_numpy() == label].mean()
    rule_activity = pd.DataFrame(activity).mul(100).round(1)
    rule_activity.columns = [f"{c} %" for c in rule_activity.columns]

    silent = sorted(flags.columns[flags.sum() == 0].tolist())

    # Where the two systems disagree, and in which direction.
    direction = np.sign(y_rule - y_true)
    label_disagreement = pd.DataFrame({
        "dataset_label": df[C.TARGET].to_numpy(),
        "direction": pd.Series(direction).map(
            {-1: "guideline milder", 0: "agree", 1: "guideline stricter"}
        ).to_numpy(),
    }).value_counts().unstack(fill_value=0)

    notes: list[str] = []
    if silent:
        notes.append(
            f"{len(silent)} criteria never fire, so they contribute nothing on this "
            f"cohort: {', '.join(silent)}. Any vital whose recorded range cannot "
            f"reach its clinical cut-off is decoration, not signal."
        )
    missed_high = int(cm[C.POSITIVE_INDEX, : C.POSITIVE_INDEX].sum())
    if missed_high:
        notes.append(
            f"{missed_high} of {int(cm[C.POSITIVE_INDEX].sum())} dataset high-risk "
            f"mothers fall below the guideline high-risk band -- guideline thresholds "
            f"alone would not refer them."
        )
    over_referred = int(cm[0, C.POSITIVE_INDEX])
    if over_referred:
        notes.append(
            f"{over_referred} dataset low-risk mothers reach the guideline high-risk "
            f"band, i.e. the guidelines are stricter than the labelling clinicians were."
        )

    return CalibrationReport(
        bs_interpretation=bs_interpretation,
        n=len(df),
        agreement=float((y_true == y_rule).mean()),
        kappa_linear=float(cohen_kappa_score(y_true, y_rule, weights="linear")),
        kappa_quadratic=float(cohen_kappa_score(y_true, y_rule, weights="quadratic")),
        confusion=confusion,
        rule_activity=rule_activity,
        silent_rules=silent,
        label_disagreement=label_disagreement,
        notes=notes,
    )


def sensitivity_to_bs_interpretation(df: pd.DataFrame) -> pd.DataFrame:
    """Both blood-sugar readings side by side.

    The point of this table is that the interpretation of one undocumented
    column moves the guideline baseline substantially -- which is itself a
    finding about the dataset, not a modelling detail.
    """
    rows = []
    for interp in _BS_CUTOFFS:
        rep = calibration_report(df, interp)
        y_rule, _ = guideline_predict(df, build_rules(interp))
        rows.append({
            "bs_interpretation": interp,
            "moderate_cutoff": _BS_CUTOFFS[interp]["moderate"],
            "severe_cutoff": _BS_CUTOFFS[interp]["severe"],
            "agreement": round(rep.agreement, 4),
            "kappa_quadratic": round(rep.kappa_quadratic, 4),
            "flagged_high_risk_%": round(100 * (y_rule == C.POSITIVE_INDEX).mean(), 1),
        })
    return pd.DataFrame(rows)


def explain_flags(row: pd.Series, rules: list[Rule] | None = None) -> list[dict]:
    """Which criteria fire for one mother, with the reason and the source."""
    rules = rules or THRESHOLDS
    one = row.to_frame().T
    out = []
    for rule in rules:
        if bool(rule.evaluate(one).iloc[0]):
            out.append({
                "rule": rule.name,
                "severity": rule.severity,
                "description": rule.description,
                "source": rule.source,
                "observed": row[rule.column],
                "column": rule.column,
            })
    return out
