"""Safety-first evaluation.

Accuracy is the wrong headline number for this problem. A model that calls
every mother low-risk scores respectably on a dataset where 37% of rows are
low-risk, and would be lethal in a clinic. So the metrics here are built around
the specific error that matters:

    **critical miss** -- a mother whose true label is high risk, predicted low risk.

That error is reported separately from ordinary recall because "mid risk" still
sends her onward for review, whereas "low risk" sends her home. Alongside it:

* **high-risk recall** -- the fraction of high-risk mothers escalated at all;
* **referral rate** -- the operational cost, i.e. how much work the tool creates.
  A tool that refers everyone has perfect recall and gets switched off in a week;
* **expected cost** -- the asymmetric cost matrix in ``config``, which prices a
  critical miss at 25x an unnecessary referral;
* **quadratic-weighted kappa** -- an ordinal-aware agreement score, since
  low/mid/high is a ranking rather than three unrelated categories.

``decide`` and ``tune_operating_point`` separate the *model* from the *decision
rule*. Fitting a classifier and then choosing where to put the referral
threshold are different acts, and conflating them is how safety-critical
thresholds end up at whatever ``argmax`` happened to give.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from . import config as C

HIGH = C.POSITIVE_INDEX                     # 2
MID = C.LABEL_TO_INT["mid risk"]            # 1
LOW = C.LABEL_TO_INT["low risk"]            # 0
_COST = np.asarray(C.COST_MATRIX, dtype=float)


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperatingPoint:
    """Where the referral thresholds sit. Fitted on training folds only."""

    t_high: float      # P(high risk) at or above this -> escalate to high
    t_escalate: float  # P(mid or high) at or above this -> at least mid

    def describe(self) -> str:
        return f"t_high={self.t_high:.2f}, t_escalate={self.t_escalate:.2f}"


DEFAULT_OPERATING_POINT = OperatingPoint(t_high=0.50, t_escalate=0.50)


def decide(proba: np.ndarray, point: OperatingPoint = DEFAULT_OPERATING_POINT) -> np.ndarray:
    """Turn class probabilities into a risk band via explicit thresholds.

    Ordered escalation rather than ``argmax``: a 40% chance of high risk should
    escalate even when "low risk" holds a 45% plurality, and ``argmax`` cannot
    express that.
    """
    proba = np.asarray(proba, dtype=float)
    p_high = proba[:, HIGH]
    p_escalate = proba[:, HIGH] + proba[:, MID]

    out = np.full(len(proba), LOW, dtype=int)
    out[p_escalate >= point.t_escalate] = MID
    out[p_high >= point.t_high] = HIGH
    return out


def tune_operating_point(
    y_true: np.ndarray,
    proba: np.ndarray,
    max_referral_rate: float = C.MAX_REFERRAL_RATE,
    grid: np.ndarray | None = None,
) -> tuple[OperatingPoint, pd.DataFrame]:
    """Choose thresholds that maximise high-risk recall within a referral budget.

    Without the budget constraint the optimum is always "refer everybody", which
    is why the constraint is part of the objective rather than a footnote. Ties
    on recall are broken on expected cost, then on the smaller referral load.

    Returns the chosen point and the full sweep, so the trade-off can be plotted
    instead of asserted.
    """
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    y_true = np.asarray(y_true)

    rows = []
    for t_high in grid:
        for t_esc in grid:
            if t_esc > t_high:
                # An escalate threshold stricter than the high threshold makes
                # the mid band unreachable; skip rather than score nonsense.
                continue
            pred = decide(proba, OperatingPoint(t_high, t_esc))
            m = safety_metrics(y_true, pred)
            rows.append({
                "t_high": round(float(t_high), 3),
                "t_escalate": round(float(t_esc), 3),
                "high_risk_recall": m["high_risk_recall"],
                "critical_miss_rate": m["critical_miss_rate"],
                "referral_rate": m["referral_rate"],
                "expected_cost": m["expected_cost"],
                "balanced_accuracy": m["balanced_accuracy"],
            })
    sweep = pd.DataFrame(rows)

    feasible = sweep[sweep["referral_rate"] <= max_referral_rate]
    if feasible.empty:
        # Every threshold pair breaches the budget (very imbalanced fold). Fall
        # back to the cheapest referral load rather than silently ignoring it.
        feasible = sweep.nsmallest(1, "referral_rate")

    best = feasible.sort_values(
        ["high_risk_recall", "expected_cost", "referral_rate"],
        ascending=[False, True, True],
    ).iloc[0]
    return OperatingPoint(float(best["t_high"]), float(best["t_escalate"])), sweep


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_matched_referral(
    y_true: np.ndarray,
    proba: np.ndarray,
    target_referral_rate: float,
    tolerance: float = 0.02,
    grid: np.ndarray | None = None,
) -> dict[str, float]:
    """High-risk recall when the model is forced to refer as many mothers as a
    reference system does.

    Comparing a threshold-tuned model against a fixed clinical rule at their own
    natural operating points is not a comparison -- whichever refers more people
    will look better on recall. This equalises the operational cost first, and
    only then compares. It is the honest form of the question "does the learned
    model add anything over the guidelines?"

    Returns the matched operating point and its metrics; ``matched_referral_rate``
    reports what was actually achieved, since the threshold grid is discrete and
    an exact match is not always reachable.
    """
    grid = np.linspace(0.02, 0.98, 49) if grid is None else grid
    y_true = np.asarray(y_true)

    rows = []
    for t_high in grid:
        for t_esc in grid:
            if t_esc > t_high:
                continue
            pred = decide(proba, OperatingPoint(t_high, t_esc))
            rows.append((t_high, t_esc, referral_rate(pred), pred))

    # Prefer points inside the tolerance band; among those, the best recall.
    within = [r for r in rows if abs(r[2] - target_referral_rate) <= tolerance]
    pool = within or [min(rows, key=lambda r: abs(r[2] - target_referral_rate))]
    best = max(pool, key=lambda r: recall_score(
        y_true, r[3], labels=list(range(len(C.LABELS))), average=None, zero_division=0)[HIGH])

    t_high, t_esc, achieved, pred = best
    out = safety_metrics(y_true, pred)
    out.update({
        "target_referral_rate": float(target_referral_rate),
        "matched_referral_rate": float(achieved),
        "t_high": float(t_high),
        "t_escalate": float(t_esc),
    })
    return out


def critical_miss_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of truly high-risk mothers sent home as low risk."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    high = y_true == HIGH
    if not high.any():
        return float("nan")
    return float((y_pred[high] == LOW).mean())


def expected_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean cost per case under the asymmetric cost matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(C.LABELS)))
    return float((cm * _COST).sum() / max(cm.sum(), 1))


def referral_rate(y_pred: np.ndarray) -> float:
    """Share of cases escalated above low risk -- the workload the tool creates."""
    return float((np.asarray(y_pred) > LOW).mean())


def safety_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """The full metric block, ordered with the safety-critical numbers first."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(C.LABELS)))
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    tp_high = int(((y_true == HIGH) & (y_pred == HIGH)).sum())
    pred_high = int((y_pred == HIGH).sum())

    return {
        # safety
        "high_risk_recall": float(per_class_recall[HIGH]),
        "critical_miss_rate": critical_miss_rate(y_true, y_pred),
        "expected_cost": expected_cost(y_true, y_pred),
        # operational
        "referral_rate": referral_rate(y_pred),
        "high_risk_precision": float(tp_high / pred_high) if pred_high else 0.0,
        # conventional, reported for comparability with published work
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "kappa_quadratic": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "recall_low": float(per_class_recall[LOW]),
        "recall_mid": float(per_class_recall[MID]),
    }


METRIC_ORDER = [
    "high_risk_recall", "critical_miss_rate", "expected_cost", "referral_rate",
    "high_risk_precision", "accuracy", "balanced_accuracy", "macro_f1",
    "kappa_quadratic", "recall_low", "recall_mid",
]

# Direction of improvement, so tables and tests do not have to hardcode it.
HIGHER_IS_BETTER = {
    "high_risk_recall": True, "critical_miss_rate": False, "expected_cost": False,
    "referral_rate": False, "high_risk_precision": True, "accuracy": True,
    "balanced_accuracy": True, "macro_f1": True, "kappa_quadratic": True,
    "recall_low": True, "recall_mid": True,
}


@dataclass
class FoldResult:
    model: str
    fold: int
    operating_point: OperatingPoint
    metrics: dict[str, float]

    def as_row(self) -> dict:
        return {"model": self.model, "fold": self.fold,
                "t_high": self.operating_point.t_high,
                "t_escalate": self.operating_point.t_escalate,
                **self.metrics}


def aggregate(results: list[FoldResult]) -> pd.DataFrame:
    """Mean +/- std per model across folds, safety metrics first."""
    df = pd.DataFrame([r.as_row() for r in results])
    cols = [c for c in METRIC_ORDER if c in df.columns]
    agg = df.groupby("model")[cols].agg(["mean", "std"])
    return agg.reindex(columns=pd.MultiIndex.from_product([cols, ["mean", "std"]]))


def format_table(agg: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Collapse mean/std pairs into readable ``0.812 +/- 0.031`` cells."""
    out = {}
    for metric in agg.columns.get_level_values(0).unique():
        mean = agg[(metric, "mean")]
        std = agg[(metric, "std")].fillna(0.0)
        out[metric] = [f"{m:.{decimals}f} +/- {s:.{decimals}f}" for m, s in zip(mean, std)]
    return pd.DataFrame(out, index=agg.index)


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(C.LABELS)))
    return pd.DataFrame(
        cm,
        index=[f"true:{l}" for l in C.LABELS],
        columns=[f"pred:{l}" for l in C.LABELS],
    )


def metrics_to_series(metrics: dict[str, float]) -> pd.Series:
    return pd.Series({k: metrics[k] for k in METRIC_ORDER if k in metrics})


def asdict_operating_point(point: OperatingPoint) -> dict:
    return asdict(point)
