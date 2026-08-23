"""End-to-end run: data -> calibration -> models -> explanation -> artifacts.

One entry point (``run``) so that every number in the write-up traces to a
single reproducible command, and so the paper's results table cannot drift away
from the code that produced it. Every artifact carries the run metadata,
including whether the data was real or synthetic.
"""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import explain as E
from . import guidelines as G
from . import localization as L
from . import metrics as M
from . import models as Mo


@dataclass
class RunResult:
    """Everything one run produced, in memory as well as on disk."""

    dataset_info: D.DatasetInfo
    cleaning_summary: str
    calibration: G.CalibrationReport
    bs_sensitivity: pd.DataFrame
    fold_results: list[M.FoldResult]
    operating_points: pd.DataFrame
    aggregate: pd.DataFrame
    results_table: pd.DataFrame
    best_model: str
    final_operating_point: M.OperatingPoint
    matched_comparison: pd.DataFrame
    importance: pd.DataFrame
    importance_method: str
    sample_messages: list[dict]
    metadata: dict[str, Any] = field(default_factory=dict)


def run(
    source: str = "auto",
    n_repeats: int = C.CV_REPEATS,
    n_splits: int = C.CV_FOLDS,
    outdir: Path | None = None,
    include_slow: bool = True,
    verbose: bool = True,
) -> RunResult:
    outdir = Path(outdir) if outdir else C.ARTIFACTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)

    # --- 1. data -----------------------------------------------------------
    raw, info = D.load_dataset(source)
    say(f"[1/6] {info.banner()}")
    clean, creport = D.clean(raw)
    say(f"      {creport.summary()}")
    X, y = D.split_xy(clean)

    # --- 2. guideline calibration -----------------------------------------
    say("[2/6] Guideline calibration ...")
    calibration = G.calibration_report(clean)
    bs_sensitivity = G.sensitivity_to_bs_interpretation(clean)
    say(f"      agreement={calibration.agreement:.1%}  "
        f"kappa_quad={calibration.kappa_quadratic:.3f}")

    # --- 3. cross-validation ----------------------------------------------
    say(f"[3/6] Cross-validation ({n_splits} folds x {n_repeats} repeats) ...")
    specs = Mo.build_models(include_slow=include_slow)
    fold_results, points = Mo.cross_validate(
        X, y, specs=specs, n_splits=n_splits, n_repeats=n_repeats, verbose=verbose
    )
    agg = M.aggregate(fold_results)
    table = M.format_table(agg)
    best = Mo.select_best(agg)
    say(f"      selected on high-risk recall: {best}")

    # --- 4. does learning beat the guidelines at equal workload? ----------
    say("[4/6] Matched-referral comparison ...")
    matched = _matched_comparison(X, y, specs, agg, n_splits=n_splits)

    # --- 5. final fit + explanation ---------------------------------------
    say(f"[5/6] Refitting {best} on all data ...")
    spec = Mo.get_spec(best, specs)
    final_model, final_point = Mo.fit_final(spec, X, y)
    importance, importance_method = E.global_importance(final_model, X, y)
    say(f"      attribution: {importance_method}")

    sample_messages = _sample_messages(final_model, final_point, X, clean)

    # --- 6. artifacts ------------------------------------------------------
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": info.source,
        "data_is_synthetic": info.is_synthetic,
        "data_sha256": info.sha256,
        "data_rows_raw": info.n_rows,
        "data_rows_clean": int(len(clean)),
        "cv": {"folds": n_splits, "repeats": n_repeats, "seed": C.RANDOM_SEED},
        "bs_interpretation": G.BS_INTERPRETATION,
        "max_referral_rate": C.MAX_REFERRAL_RATE,
        "selected_model": best,
        "final_operating_point": M.asdict_operating_point(final_point),
        "attribution_method": importance_method,
        "template_review_status": L.review_summary(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "warning": info.banner(),
    }

    result = RunResult(
        dataset_info=info, cleaning_summary=creport.summary(), calibration=calibration,
        bs_sensitivity=bs_sensitivity, fold_results=fold_results, operating_points=points,
        aggregate=agg, results_table=table, best_model=best,
        final_operating_point=final_point, matched_comparison=matched,
        importance=importance, importance_method=importance_method,
        sample_messages=sample_messages, metadata=metadata,
    )

    _write_artifacts(result, final_model, outdir, say)
    say(f"[6/6] Artifacts written to {outdir}")
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _matched_comparison(
    X: pd.DataFrame, y: np.ndarray, specs: list[Mo.ModelSpec],
    agg: pd.DataFrame, n_splits: int, seed: int = C.RANDOM_SEED,
) -> pd.DataFrame:
    """Recall for each learned model at the guideline baseline's referral load.

    The guideline rule refers whoever it refers -- it has no threshold to tune.
    So to ask whether learning adds anything, every learned model is pushed to
    that same referral rate and compared there. Single held-out split rather than
    the full repeated CV: this is a comparison at a *fixed* operating point, and
    running it inside the tuning loop would re-introduce the leakage that loop
    exists to avoid.
    """
    from sklearn.model_selection import train_test_split

    target = float(agg.loc["guideline_baseline", ("referral_rate", "mean")])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    rows = []
    for spec in specs:
        model = Mo._fit(spec, spec.factory(), X_tr, y_tr)
        proba = model.predict_proba(X_te)

        if spec.tunable_threshold:
            m = M.recall_at_matched_referral(y_te, proba, target)
        else:
            pred = M.decide(proba, M.DEFAULT_OPERATING_POINT)
            m = M.safety_metrics(y_te, pred)
            m.update({"target_referral_rate": target,
                      "matched_referral_rate": M.referral_rate(pred),
                      "t_high": np.nan, "t_escalate": np.nan})

        rows.append({
            "model": spec.name,
            "referral_rate": round(m["matched_referral_rate"], 3),
            "high_risk_recall": round(m["high_risk_recall"], 3),
            "critical_miss_rate": round(m["critical_miss_rate"], 3),
            "expected_cost": round(m["expected_cost"], 3),
            "balanced_accuracy": round(m["balanced_accuracy"], 3),
            "t_high": m["t_high"], "t_escalate": m["t_escalate"],
        })

    return pd.DataFrame(rows).sort_values("high_risk_recall", ascending=False)


def _sample_messages(
    model, point: M.OperatingPoint, X: pd.DataFrame, clean: pd.DataFrame, n: int = 3
) -> list[dict]:
    """One worked example per predicted band: vitals -> flags -> bilingual text.

    These go in the artifacts because they are the only place the whole chain is
    visible end to end, and because a reviewer reading Urdu should not have to
    run Streamlit to see what the tool would actually say.
    """
    proba = model.predict_proba(X)
    preds = M.decide(proba, point)

    out = []
    for band_idx, band in enumerate(C.LABELS):
        idx = np.flatnonzero(preds == band_idx)
        if not len(idx):
            continue
        i = int(idx[0])
        row = X.iloc[i]
        expl = E.explain_instance(model, row, X, predicted_class=band_idx)
        drivers = expl.drivers(3)
        messages = L.render_both(band, drivers, row.to_dict())
        out.append({
            "row_index": i,
            "vitals": {k: float(v) for k, v in row.to_dict().items()},
            "true_label": clean.iloc[i][C.TARGET],
            "predicted_band": band,
            "probabilities": {l: round(float(p), 3) for l, p in zip(C.LABELS, proba[i])},
            "drivers": drivers,
            "attribution_method": expl.method,
            "guideline_flags": [f["rule"] for f in G.explain_flags(row)],
            "message_en": messages["en"].text,
            "message_ur": messages["ur"].text,
            "template_review": messages["ur"].review,
        })
        if len(out) >= n:
            break
    return out


def _md(frame: pd.DataFrame, index: bool = True) -> str:
    """Markdown if ``tabulate`` is available, fixed-width text otherwise.

    ``DataFrame.to_markdown`` needs an optional dependency. Losing a completed
    run -- minutes of cross-validation -- because a formatting library is absent
    is not an acceptable failure mode, so degrade the formatting instead.
    """
    try:
        return frame.to_markdown(index=index)
    except ImportError:
        return "```\n" + frame.to_string(index=index) + "\n```"


def _write_artifacts(result: RunResult, model, outdir: Path, say) -> None:
    banner = result.dataset_info.banner()

    def stamped(text: str) -> str:
        return f"<!-- {banner} -->\n\n{text}\n"

    pd.DataFrame([r.as_row() for r in result.fold_results]).to_csv(
        outdir / "fold_results.csv", index=False)
    result.operating_points.to_csv(outdir / "operating_points.csv", index=False)
    result.results_table.to_csv(outdir / "results_table.csv")
    result.matched_comparison.to_csv(outdir / "matched_referral_comparison.csv", index=False)
    result.importance.to_csv(outdir / "feature_importance.csv")
    result.bs_sensitivity.to_csv(outdir / "bs_interpretation_sensitivity.csv", index=False)
    result.calibration.confusion.to_csv(outdir / "calibration_confusion.csv")
    result.calibration.rule_activity.to_csv(outdir / "guideline_rule_activity.csv")

    (outdir / "calibration_report.txt").write_text(
        f"{banner}\n\n{result.calibration.to_text()}\n\n"
        f"Blood-sugar interpretation sensitivity:\n"
        f"{result.bs_sensitivity.to_string(index=False)}\n",
        encoding="utf-8")

    (outdir / "results_table.md").write_text(
        stamped("## Cross-validated results (mean +/- std)\n\n"
                + _md(result.results_table)
                + "\n\n## At the guideline baseline's referral load\n\n"
                + _md(result.matched_comparison, index=False)),
        encoding="utf-8")

    (outdir / "sample_messages.md").write_text(
        stamped("# Worked examples\n\n" + "\n".join(_format_message(m) for m in result.sample_messages)),
        encoding="utf-8")

    (outdir / "run_metadata.json").write_text(
        json.dumps(result.metadata, indent=2), encoding="utf-8")

    joblib.dump(
        {"model": model,
         "operating_point": M.asdict_operating_point(result.final_operating_point),
         "features": C.FEATURES, "labels": C.LABELS, "metadata": result.metadata},
        outdir / "model.joblib")

    say(f"      wrote {len(list(outdir.glob('*')))} files")


def _format_message(m: dict) -> str:
    vitals = ", ".join(f"{k}={v:g}" for k, v in m["vitals"].items())
    flags = ", ".join(m["guideline_flags"]) or "none"
    return (
        f"### Predicted: {m['predicted_band']}  (dataset label: {m['true_label']})\n\n"
        f"- **Vitals:** {vitals}\n"
        f"- **Probabilities:** {m['probabilities']}\n"
        f"- **Drivers ({m['attribution_method']}):** {', '.join(m['drivers'])}\n"
        f"- **Guideline criteria met:** {flags}\n"
        f"- **Template review status:** {m['template_review']}\n\n"
        f"**English:** {m['message_en']}\n\n"
        f"**Urdu:** {m['message_ur']}\n"
    )


def load_bundle(path: Path | None = None) -> dict:
    """Load a saved model bundle, for the app and for downstream scripts."""
    path = Path(path) if path else C.ARTIFACTS_DIR / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run: python scripts/train.py"
        )
    bundle = joblib.load(path)
    bundle["operating_point"] = M.OperatingPoint(**bundle["operating_point"])
    return bundle
