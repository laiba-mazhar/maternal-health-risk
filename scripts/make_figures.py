"""Generate the figures for the write-up.

    python scripts/make_figures.py

Reads the artifacts left by ``train.py`` where possible and recomputes only what
it must, so figures cannot disagree with the results table. Saves to
``artifacts/figures/``. Every figure produced from synthetic data gets a
SYNTHETIC stamp burned into the image, not just the caption -- a figure pasted
into a slide deck loses its caption long before it loses its axes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from mhrisk import config as C
from mhrisk import data as D
from mhrisk import guidelines as G
from mhrisk import metrics as M
from mhrisk import models as Mo

PALETTE = {"low risk": "#4c9f70", "mid risk": "#e0a458", "high risk": "#c0504d"}
GREY = "#5a5a5a"


def _stamp(fig, is_synthetic: bool) -> None:
    """Burn the provenance warning into the image itself.

    Top-left rather than bottom-right: axis labels live along the bottom edge and
    the stamp collided with them. Inside the image rather than in the caption,
    because a figure pasted into a slide deck loses its caption long before it
    loses its axes.
    """
    if is_synthetic:
        fig.text(0.005, 0.995, "SYNTHETIC DATA - not a clinical result",
                 ha="left", va="top", fontsize=7, color="#b00020", alpha=0.9)


def _save(fig, outdir: Path, name: str, is_synthetic: bool) -> Path:
    _stamp(fig, is_synthetic)
    path = outdir / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")
    return path


# ---------------------------------------------------------------------------
def fig_class_distribution(clean: pd.DataFrame, outdir: Path, syn: bool) -> None:
    counts = clean[C.TARGET].value_counts().reindex(C.LABELS)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(range(len(counts)), counts.values,
           color=[PALETTE[l] for l in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, f"{v}\n({v / counts.sum():.0%})", ha="center", fontsize=8)
    ax.set_xticks(range(len(counts)), counts.index)
    ax.set_ylabel("Mothers")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_title("Risk-label distribution", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, outdir, "fig1_class_distribution.png", syn)


def fig_vitals_by_class(clean: pd.DataFrame, outdir: Path, syn: bool) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    for ax, col in zip(axes.ravel(), C.FEATURES):
        groups = [clean.loc[clean[C.TARGET] == l, col].values for l in C.LABELS]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.6,
                        medianprops={"color": "black"}, flierprops={"markersize": 2})
        for patch, l in zip(bp["boxes"], C.LABELS):
            patch.set_facecolor(PALETTE[l])
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, 4), ["low", "mid", "high"], fontsize=8)
        ax.set_title(f"{col}  [{C.UNITS[col].split('(')[0].strip()}]", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Vitals by risk label", fontsize=11)
    fig.tight_layout()
    _save(fig, outdir, "fig2_vitals_by_class.png", syn)


def fig_calibration_confusion(clean: pd.DataFrame, outdir: Path, syn: bool) -> None:
    rep = G.calibration_report(clean)
    cm = rep.confusion.values
    row_pct = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{row_pct[i, j]:.0%}", ha="center", va="center",
                    fontsize=8, color="white" if row_pct[i, j] > 0.5 else "black")
    ax.set_xticks(range(3), [l.replace(" risk", "") for l in C.LABELS])
    ax.set_yticks(range(3), [l.replace(" risk", "") for l in C.LABELS])
    ax.set_xlabel("Guideline band")
    ax.set_ylabel("Dataset label")
    ax.set_title(f"Dataset labels vs published thresholds\n"
                 f"agreement {rep.agreement:.0%}, quadratic kappa {rep.kappa_quadratic:.2f}",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row share")
    _save(fig, outdir, "fig3_guideline_calibration.png", syn)


def fig_rule_activity(clean: pd.DataFrame, outdir: Path, syn: bool) -> None:
    rep = G.calibration_report(clean)
    act = rep.rule_activity[[c for c in rep.rule_activity.columns if c != "overall %"]]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(act.values, cmap="OrRd", aspect="auto", vmin=0)
    for i in range(act.shape[0]):
        for j in range(act.shape[1]):
            v = act.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > act.values.max() * 0.6 else "black")
    ax.set_xticks(range(act.shape[1]), [c.replace(" %", "") for c in act.columns], fontsize=8)
    ax.set_yticks(range(act.shape[0]), act.index, fontsize=7)
    ax.set_title("How often each guideline criterion fires (%)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, label="% of mothers")
    _save(fig, outdir, "fig4_rule_activity.png", syn)


def fig_recall_referral_tradeoff(
    clean: pd.DataFrame, outdir: Path, syn: bool, seed: int = C.RANDOM_SEED
) -> None:
    """The central figure: what recall costs in referral load.

    A single accuracy number cannot express the decision a programme manager
    actually faces, which is how many extra referrals buys how many more
    high-risk mothers caught. This curve can.
    """
    from sklearn.model_selection import train_test_split

    X, y = D.split_xy(clean)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y,
                                              random_state=seed)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))

    for spec in Mo.build_models(include_slow=False):
        model = Mo._fit(spec, spec.factory(), X_tr, y_tr)
        proba = model.predict_proba(X_te)

        if not spec.tunable_threshold:
            pred = M.decide(proba, M.DEFAULT_OPERATING_POINT)
            m = M.safety_metrics(y_te, pred)
            ax.scatter(m["referral_rate"], m["high_risk_recall"], s=90, zorder=5,
                       marker="D" if "guideline" in spec.name else "s",
                       label=f"{spec.name} (fixed)")
            continue

        _, sweep = M.tune_operating_point(y_te, proba)
        # Upper envelope: best achievable recall at each referral load.
        env = (sweep.sort_values("referral_rate")
                    .groupby(sweep["referral_rate"].round(2))["high_risk_recall"]
                    .max())
        ax.plot(env.index, env.values, marker="o", ms=3, lw=1.6, label=spec.name)

    ax.axvline(C.MAX_REFERRAL_RATE, ls="--", c=GREY, lw=1)
    # Annotate at the top: the legend occupies the lower right, where this label
    # was previously hidden behind it.
    ax.text(C.MAX_REFERRAL_RATE - 0.01, 0.99, f"referral budget {C.MAX_REFERRAL_RATE:.0%}",
            fontsize=7, color=GREY, ha="right", va="top", rotation=90)
    ax.set_xlabel("Referral rate (share of mothers escalated)")
    ax.set_ylabel("High-risk recall")
    ax.set_title("What extra recall costs in referral load", fontsize=10)
    ax.set_ylim(0, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, outdir, "fig5_recall_vs_referral.png", syn)


def fig_feature_importance(outdir: Path, syn: bool) -> None:
    path = C.ARTIFACTS_DIR / "feature_importance.csv"
    if not path.exists():
        print("  (skipping feature importance: run train.py first)")
        return
    table = pd.read_csv(path, index_col=0)
    col = table.columns[-1] if "mean" in table.columns[-1] else table.columns[0]
    series = table[col].sort_values()

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.barh(series.index, series.values, color="#3b6ea5")
    ax.set_xlabel(col)
    ax.set_title("Feature attribution, selected model", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, outdir, "fig6_feature_importance.png", syn)


def fig_threshold_stability(outdir: Path, syn: bool) -> None:
    """Do the tuned thresholds agree across folds?

    A threshold that swings from 0.15 to 0.75 fold to fold is not a threshold
    anyone should deploy, and this is the plot that reveals it.
    """
    path = C.ARTIFACTS_DIR / "operating_points.csv"
    if not path.exists():
        print("  (skipping threshold stability: run train.py first)")
        return
    pts = pd.read_csv(path)
    tunable = pts[pts["t_high"].notna() & ~pts["model"].str.endswith("_baseline")]
    if tunable.empty:
        print("  (skipping threshold stability: no tuned models)")
        return

    models_ = sorted(tunable["model"].unique())
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    data = [tunable.loc[tunable["model"] == m, "t_high"].values for m in models_]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops={"color": "black"})
    for patch in bp["boxes"]:
        patch.set_facecolor("#8fb8de")
    ax.set_xticks(range(1, len(models_) + 1), models_, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("tuned t_high")
    ax.set_ylim(0, 1)
    ax.set_title("Stability of the tuned referral threshold across folds", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, outdir, "fig7_threshold_stability.png", syn)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["auto", "real", "bundled"], default="auto")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else C.ARTIFACTS_DIR / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    raw, info = D.load_dataset(args.source)
    clean, _ = D.clean(raw)
    syn = info.is_synthetic
    print(f"{info.banner()}\nWriting figures to {outdir}")

    fig_class_distribution(clean, outdir, syn)
    fig_vitals_by_class(clean, outdir, syn)
    fig_calibration_confusion(clean, outdir, syn)
    fig_rule_activity(clean, outdir, syn)
    fig_recall_referral_tradeoff(clean, outdir, syn)
    fig_feature_importance(outdir, syn)
    fig_threshold_stability(outdir, syn)

    meta = C.ARTIFACTS_DIR / "run_metadata.json"
    if meta.exists():
        info_json = json.loads(meta.read_text())
        print(f"\nFigures correspond to run {info_json['generated_utc']} "
              f"({info_json['data_source']}, model={info_json['selected_model']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
