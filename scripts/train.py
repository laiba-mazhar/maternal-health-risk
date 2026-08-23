"""Run the full study and write every artifact.

    python scripts/train.py                    # bundled synthetic data
    python scripts/train.py --source real      # the real UCI dataset
    python scripts/train.py --quick            # 1 repeat, skip the slow model

Everything the paper cites comes out of this one command.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from mhrisk import config as C
from mhrisk import localization as L
from mhrisk import pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["auto", "real", "bundled"], default="auto",
                    help="auto prefers the real dataset and falls back to synthetic")
    ap.add_argument("--repeats", type=int, default=C.CV_REPEATS)
    ap.add_argument("--folds", type=int, default=C.CV_FOLDS)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="1 repeat and no neural model, for a fast sanity run")
    args = ap.parse_args()

    repeats = 1 if args.quick else args.repeats

    result = pipeline.run(
        source=args.source,
        n_repeats=repeats,
        n_splits=args.folds,
        outdir=args.outdir,
        include_slow=not args.quick,
    )

    print("\n" + "=" * 78)
    print("CROSS-VALIDATED RESULTS  (safety metrics first)")
    print("=" * 78)
    cols = ["high_risk_recall", "critical_miss_rate", "expected_cost",
            "referral_rate", "balanced_accuracy", "accuracy"]
    print(result.results_table[cols].to_string())

    print("\n" + "=" * 78)
    print("AT THE GUIDELINE BASELINE'S REFERRAL LOAD")
    print("=" * 78)
    print(result.matched_comparison.to_string(index=False))

    print("\n" + "=" * 78)
    print("GUIDELINE CALIBRATION")
    print("=" * 78)
    print(result.calibration.to_text())

    print("\n" + "=" * 78)
    print(f"Selected model : {result.best_model}")
    print(f"Operating point: {result.final_operating_point.describe()}")
    print(f"Attribution    : {result.importance_method}")

    unreviewed = L.unreviewed_bands()
    if unreviewed:
        print(f"\nNOTE: Urdu templates for {unreviewed} are not clinically signed off.")
    if result.dataset_info.is_synthetic:
        print(f"\n{result.dataset_info.banner()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
