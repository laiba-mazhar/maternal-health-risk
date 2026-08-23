"""Score one set of vitals from the command line.

    python scripts/predict.py --age 34 --sbp 148 --dbp 96 --bs 13.2 --temp 99.1 --hr 88

Prints the risk band, the drivers behind it, which guideline criteria fired, and
the bilingual message a health worker would actually deliver. Useful for spot
checks and for demonstrating the tool without launching the web app.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

import _bootstrap  # noqa: F401

from mhrisk import config as C
from mhrisk import explain as E
from mhrisk import guidelines as G
from mhrisk import localization as L
from mhrisk import metrics as M
from mhrisk import data as D
from mhrisk import pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--age", type=float, required=True)
    ap.add_argument("--sbp", type=float, required=True, help="systolic BP, mmHg")
    ap.add_argument("--dbp", type=float, required=True, help="diastolic BP, mmHg")
    ap.add_argument("--bs", type=float, required=True, help="blood glucose, mmol/L")
    ap.add_argument("--temp", type=float, required=True, help="body temperature, F")
    ap.add_argument("--hr", type=float, required=True, help="resting heart rate, bpm")
    ap.add_argument("--lang", choices=["ur", "en", "both"], default="both")
    ap.add_argument("--model", default=None, help="path to model.joblib")
    args = ap.parse_args()

    bundle = pipeline.load_bundle(args.model)
    model, point = bundle["model"], bundle["operating_point"]

    row = pd.Series(L.round_vitals({
        "Age": args.age, "SystolicBP": args.sbp, "DiastolicBP": args.dbp,
        "BS": args.bs, "BodyTemp": args.temp, "HeartRate": args.hr,
    }))[C.FEATURES]
    frame = row.to_frame().T.astype(float)

    proba = model.predict_proba(frame)[0]
    band_idx = int(M.decide(proba.reshape(1, -1), point)[0])
    band = C.INT_TO_LABEL[band_idx]

    # Background for the attribution method, from whichever dataset is available.
    raw, info = D.load_dataset("auto")
    background, _ = D.split_xy(D.clean(raw)[0])
    expl = E.explain_instance(model, row, background, predicted_class=band_idx)
    drivers = expl.drivers(3)

    print("=" * 72)
    print(f"Risk band      : {band.upper()}   ({L.BAND_NAMES[band]['ur']})")
    print(f"Probabilities  : " + "  ".join(
        f"{l}={p:.2f}" for l, p in zip(C.LABELS, proba)))
    print(f"Operating point: {point.describe()}")
    print(f"Model          : {bundle['metadata'].get('selected_model', '?')}"
          f"  (trained on {bundle['metadata'].get('data_source', '?')} data)")
    print()

    print("Drivers (%s):" % expl.method)
    print(expl.to_frame().to_string(index=False))
    print()

    flags = G.explain_flags(row)
    print("Guideline criteria met:")
    if flags:
        for f in flags:
            print(f"  [{f['severity']:8s}] {f['description']}  "
                  f"(observed {f['column']}="
                  f"{L.format_vital(f['column'], f['observed'])})")
            print(f"             source: {f['source']}")
    else:
        print("  none")
    print()

    messages = L.render_both(band, drivers, row.to_dict())
    for lang in (["en", "ur"] if args.lang == "both" else [args.lang]):
        label = "English" if lang == "en" else "Urdu"
        print(f"--- {label} message ({messages[lang].review}) ---")
        print(messages[lang].text)
        print()

    print(L.NOT_A_DEVICE["en"])
    if bundle["metadata"].get("data_is_synthetic"):
        print("WARNING: this model was trained on synthetic data. Output is a "
              "software demonstration, not a clinical opinion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
