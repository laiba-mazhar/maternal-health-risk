"""Fetch the real UCI Maternal Health Risk Data Set into ``data/raw/``.

The dataset is free and needs no registration, so this is a plain download. If
the network is unavailable the script says exactly where to put a manually
downloaded file rather than failing silently -- and the pipeline keeps working on
the synthetic stand-in in the meantime.
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile

import _bootstrap  # noqa: F401  (path setup)

from mhrisk import config as C

UCI_ID = 863
SOURCES = [
    f"https://archive.ics.uci.edu/static/public/{UCI_ID}/maternal+health+risk.zip",
    f"https://archive.ics.uci.edu/static/public/{UCI_ID}/data.csv",
]

MANUAL_INSTRUCTIONS = f"""
Could not download automatically. To install the dataset by hand:

  1. Open https://archive.ics.uci.edu/dataset/{UCI_ID}/maternal+health+risk
     (or the Kaggle mirror: search "Maternal Health Risk Data Set")
  2. Download the CSV.
  3. Save it as:
       {C.RAW_CSV}

The file needs these columns:
  {", ".join(C.FEATURES + [C.TARGET])}

Until then everything still runs on the bundled synthetic dataset -- results are
just marked SYNTHETIC and must not be quoted as findings.
"""


def _try_download(url: str, timeout: int) -> bytes | None:
    import requests

    print(f"  trying {url}")
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "mhrisk/1.0"})
        resp.raise_for_status()
        return resp.content
    except Exception as exc:  # noqa: BLE001 - any failure means try the next source
        print(f"    failed: {type(exc).__name__}: {exc}")
        return None


def _extract_csv(payload: bytes) -> bytes | None:
    """Return CSV bytes from either a zip archive or a raw CSV response."""
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                print(f"    archive has no CSV: {zf.namelist()}")
                return None
            print(f"    extracting {names[0]}")
            return zf.read(names[0])
    if b"," in payload[:2000]:
        return payload
    print("    response is neither a zip nor a CSV")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    C.RAW_DIR.mkdir(parents=True, exist_ok=True)

    if C.RAW_CSV.exists() and not args.force:
        print(f"Already present: {C.RAW_CSV}  (use --force to re-download)")
        return 0

    print("Fetching the UCI Maternal Health Risk Data Set ...")
    for url in SOURCES:
        payload = _try_download(url, args.timeout)
        if payload is None:
            continue
        csv_bytes = _extract_csv(payload)
        if csv_bytes is None:
            continue

        C.RAW_CSV.write_bytes(csv_bytes)

        # Verify before declaring success -- a 200 response is not a valid dataset.
        import pandas as pd

        try:
            df = pd.read_csv(C.RAW_CSV)
        except Exception as exc:  # noqa: BLE001
            print(f"    downloaded file is not readable CSV: {exc}")
            C.RAW_CSV.unlink(missing_ok=True)
            continue

        missing = [c for c in C.FEATURES + [C.TARGET] if c not in df.columns]
        if missing:
            print(f"    unexpected columns; missing {missing}")
            print(f"    got: {list(df.columns)}")
            C.RAW_CSV.unlink(missing_ok=True)
            continue

        print(f"\nSaved {len(df)} rows to {C.RAW_CSV}")
        print(df[C.TARGET].value_counts().to_string())
        return 0

    print(MANUAL_INSTRUCTIONS)
    return 1


if __name__ == "__main__":
    sys.exit(main())
