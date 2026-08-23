"""Capture interface screenshots for the README.

    python scripts/make_screenshots.py

Starts the Streamlit app on a spare port, drives it through Playwright, and
writes PNGs to ``docs/img/``. Screenshots are captured from deep links
(``?case=...&show=1``) rather than from scripted clicking, so a regenerated set
is byte-comparable rather than dependent on where a button happened to land.

Requires ``playwright`` and its Chromium build:

    pip install playwright && playwright install chromium
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from mhrisk import config as C

OUT = C.ROOT / "docs" / "img"
APP = C.ROOT / "app" / "streamlit_app.py"

# (filename, url suffix, viewport)
#
# Viewports are deliberately tall. Streamlit scrolls inside a fixed-height
# container rather than growing the document, so Playwright's ``full_page`` does
# not reach past the fold -- the only way to capture the whole screen is to give
# it a viewport tall enough that nothing has to scroll.
SHOTS = [
    ("ui-form.png", "", (1180, 1500)),
    ("ui-high-risk.png", "?case=high-glucose&show=1", (1180, 2950)),
    ("ui-low-risk.png", "?case=routine&show=1", (1180, 2650)),
    ("ui-raised-bp.png", "?case=raised-bp&show=1", (1180, 2950)),
    ("ui-mobile.png", "?case=high-glucose&show=1", (414, 3400)),
]

# Streamlit hydrates its widgets after the first paint, so a screenshot taken on
# ``load`` catches labelled boxes with no labels in them. Wait until the inputs
# actually carry values before shooting.
READY_JS = """() => {
  const inputs = [...document.querySelectorAll('input[type=number]')];
  if (inputs.length < 6) return false;
  if (inputs.some(i => i.value === '')) return false;
  const buttons = [...document.querySelectorAll('button')]
      .filter(b => (b.innerText || '').trim().length > 0);
  return buttons.length >= 4;
}"""

VALUES_JS = "() => [...document.querySelectorAll('input[type=number]')].map(i => i.value)"


def wait_until_settled(page, tries: int = 20, gap_ms: int = 500) -> list[str]:
    """Block until the form values stop changing.

    Streamlit paints a widget at its *minimum* before the session-state value
    arrives in a follow-up delta. Readiness alone is therefore not enough -- it
    happily passes on that first frame, and the screenshot then shows a form
    reading "Age 10, BP 60/30", which is not a state the app ever really shows a
    user. Sampling until two consecutive reads agree waits out the correction.
    """
    previous = None
    for _ in range(tries):
        current = page.evaluate(VALUES_JS)
        if current and current == previous:
            return current
        previous = current
        page.wait_for_timeout(gap_ms)
    return previous or []


def free_port(preferred: int = 8532) -> int:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def wait_for(port: int, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=0, help="0 picks a free port")
    ap.add_argument("--keep-open", action="store_true")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is required:\n"
              "  pip install playwright && playwright install chromium")
        return 1

    if not (C.ARTIFACTS_DIR / "model.joblib").exists():
        print("No trained model. Run: python scripts/train.py")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    port = args.port or free_port()

    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP),
         "--server.port", str(port), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        cwd=str(C.ROOT),
    )
    print(f"streamlit starting on :{port} (pid {server.pid})")

    try:
        if not wait_for(port):
            print("server did not come up in time")
            return 1

        with sync_playwright() as p:
            browser = p.chromium.launch()

            # Warm-up load, discarded. The first request pays for loading the
            # model and the dataset, and a screenshot taken during that shows an
            # empty shell -- which is exactly what happened before this existed.
            warm = browser.new_page()
            warm.goto(f"http://127.0.0.1:{port}/", wait_until="load")
            try:
                warm.wait_for_function(READY_JS, timeout=90_000)
            except Exception:  # noqa: BLE001
                print("  ! warm-up never became ready; shots may be incomplete")
            warm.close()

            for name, suffix, (w, h) in SHOTS:
                page = browser.new_page(viewport={"width": w, "height": h},
                                        device_scale_factor=2)
                page.goto(f"http://127.0.0.1:{port}/{suffix}", wait_until="load")

                try:
                    page.wait_for_function(READY_JS, timeout=60_000)
                except Exception:  # noqa: BLE001
                    print(f"  ! {name}: widgets never finished hydrating")

                if "show=1" in suffix:
                    try:
                        page.wait_for_selector(".hero", timeout=30_000)
                        page.wait_for_selector(".vitals .chip", timeout=30_000)
                    except Exception:  # noqa: BLE001
                        print(f"  ! {name}: result panel never appeared")

                values = wait_until_settled(page)

                # Nastaliq is a webfont; shooting before the swap gives a
                # fallback face, which misrepresents the whole point of the UI.
                try:
                    page.evaluate("() => document.fonts.ready")
                except Exception:  # noqa: BLE001
                    pass
                page.wait_for_timeout(1500)

                path = OUT / name
                page.screenshot(path=str(path))
                print(f"  {name}  ({w}x{h})  values={values}")
                page.close()
            browser.close()
    finally:
        if not args.keep_open:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
            print("streamlit stopped")

    print(f"\nWrote {len(SHOTS)} screenshots to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
