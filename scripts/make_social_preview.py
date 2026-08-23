"""Generate the GitHub social preview card.

    python scripts/make_social_preview.py

Writes ``docs/img/social-preview.png`` at 1280x640, the size GitHub asks for.
Upload it at **Settings > General > Social preview** on the repository.

Why this exists: without an uploaded preview, GitHub advertises an
``og:image`` URL that can 404, so every link unfurl (LinkedIn, Slack, X,
WhatsApp) falls back to no image at all. A repository link is often the first
thing someone sees of a project; a blank card wastes that.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import _bootstrap  # noqa: F401

from mhrisk import config as C

OUT = C.ROOT / "docs" / "img" / "social-preview.png"
CHART = C.ROOT / "docs" / "img" / "fig5_recall_vs_referral.png"

W, H = 1280, 640
INK = "#16202a"
SOFT = "#5a6b78"
ACCENT = "#1f6f8b"
RULE = "#dde4e9"
BG = "#ffffff"

# Segoe UI ships with Windows; DejaVu with most Linux Pillow builds. Fall back
# rather than crash, since a slightly different face is better than no card.
FONT_CANDIDATES = {
    "bold": ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "semibold": ["C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeuib.ttf",
                 "C:/Windows/Fonts/arialbd.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "regular": ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES[kind]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def wrap(draw, text: str, fnt, max_width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def main() -> int:
    if not CHART.exists():
        print(f"Missing {CHART}. Run: python scripts/make_figures.py")
        return 1

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Accent spine down the left edge.
    d.rectangle([0, 0, 10, H], fill=ACCENT)

    pad = 56
    x = pad + 10
    text_w = 700

    # --- eyebrow ---
    f_eyebrow = font("semibold", 21)
    d.text((x, 54), "R&D  ·  APPLIED ML IN HEALTHCARE", font=f_eyebrow, fill=ACCENT)

    # --- title ---
    f_title = font("bold", 55)
    y = 96
    for line in wrap(d, "Maternal Health Risk Screening", f_title, text_w):
        d.text((x, y), line, font=f_title, fill=INK)
        y += 64

    # --- subtitle ---
    f_sub = font("regular", 26)
    y += 6
    for line in wrap(d, "Safety-first modelling, guideline calibration, "
                        "and Urdu risk communication", f_sub, text_w):
        d.text((x, y), line, font=f_sub, fill=SOFT)
        y += 36

    # --- the finding, which is the reason to click ---
    y += 22
    d.line([(x, y), (x + 96, y)], fill=RULE, width=3)
    y += 24
    f_find = font("semibold", 29)
    for line in wrap(d, "11 documented clinical thresholds beat "
                        "every model I trained.", f_find, text_w):
        d.text((x, y), line, font=f_find, fill=INK)
        y += 40

    # --- footer ---
    f_foot = font("regular", 22)
    d.text((x, H - 74), "github.com/laiba-mazhar/maternal-health-risk",
           font=f_foot, fill=ACCENT)
    d.text((x, H - 44), "Research prototype, not a medical device",
           font=font("regular", 19), fill=SOFT)

    # --- chart, bottom right ---
    chart = Image.open(CHART).convert("RGB")
    target_w = 430
    chart = chart.resize(
        (target_w, round(chart.height * target_w / chart.width)), Image.LANCZOS)
    img.paste(chart, (W - target_w - 34, H - chart.height - 30))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, optimize=True)
    print(f"Wrote {OUT}  ({W}x{H}, {OUT.stat().st_size // 1024} KB)")
    print("Upload at: repo > Settings > General > Social preview")
    return 0


if __name__ == "__main__":
    sys.exit(main())
