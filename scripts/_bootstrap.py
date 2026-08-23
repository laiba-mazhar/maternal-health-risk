"""Put ``src/`` on the path so the scripts run from a clean checkout.

Avoids requiring ``pip install -e .`` just to reproduce a result. Every script
imports this first.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
