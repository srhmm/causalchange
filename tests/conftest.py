from __future__ import annotations

import sys
from pathlib import Path

# Lets the tests run directly from a checkout/zip without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
