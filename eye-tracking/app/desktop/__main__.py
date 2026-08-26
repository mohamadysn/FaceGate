"""Launch: python -m app.desktop"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.desktop import run_app

if __name__ == "__main__":
    run_app()
