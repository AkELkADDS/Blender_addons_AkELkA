"""GR2 codec package — flat imports (paths, bitknit, granny_anim, …)."""

from __future__ import annotations

import sys
from pathlib import Path

_CODEC_DIR = Path(__file__).resolve().parent
if str(_CODEC_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEC_DIR))
