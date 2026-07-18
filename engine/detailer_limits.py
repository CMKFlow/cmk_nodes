from __future__ import annotations

import math

DETAILER_DENOISE_MIN = 0.0001
DETAILER_DENOISE_MAX = 0.5
DETAILER_DENOISE_DEFAULT = 0.5


def clamp_detailer_denoise(value, default: float = DETAILER_DENOISE_DEFAULT) -> float:
    """Return the shared effective denoise range for every CMK detailer path."""
    try:
        result = float(value)
    except Exception:
        result = float(default)
    if not math.isfinite(result):
        result = float(default)
    return min(DETAILER_DENOISE_MAX, max(DETAILER_DENOISE_MIN, result))
