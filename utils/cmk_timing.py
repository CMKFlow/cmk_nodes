from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from time import perf_counter


@contextmanager
def cmk_timed(stage: str, detail: str = ""):
    """Emit lightweight wall-clock markers without touching workflow values."""
    label = str(stage).strip() or "UNNAMED"
    suffix = f" | {detail}" if detail else ""
    started = perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
    print(f"[CMK Timing] START {label} | {started_at}{suffix}")
    try:
        yield
    except BaseException:
        elapsed = perf_counter() - started
        ended_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        print(f"[CMK Timing] FAIL  {label} | {elapsed:.2f}s | {ended_at}{suffix}")
        raise
    else:
        elapsed = perf_counter() - started
        ended_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        print(f"[CMK Timing] END   {label} | {elapsed:.2f}s | {ended_at}{suffix}")


def cmk_timed_call(stage: str):
    """Decorate a ComfyUI node method while preserving its public signature."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with cmk_timed(stage):
                return function(*args, **kwargs)
        return wrapped
    return decorate
