"""Run a harness scenario in a fresh interpreter (simulated ComfyUI restart).

A fresh process starts with an empty ComfyUI output cache and empty
module-level session state, while the work directory (input/output/temp and
the redirected package cache) is preserved between calls.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent
TESTS_ROOT = HARNESS_ROOT.parent
RESULT_MARKER = "CMK_HARNESS_RESULT="


def run_scenario(name: str, workdir: Path, timeout: int = 600, **kwargs) -> dict:
    bootstrap = (
        "import json, sys; "
        f"sys.path.insert(0, {str(TESTS_ROOT)!r}); "
        "from harness import scenarios; "
        f"result = getattr(scenarios, {name!r})({str(workdir)!r}, **json.loads({json.dumps(json.dumps(kwargs))})); "
        f"print({RESULT_MARKER!r} + json.dumps(result))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", bootstrap],
        capture_output=True, text=True, timeout=timeout,
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER):])
    raise RuntimeError(
        f"scenario {name} produced no result (exit {completed.returncode}):\n"
        f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
    )
