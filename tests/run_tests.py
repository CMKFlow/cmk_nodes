"""Single entry point for the CMK Flow test suite.

Every test file runs in its own interpreter because several unit tests install
module aliases and stubs in ``sys.modules``.  The runner never deletes caches:
it snapshots the real persistent cache locations before and after the run and
fails when any of them changed.

Usage:
    python -I tests/run_tests.py [--tier T0,T1] [--only PATTERN] [--json PATH]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = TESTS_ROOT.parent
COMFY_ROOT = PACKAGE_ROOT.parents[1]
REPORTS = TESTS_ROOT / "_reports"

# Real cache locations that a test run must never modify.
PROTECTED_CACHE_ROOTS = (
    PACKAGE_ROOT / "cache",
    COMFY_ROOT / "temp" / "cmk",
)

# Files without an explicit CMK_TEST_TIER constant predate S1.  They neither
# load models nor start ComfyUI execution, so they belong to the static/unit tier.
DEFAULT_TIER = "T0"
FILE_TIMEOUT_SECONDS = 900


def declared_tier(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return DEFAULT_TIER
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "CMK_TEST_TIER" for t in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            return str(node.value.value)
    return DEFAULT_TIER


def snapshot(roots) -> dict[str, tuple[int, int]]:
    state = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name != ".DS_Store":
                stat = path.stat()
                state[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return state


def find_node() -> str | None:
    """Locate node without changing the system configuration."""
    explicit = os.environ.get("CMK_NODE")
    if explicit and Path(explicit).is_file():
        return explicit
    return shutil.which("node")


def run_python(path: Path) -> dict:
    # -I keeps user site-packages and PYTHON* variables out; the tests directory
    # and the ComfyUI root are therefore put on sys.path explicitly instead of
    # relying on the cwd (some tests import `comfy` as ComfyUI itself does).
    bootstrap = (
        "import sys, unittest; "
        f"sys.path[:0] = [{str(TESTS_ROOT)!r}, {str(COMFY_ROOT)!r}]; "
        f"unittest.main(module={path.stem!r}, argv=['unittest', '-v'])"
    )
    command = [sys.executable, "-I", "-c", bootstrap]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, cwd=PACKAGE_ROOT, capture_output=True, text=True,
            timeout=FILE_TIMEOUT_SECONDS,
        )
        output = completed.stdout + completed.stderr
        status = "PASS" if completed.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired as exc:
        output = f"{exc.stdout or ''}{exc.stderr or ''}\nTIMEOUT after {FILE_TIMEOUT_SECONDS}s"
        status = "TIMEOUT"
    return {
        "file": path.name,
        "kind": "python",
        "status": status,
        "seconds": round(time.perf_counter() - started, 2),
        "summary": unittest_summary(output),
        "output": output,
    }


def unittest_summary(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    tail = [
        line for line in lines
        if line.startswith(("Ran ", "OK", "FAILED"))
    ]
    return " | ".join(tail[-2:])


def run_node(path: Path, node: str | None) -> dict:
    if node is None:
        return {
            "file": path.name, "kind": "node", "status": "NOT RUN",
            "seconds": 0.0, "summary": "node executable not found (set CMK_NODE)",
            "output": "",
        }
    started = time.perf_counter()
    completed = subprocess.run(
        [node, str(path)], cwd=PACKAGE_ROOT, capture_output=True, text=True,
        timeout=FILE_TIMEOUT_SECONDS,
    )
    output = completed.stdout + completed.stderr
    return {
        "file": path.name, "kind": "node",
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "seconds": round(time.perf_counter() - started, 2),
        "summary": output.strip().splitlines()[-1] if output.strip() else "",
        "output": output,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="T0,T1", help="comma separated tiers to run")
    parser.add_argument("--only", default="", help="substring filter on file names")
    parser.add_argument("--json", default="", help="write a JSON report to this path")
    parser.add_argument("--no-node", action="store_true", help="skip .mjs tests")
    args = parser.parse_args(argv)
    tiers = {item.strip() for item in args.tier.split(",") if item.strip()}

    node = None if args.no_node else find_node()
    before = snapshot(PROTECTED_CACHE_ROOTS)
    results = []
    for path in sorted(TESTS_ROOT.glob("test_*.py")):
        if args.only and args.only not in path.name:
            continue
        tier = declared_tier(path)
        if tier not in tiers:
            results.append({
                "file": path.name, "kind": "python", "status": "SKIPPED",
                "seconds": 0.0, "summary": f"tier {tier} not selected", "output": "",
            })
            continue
        result = run_python(path)
        result["tier"] = tier
        results.append(result)
    if not args.no_node:
        for path in sorted(TESTS_ROOT.glob("test_*.mjs")):
            if args.only and args.only not in path.name:
                continue
            results.append(run_node(path, node))
    after = snapshot(PROTECTED_CACHE_ROOTS)

    changed = sorted(
        path for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )
    for result in results:
        print(f"{result['status']:8} {result['seconds']:8.2f}s  {result['file']:52} {result['summary']}")
    print(f"node: {node or 'not found'}")
    if changed:
        print("PROTECTED CACHE CHANGED:")
        for path in changed:
            print(f"  {path}")
    else:
        print("protected caches unchanged")

    report = {
        "python": sys.executable,
        "node": node,
        "tiers": sorted(tiers),
        "protected_cache_changes": changed,
        "results": results,
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    failed = [r for r in results if r["status"] in {"FAIL", "TIMEOUT"}]
    return 1 if failed or changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
