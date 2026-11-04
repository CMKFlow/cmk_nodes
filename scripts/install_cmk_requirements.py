#!/usr/bin/env python3
"""Install and verify CMK dependencies in the Python used by ComfyUI.

The script is intentionally safe to launch with a system Python. It locates the
ComfyUI root from the installed repository path and delegates pip to ComfyUI's
own virtual environment instead of guessing from the caller's shell.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


REQUIRED_IMPORTS = (
    "cv2",
    "imageio_ffmpeg",
    "insightface",
    "nudenet",
    "onnxruntime",
    "segment_anything",
    "ultralytics",
)


def find_comfy_root(repo_root: Path) -> Path:
    """Return the ComfyUI directory containing this custom-node checkout."""

    repo_root = repo_root.resolve()
    direct = repo_root.parent.parent
    if repo_root.parent.name == "custom_nodes" and (direct / "main.py").is_file():
        return direct

    for parent in repo_root.parents:
        if (parent / "main.py").is_file() and (parent / "custom_nodes").is_dir():
            return parent

    raise RuntimeError(
        "CMK must be installed below a ComfyUI custom_nodes directory before its "
        "runtime dependencies can be selected safely."
    )


def python_candidates(comfy_root: Path, current_executable: Path) -> Iterable[Path]:
    """Yield supported ComfyUI interpreters in deterministic preference order."""

    install_root = comfy_root.parent
    candidates = (
        comfy_root / ".venv" / "bin" / "python3",
        comfy_root / ".venv" / "bin" / "python",
        comfy_root / ".venv" / "Scripts" / "python.exe",
        comfy_root / "venv" / "bin" / "python3",
        comfy_root / "venv" / "bin" / "python",
        comfy_root / "venv" / "Scripts" / "python.exe",
        install_root / "python_embeded" / "python.exe",
        install_root / "standalone-env" / "bin" / "python3",
        install_root / "standalone-env" / "bin" / "python",
        current_executable,
    )
    seen: set[Path] = set()
    for candidate in candidates:
        # Preserve virtual-environment launcher symlinks. Resolving
        # ComfyUI/.venv/bin/python to the base interpreter makes Python lose
        # the .venv context and installs into the wrong environment.
        candidate = candidate.expanduser().absolute()
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def find_comfy_python(comfy_root: Path, current_executable: Path | None = None) -> Path:
    current = current_executable or Path(sys.executable)
    for candidate in python_candidates(comfy_root, current):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("No usable ComfyUI Python interpreter was found.")


def verify_dependencies(python: Path) -> None:
    imports = "; ".join(f"import {name}" for name in REQUIRED_IMPORTS)
    subprocess.run([str(python), "-c", imports], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify the required imports without running pip",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selected interpreter and pip command without changing it",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    requirements = repo_root / "requirements.txt"
    comfy_root = find_comfy_root(repo_root)
    python = find_comfy_python(comfy_root)
    pip_command = [str(python), "-m", "pip", "install", "-r", str(requirements)]

    print(f"CMK repository : {repo_root}")
    print(f"ComfyUI root   : {comfy_root}")
    print(f"ComfyUI Python : {python}")

    if args.dry_run:
        print("Pip command    : " + " ".join(pip_command))
        return 0

    if not args.check_only:
        subprocess.run(pip_command, check=True)

    verify_dependencies(python)
    print("CMK dependency check: OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"CMK dependency setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
