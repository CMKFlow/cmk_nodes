#!/usr/bin/env python3
"""Update the generated compatibility blocks in both CMK Flow READMEs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


START_MARKER = "<!-- cmk-reference-environment:start -->"
END_MARKER = "<!-- cmk-reference-environment:end -->"
REFERENCE_FILENAME = "compatibility_reference.json"
EXPECTED_KEYS = {
    "schema_version",
    "comfyui",
    "python",
    "pytorch",
    "compute_backend",
}
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")


class CompatibilityUpdateError(RuntimeError):
    """Raised when compatibility documentation cannot be updated safely."""


@dataclass(frozen=True)
class ReferenceEnvironment:
    comfyui: str
    python: str
    pytorch: str
    compute_backend: str


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CompatibilityUpdateError(
                f"duplicate key in {REFERENCE_FILENAME}: {key}"
            )
        result[key] = value
    return result


def load_reference(path: Path) -> ReferenceEnvironment:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except FileNotFoundError as exc:
        raise CompatibilityUpdateError(f"missing reference data: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CompatibilityUpdateError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise CompatibilityUpdateError(f"reference data must be a JSON object: {path}")
    missing = sorted(EXPECTED_KEYS - payload.keys())
    unexpected = sorted(payload.keys() - EXPECTED_KEYS)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected keys: {', '.join(unexpected)}")
        raise CompatibilityUpdateError(f"invalid reference data in {path}: {'; '.join(details)}")
    if payload["schema_version"] != 1:
        raise CompatibilityUpdateError(
            f"unsupported schema_version in {path}: {payload['schema_version']!r}"
        )

    for key in ("comfyui", "python", "pytorch"):
        value = payload[key]
        if not isinstance(value, str) or not VERSION_PATTERN.fullmatch(value):
            raise CompatibilityUpdateError(
                f"invalid confirmed version for {key} in {path}: {value!r}"
            )
    backend = payload["compute_backend"]
    if not isinstance(backend, str) or not backend.strip() or "\n" in backend:
        raise CompatibilityUpdateError(
            f"invalid confirmed compute_backend in {path}: {backend!r}"
        )

    return ReferenceEnvironment(
        comfyui=payload["comfyui"],
        python=payload["python"],
        pytorch=payload["pytorch"],
        compute_backend=backend,
    )


def load_package_version(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CompatibilityUpdateError(f"missing package metadata: {path}") from exc
    project = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", text)
    if not project:
        raise CompatibilityUpdateError(f"pyproject.toml has no [project] table: {path}")
    matches = re.findall(r'(?m)^version\s*=\s*"([^"]+)"\s*$', project.group(1))
    if len(matches) != 1 or not VERSION_PATTERN.fullmatch(matches[0]):
        raise CompatibilityUpdateError(
            f"pyproject.toml must contain exactly one valid static project version: {path}"
        )
    return matches[0]


def render_block(language: str, cmk_version: str, reference: ReferenceEnvironment) -> str:
    if language == "de":
        return f"""| Komponente | Referenzwert |
| --- | --- |
| CMK Flow | `{cmk_version}` |
| ComfyUI | `{reference.comfyui}` |
| Python | `{reference.python}` |
| PyTorch | `{reference.pytorch}` |
| Compute-Backend | `{reference.compute_backend}` |

Diese Werte beschreiben die aktuelle validierte Referenzumgebung. Sie sind weder
pauschale Mindestanforderungen noch eine vollständige Kompatibilitätsgarantie
und implizieren keine vollständige Prüfung sämtlicher Workflows, Erweiterungen
oder Hardwareplattformen."""
    if language == "en":
        return f"""| Component | Reference value |
| --- | --- |
| CMK Flow | `{cmk_version}` |
| ComfyUI | `{reference.comfyui}` |
| Python | `{reference.python}` |
| PyTorch | `{reference.pytorch}` |
| Compute backend | `{reference.compute_backend}` |

These values describe the current validated reference environment. They are
neither general minimum requirements nor a complete compatibility guarantee,
and they do not imply complete testing of every workflow, extension, or
hardware platform."""
    raise CompatibilityUpdateError(f"unsupported README language: {language}")


def update_marked_block(text: str, block: str, path: Path) -> str:
    if text.count(START_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise CompatibilityUpdateError(
            f"{path} must contain exactly one compatibility marker pair"
        )
    start = text.index(START_MARKER) + len(START_MARKER)
    end = text.index(END_MARKER)
    if start >= end:
        raise CompatibilityUpdateError(f"compatibility markers are out of order: {path}")
    return text[:start] + "\n" + block.rstrip() + "\n" + text[end:]


def generated_readmes(root: Path, reference_path: Path | None = None):
    reference = load_reference(reference_path or root / REFERENCE_FILENAME)
    cmk_version = load_package_version(root / "pyproject.toml")
    result = {}
    for filename, language in (("README.md", "de"), ("README.en.md", "en")):
        path = root / filename
        try:
            original = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise CompatibilityUpdateError(f"missing README: {path}") from exc
        result[path] = update_marked_block(
            original, render_block(language, cmk_version, reference), path
        )
    return result


def run(root: Path, reference_path: Path | None = None, check: bool = False) -> int:
    generated = generated_readmes(root, reference_path)
    stale = [path for path, text in generated.items() if path.read_text(encoding="utf-8") != text]
    if check:
        if stale:
            raise CompatibilityUpdateError(
                "compatibility documentation is out of date: "
                + ", ".join(str(path) for path in stale)
            )
        print("README compatibility blocks are up to date.")
        return 0
    for path in stale:
        path.write_text(generated[path], encoding="utf-8")
        print(f"Updated {path}")
    if not stale:
        print("README compatibility blocks already match the reference data.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="CMK Flow repository root (defaults to the script's repository)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help=f"explicit {REFERENCE_FILENAME} path (primarily for validation)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if either README does not match the confirmed reference data",
    )
    args = parser.parse_args(argv)
    try:
        return run(args.root.resolve(), args.reference, args.check)
    except (CompatibilityUpdateError, OSError) as exc:
        print(f"README compatibility update FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
