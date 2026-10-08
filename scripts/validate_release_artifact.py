#!/usr/bin/env python3
"""Validate a CMK source tree or packed Registry artifact before publishing."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path


class ValidationError(RuntimeError):
    pass


class ContentReader:
    def __init__(self, path: Path):
        self.path = path
        self.is_archive = path.is_file()
        self._zip = zipfile.ZipFile(path) if self.is_archive else None
        self._names = set(self._zip.namelist()) if self._zip else set()

    def close(self):
        if self._zip:
            self._zip.close()

    def names(self):
        if self._zip:
            return sorted(name for name in self._names if not name.endswith("/"))
        return sorted(
            item.relative_to(self.path).as_posix()
            for item in self.path.rglob("*")
            if item.is_file() and ".git" not in item.relative_to(self.path).parts
        )

    def exists(self, relative: str):
        return relative in self._names if self._zip else (self.path / relative).is_file()

    def read_bytes(self, relative: str):
        if not self.exists(relative):
            raise ValidationError(f"missing required file: {relative}")
        return self._zip.read(relative) if self._zip else (self.path / relative).read_bytes()

    def read_text(self, relative: str):
        return self.read_bytes(relative).decode("utf-8")


def nested_key(data, dotted_key: str):
    current = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def sha256(data: bytes):
    return hashlib.sha256(data).hexdigest()


def project_version(pyproject_text: str):
    project = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", pyproject_text)
    if not project:
        raise ValidationError("pyproject.toml has no [project] table")
    version = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', project.group(1))
    if not version:
        raise ValidationError("pyproject.toml has no static project version")
    return version.group(1)


def expected_source_files(source: Path):
    patterns = []
    ignore_file = source / ".comfyignore"
    if ignore_file.is_file():
        patterns = [
            line.strip()
            for line in ignore_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def ignored(relative: str):
        parts = Path(relative).parts
        if any(part in {".git", ".ruff_cache", "__pycache__"} for part in parts):
            return True
        if relative.endswith((".pyc", ".DS_Store")) or relative == "node.zip":
            return True
        for pattern in patterns:
            prefix = pattern.rstrip("/")
            if pattern.endswith("/") and (relative == prefix or relative.startswith(prefix + "/")):
                return True
            if fnmatch.fnmatch(relative, pattern):
                return True
        return False

    return {
        item.relative_to(source).as_posix()
        for item in source.rglob("*")
        if item.is_file() and not ignored(item.relative_to(source).as_posix())
    }


def validate(reader: ContentReader, manifest: dict, source: Path | None = None):
    errors = []
    required = list(manifest.get("requiredFiles", []))
    for feature in manifest.get("releaseFeatures", []):
        required.extend(feature.get("requiredFiles", []))
    required.append("release_manifest.json")

    for relative in dict.fromkeys(required):
        if not reader.exists(relative):
            errors.append(f"missing required file: {relative}")

    names = reader.names()
    if reader.is_archive:
        for forbidden in manifest.get("forbiddenArtifactPaths", []):
            prefix = forbidden.rstrip("/")
            if any(name == prefix or name.startswith(prefix + "/") for name in names):
                errors.append(f"forbidden artifact content: {prefix}")

    for feature in manifest.get("releaseFeatures", []):
        for relative, markers in feature.get("requiredMarkers", {}).items():
            try:
                content = reader.read_text(relative)
            except ValidationError as exc:
                errors.append(str(exc))
                continue
            for marker in markers:
                if marker not in content:
                    errors.append(f"{feature['name']}: marker missing in {relative}: {marker}")
        for relative, dotted_key in feature.get("requiredJsonKeys", {}).items():
            try:
                data = json.loads(reader.read_text(relative))
            except (ValidationError, json.JSONDecodeError) as exc:
                errors.append(f"invalid JSON {relative}: {exc}")
                continue
            if not nested_key(data, dotted_key):
                errors.append(f"{feature['name']}: JSON key missing in {relative}: {dotted_key}")

    try:
        version = project_version(reader.read_text("pyproject.toml"))
    except ValidationError as exc:
        errors.append(f"invalid pyproject.toml: {exc}")
        version = None
    if version:
        for relative, phrase in (
            ("README.md", f"CMK {version}"),
            ("README.en.md", f"CMK {version}"),
            ("CHANGELOG.md", f"CMK {version}"),
        ):
            try:
                if phrase not in reader.read_text(relative):
                    errors.append(f"version {version} missing from {relative}")
            except ValidationError as exc:
                errors.append(str(exc))

    if source is not None:
        expected = expected_source_files(source)
        artifact_names = set(names)
        for relative in sorted(expected - artifact_names):
            errors.append(f"source file missing from artifact: {relative}")
        for relative in sorted(artifact_names - expected):
            errors.append(f"artifact contains unexpected file: {relative}")
        for relative in sorted(expected & artifact_names):
            source_file = source / relative
            if sha256(source_file.read_bytes()) != sha256(reader.read_bytes(relative)):
                errors.append(f"artifact/source mismatch: {relative}")

    if errors:
        raise ValidationError("\n".join(f"- {error}" for error in errors))
    return version, len(names), len(set(required))


def git_identity(source: Path):
    try:
        root = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "--show-toplevel"], text=True
        ).strip()
        head = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(source), "branch", "--show-current"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError(f"source is not an identifiable Git worktree: {exc}") from exc
    return root, branch or "DETACHED", head


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path, help="source directory or packed .zip artifact")
    parser.add_argument("--source", type=Path, help="source tree used to build an artifact")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args(argv)

    target = args.target.resolve()
    source = (args.source or (target if target.is_dir() else None))
    if source is None:
        parser.error("--source is required when target is an archive")
    source = source.resolve()
    manifest_path = source / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root, branch, head = git_identity(source)
    if Path(root) != source:
        raise ValidationError(f"source must be the Git worktree root: {source} (found {root})")
    if args.require_clean:
        status = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        )
        if status.strip():
            raise ValidationError("release worktree is not clean")

    reader = ContentReader(target)
    try:
        version, files, required = validate(
            reader, manifest, source if target.is_file() else None
        )
    finally:
        reader.close()
    print(f"CMK release validation OK: version={version} branch={branch} head={head}")
    print(f"Validated {required} required components across {files} packaged files.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, OSError, json.JSONDecodeError) as exc:
        print(f"CMK release validation FAILED:\n{exc}", file=sys.stderr)
        raise SystemExit(1)
