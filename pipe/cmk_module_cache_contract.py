from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


ARTIFACTS_FIELD = "cmk_cache_artifacts"
CONTRACT_VERSION = 1


def _canonical(value: Any) -> Any:
    """Return a deterministic JSON value for effective cache inputs."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    raise TypeError(
        "CMK module cache identity accepts only deterministic scalar, mapping, "
        f"list, or tuple values; received {type(value).__name__}"
    )


def build_artifact_key(
    stage_key: str,
    upstream_artifact: str,
    settings: Mapping[str, Any],
    *,
    schema: str,
) -> str:
    """Build a topology-independent identity for one module result.

    ``upstream_artifact`` identifies the effective public result consumed by the
    module. ``settings`` contains only options that can alter the public result.
    Runtime node ids, graph paths, UI state, VISUAL wiring, and log wiring are
    deliberately not accepted as separate identity components.
    """
    stage = str(stage_key or "").strip()
    upstream = str(upstream_artifact or "").strip()
    schema_name = str(schema or "").strip()
    if not stage:
        raise ValueError("stage_key must not be empty")
    if not upstream:
        raise ValueError("upstream_artifact must not be empty")
    if not schema_name:
        raise ValueError("schema must not be empty")

    payload = {
        "contract": CONTRACT_VERSION,
        "schema": schema_name,
        "stage": stage,
        "upstream": upstream,
        "settings": _canonical(settings),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_for(process: Mapping[str, Any] | None, stage_key: str) -> str | None:
    if not isinstance(process, Mapping):
        return None
    artifacts = process.get(ARTIFACTS_FIELD)
    if not isinstance(artifacts, Mapping):
        return None
    value = artifacts.get(str(stage_key))
    return str(value) if isinstance(value, str) and value else None


def stamp_artifact(
    process: Mapping[str, Any],
    stage_key: str,
    artifact_key: str,
) -> dict[str, Any]:
    """Copy a PROCESS pipe and add one public module-result identity."""
    if not isinstance(process, Mapping):
        raise TypeError("PROCESS must be a mapping")
    stage = str(stage_key or "").strip()
    artifact = str(artifact_key or "").strip()
    if not stage or not artifact:
        raise ValueError("stage_key and artifact_key must not be empty")

    result = dict(process)
    existing = process.get(ARTIFACTS_FIELD)
    artifacts = dict(existing) if isinstance(existing, Mapping) else {}
    artifacts[stage] = artifact
    result[ARTIFACTS_FIELD] = artifacts
    return result
