from __future__ import annotations

from collections.abc import Mapping
from typing import Any


BASELINE = "BASELINE"
RELEASE = "RELEASE"

# This is the single source-controlled distribution switch.  A public release
# must set it to RELEASE before the release commit is built and validated.
DISTRIBUTION_MODE = RELEASE

CONTENT_GUARD_PROCESS_KEY = "content_guard_enabled"
CONTENT_GUARD_WIDGET = "content_guard_enabled"
CONTENT_GUARD_EXECUTION_KEY = "_cmk_content_guard_enabled"


def distribution_mode() -> str:
    mode = str(DISTRIBUTION_MODE or "").strip().upper()
    if mode not in {BASELINE, RELEASE}:
        raise RuntimeError(f"Invalid CMK distribution mode: {DISTRIBUTION_MODE!r}")
    return mode


def content_guard_control_visible() -> bool:
    return distribution_mode() == BASELINE


def effective_content_guard_enabled(
    process: Mapping[str, Any] | None = None,
    execution_value: Any = None,
) -> bool:
    """Resolve the authoritative guard state at an execution boundary.

    RELEASE deliberately ignores serialized and runtime PROCESS values.  In
    BASELINE, missing/legacy values remain fail-closed by defaulting to ON.
    """
    if distribution_mode() == RELEASE:
        return True
    if execution_value is not None:
        return bool(execution_value)
    if not isinstance(process, Mapping):
        return True
    return bool(process.get(CONTENT_GUARD_PROCESS_KEY, True))


def content_guard_state_token(
    process: Mapping[str, Any] | None = None,
    execution_value: Any = None,
) -> str:
    enabled = effective_content_guard_enabled(process, execution_value)
    return f"{distribution_mode().lower()}:{'on' if enabled else 'off'}"


def _prompt_nodes_upstream_of(prompt: Mapping[Any, Any], root_node_id: Any):
    nodes_by_id = {str(node_id): node for node_id, node in prompt.items()}
    pending = [str(root_node_id)]
    visited = set()
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = nodes_by_id.get(node_id)
        if not isinstance(node, Mapping):
            continue
        yield node
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        for value in inputs.values():
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and str(value[0]) in nodes_by_id
            ):
                pending.append(str(value[0]))


def content_guard_prompt_state(prompt: Any, root_node_id: Any = None) -> bool:
    """Read module 01's state globally or from one node's upstream graph."""
    if distribution_mode() == RELEASE:
        return True
    if not isinstance(prompt, Mapping):
        return True
    nodes = (
        _prompt_nodes_upstream_of(prompt, root_node_id)
        if root_node_id is not None
        else prompt.values()
    )
    nodes = list(nodes)
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, Mapping) and CONTENT_GUARD_EXECUTION_KEY in inputs:
            return effective_content_guard_enabled(
                execution_value=inputs.get(CONTENT_GUARD_EXECUTION_KEY),
            )
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        if str(node.get("class_type", "")) != "CMKPipeCreateImage":
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, Mapping):
            return bool(inputs.get(CONTENT_GUARD_WIDGET, True))
    return True


def content_guard_prompt_token(prompt: Any, root_node_id: Any = None) -> str:
    enabled = content_guard_prompt_state(prompt, root_node_id)
    return f"{distribution_mode().lower()}:{'on' if enabled else 'off'}"
