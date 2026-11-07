from __future__ import annotations

from collections.abc import Mapping
from typing import Any


BASELINE = "BASELINE"
RELEASE = "RELEASE"

# This is the single source-controlled distribution switch.  A public release
# must set it to RELEASE before the release commit is built and validated.
DISTRIBUTION_MODE = BASELINE

CONTENT_GUARD_PROCESS_KEY = "content_guard_enabled"
CONTENT_GUARD_WIDGET = "content_guard_enabled"
STOP_AFTER_HANDOFF_PROCESS_KEY = "stop_after_handoff"
STOP_AFTER_HANDOFF_WIDGET = "stop_after_handoff"


def distribution_mode() -> str:
    mode = str(DISTRIBUTION_MODE or "").strip().upper()
    if mode not in {BASELINE, RELEASE}:
        raise RuntimeError(f"Invalid CMK distribution mode: {DISTRIBUTION_MODE!r}")
    return mode


def content_guard_control_visible() -> bool:
    return distribution_mode() == BASELINE


def effective_content_guard_enabled(
    process: Mapping[str, Any] | None = None,
) -> bool:
    """Resolve the authoritative guard state at an execution boundary.

    RELEASE deliberately ignores serialized and runtime PROCESS values.  In
    BASELINE, missing/legacy values remain fail-closed by defaulting to ON.
    """
    if distribution_mode() == RELEASE:
        return True
    if not isinstance(process, Mapping):
        return True
    return bool(process.get(CONTENT_GUARD_PROCESS_KEY, True))


def content_guard_state_token(
    process: Mapping[str, Any] | None = None,
) -> str:
    enabled = effective_content_guard_enabled(process)
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
    # Module 01 owns the switch in a generation flow; a standalone FaceSwap
    # image flow has it on its image loader instead.
    for class_type in ("CMKPipeCreateImage", "CMKSwapImageLoaderPipe"):
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            if str(node.get("class_type", "")) != class_type:
                continue
            inputs = node.get("inputs")
            if isinstance(inputs, Mapping):
                return bool(inputs.get(CONTENT_GUARD_WIDGET, True))
    return True


def content_guard_prompt_token(prompt: Any, root_node_id: Any = None) -> str:
    enabled = content_guard_prompt_state(prompt, root_node_id)
    return f"{distribution_mode().lower()}:{'on' if enabled else 'off'}"


def _start_inpaint_mode(value: Any) -> bool:
    """Module 01's INPAINT_MODE, as CMKPipeCreateImage resolves it."""
    if isinstance(value, bool):
        return value
    return str(value or "Text2Image").strip().lower() == "inpaint"


def _start_model_family(value: Any) -> str:
    """Module 01's model family, as CMKPipeCreateImage resolves it."""
    family = str(value or "SDXL").strip().lower()
    if family == "hybrid":
        return "hybrid"
    if family in {"z-image turbo", "z image turbo", "z_image_turbo"}:
        return "z_image_turbo"
    return "sdxl"


def start_instantid_mode(inpaint_mode: Any, requested_mode: Any) -> str:
    """Effective InstantID mode.  Inpaint ignores the hidden mode widget."""
    if _start_inpaint_mode(inpaint_mode):
        return "HYBRID INPAINT"
    requested = str(requested_mode or "HYBRID").strip().upper()
    return "ZERO-PASS" if requested == "ZERO-PASS" else "HYBRID"


def stop_after_handoff_active(process: Mapping[str, Any] | None = None) -> bool:
    """Return the effective preview-only state for a supported handoff."""
    if not isinstance(process, Mapping):
        return False
    if not process.get(STOP_AFTER_HANDOFF_PROCESS_KEY, False):
        return False
    if process.get("hybrid_mode", False):
        return True
    return bool(
        process.get("instantid_enabled", False)
        and str(process.get("instantid_mode", "HYBRID") or "HYBRID")
        .strip()
        .upper()
        in {"HYBRID", "HYBRID INPAINT"}
        and not process.get("hybrid_mode", False)
        and str(process.get("model_family", "sdxl")).strip().lower() == "sdxl"
    )


def instantid_inpaint_prompt_active(prompt: Any) -> bool:
    """True when module 01 selects SDXL -> InstantID in Inpaint mode."""
    if not isinstance(prompt, Mapping):
        return False
    for node in prompt.values():
        if not isinstance(node, Mapping):
            continue
        if str(node.get("class_type", "")) != "CMKPipeCreateImage":
            continue
        inputs = node.get("inputs")
        if (
            isinstance(inputs, Mapping)
            and inputs.get("instantid_enabled", False)
            and _start_inpaint_mode(inputs.get("INPAINT_MODE", "Text2Image"))
            and _start_model_family(inputs.get("model_family", "SDXL")) == "sdxl"
        ):
            return True
    return False


def stop_after_handoff_prompt_state(
    prompt: Any,
    root_node_id: Any = None,
) -> bool:
    """Read module 01's preview-only request without executing the graph."""
    if not isinstance(prompt, Mapping):
        return False
    nodes = (
        _prompt_nodes_upstream_of(prompt, root_node_id)
        if root_node_id is not None
        else prompt.values()
    )
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        if str(node.get("class_type", "")) != "CMKPipeCreateImage":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        # The same normalization CMKPipeCreateImage applies to PROCESS, so a
        # hidden or unsupported widget value cannot produce another decision.
        family = _start_model_family(inputs.get("model_family", "SDXL"))
        return stop_after_handoff_active({
            STOP_AFTER_HANDOFF_PROCESS_KEY: inputs.get(
                STOP_AFTER_HANDOFF_WIDGET, False
            ),
            "instantid_enabled": inputs.get("instantid_enabled", False),
            "instantid_mode": start_instantid_mode(
                inputs.get("INPAINT_MODE", "Text2Image"),
                inputs.get("instantid_mode", "HYBRID"),
            ),
            "model_family": family,
            "hybrid_mode": family == "hybrid",
        })
    return False
