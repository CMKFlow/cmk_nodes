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


# Gate decisions read from the expanded prompt before ComfyUI forms cache keys.
# The prompt gate pruning and the SDXL completion graph share these helpers so
# that a gate's activation rule exists only once.
GATE_ACTIVE = "active"
GATE_INACTIVE = "inactive"
GATE_UNKNOWN = "unknown"
MODULE_GATE_CLASS = "CMKModuleBypassGate"
INSTANTID_GATE_CLASS = "CMKSamplerBypassGate"
PRUNABLE_GATE_CLASSES = frozenset({MODULE_GATE_CLASS, INSTANTID_GATE_CLASS})


def prompt_node(prompt: Any, node_id: Any):
    """The node behind an id of an expanded prompt, or None."""
    if not isinstance(prompt, Mapping):
        return None
    node = prompt.get(node_id)
    if not isinstance(node, Mapping):
        node = prompt.get(str(node_id))
    return node if isinstance(node, Mapping) else None


def prompt_link(prompt: Any, value: Any) -> bool:
    """True for an input value that links to a node of the prompt."""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], int)
        and prompt_node(prompt, value[0]) is not None
    )


def prompt_process_sources(prompt: Any, process_link: Any) -> dict:
    """Every module 01 node a PROCESS link originates from, keyed by node id."""
    pending = [process_link[0]] if prompt_link(prompt, process_link) else []
    visited = set()
    sources = {}
    while pending:
        node_id = pending.pop()
        key = str(node_id)
        if key in visited:
            continue
        visited.add(key)
        node = prompt_node(prompt, node_id)
        if node is None:
            continue
        if str(node.get("class_type", "")) == "CMKPipeCreateImage":
            sources[key] = node
            continue
        for value in (node.get("inputs", {}) or {}).values():
            if prompt_link(prompt, value):
                pending.append(value[0])
    return sources


def _instantid_gate_prompt_state(prompt: Mapping[Any, Any], gate_id: Any, enable: Any) -> str:
    """CMKProcessEnableFlag(instantid_enabled) as module 01 will resolve it.

    PROCESS["instantid_enabled"] is written only by CMKPipeCreateImage, as
    ``bool(widget) and model_family == "sdxl"``; the flag reads False while
    STOP AFTER HANDOFF is effective.  The prompt decides only when every
    deciding input is a literal of one unambiguous module 01 node.
    """
    if not prompt_link(prompt, enable):
        return GATE_UNKNOWN
    flag = prompt_node(prompt, enable[0])
    flag_inputs = flag.get("inputs") if flag is not None else None
    if (
        str(flag.get("class_type", "")) != "CMKProcessEnableFlag"
        or not isinstance(flag_inputs, Mapping)
        or flag_inputs.get("flag") != "instantid_enabled"
    ):
        return GATE_UNKNOWN
    sources = prompt_process_sources(prompt, flag_inputs.get("PROCESS"))
    starts = [
        node for node in _prompt_nodes_upstream_of(prompt, gate_id)
        if isinstance(node, Mapping) and str(node.get("class_type", "")) == "CMKPipeCreateImage"
    ]
    if len(sources) != 1 or len(starts) != 1 or next(iter(sources.values())) is not starts[0]:
        return GATE_UNKNOWN
    start_inputs = starts[0].get("inputs")
    if not isinstance(start_inputs, Mapping):
        return GATE_UNKNOWN
    enabled = start_inputs.get("instantid_enabled")
    if enabled is False:
        return GATE_INACTIVE
    if enabled is not True:
        return GATE_UNKNOWN
    family = start_inputs.get("model_family")
    if not isinstance(family, str):
        return GATE_UNKNOWN
    if _start_model_family(family) != "sdxl":
        return GATE_INACTIVE
    stop = start_inputs.get(STOP_AFTER_HANDOFF_WIDGET, False)
    if not isinstance(stop, bool):
        return GATE_UNKNOWN
    if not stop:
        return GATE_ACTIVE
    for name in ("INPAINT_MODE", "instantid_mode"):
        if name in start_inputs and not isinstance(start_inputs[name], (str, bool)):
            return GATE_UNKNOWN
    return GATE_INACTIVE if stop_after_handoff_prompt_state(prompt, gate_id) else GATE_ACTIVE


def gate_prompt_state(prompt: Any, gate_id: Any) -> str:
    """Whether a CMK gate will select its ACTIVE inputs in this prompt.

    Module gates decide ``bool(ENABLE) and not STOP`` at runtime; with a
    literal ENABLE the prompt holds exactly that decision.  Anything that the
    prompt cannot decide is GATE_UNKNOWN.
    """
    node = prompt_node(prompt, gate_id)
    if node is None:
        return GATE_UNKNOWN
    class_type = str(node.get("class_type", ""))
    inputs = node.get("inputs")
    if class_type not in PRUNABLE_GATE_CLASSES or not isinstance(inputs, Mapping):
        return GATE_UNKNOWN
    enable = inputs.get("ENABLE")
    if enable is False:
        return GATE_INACTIVE
    if class_type == MODULE_GATE_CLASS:
        if enable is True:
            return GATE_INACTIVE if stop_after_handoff_prompt_state(prompt, gate_id) else GATE_ACTIVE
        return GATE_UNKNOWN
    if enable is True:
        return GATE_ACTIVE
    return _instantid_gate_prompt_state(prompt, gate_id, enable)
