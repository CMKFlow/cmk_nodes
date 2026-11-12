"""Inline the selection of CMK Detailer Family Enable into its consumers.

A control node linked to the ``model_family`` widget of the Multi-Model
Detailer Prepare would otherwise reach the backend as a link.  Gate pruning
and the Detailer boundary (disabled state, retention slot) and the persistent
fingerprints read ``model_family`` as the literal family name, exactly as the
Prepare dropdown delivers it.  This prompt handler replaces every link from
the control node's ``MODEL FAMILY`` output by the node's own literal
selection, so a linked control node and a set dropdown produce the identical
prompt.  A selection that is itself a link stays untouched.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

FAMILY_CONTROL_CLASS = "CMKDetailerFamilyEnable"
FAMILY_OUTPUT_SLOT = 1
FAMILY_VALUES = ("None", "SDXL", "Z-Image Turbo")


def inline_family_links(prompt: dict) -> tuple[dict, int]:
    """Return a copy of ``prompt`` with control links inlined and their count."""
    selections = {
        str(node_id): (node.get("inputs") or {}).get("model_family")
        for node_id, node in prompt.items()
        if isinstance(node, dict) and str(node.get("class_type", "")) == FAMILY_CONTROL_CLASS
    }
    selections = {key: value for key, value in selections.items() if value in FAMILY_VALUES}
    if not selections:
        return prompt, 0
    result = copy.deepcopy(prompt)
    count = 0
    for node in result.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        for name, value in list(inputs.items()):
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and str(value[0]) in selections
                and value[1] == FAMILY_OUTPUT_SLOT
            ):
                inputs[name] = selections[str(value[0])]
                count += 1
    return (result, count) if count else (prompt, 0)


def on_prompt(json_data: Any) -> Any:
    """ComfyUI on_prompt handler; returns the original request on any failure."""
    try:
        if not isinstance(json_data, dict) or not isinstance(json_data.get("prompt"), dict):
            return json_data
        prompt, count = inline_family_links(json_data["prompt"])
        if not count:
            return json_data
        result = dict(json_data)
        result["prompt"] = prompt
        return result
    except Exception:
        logging.exception("[CMK Detailer Family] inline failed · prompt left unchanged")
        return json_data


def register(server) -> bool:
    """Register the handler once on a PromptServer instance."""
    add_handler = getattr(server, "add_on_prompt_handler", None)
    if not callable(add_handler) or getattr(server, "_cmk_family_inline_registered", False):
        return False
    add_handler(on_prompt)
    server._cmk_family_inline_registered = True
    return True
