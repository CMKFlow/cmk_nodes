"""Inline the selection of the CMK mode controls into their consumers.

Controls: CMK Detailer Family Enable (``model_family``: None, SDXL, Z-Image
Turbo) and, since FaceProcess 2.8, CMK FaceProcess Mode Enable
(``process_mode``: None, Restore, SDXL, Z-Image Turbo).  Each control inlines
only its own values; the Detailer control and its prompts are unchanged.

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

from .pipe.cmk_faceprocess_mode import FACEPROCESS_MODE_WIDGET, FACEPROCESS_MODES

FAMILY_CONTROL_CLASS = "CMKDetailerFamilyEnable"
FAMILY_OUTPUT_SLOT = 1
FAMILY_VALUES = ("None", "SDXL", "Z-Image Turbo")
FACEPROCESS_CONTROL_CLASS = "CMKFaceProcessModeEnable"

# control class -> (selection widget, output slot of the selection, valid values)
CONTROL_CLASSES = {
    FAMILY_CONTROL_CLASS: ("model_family", FAMILY_OUTPUT_SLOT, FAMILY_VALUES),
    FACEPROCESS_CONTROL_CLASS: (FACEPROCESS_MODE_WIDGET, 1, tuple(FACEPROCESS_MODES)),
}


def inline_family_links(prompt: dict) -> tuple[dict, int]:
    """Return a copy of ``prompt`` with control links inlined and their count."""
    selections = {}
    for node_id, node in prompt.items():
        control = CONTROL_CLASSES.get(str(node.get("class_type", ""))) if isinstance(node, dict) else None
        if control is None:
            continue
        widget, slot, values = control
        value = (node.get("inputs") or {}).get(widget)
        if value in values:
            selections[str(node_id)] = (value, slot)
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
                and value[1] == selections[str(value[0])][1]
            ):
                inputs[name] = selections[str(value[0])][0]
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
        logging.exception("[CMK Mode Control] inline failed · prompt left unchanged")
        return json_data


def register(server) -> bool:
    """Register the handler once on a PromptServer instance."""
    add_handler = getattr(server, "add_on_prompt_handler", None)
    if not callable(add_handler) or getattr(server, "_cmk_family_inline_registered", False):
        return False
    add_handler(on_prompt)
    server._cmk_family_inline_registered = True
    return True
