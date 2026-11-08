"""Remove provably unused gate links before ComfyUI forms cache keys.

ComfyUI's cache signature of a node contains every linked ancestor, whether a
lazy gate will ever request it or not.  A CMK gate that selects its BYPASS
inputs therefore still made the parameters and image references of its
inactive module part of every downstream key.  This prompt handler removes the
``* ACTIVE`` links of gates whose decision the prompt already holds
(``cmk_distribution.gate_prompt_state``); gates in any other state keep their
links.

The removed links stay in the node's ``_meta`` (ignored by ComfyUI's and CMK's
cache identities) so that a prompt replayed from ``/history`` gets them back
before it is pruned again.  ``CMK_PROMPT_GATE_PRUNING=0`` disables the pruning;
restoring stored links still happens then.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

try:
    from .cmk_distribution import GATE_INACTIVE, PRUNABLE_GATE_CLASSES, gate_prompt_state
except ImportError:  # Direct source loading in tests.
    from cmk_distribution import GATE_INACTIVE, PRUNABLE_GATE_CLASSES, gate_prompt_state

ENVIRONMENT_SWITCH = "CMK_PROMPT_GATE_PRUNING"
PRUNED_META_KEY = "cmk_pruned_inputs"
ACTIVE_SUFFIX = " ACTIVE"


def pruning_enabled() -> bool:
    value = os.environ.get(ENVIRONMENT_SWITCH, "1")
    return str(value).strip().lower() not in {"0", "false", "off", "no"}


def _gate_ids(prompt: dict) -> list:
    return [
        node_id for node_id, node in prompt.items()
        if isinstance(node, dict) and str(node.get("class_type", "")) in PRUNABLE_GATE_CLASSES
    ]


def restore_pruned_links(prompt: dict) -> dict:
    """Put stored links back; a link to a node the prompt no longer holds stays out."""
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        meta = node.get("_meta")
        if not isinstance(meta, dict) or PRUNED_META_KEY not in meta:
            continue
        stored = meta.pop(PRUNED_META_KEY)
        inputs = node.setdefault("inputs", {})
        if isinstance(stored, dict) and isinstance(inputs, dict):
            for name, link in stored.items():
                if (
                    name not in inputs
                    and isinstance(link, (list, tuple))
                    and len(link) == 2
                    and str(link[0]) in {str(key) for key in prompt}
                ):
                    inputs[name] = list(link)
        if not meta:
            node.pop("_meta")
    return prompt


def _remove_active_links(prompt: dict, gate_id) -> dict:
    node = prompt[gate_id]
    inputs = node.get("inputs", {})
    removed = {name: inputs.pop(name) for name in [n for n in inputs if str(n).endswith(ACTIVE_SUFFIX)]}
    if removed:
        node.setdefault("_meta", {})[PRUNED_META_KEY] = removed
    return removed


def prune_inactive_gate_links(prompt: dict, enabled: bool = True) -> tuple[dict, dict]:
    """Return a pruned copy of ``prompt`` and the removed links per gate.

    The input prompt is never modified.  If pruning would change the decision
    of any gate (for example because module 01 was reachable only through an
    ACTIVE link), nothing is pruned.
    """
    work = restore_pruned_links(copy.deepcopy(prompt))
    if not enabled:
        return work, {}
    gates = _gate_ids(work)
    decisions = {gate_id: gate_prompt_state(work, gate_id) for gate_id in gates}
    pruned = copy.deepcopy(work)
    removed = {}
    for gate_id in gates:
        if decisions[gate_id] == GATE_INACTIVE:
            links = _remove_active_links(pruned, gate_id)
            if links:
                removed[str(gate_id)] = links
    if removed and any(gate_prompt_state(pruned, gate_id) != decisions[gate_id] for gate_id in gates):
        logging.warning("[CMK Gate Pruning] skipped · pruning would change a gate decision")
        return work, {}
    return pruned, removed


def on_prompt(json_data: Any) -> Any:
    """ComfyUI on_prompt handler; returns the original request on any failure."""
    try:
        if not isinstance(json_data, dict) or not isinstance(json_data.get("prompt"), dict):
            return json_data
        prompt, removed = prune_inactive_gate_links(json_data["prompt"], pruning_enabled())
        if prompt == json_data["prompt"]:
            return json_data
        result = dict(json_data)
        result["prompt"] = prompt
        if removed:
            logging.info(
                "[CMK Gate Pruning] inactive gates decoupled · %s",
                ", ".join(sorted(removed)),
            )
        return result
    except Exception:
        logging.exception("[CMK Gate Pruning] failed · prompt left unchanged")
        return json_data


def register(server) -> bool:
    """Register the handler once on a PromptServer instance."""
    add_handler = getattr(server, "add_on_prompt_handler", None)
    if not callable(add_handler) or getattr(server, "_cmk_gate_pruning_registered", False):
        return False
    add_handler(on_prompt)
    server._cmk_gate_pruning_registered = True
    return True
