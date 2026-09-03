from __future__ import annotations

import json
from typing import Any

VISUAL_TYPE = "CMK_VISUAL_PIPE"
VISUAL_VERSION = 1
_STATUSES = {
    "registered", "available", "waiting", "active", "updated", "invalid",
    "cleared", "live_started", "live_updated", "live_ended", "completed",
    "skipped", "disabled",
}


def empty_visual(execution_id: Any = None) -> dict[str, Any]:
    result = {"type": VISUAL_TYPE, "version": VISUAL_VERSION, "providers": []}
    if execution_id is not None:
        result["execution_id"] = str(execution_id)
    return result


def normalize_visual(value: Any) -> dict[str, Any]:
    if value is None:
        return empty_visual()
    if not isinstance(value, dict) or value.get("type") != VISUAL_TYPE:
        raise TypeError("VISUAL must be a CMK_VISUAL_PIPE dictionary")
    providers = value.get("providers", [])
    if not isinstance(providers, list):
        raise TypeError("VISUAL['providers'] must be a list")
    result = {"type": VISUAL_TYPE, "version": VISUAL_VERSION, "providers": list(providers)}
    if value.get("execution_id") is not None:
        result["execution_id"] = str(value["execution_id"])
    return result


def _provider_id(module_instance_id: Any, module_type: str) -> str:
    safe_type = "".join(ch if ch.isalnum() else "-" for ch in str(module_type)).strip("-")
    safe_id = "".join(ch if ch.isalnum() else "-" for ch in str(module_instance_id)).strip("-")
    return f"cmk-{safe_type}-{safe_id}"


def register_provider(
    visual: Any,
    *,
    module_instance_id: Any,
    module_type: str,
    module_label: str,
    sequence: int,
    channels: dict[str, Any],
    status: str = "completed",
    live_node_id: Any = None,
    branch: str = "",
    stage_key: str = "",
) -> dict[str, Any]:
    result = normalize_visual(visual)
    provider_id = _provider_id(module_instance_id, module_type)
    clean_channels = {str(name): image for name, image in channels.items() if image is not None}
    # A provider is evidence of an actual image signal.  In particular, an
    # optional/bypassed module must not create an empty tab merely because its
    # provider node is evaluated at the end of the workflow.
    if not clean_channels:
        return result
    # Providers describe the processing chain in ascending sequence order.  A
    # downstream module may still expose a convenience view of an earlier stage
    # (the Refiner historically publishes its input as "1st Pass").  Once an
    # intermediate stage such as Identity is already present, that convenience
    # provider is stale and must not overwrite the real earlier stage.
    if any(int(item.get("sequence", 0)) > int(sequence) for item in result["providers"]):
        return result
    provider = {
        "provider_id": provider_id,
        "module_instance_id": str(module_instance_id),
        "module_type": str(module_type),
        "module_label": str(module_label or module_type),
        "sequence": int(sequence),
        "capabilities": {
            "preview": bool(clean_channels),
            "multi_source": len(clean_channels) > 1,
            "compare": len(clean_channels) > 1,
            "live": live_node_id is not None,
        },
        "channels": clean_channels,
        "status": status if status in _STATUSES else "completed",
    }
    if branch:
        provider["branch"] = str(branch)
    if stage_key:
        provider["stage_key"] = str(stage_key)
    if live_node_id is not None:
        provider["live_node_id"] = str(live_node_id)

    providers = [item for item in result["providers"] if item.get("provider_id") != provider_id]
    providers.append(provider)
    result["providers"] = sorted(providers, key=lambda item: (item.get("sequence", 0), item.get("provider_id", "")))
    return result


def _upstream_class_node(prompt: Any, value: Any, class_type: str, visited=None):
    if not isinstance(prompt, dict) or not isinstance(value, (list, tuple)) or not value:
        return None
    node_id = str(value[0])
    visited = visited or set()
    if node_id in visited:
        return None
    visited.add(node_id)
    node = prompt.get(node_id) or prompt.get(value[0])
    if not isinstance(node, dict):
        return None
    if node.get("class_type") == class_type:
        return node_id
    for upstream in (node.get("inputs", {}) or {}).values():
        found = _upstream_class_node(prompt, upstream, class_type, visited)
        if found is not None:
            return found
    return None


class CMKVisualPass:
    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"VISUAL": (VISUAL_TYPE,)}}

    RETURN_TYPES = (VISUAL_TYPE,)
    RETURN_NAMES = ("VISUAL",)
    FUNCTION = "forward"
    CATEGORY = "CMK/Developer/Visual"

    @staticmethod
    def forward(VISUAL=None):
        return (empty_visual() if VISUAL is None else VISUAL,)


class CMKVisualProvider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "label": ("STRING", {"default": "Result"}),
                "sequence": ("INT", {"default": 10, "min": 0, "max": 999}),
            },
            "optional": {
                "VISUAL": (VISUAL_TYPE,),
                "SOURCE": ("IMAGE",),
                "BEFORE": ("IMAGE",),
                "AFTER": ("IMAGE",),
                "enable": ("BOOLEAN", {"default": True}),
                "live_node_type": ("STRING", {"default": ""}),
                "branch": ("STRING", {"default": ""}),
                "stage_key": ("STRING", {"default": ""}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (VISUAL_TYPE,)
    RETURN_NAMES = ("VISUAL",)
    FUNCTION = "publish"
    CATEGORY = "CMK/Developer/Visual"

    @staticmethod
    def publish(
        IMAGE,
        label,
        sequence,
        VISUAL=None,
        SOURCE=None,
        BEFORE=None,
        AFTER=None,
        enable=True,
        live_node_type="",
        branch="",
        stage_key="",
        prompt=None,
        unique_id=None,
    ):
        if not bool(enable):
            return (normalize_visual(VISUAL),)
        current = prompt.get(str(unique_id), {}) if isinstance(prompt, dict) else {}
        image_link = (current.get("inputs", {}) or {}).get("IMAGE")
        live_node_id = _upstream_class_node(prompt, image_link, live_node_type) if live_node_type else None
        channels = {"result": IMAGE, "source": SOURCE, "before": BEFORE, "after": AFTER}
        return (register_provider(
            VISUAL,
            module_instance_id=unique_id,
            module_type="CMKVisualProvider",
            module_label=label,
            sequence=sequence,
            channels=channels,
            status="completed",
            live_node_id=live_node_id,
            branch=branch,
            stage_key=stage_key,
        ),)


class CMKVisualizer:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"VISUAL": (VISUAL_TYPE,)}, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ()
    FUNCTION = "show"
    CATEGORY = "CMK/Visual"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @staticmethod
    def show(VISUAL=None, unique_id=None):
        from ..utils.comfy_preview_output import save_preview_png
        from ..utils.preview_payload import _as_rgb_uint8

        visual = normalize_visual(VISUAL)
        providers = []
        ui_images = []
        for provider in visual["providers"]:
            channels = []
            for name, image in (provider.get("channels", {}) or {}).items():
                info = save_preview_png(
                    _as_rgb_uint8(image),
                    prefix=f"CMK_visual_{provider['provider_id']}_{name}",
                )
                index = len(ui_images)
                ui_images.append(info)
                channels.append({"name": name, "image_index": index})
            item = {key: value for key, value in provider.items() if key != "channels"}
            item["channels"] = channels
            providers.append(item)
        payload = {"version": VISUAL_VERSION, "providers": providers}
        return {"ui": {"cmk_visual_images": ui_images, "cmk_visual": [json.dumps(payload)]}}
