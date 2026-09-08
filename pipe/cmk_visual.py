from __future__ import annotations

import json
from typing import Any

VISUAL_TYPE = "CMK_VISUAL_PIPE"
VISUAL_VERSION = 1


class _CMKAnyType(str):
    def __ne__(self, other):
        return False


CMK_FINISH_INPUT = _CMKAnyType("*")
_STATUSES = {
    "registered", "available", "waiting", "active", "updated", "invalid",
    "cleared", "live_started", "live_updated", "live_ended", "completed",
    "skipped", "disabled",
}
_LIVE_TYPE_ALIASES = {
    "FaceProcess": {"CMKFaceProcessPipe", "CMK_FaceProcess", "CMKFaceProcess"},
}


def _matches_live_type(actual: Any, declared: Any) -> bool:
    actual_text = str(actual or "")
    declared_text = str(declared or "")
    return actual_text == declared_text or actual_text in _LIVE_TYPE_ALIASES.get(
        declared_text, set()
    )


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
    live_node_ids: list[Any] | None = None,
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
    # Provider order is the order of the incoming VISUAL chain. ``sequence``
    # identifies a module in the UI, but must never prescribe wiring order:
    # modules such as 23, 25 and 30 are intentionally freely composable.
    resolved_live_ids = [str(item) for item in (live_node_ids or []) if item is not None]
    if live_node_id is not None and str(live_node_id) not in resolved_live_ids:
        resolved_live_ids.insert(0, str(live_node_id))
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
            "live": bool(resolved_live_ids),
        },
        "channels": clean_channels,
        "status": status if status in _STATUSES else "completed",
    }
    if branch:
        provider["branch"] = str(branch)
    if stage_key:
        provider["stage_key"] = str(stage_key)
    if resolved_live_ids:
        provider["live_node_id"] = resolved_live_ids[0]
        provider["live_node_ids"] = resolved_live_ids

    providers = list(result["providers"])
    for index, item in enumerate(providers):
        if item.get("provider_id") == provider_id:
            providers[index] = provider
            break
    else:
        providers.append(provider)
    result["providers"] = providers
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
    if _matches_live_type(node.get("class_type"), class_type):
        return node_id
    for upstream in (node.get("inputs", {}) or {}).values():
        found = _upstream_class_node(prompt, upstream, class_type, visited)
        if found is not None:
            return found
    return None


def _upstream_class_nodes(prompt: Any, value: Any, class_type: str, scope: str = "", visited=None):
    """Return every matching upstream node inside the current subgraph instance."""
    if not isinstance(prompt, dict) or not isinstance(value, (list, tuple)) or not value:
        return []
    node_id = str(value[0])
    if scope and not (node_id == scope or node_id.startswith(f"{scope}:")):
        return []
    visited = visited or set()
    if node_id in visited:
        return []
    visited.add(node_id)
    node = prompt.get(node_id) or prompt.get(value[0])
    if not isinstance(node, dict):
        return []
    if _matches_live_type(node.get("class_type"), class_type):
        return [node_id]
    found = []
    for upstream in (node.get("inputs", {}) or {}).values():
        found.extend(_upstream_class_nodes(prompt, upstream, class_type, scope, visited))
    return found


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
                "label": ("STRING", {"default": "Result"}),
                "sequence": (
                    "INT",
                    {"default": 10, "min": 0, "max": 999, "advanced": True},
                ),
            },
            "optional": {
                "VISUAL": (VISUAL_TYPE,),
                "IMAGE": ("IMAGE", {"lazy": True}),
                "SOURCE": ("IMAGE", {"lazy": True}),
                "BEFORE": ("IMAGE", {"lazy": True}),
                "AFTER": ("IMAGE", {"lazy": True}),
                "enable": ("BOOLEAN", {"default": True, "forceInput": True}),
                "live_node_type": ("STRING", {"default": "", "advanced": True}),
                "branch": ("STRING", {"default": "", "advanced": True}),
                "stage_key": ("STRING", {"default": "", "advanced": True}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (VISUAL_TYPE,)
    RETURN_NAMES = ("VISUAL",)
    FUNCTION = "publish"
    CATEGORY = "CMK/Developer/Visual"

    @staticmethod
    def check_lazy_status(
        enable=True,
        prompt=None,
        unique_id=None,
        **inputs,
    ):
        if not bool(enable):
            return []
        current = prompt.get(str(unique_id), {}) if isinstance(prompt, dict) else {}
        connected = (current.get("inputs", {}) or {})
        for name in ("IMAGE", "SOURCE", "BEFORE", "AFTER"):
            if name in connected and inputs.get(name) is None:
                return [name]
        return []

    @staticmethod
    def publish(
        IMAGE=None,
        label="Result",
        sequence=10,
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
        if IMAGE is None:
            raise ValueError("CMK Visual Provider requires IMAGE while enabled")
        current = prompt.get(str(unique_id), {}) if isinstance(prompt, dict) else {}
        image_link = (current.get("inputs", {}) or {}).get("IMAGE")
        unique_text = str(unique_id or "")
        scope = unique_text.rsplit(":", 1)[0] if ":" in unique_text else ""
        live_node_ids = _upstream_class_nodes(
            prompt, image_link, live_node_type, scope
        ) if live_node_type else []
        live_node_id = live_node_ids[0] if live_node_ids else None
        channels = {"result": IMAGE, "source": SOURCE, "before": BEFORE, "after": AFTER}
        result = register_provider(
            VISUAL,
            module_instance_id=unique_id,
            module_type="CMKVisualProvider",
            module_label=label,
            sequence=sequence,
            channels=channels,
            status="completed",
            live_node_id=live_node_id,
            live_node_ids=live_node_ids,
            branch=branch,
            stage_key=stage_key,
        )
        return (result,)


class CMKVisualizer:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            models = list(folder_paths.get_filename_list("upscale_models")) or ["none"]
        except ImportError:
            models = ["RealESRGAN_x4plus.pth", "RealESRGAN-x2plus.pth"]
        default_4x = "RealESRGAN_x4plus.pth" if "RealESRGAN_x4plus.pth" in models else models[0]
        default_2x = "RealESRGAN-x2plus.pth" if "RealESRGAN-x2plus.pth" in models else models[0]
        advanced = {"advanced": True}
        return {
            "optional": {
                "MODEL (opt)": (CMK_FINISH_INPUT,),
                "PROCESS": (CMK_FINISH_INPUT,),
                "IMAGE": (CMK_FINISH_INPUT,),
                "LOG": (CMK_FINISH_INPUT,),
                "VISUAL": (VISUAL_TYPE,),
                "SAVE ENABLED": ("BOOLEAN", {"default": True}),
                "filename prefix": ("STRING", {"default": "image"}),
                "output folder": ("STRING", {"default": "", **advanced}),
                "use date folder": ("BOOLEAN", {"default": True, **advanced}),
                "enable upscale": ("BOOLEAN", {"default": False, **advanced}),
                "limit 4x MP": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 100.0, "step": 0.5, **advanced}),
                "limit 2x MP": ("FLOAT", {"default": 8.0, "min": 0.5, "max": 100.0, "step": 0.5, **advanced}),
                "model 4x": (models, {"default": default_4x, **advanced}),
                "model 2x": (models, {"default": default_2x, **advanced}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "show"
    CATEGORY = "CMK/Flow/Core"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @staticmethod
    def show(VISUAL=None, unique_id=None, **kwargs):
        from ..utils.comfy_preview_output import save_preview_png
        from ..utils.preview_payload import _as_rgb_uint8

        visual = normalize_visual(VISUAL)
        save_text = None
        image = kwargs.get("IMAGE")
        process = kwargs.get("PROCESS")
        log = kwargs.get("LOG")
        model = kwargs.get("MODEL (opt)")
        save_enabled = bool(kwargs.get("SAVE ENABLED", True))
        upscale_enabled = bool(kwargs.get("enable upscale", False))

        if image is not None and (save_enabled or upscale_enabled):
            from .cmk_family_result import CMKResultUnpackPipe

            model, process, image, log = CMKResultUnpackPipe.unpack(
                PROCESS=process,
                IMAGE=image,
                LOG=log,
                **{"MODEL (opt)": model},
            )

        if image is not None and upscale_enabled:
            from ..nodes.image.smart_upscale import CMK_SmartUpscalerPipe

            image, log, _upscale_diagnostic = CMK_SmartUpscalerPipe().run_pipe(
                IMAGE=image,
                LOG=log,
                enable=True,
                limit_4x_mp=float(kwargs.get("limit 4x MP", 1.5)),
                limit_2x_mp=float(kwargs.get("limit 2x MP", 8.0)),
                model_4x=kwargs.get("model 4x"),
                model_2x=kwargs.get("model 2x"),
            )
            visual = register_provider(
                visual,
                module_instance_id=f"{unique_id}:upscale",
                module_type="CMKVisualizerUpscale",
                module_label="Upscale",
                sequence=90,
                channels={"result": image},
                status="completed",
                branch="result",
                stage_key="result.upscale",
            )

        if image is not None and save_enabled:
            from ..nodes.io.save_project_image import CMK_SaveProjectImage

            save_result = CMK_SaveProjectImage().run(
                PROCESS=process,
                IMAGE=image,
                LOG=log,
                **{
                    "MODEL (opt)": model,
                    "SAVE ENABLED": True,
                    "FILENAME PREFIX": kwargs.get("filename prefix", "image"),
                    "OUTPUT FOLDER": kwargs.get("output folder", ""),
                    "USE DATE FOLDER": bool(kwargs.get("use date folder", True)),
                    "PROJECT FOLDER": "",
                },
            )
            save_text = (save_result.get("ui", {}) or {}).get("text")

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
        ui = {"cmk_visual_images": ui_images, "cmk_visual": [json.dumps(payload)]}
        if save_text:
            ui["text"] = save_text
        return {"ui": ui}
