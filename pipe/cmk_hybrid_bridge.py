from __future__ import annotations

import gc

from comfy_execution.graph_utils import ExecutionBlocker

from .cmk_log_pipe import cmk_add_block
from .cmk_visual import empty_visual


def _release_completed_sdxl_patch(patcher):
    """Drop patch-only tensors from a completed Hybrid-Inpaint clone."""
    released_entries = sum(
        len(entries) for entries in getattr(patcher, "patches", {}).values()
    )
    try:
        patcher.cleanup()
    except Exception:
        pass
    for name in (
        "patches",
        "object_patches",
        "weight_wrapper_patches",
        "attachments",
        "additional_models",
    ):
        value = getattr(patcher, name, None)
        if hasattr(value, "clear"):
            value.clear()
    # Fooocus' spatial block patch retains its computed feature through the
    # patcher's transformer options. The clone is terminal at this point.
    patcher.model_options = {"transformer_options": {}}
    gc.collect()
    return released_entries


class CMKHybridZITInputPipe:
    """Select the ordinary ZIT source or the completed SDXL hybrid handoff."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "PROCESS ZIT": ("CMK_PROCESS_Z_IMAGE",),
            },
            "optional": {
                "IMAGE SOURCE": ("IMAGE", {"lazy": True}),
                "LOG SOURCE": ("CMK_LOG_PIPE", {"lazy": True}),
                "VISUAL SOURCE": ("CMK_VISUAL_PIPE", {"lazy": True}),
                "SAMPLED SDXL": ("CMK_SAMPLED_PIPE", {"lazy": True}),
                "LOG SDXL": ("CMK_LOG_PIPE", {"lazy": True}),
                "VISUAL SDXL": ("CMK_VISUAL_PIPE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "CMK_LOG_PIPE", "CMK_VISUAL_PIPE")
    RETURN_NAMES = ("IMAGE", "LOG", "VISUAL")
    FUNCTION = "select"
    CATEGORY = "CMK/Developer/Pipe/Bridge"
    DEV_ONLY = True

    def check_lazy_status(self, **inputs):
        process = inputs.get("PROCESS ZIT")
        if process is None:
            return ["PROCESS ZIT"]
        if isinstance(process, dict) and not process.get("family_active", True):
            return []
        if bool(process.get("hybrid_mode", False)):
            for name in ("SAMPLED SDXL", "LOG SDXL", "VISUAL SDXL"):
                if inputs.get(name) is None:
                    return [name]
            return []
        required = ["LOG SOURCE", "VISUAL SOURCE"]
        if bool(process.get("boolean_inpaint_mode", False)):
            required.insert(0, "IMAGE SOURCE")
        for name in required:
            if inputs.get(name) is None:
                return [name]
        return []

    @staticmethod
    def select(**inputs):
        process = inputs.get("PROCESS ZIT")
        if process is None or (
            isinstance(process, dict) and not process.get("family_active", True)
        ):
            blocked = ExecutionBlocker(None)
            return (blocked, blocked, blocked)
        if not isinstance(process, dict):
            raise TypeError("CMK Hybrid ZIT Input requires PROCESS ZIT")

        if not bool(process.get("hybrid_mode", False)):
            return (
                inputs.get("IMAGE SOURCE"),
                inputs.get("LOG SOURCE"),
                inputs.get("VISUAL SOURCE") or empty_visual(),
            )

        sampled = inputs.get("SAMPLED SDXL")
        if not isinstance(sampled, dict):
            raise TypeError("CMK Hybrid ZIT Input requires SAMPLED SDXL")
        image = sampled.get("image_1st_pass")
        if image is None:
            raise ValueError(
                "CMK Hybrid ZIT Input: SDXL handoff contains no clean x0 image"
            )
        unload_requested = bool(process.get("unload_models_after_use", True))
        sdxl_model = sampled.get("model_patched") or sampled.get("model")
        sdxl_unload_status = "NOT AVAILABLE"
        patch_release_status = "NOT REQUIRED"
        cache_release_status = "KEPT"
        if unload_requested and sdxl_model is not None:
            if bool(process.get("hybrid_inpaint_mode", False)):
                released_patch_entries = _release_completed_sdxl_patch(sdxl_model)
                patch_release_status = f"RELEASED ({released_patch_entries} entries)"
            from .cmk_family_result import clear_sdxl_sampled_boundary_cache
            from .loaders.checkpoint_vae_loader import unload_all_phase_models

            cleared = clear_sdxl_sampled_boundary_cache()
            cache_release_status = f"CLEARED ({cleared})"
            sdxl_unload_status = unload_all_phase_models("Hybrid SDXL -> ZIT handoff")
        elif not unload_requested:
            sdxl_unload_status = "KEPT LOADED"
        handoff = int(process.get("hybrid_sdxl_handoff", 90))
        denoise = float(process.get("hybrid_zit_denoise", 0.20))
        log = cmk_add_block(
            inputs.get("LOG SDXL"),
            "Hybrid Handoff",
            35,
            [
                "MODE            : SDXL -> ZIT",
                f"SDXL HANDOFF   : {handoff}%",
                "BRIDGE          : SDXL x0 decode -> ZIT VAE encode",
                f"SDXL MODEL     : {sdxl_unload_status}",
                f"SDXL PATCH DATA: {patch_release_status}",
                f"SDXL CACHE      : {cache_release_status}",
                f"ZIT DENOISE     : {denoise:.2f}",
            ],
            True,
        )
        return (image, log, inputs.get("VISUAL SDXL") or empty_visual())
