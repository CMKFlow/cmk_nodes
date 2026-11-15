"""CMK Flow · Image Resize v2 (2.8): the unchanged Image Resize with the CMK
port contract MODEL ZIT, MODEL SDXL, PROCESS, IMAGE, LOG, VISUAL.

Resize, bypass, cache and diagnostic are those of CMK Flow · Image Resize;
MODE (``scale_mode`` ENABLE / DISABLE) is its SCALE ENABLE switch.  Both MODEL
inputs are transport only: forwarded unchanged, never read and, like PROCESS
and VISUAL, not part of the cache key.
"""

from __future__ import annotations

from ...pipe.cmk_persistent_cache import build_node_fingerprint
from .image_resize import METHOD_MAX_DIMENSION, METHOD_MEGAPIXELS, METHOD_RESIZE_PAD, CMKImageResize

TRANSPORT_INPUTS = ("MODEL ZIT", "MODEL SDXL", "PROCESS", "VISUAL")
SCALE_MODE_WIDGET = "scale_mode"
SCALE_MODES = ["ENABLE", "DISABLE"]
_INACTIVE = {
    METHOD_MAX_DIMENSION: ("megapixels", "target_width", "target_height", "padding_color"),
    METHOD_MEGAPIXELS: ("largest_size", "target_width", "target_height", "padding_color"),
    METHOD_RESIZE_PAD: ("largest_size", "megapixels"),
}


class CMKImageResizeV2(CMKImageResize):
    """CMK Flow · Image Resize with MODEL ZIT and MODEL SDXL pass-through."""

    RETURN_TYPES = ("CMK_MODEL_PIPE", *CMKImageResize.RETURN_TYPES)
    RETURN_NAMES = ("MODEL ZIT", "MODEL SDXL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic")

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        optional = types["optional"]
        optional.pop("MODEL (opt)")
        types["optional"] = {"MODEL ZIT": ("CMK_MODEL_PIPE",), "MODEL SDXL": ("CMK_MODEL_PIPE",), **optional}
        # MODE ENABLE / DISABLE in place of the SCALE ENABLE toggle (same meaning).
        types["required"] = {(SCALE_MODE_WIDGET if name == "SCALE ENABLE" else name):
                             ((SCALE_MODES, {"default": "ENABLE"}) if name == "SCALE ENABLE" else spec)
                             for name, spec in types["required"].items()}
        return types

    @classmethod
    def _cache_key(cls, prompt, unique_id, method: str):
        # Resolved by the v2 class; the legacy schema and the same excluded
        # inputs, with both MODEL pipes as transport.
        return build_node_fingerprint(
            prompt,
            unique_id,
            ("CMKImageResizeV2",),
            cls._CACHE_SCHEMA,
            exclude_inputs=(*_INACTIVE[method], *TRANSPORT_INPUTS),
            include_node_identity=True,
        )

    def resize(self, PROCESS, IMAGE, LOG, **inputs):
        model_zit = inputs.pop("MODEL ZIT", None)
        inputs["MODEL (opt)"] = inputs.pop("MODEL SDXL", None)
        inputs["SCALE ENABLE"] = str(inputs.pop(SCALE_MODE_WIDGET, "ENABLE")) != "DISABLE"
        return (model_zit, *super().resize(PROCESS, IMAGE, LOG, **inputs))
