from __future__ import annotations

import numpy as np
import torch

from ...utils.comfy_preview_output import render_preview_panel, save_preview_png, ui_images
from ...utils.preview_payload import normalize_preview_payload, preview_summary


_CAPTION_MODES = ["Off", "Standard", "Details"]


def _panel_to_image_tensor(panel):
    arr = np.asarray(panel)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr = np.clip(arr[..., :3], 0, 255).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


class CMKPreviewRender:
    """Quick diagnostic output: render one diagnostic directly in the node UI."""

    CATEGORY = "CMK/Toolbox/Diagnostics"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"diagnostic": ("CMK_DIAGNOSTIC",), "caption": (_CAPTION_MODES, {"default": "Standard"})}}

    def run(self, diagnostic, caption: str):
        data = normalize_preview_payload(diagnostic)
        # Older renderer names the compact mode Summary. Standard is the UI name.
        render_caption = "Summary" if str(caption) == "Standard" else str(caption)
        panel = render_preview_panel(data, caption=render_caption)
        image_info = save_preview_png(panel, prefix="CMK_preview_render")
        return {
            "ui": ui_images([image_info], preview_summary(data))["ui"],
            "result": (_panel_to_image_tensor(panel),),
        }
