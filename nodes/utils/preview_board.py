from __future__ import annotations

import numpy as np
import torch

from ...utils.comfy_preview_output import combine_preview_panels, render_preview_panel, save_preview_png, ui_images
from ...utils.preview_payload import normalize_diagnostic_payload, preview_summary


_CAPTION_MODES = ["Off", "Standard", "Details"]
_MAX_DIAGNOSTICS = 32


def _panel_to_image_tensor(panel):
    arr = np.asarray(panel)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr = np.clip(arr[..., :3], 0, 255).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


class CMKPreviewBoard:
    """Combine several local diagnostics and display them as a compact visual board."""

    CATEGORY = "CMK/Toolbox/Diagnostics"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        optional = {"diagnostic_1": ("CMK_DIAGNOSTIC",)}
        for index in range(2, _MAX_DIAGNOSTICS + 1):
            optional[f"diagnostic_{index}"] = ("CMK_DIAGNOSTIC",)
        return {
            "required": {"caption": (_CAPTION_MODES, {"default": "Standard"})},
            "optional": optional,
        }

    def run(self, caption: str, diagnostic_1=None, **kwargs):
        diagnostics = []
        if diagnostic_1 is not None:
            diagnostics.append(diagnostic_1)
        for index in range(2, _MAX_DIAGNOSTICS + 1):
            value = kwargs.get(f"diagnostic_{index}")
            if value is not None:
                diagnostics.append(value)

        if not diagnostics:
            blank = np.zeros((240, 720, 3), dtype=np.uint8)
            image_info = save_preview_png(blank, prefix="CMK_preview_board")
            return {
                "ui": ui_images([image_info], "No diagnostic input connected.")["ui"],
                "result": (_panel_to_image_tensor(blank),),
            }

        render_caption = "Summary" if str(caption) == "Standard" else str(caption)
        panels = []
        summaries = []
        for idx, p in enumerate(diagnostics, start=1):
            data = normalize_diagnostic_payload(p)
            panels.append(render_preview_panel(data, caption=render_caption))
            summaries.append(f"[{idx}] {preview_summary(data)}")

        board = combine_preview_panels(panels)
        image_info = save_preview_png(board, prefix="CMK_preview_board")
        return {
            "ui": ui_images([image_info], "\n\n".join(summaries))["ui"],
            "result": (_panel_to_image_tensor(board),),
        }
