"""Lightweight PROCESS configuration for CMK regional conditioning."""

from __future__ import annotations

from .cmk_log_pipe import cmk_add_block, cmk_clean_text
from .cmk_visual import normalize_visual
from ..utils.cmk_diagnostic import make_diagnostic_payload


POSITIONS = (
    "TOP LEFT", "TOP", "TOP RIGHT",
    "LEFT", "CENTER", "RIGHT",
    "BOTTOM LEFT", "BOTTOM", "BOTTOM RIGHT",
    "CUSTOM",
)
SIZES = ("SMALL", "MEDIUM", "LARGE")
SIZE_FRACTIONS = {"SMALL": 0.33, "MEDIUM": 0.50, "LARGE": 0.67}


def preset_geometry(position: str, size: str):
    """Map a semantic position and size to symmetric normalized geometry."""
    position = str(position or "CENTER").upper()
    extent = SIZE_FRACTIONS.get(str(size or "MEDIUM").upper(), SIZE_FRACTIONS["MEDIUM"])
    horizontal = "CENTER"
    vertical = "CENTER"
    if "LEFT" in position:
        horizontal = "LEFT"
    elif "RIGHT" in position:
        horizontal = "RIGHT"
    if position.startswith("TOP") or position == "TOP":
        vertical = "TOP"
    elif position.startswith("BOTTOM") or position == "BOTTOM":
        vertical = "BOTTOM"
    x = {"LEFT": 0.0, "CENTER": (1.0 - extent) / 2.0, "RIGHT": 1.0 - extent}[horizontal]
    y = {"TOP": 0.0, "CENTER": (1.0 - extent) / 2.0, "BOTTOM": 1.0 - extent}[vertical]
    return x, y, extent, extent


def _clamp(value, minimum=0.0, maximum=1.0):
    return min(maximum, max(minimum, float(value)))


class CMKRegionalConditioningSDXL:
    """Append up to three region specifications to an SDXL PROCESS pipe.

    This node deliberately does not encode text. Module 10 owns CLIP and
    materializes every active region immediately before sampling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "PROCESS": ("CMK_PROCESS_SDXL",),
            "IMAGE": ("IMAGE",),
            "LOG": ("CMK_LOG_PIPE",),
        }
        defaults = (
            ("LEFT", "MEDIUM"),
            ("CENTER", "MEDIUM"),
            ("RIGHT", "MEDIUM"),
        )
        for index, (position, size) in enumerate(defaults, 1):
            x, y, width, height = preset_geometry(position, size)
            prefix = f"region_{index}_"
            required.update({
                prefix + "position": (POSITIONS, {"default": position}),
                prefix + "size": (SIZES, {"default": size}),
                prefix + "x": ("FLOAT", {"default": x, "min": 0.0, "max": 1.0, "step": 0.01}),
                prefix + "y": ("FLOAT", {"default": y, "min": 0.0, "max": 1.0, "step": 0.01}),
                prefix + "width": ("FLOAT", {"default": width, "min": 0.01, "max": 1.0, "step": 0.01}),
                prefix + "height": ("FLOAT", {"default": height, "min": 0.01, "max": 1.0, "step": 0.01}),
                prefix + "end": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How long this regional conditioning may influence sampling.",
                }),
                prefix + "area_prompt": ("STRING", {"default": "", "multiline": True}),
                prefix + "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            })
        return {
            "required": required,
            "optional": {"VISUAL": ("CMK_VISUAL_PIPE",)},
        }

    RETURN_TYPES = (
        "CMK_PROCESS_SDXL", "IMAGE", "CMK_LOG_PIPE", "CMK_VISUAL_PIPE",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic")
    FUNCTION = "configure"
    CATEGORY = "CMK/Flow/Conditioning"
    DESCRIPTION = "Adds up to three spatial prompt regions to the SDXL process before ControlNet and sampling."

    def configure(self, PROCESS, IMAGE, LOG, **kwargs):
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Regional Conditioning SDXL: PROCESS must be a CMK SDXL process")
        process = dict(PROCESS)
        regions = [dict(region) for region in process.get("regional_conditioning", []) if isinstance(region, dict)]
        added = []
        for index in range(1, 4):
            prefix = f"region_{index}_"
            prompt = cmk_clean_text(kwargs.get(prefix + "area_prompt", ""))
            if not prompt:
                continue
            position = str(kwargs.get(prefix + "position", "CENTER")).upper()
            size = str(kwargs.get(prefix + "size", "MEDIUM")).upper()
            if position == "CUSTOM":
                x = _clamp(kwargs.get(prefix + "x", 0.25))
                y = _clamp(kwargs.get(prefix + "y", 0.25))
                width = _clamp(kwargs.get(prefix + "width", 0.50), 0.01)
                height = _clamp(kwargs.get(prefix + "height", 0.50), 0.01)
                x = min(x, 1.0 - width)
                y = min(y, 1.0 - height)
                geometry_source = "CUSTOM"
            else:
                x, y, width, height = preset_geometry(position, size)
                geometry_source = f"{position} · {size}"
            region = {
                "prompt": prompt,
                "x": x, "y": y, "width": width, "height": height,
                "strength": _clamp(kwargs.get(prefix + "strength", 1.0), 0.0, 2.0),
                "start": 0.0,
                "end": _clamp(kwargs.get(prefix + "end", 1.0)),
                "source": geometry_source,
                "module_region": index,
            }
            regions.append(region)
            added.append(region)
        process["regional_conditioning"] = regions
        process["regional_conditioning_count"] = len(regions)
        lines = [
            f"ACTIVE REGIONS  : {len(added)}",
            f"TOTAL CHAINED   : {len(regions)}",
        ]
        for index, region in enumerate(added, 1):
            lines.append(
                f"REGION {index}       : {region['source']} | "
                f"x={region['x']:.2f} y={region['y']:.2f} w={region['width']:.2f} h={region['height']:.2f} | "
                f"strength={region['strength']:.2f} end={region['end']:.2f}"
            )
            lines.append(f"REGION {index} PROMPT: {region['prompt']}")
        log = cmk_add_block(LOG, "Regional Conditioning SDXL", 20, lines, True)
        diagnostic = make_diagnostic_payload(
            title="Regional Conditioning SDXL",
            node="CMK Flow · 02 Regional Conditioning SDXL",
            previews=[],
            summary=f"{len(added)} active regions | {len(regions)} chained total",
            details="\n".join(lines),
            mode="regional conditioning",
            metadata={"active_regions": len(added), "total_regions": len(regions)},
        )
        return process, IMAGE, log, normalize_visual(kwargs.get("VISUAL")), diagnostic
