"""Visual layout configuration for the existing SDXL regional conditioning path."""

from __future__ import annotations

import json

from .cmk_log_pipe import cmk_add_block, cmk_clean_text
from .cmk_visual import normalize_visual
from ..utils.cmk_diagnostic import make_diagnostic_payload


DEFAULT_LAYOUT = {
    "version": 1,
    "width": 1024,
    "height": 1024,
    "strength_offset": 0.0,
    "regions": [],
}


def _clamp(value, minimum=0.0, maximum=1.0):
    return min(maximum, max(minimum, float(value)))


def process_dimensions(process):
    """Return the authoritative dimensions established by 01 START HERE."""
    if not isinstance(process, dict):
        return 1024, 1024

    def dimension(primary, secondary):
        try:
            return max(64, int(process.get(primary, process.get(secondary, 1024))))
        except (TypeError, ValueError):
            return 1024

    return dimension("width", "target_width"), dimension("height", "target_height")


def _layout_payload(value):
    try:
        payload = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError(f"CMK Regional Layout: invalid layout data: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("CMK Regional Layout: layout data must be an object")
    if not isinstance(payload.get("regions", []), list):
        raise TypeError("CMK Regional Layout: regions must be a list")
    return payload


def parse_layout(value, width, height):
    """Validate workflow-owned editor data and return executable regions."""
    payload = _layout_payload(value)
    raw_regions = payload.get("regions", [])
    strength_offset = _clamp(payload.get("strength_offset", 0.0), 0.0, 2.0)

    regions = []
    used_ids = set()
    for index, raw in enumerate(raw_regions, 1):
        if not isinstance(raw, dict):
            continue
        prompt = cmk_clean_text(raw.get("prompt", ""))
        if not prompt:
            continue
        region_id = str(raw.get("id") or f"region-{index}").strip() or f"region-{index}"
        if region_id in used_ids:
            region_id = f"{region_id}-{index}"
        used_ids.add(region_id)
        name = cmk_clean_text(raw.get("name", "")) or f"Region {index}"
        box = raw.get("box", raw)
        if not isinstance(box, dict):
            box = {}
        box_width = _clamp(box.get("width", 0.5), 0.01)
        box_height = _clamp(box.get("height", 0.5), 0.01)
        x = min(_clamp(box.get("x", 0.25)), 1.0 - box_width)
        y = min(_clamp(box.get("y", 0.25)), 1.0 - box_height)
        individual_strength = _clamp(raw.get("strength", 1.0), 0.0, 2.0)
        regions.append({
            "id": region_id,
            "name": name,
            "prompt": prompt,
            "x": x,
            "y": y,
            "width": box_width,
            "height": box_height,
            "strength": individual_strength + strength_offset,
            "start": 0.0,
            "end": _clamp(raw.get("end", 1.0)),
            "source": f"LAYOUT {int(width)}x{int(height)}",
            "module_region": index,
        })
    return regions


class CMKRegionalLayoutSDXL:
    """Append editor-defined regions to the established SDXL PROCESS pipe."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "PROCESS": ("CMK_PROCESS_SDXL",),
                "IMAGE": ("IMAGE",),
                "LOG": ("CMK_LOG_PIPE",),
                "width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 64}),
                "regions_json": ("STRING", {
                    "default": json.dumps(DEFAULT_LAYOUT, separators=(",", ":")),
                    "multiline": True,
                }),
            },
            "optional": {"VISUAL": ("CMK_VISUAL_PIPE",)},
        }

    RETURN_TYPES = (
        "CMK_PROCESS_SDXL", "IMAGE", "CMK_LOG_PIPE", "CMK_VISUAL_PIPE",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("PROCESS SDXL", "IMAGE", "LOG", "VISUAL", "diagnostic")
    FUNCTION = "configure"
    CATEGORY = "CMK/Flow/Process"
    DESCRIPTION = "Defines visual prompt regions and appends them to the existing SDXL regional-conditioning process."

    def configure(self, PROCESS, IMAGE, LOG, width, height, regions_json, **kwargs):
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Regional Layout: PROCESS must be a CMK SDXL process")
        payload = _layout_payload(regions_json)
        if not payload.get("regions"):
            diagnostic = make_diagnostic_payload(
                title="07 Regional Layout",
                node="CMK Flow · 07 Regional Layout",
                previews=[],
                summary="0 regions | PROCESS unchanged",
                details="No regions configured. The incoming PROCESS was returned unchanged.",
                mode="regional layout",
                metadata={"active_regions": 0, "total_regions": len(PROCESS.get("regional_conditioning", []))},
            )
            return PROCESS, IMAGE, LOG, normalize_visual(kwargs.get("VISUAL")), diagnostic

        process_width, process_height = process_dimensions(PROCESS)
        strength_offset = _clamp(payload.get("strength_offset", 0.0), 0.0, 2.0)
        process = dict(PROCESS)
        chained = [dict(region) for region in process.get("regional_conditioning", []) if isinstance(region, dict)]
        added = parse_layout(payload, process_width, process_height)
        chained.extend(added)
        process["regional_conditioning"] = chained
        process["regional_conditioning_count"] = len(chained)

        lines = [
            f"PROCESS SIZE    : {process_width} x {process_height}",
            f"STRENGTH OFFSET : +{strength_offset:.2f}",
            f"ACTIVE REGIONS : {len(added)}",
            f"TOTAL CHAINED  : {len(chained)}",
        ]
        for index, region in enumerate(added, 1):
            lines.append(
                f"REGION {index}       : {region['name']} | "
                f"x={region['x']:.3f} y={region['y']:.3f} "
                f"w={region['width']:.3f} h={region['height']:.3f} | "
                f"strength={region['strength']:.2f} end={region['end']:.2f}"
            )
            lines.append(f"REGION {index} PROMPT: {region['prompt']}")
        log = cmk_add_block(LOG, "07 Regional Layout", 20, lines, True)
        diagnostic = make_diagnostic_payload(
            title="07 Regional Layout",
            node="CMK Flow · 07 Regional Layout",
            previews=[],
            summary=f"{len(added)} active regions | {len(chained)} chained total",
            details="\n".join(lines),
            mode="regional layout",
            metadata={
                "active_regions": len(added),
                "total_regions": len(chained),
                "width": process_width,
                "height": process_height,
                "strength_offset": strength_offset,
            },
        )
        return process, IMAGE, log, normalize_visual(kwargs.get("VISUAL")), diagnostic
