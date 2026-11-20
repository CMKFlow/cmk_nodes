"""Joint latent weight profile for SDXL regional conditioning.

ComfyUI attenuates an AREA conditioning over up to eight latent cells at every
edge that does not touch the image border, so the user's rectangle never acts
at its full strength near its edges.  That attenuation is skipped as soon as
the conditioning carries a ``mask``.  CMK therefore hands every region a native
mask whose weight profile is computed for all regions together, plus an AREA
crop that bounds the UNet evaluation:

* inside its own rectangle a region weighs 1.0;
* towards free space the transition lies outside the rectangle
  (``FREE_SPACE_RAMP`` cells, linear);
* a neighbouring region may reach at most ``SHARED_CROSSFADE`` cells into a
  rectangle with at most ``MAX_FOREIGN_SHARE`` of the cell; within a rectangle
  the weights are normalised to sum 1, so the crossfade only suppresses seams
  and never doubles the regional share;
* in free space the regional weights sum to at most 1;
* cells inside several rectangles (user-drawn overlaps) keep 1.0 per region,
  exactly as before.

CONDITIONING BLEND (07 Regional Layout) steps between the native AREA
(step 0, ComfyUI's own path) and this profile (step ``BLEND_STEPS``).  The MASK
profile widens the UNet crop by FREE_SPACE_RAMP + CONTEXT_MARGIN = 6 latent
cells per free side, so there are exactly six steps: step k uses the integer
ramp, margin and crossfade of ``blend_parameters(k)`` and mixes the weights as
``(1 - k/6) * native AREA + k/6 * profile(k)``.  Every crop change therefore
happens on a step, never between steps.

The user's geometry is read only; nothing is written back.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


PROFILE_VERSION = "cmk-regional-edge-true-v1"
FREE_SPACE_RAMP = 4
SHARED_CROSSFADE = 2
MAX_FOREIGN_SHARE = 0.25
CONTEXT_MARGIN = 2
NATIVE_AREA_FUZZ = 8  # comfy.samplers.get_area_and_mult

BLEND_STEPS = FREE_SPACE_RAMP + CONTEXT_MARGIN
# Regions without a stored step (older layouts, CMKRegionalConditioningSDXL)
# keep the full profile they ran with before the blend existed.
DEFAULT_CONDITIONING_BLEND = BLEND_STEPS


def active_regions(regions) -> list:
    """Regions Sampler Prepare encodes, in the order of their conditioning entries.

    Shared with the InstantID reference conditioning, which must assign each
    regional entry the prompt it was encoded from.
    """
    return [region for region in (regions or []) if isinstance(region, dict) and str(region.get("prompt", "")).strip()]


def regional_prompt_translation(region, translate):
    """The translation of a region's prompt that is handed to CLIP (``translate`` = translate_prompt)."""
    return translate(region.get("prompt", ""))


def conditioning_blend(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) or not 0 <= value <= BLEND_STEPS:
        raise ValueError(f"CMK regional conditioning: blend step {value!r} must be an integer 0-{BLEND_STEPS}")
    return int(value)


def blend_label(step):
    if step == 0:
        return f"BLEND 0/{BLEND_STEPS} · AREA"
    if step == BLEND_STEPS:
        return f"BLEND {BLEND_STEPS}/{BLEND_STEPS} · MASK"
    mask = round(100 * step / BLEND_STEPS)
    return f"BLEND {step}/{BLEND_STEPS} · AREA {100 - mask} % / MASK {mask} %"


def blend_parameters(step: int) -> tuple[int, int, int, float]:
    """Ramp, context margin, crossfade (latent cells) and foreign share cap of a blend step."""
    ramp = round(FREE_SPACE_RAMP * step / BLEND_STEPS)
    return ramp, step - ramp, round(SHARED_CROSSFADE * step / BLEND_STEPS), MAX_FOREIGN_SHARE * step / BLEND_STEPS


@dataclass(frozen=True)
class RegionWeight:
    mask: torch.Tensor
    box: tuple[int, int, int, int]
    crop: tuple[int, int, int, int]
    crop_percent: tuple[float, float, float, float]
    mean_weight: float
    min_weight: float
    max_foreign_weight: float


def latent_box(region, latent_width: int, latent_height: int) -> tuple[int, int, int, int]:
    """Rasterise a normalised rectangle exactly like ComfyUI's percentage AREA.

    Returns ``(x0, y0, x1, y1)`` with exclusive end cells.
    """
    width = max(1, round(float(region.get("width", 0.5)) * latent_width))
    height = max(1, round(float(region.get("height", 0.5)) * latent_height))
    x0 = round(float(region.get("x", 0.25)) * latent_width)
    y0 = round(float(region.get("y", 0.25)) * latent_height)
    x0 = min(max(0, x0), latent_width - 1)
    y0 = min(max(0, y0), latent_height - 1)
    return x0, y0, min(latent_width, x0 + width), min(latent_height, y0 + height)


def native_area_profile(box, latent_width: int, latent_height: int) -> torch.Tensor:
    """ComfyUI's AREA edge falloff for ``box`` as a full-frame weight (strength 1)."""
    x0, y0, x1, y1 = box
    profile = torch.zeros(latent_height, latent_width)
    core = torch.ones(y1 - y0, x1 - x0)
    for dim, (start, end, full) in enumerate(((y0, y1, latent_height), (x0, x1, latent_width))):
        length = end - start
        rr = min(NATIVE_AREA_FUZZ, length // 4)
        ramp = torch.ones(length)
        for t in range(rr):
            if start != 0:
                ramp[t] *= (1.0 / rr) * (t + 1)
            if end < full:
                ramp[length - 1 - t] *= (1.0 / rr) * (t + 1)
        core *= ramp.view(-1, 1) if dim == 0 else ramp.view(1, -1)
    profile[y0:y1, x0:x1] = core
    return profile


def _box_distance(box, grid_x, grid_y):
    x0, y0, x1, y1 = box
    dx = torch.clamp(x0 - grid_x, min=0) + torch.clamp(grid_x - (x1 - 1), min=0)
    dy = torch.clamp(y0 - grid_y, min=0) + torch.clamp(grid_y - (y1 - 1), min=0)
    return torch.sqrt(dx * dx + dy * dy)


def build_region_weights(regions, latent_width: int, latent_height: int, step: int = BLEND_STEPS) -> list[RegionWeight]:
    """Return one native-mask weight profile and AREA crop per region, in order.

    ``step`` is the CONDITIONING BLEND step 1..BLEND_STEPS; step 0 is ComfyUI's
    native AREA and has no profile.
    """
    latent_width = int(latent_width)
    latent_height = int(latent_height)
    if latent_width < 1 or latent_height < 1:
        raise ValueError(f"CMK regional weighting: invalid latent size {latent_width}x{latent_height}")
    if not 1 <= step <= BLEND_STEPS:
        raise ValueError(f"CMK regional weighting: blend step {step} has no mask profile")
    boxes = [latent_box(region, latent_width, latent_height) for region in regions]
    if not boxes:
        return []
    free_space_ramp, context_margin, shared_crossfade, max_foreign_share = blend_parameters(step)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(latent_height, dtype=torch.float32),
        torch.arange(latent_width, dtype=torch.float32),
        indexing="ij",
    )
    distance = torch.stack([_box_distance(box, grid_x, grid_y) for box in boxes])
    inside = distance == 0
    inside_count = inside.sum(dim=0)

    free_ramp = torch.clamp(1.0 - distance / (free_space_ramp + 1), min=0.0)
    shared_ramp = torch.where(
        distance <= shared_crossfade,
        1.0 - distance / (shared_crossfade + 1),
        torch.zeros_like(distance),
    )

    # Exactly one owning rectangle: owner 1.0 plus a short neighbour crossfade,
    # normalised to a total of 1.  The neighbours together never exceed
    # the foreign share cap of a cell, also for touching rectangles.
    foreign = torch.where(inside, torch.zeros_like(shared_ramp), shared_ramp)
    foreign_cap = max_foreign_share / (1.0 - max_foreign_share)
    foreign_sum = foreign.sum(dim=0, keepdim=True)
    foreign = foreign * torch.clamp(foreign_cap / torch.clamp(foreign_sum, min=1e-12), max=1.0)
    owned = torch.where(inside, torch.ones_like(foreign), foreign)
    owned = owned / owned.sum(dim=0, keepdim=True)
    # Free space: outward ramps, together never above 1.
    free = free_ramp / torch.clamp(free_ramp.sum(dim=0, keepdim=True), min=1.0)
    # User overlaps: every containing region keeps its full weight.
    overlap = inside.to(torch.float32)

    weights = torch.where(
        inside_count >= 2,
        overlap,
        torch.where(inside_count == 1, owned, free),
    )
    if step < BLEND_STEPS:
        mix = step / BLEND_STEPS
        native = torch.stack([native_area_profile(box, latent_width, latent_height) for box in boxes])
        weights = (1.0 - mix) * native + mix * weights

    results = []
    for index, box in enumerate(boxes):
        mask = weights[index]
        rows = torch.nonzero(mask.sum(dim=1) > 0).flatten()
        cols = torch.nonzero(mask.sum(dim=0) > 0).flatten()
        cx0 = max(0, int(cols.min()) - context_margin)
        cx1 = min(latent_width, int(cols.max()) + 1 + context_margin)
        cy0 = max(0, int(rows.min()) - context_margin)
        cy1 = min(latent_height, int(rows.max()) + 1 + context_margin)
        x0, y0, x1, y1 = box
        own = mask[y0:y1, x0:x1]
        foreign = [
            float(weights[other, y0:y1, x0:x1].max())
            for other in range(len(boxes)) if other != index
        ]
        results.append(RegionWeight(
            mask=mask.unsqueeze(0).contiguous(),
            box=box,
            crop=(cx0, cy0, cx1, cy1),
            crop_percent=(
                cx0 / latent_width,
                cy0 / latent_height,
                (cx1 - cx0) / latent_width,
                (cy1 - cy0) / latent_height,
            ),
            mean_weight=float(own.mean()),
            min_weight=float(own.min()),
            max_foreign_weight=max(foreign) if foreign else 0.0,
        ))
    return results
