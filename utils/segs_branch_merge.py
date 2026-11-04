from __future__ import annotations

from typing import Sequence

import torch

from ..engine.native_detailer import SEGSPaste

from .stable_segs import image_signature, stable_branch_components


def extract_image(value):
    """Return the first ComfyUI IMAGE tensor contained in an Impact result."""
    if value is None:
        return None

    if hasattr(value, "shape") and len(getattr(value, "shape", ())) == 4:
        return value

    if isinstance(value, (tuple, list)):
        for item in value:
            image = extract_image(item)
            if image is not None:
                return image

    return None


def valid_segs(value) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], tuple)
        and len(value[0]) == 2
        and isinstance(value[1], list)
    )


def segment_count(value) -> int:
    return len(value[1]) if valid_segs(value) else 0


def _validate_image(name: str, image) -> None:
    if image is None:
        raise ValueError(f"{name} is missing")
    if not hasattr(image, "shape") or len(image.shape) != 4:
        raise ValueError(f"{name} must be a ComfyUI IMAGE tensor in BHWC format")


def merge_stable_branch(
    authoritative_image,
    accumulated_image,
    branch_image,
    branch_mask,
    source_signature=None,
    authoritative_signature=None,
):
    """Merge a precomposed cache-stable branch over its spatial support."""
    _validate_image("authoritative image", authoritative_image)
    _validate_image("accumulated image", accumulated_image)
    _validate_image("stable branch image", branch_image)

    if tuple(authoritative_image.shape) != tuple(branch_image.shape):
        raise ValueError(
            "CMK SEGS CONCAT: stable branch image shape does not match authoritative image "
            f"({tuple(branch_image.shape)} != {tuple(authoritative_image.shape)})"
        )
    if tuple(accumulated_image.shape) != tuple(branch_image.shape):
        raise ValueError(
            "CMK SEGS CONCAT: accumulated image shape does not match stable branch image "
            f"({tuple(accumulated_image.shape)} != {tuple(branch_image.shape)})"
        )
    if not hasattr(branch_mask, "shape") or len(branch_mask.shape) != 4:
        raise ValueError("CMK SEGS CONCAT: stable branch mask is invalid")
    if tuple(branch_mask.shape[:3]) != tuple(branch_image.shape[:3]) or branch_mask.shape[-1] != 1:
        raise ValueError(
            "CMK SEGS CONCAT: stable branch mask shape does not match branch image "
            f"({tuple(branch_mask.shape)} vs {tuple(branch_image.shape)})"
        )

    if source_signature:
        current_signature = authoritative_signature or image_signature(authoritative_image)
        if current_signature != source_signature:
            raise RuntimeError(
                "CMK SEGS CONCAT: cached branch was created from a different authoritative image; "
                "the branch cache must be recomputed"
            )

    device = accumulated_image.device
    dtype = accumulated_image.dtype
    candidate = branch_image.to(device=device, dtype=dtype)
    mask = branch_mask.to(device=device, dtype=torch.bool)
    return torch.where(mask, candidate, accumulated_image)


def merge_segs_collections(
    authoritative_image,
    segs_collections: Sequence,
    *,
    feather: int = 5,
    alpha: int = 255,
):
    """Compose independent SEGS branches without source/result interleaving.

    CMK stable branches contribute their precomposed image over a spatial SEG
    support. Ordinary Impact SEGS follow native sequential paste semantics.
    Input order defines priority for genuinely overlapping processed regions.
    """
    _validate_image("CMK SEGS CONCAT: image", authoritative_image)

    result_image = authoritative_image
    authoritative_signature = None

    for index, current_segs in enumerate(segs_collections, start=1):
        if current_segs is None:
            continue
        if not valid_segs(current_segs):
            raise ValueError(f"CMK SEGS CONCAT: SEGS {index} is invalid")

        stable = stable_branch_components(current_segs)
        if stable is not None:
            branch_image, branch_mask, source_signature = stable
            if source_signature and authoritative_signature is None:
                authoritative_signature = image_signature(authoritative_image)
            result_image = merge_stable_branch(
                authoritative_image,
                result_image,
                branch_image,
                branch_mask,
                source_signature,
                authoritative_signature,
            )
            continue

        if segment_count(current_segs) == 0:
            continue

        # Ordinary Impact SEGS already identify their actual contribution.
        # Native sequential paste is correct here. A pixel-delta mask is not:
        # it interleaves source and processed pixels and creates the observed
        # salt-and-pepper pattern.
        paste_result = SEGSPaste.doit(
            image=result_image,
            segs=current_segs,
            feather=int(feather),
            alpha=int(alpha),
        )
        pasted_image = extract_image(paste_result)
        if pasted_image is None:
            raise RuntimeError(
                f"CMK SEGS CONCAT: SEGSPaste returned no image for SEGS {index}"
            )
        result_image = pasted_image

    return result_image
