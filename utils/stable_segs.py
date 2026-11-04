from __future__ import annotations

import hashlib
from typing import Any

import torch


_STABLE_SEGS_SCHEMA = 2
_DELTA_EPSILON = 1.0 / 65535.0


class CMKStableSEGS(tuple):
    """Impact-compatible SEGS with a cache-stable full branch composition.

    The tuple payload remains ``((width, height), [SEG, ...])``. CMK adds the
    independently composed branch image and a *spatial branch support mask*.
    The support is derived from the selected/processed SEG crop regions, never
    from per-pixel differences. This is essential: a diffusion result and its
    source naturally share individual pixels; a pixel-delta mask would combine
    both images in a salt-and-pepper pattern.
    """

    def __new__(
        cls,
        header,
        items=None,
        branch_image=None,
        branch_mask=None,
        source_signature=None,
        schema: int = _STABLE_SEGS_SCHEMA,
    ):
        if items is None and isinstance(header, (tuple, list)) and len(header) == 2:
            header, items = header
        obj = super().__new__(cls, (header, list(items or [])))
        obj.cmk_branch_image = branch_image
        obj.cmk_branch_mask = branch_mask
        obj.cmk_source_signature = source_signature
        obj.cmk_stable_schema = int(schema)
        return obj

    def __reduce__(self):
        return (
            self.__class__,
            (
                self[0],
                self[1],
                self.cmk_branch_image,
                self.cmk_branch_mask,
                self.cmk_source_signature,
                self.cmk_stable_schema,
            ),
        )


def _validate_image(name: str, image) -> None:
    if image is None:
        raise ValueError(f"{name} is missing")
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError(f"{name} must be a ComfyUI IMAGE tensor in BHWC format")


def _valid_segs(value) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], tuple)
        and len(value[0]) == 2
        and isinstance(value[1], list)
    )


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").contiguous()


def image_signature(image: torch.Tensor) -> str:
    """Return a deterministic source signature without retaining the source image."""
    _validate_image("CMK stable SEGS source image", image)
    tensor = _cpu_tensor(image).to(torch.float32)
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _crop_region(seg, width: int, height: int):
    region = getattr(seg, "crop_region", None)
    if region is None or len(region) < 4:
        return None
    try:
        x1 = max(0, min(width, int(round(float(region[0])))))
        y1 = max(0, min(height, int(round(float(region[1])))))
        x2 = max(0, min(width, int(round(float(region[2])))))
        y2 = max(0, min(height, int(round(float(region[3])))))
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _spatial_support_mask(segs, image: torch.Tensor) -> torch.Tensor:
    """Return the union of the actual branch SEG crop regions.

    The branch image is already fully composed, including its native mask and
    feather. Copying that precomposed image over the rectangular support keeps
    the exact feathered pixels while avoiding any pixel-by-pixel source/result
    interleaving. Overlapping branch regions retain the documented input-order
    priority: the later branch wins.
    """
    batch, height, width, _ = image.shape
    mask = torch.zeros(
        (batch, height, width, 1),
        dtype=torch.bool,
        device=image.device,
    )
    if not _valid_segs(segs):
        return mask

    for seg in segs[1]:
        region = _crop_region(seg, width, height)
        if region is None:
            continue
        x1, y1, x2, y2 = region
        mask[:, y1:y2, x1:x2, :] = True
    return mask


def _delta_bbox_support(authoritative_image, branch_image) -> torch.Tensor:
    """Fallback support for a processed branch without usable SEGS.

    A bounding rectangle is used per batch item. The raw delta itself is never
    used as the merge mask because that creates the observed speckled overlay.
    """
    source = authoritative_image.detach().to(torch.float32)
    candidate = branch_image.detach().to(device=source.device, dtype=torch.float32)
    changed = torch.amax(torch.abs(candidate - source), dim=-1) > _DELTA_EPSILON
    batch, height, width = changed.shape
    mask = torch.zeros((batch, height, width, 1), dtype=torch.bool, device=source.device)

    for index in range(batch):
        coords = torch.nonzero(changed[index], as_tuple=False)
        if coords.numel() == 0:
            continue
        y1 = max(0, int(coords[:, 0].min().item()) - 1)
        y2 = min(height, int(coords[:, 0].max().item()) + 2)
        x1 = max(0, int(coords[:, 1].min().item()) - 1)
        x2 = min(width, int(coords[:, 1].max().item()) + 2)
        mask[index, y1:y2, x1:x2, :] = True
    return mask


def segs_with_image_crops(segs, image: torch.Tensor, *, full_crop_mask: bool = False):
    """Bind selected SEG regions to crops from the final branch image.

    FaceProcess previously re-detected all faces after processing. That made an
    individual branch claim unchanged sibling faces. This helper instead builds
    processed SEGS strictly from the selected branch regions and the final image.
    """
    if not _valid_segs(segs):
        raise ValueError("CMK processed SEGS: invalid SEGS value")
    _validate_image("CMK processed SEGS image", image)

    _, height, width, _ = image.shape
    items = []
    for seg in segs[1]:
        region = _crop_region(seg, width, height)
        if region is None:
            continue
        x1, y1, x2, y2 = region
        cropped_image = _cpu_tensor(image[:, y1:y2, x1:x2, :])

        cropped_mask = getattr(seg, "cropped_mask", None)
        if full_crop_mask:
            cropped_mask = torch.ones(
                (image.shape[0], y2 - y1, x2 - x1),
                dtype=torch.float32,
                device="cpu",
            )
        elif isinstance(cropped_mask, torch.Tensor):
            cropped_mask = _cpu_tensor(cropped_mask)

        if hasattr(seg, "_replace"):
            try:
                seg = seg._replace(
                    cropped_image=cropped_image,
                    cropped_mask=cropped_mask,
                )
            except Exception:
                try:
                    seg = seg._replace(cropped_image=cropped_image)
                except Exception:
                    pass
        items.append(seg)

    return (segs[0], items)


def make_stable_segs(
    segs,
    authoritative_image,
    branch_image,
    *,
    coverage_segs=None,
):
    """Attach a cache-stable full-image branch and spatial support to SEGS."""
    if not _valid_segs(segs):
        raise ValueError("CMK stable SEGS: invalid SEGS value")

    _validate_image("CMK stable SEGS authoritative image", authoritative_image)
    _validate_image("CMK stable SEGS branch image", branch_image)
    if tuple(authoritative_image.shape) != tuple(branch_image.shape):
        raise ValueError(
            "CMK stable SEGS: branch image shape does not match authoritative image "
            f"({tuple(branch_image.shape)} != {tuple(authoritative_image.shape)})"
        )

    support_source = coverage_segs if _valid_segs(coverage_segs) else segs
    mask = _spatial_support_mask(support_source, authoritative_image)
    if not bool(mask.any().item()):
        mask = _delta_bbox_support(authoritative_image, branch_image)

    return CMKStableSEGS(
        segs[0],
        segs[1],
        branch_image=_cpu_tensor(branch_image),
        branch_mask=_cpu_tensor(mask),
        source_signature=image_signature(authoritative_image),
    )


def stable_branch_components(value) -> tuple[Any, Any, str | None] | None:
    """Return ``(branch_image, mask, source_signature)`` for CMK stable SEGS."""
    if not isinstance(value, CMKStableSEGS):
        return None
    if int(getattr(value, "cmk_stable_schema", 0)) != _STABLE_SEGS_SCHEMA:
        return None

    branch_image = getattr(value, "cmk_branch_image", None)
    branch_mask = getattr(value, "cmk_branch_mask", None)
    if not isinstance(branch_image, torch.Tensor) or branch_image.ndim != 4:
        return None
    if not isinstance(branch_mask, torch.Tensor) or branch_mask.ndim != 4:
        return None
    if branch_mask.shape[-1] != 1:
        return None
    return (
        branch_image,
        branch_mask,
        getattr(value, "cmk_source_signature", None),
    )
