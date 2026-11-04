from __future__ import annotations

"""CMK-native Kontext Reference Latent Mask support.

Algorithm adapted from ``AILab_ReferenceLatentMask`` in
1038lab/ComfyUI-RMBG (GPL-3.0). It is intentionally kept as an internal
engine helper and is consumed by ``CMK Sampler Prepare SDXL -Pipe-``.
"""

from typing import Any
import math

import torch
import torch.nn.functional as F
import node_helpers


def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("Kontext Reference Latent Mask requires a MASK tensor")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")
    return mask.float().clamp(0.0, 1.0)


def expand_mask(mask: torch.Tensor, expand_amount: int) -> torch.Tensor:
    mask = _normalize_mask(mask)
    amount = int(expand_amount)
    if amount == 0:
        return mask

    binary_mask = (mask > 0.5).float()
    kernel_size = max(3, abs(amount) * 2 + 1)
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device, dtype=mask.dtype)
    x = binary_mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])

    filtered = F.conv2d(x, kernel, padding=kernel_size // 2)
    if amount > 0:
        result = (filtered > 0).float()
    else:
        result = (filtered >= kernel_size * kernel_size).float()
    return result.squeeze(1)


def blur_mask(mask: torch.Tensor, blur_amount: float) -> torch.Tensor:
    mask = _normalize_mask(mask)
    amount = float(blur_amount)
    if amount <= 0.0:
        return mask

    x = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])
    kernel_size = max(3, math.ceil(amount * 3.0) * 2 + 1)
    half_kernel = kernel_size // 2
    grid = torch.arange(-half_kernel, half_kernel + 1, device=mask.device, dtype=torch.float32)
    gaussian = torch.exp(-0.5 * (grid / amount) ** 2)
    gaussian = gaussian / gaussian.sum()
    gaussian_x = gaussian.to(mask.dtype).view(1, 1, 1, kernel_size)
    gaussian_y = gaussian.to(mask.dtype).view(1, 1, kernel_size, 1)
    blurred = F.conv2d(x, gaussian_x, padding=(0, half_kernel))
    blurred = F.conv2d(blurred, gaussian_y, padding=(half_kernel, 0))
    return blurred.squeeze(1).clamp(0.0, 1.0)


def mask_reference_latent(
    samples: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Remove the generation area from a reference latent.

    Supplying the untouched source latent as ``reference_latents`` would
    reintroduce the very object an inpaint operation is meant to replace.
    The unmasked surroundings remain available as structural context.
    """
    if not isinstance(samples, torch.Tensor) or samples.ndim != 4:
        raise ValueError("Kontext Reference requires 4D latent samples")
    normalized = _normalize_mask(mask).to(device=samples.device, dtype=samples.dtype)
    latent_mask = F.interpolate(
        normalized.unsqueeze(1),
        size=samples.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    if latent_mask.shape[0] not in (1, samples.shape[0]):
        raise ValueError(
            "Kontext Reference mask batch does not match latent batch: "
            f"{latent_mask.shape[0]} vs {samples.shape[0]}"
        )
    return samples * (1.0 - latent_mask)


class CMKContextReferenceLatentMask:
    """Attach reference latent and processed mask to positive conditioning."""

    def prepare(
        self,
        conditioning: list,
        latent: dict[str, Any],
        mask: torch.Tensor,
        *,
        expand: int = 3,
        blur: float = 5.0,
        mask_only: bool = True,
    ) -> tuple[list, dict[str, Any], torch.Tensor]:
        if not isinstance(conditioning, list):
            raise TypeError("Kontext Reference Latent Mask requires CONDITIONING")
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("Kontext Reference Latent Mask requires latent with 'samples'")

        source_mask = _normalize_mask(mask)
        processed_mask = source_mask
        if int(expand) != 0:
            processed_mask = expand_mask(processed_mask, int(expand))
        if float(blur) > 0.0:
            processed_mask = blur_mask(processed_mask, float(blur))
            # Feathering must never erode the hand-drawn generation area.
            # Keep every source-mask pixel fully generated and place the soft
            # transition exclusively outside that boundary. Negative expand
            # deliberately requests erosion and therefore remains untouched.
            if int(expand) >= 0:
                processed_mask = torch.maximum(processed_mask, source_mask)

        reference_samples = mask_reference_latent(
            latent["samples"],
            processed_mask,
        )
        modified = node_helpers.conditioning_set_values(
            conditioning,
            {
                "concat_latent_image": reference_samples,
                "concat_mask": processed_mask,
            },
        )
        final_conditioning = node_helpers.conditioning_set_values(
            modified,
            {"reference_latents": [reference_samples]},
            append=True,
        )

        output_latent: dict[str, Any] = {"samples": latent["samples"]}
        if bool(mask_only):
            output_latent["noise_mask"] = processed_mask

        return final_conditioning, output_latent, processed_mask
