from __future__ import annotations

import numpy as np
import torch


def tensor_to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    """Convert one ComfyUI IMAGE tensor [H,W,C] float 0..1 to RGB uint8."""
    if image is None:
        raise ValueError("image is None")
    if image.ndim != 3:
        raise ValueError(f"expected image tensor [H,W,C], got shape {tuple(image.shape)}")
    array = image.detach().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"expected 3 RGB channels, got shape {array.shape}")
    return array


def uint8_rgb_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert RGB uint8 [H,W,C] to one ComfyUI IMAGE tensor [H,W,C] float 0..1."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected RGB array [H,W,3], got shape {image.shape}")
    image = np.clip(image, 0, 255).astype(np.uint8)
    return torch.from_numpy(image.astype(np.float32) / 255.0)
