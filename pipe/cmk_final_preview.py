from __future__ import annotations


def send_final_preview(image) -> None:
    """Replace the current node's latent preview with its decoded result."""
    if image is None:
        return
    try:
        import numpy as np
        from PIL import Image
        from comfy.utils import ProgressBar

        frame = image[0] if getattr(image, "ndim", 0) == 4 else image
        pixels = (
            frame.detach()
            .float()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .byte()
            .cpu()
            .numpy()
        )
        if pixels.ndim == 3 and pixels.shape[-1] == 1:
            pixels = pixels[..., 0]
        preview = Image.fromarray(np.ascontiguousarray(pixels))
        ProgressBar(1).update_absolute(1, 1, ("PNG", preview, None))
    except Exception:
        # Preview delivery must never invalidate an otherwise finished render.
        return
