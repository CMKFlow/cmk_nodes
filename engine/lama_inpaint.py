from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import threading
import urllib.request

import torch
import torch.nn.functional as F


LAMA_MODEL_FILENAME = "big-lama.pt"
LAMA_MODEL_URL = (
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
)
LAMA_MODEL_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"
_DOWNLOAD_LOCK = threading.Lock()


def lama_model_path() -> Path:
    try:
        import folder_paths  # type: ignore

        root = Path(folder_paths.models_dir)
    except Exception:
        root = Path.cwd() / "models"
    return root / "inpaint" / LAMA_MODEL_FILENAME


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_lama_model() -> Path:
    path = lama_model_path()
    if path.is_file() and _md5(path) == LAMA_MODEL_MD5:
        return path

    with _DOWNLOAD_LOCK:
        if path.is_file() and _md5(path) == LAMA_MODEL_MD5:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".download")
        try:
            if partial.exists():
                partial.unlink()
            urllib.request.urlretrieve(LAMA_MODEL_URL, partial)
            if _md5(partial) != LAMA_MODEL_MD5:
                raise RuntimeError("downloaded model checksum does not match")
            os.replace(partial, path)
        except Exception as exc:
            if partial.exists():
                partial.unlink()
            raise RuntimeError(
                "CMK Remove Object requires big-lama.pt. Automatic download failed. "
                f"Place the model at: {path}"
            ) from exc
    return path


@lru_cache(maxsize=1)
def _load_lama():
    model = torch.jit.load(str(ensure_lama_model()), map_location="cpu")
    return model.eval()


def lama_inpaint_tensor(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Run local prompt-free LaMa object removal and preserve pixels outside MASK."""
    if image is None or mask is None:
        return image
    if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
        raise ValueError("CMK LaMa expects IMAGE as [B,H,W,C].")

    work_image = image[..., :3].detach().float().clamp(0.0, 1.0).cpu()
    work_mask = mask.detach().float().cpu()
    if work_mask.ndim == 2:
        work_mask = work_mask.unsqueeze(0)
    if work_mask.ndim == 4:
        work_mask = work_mask[..., 0] if work_mask.shape[-1] == 1 else work_mask[:, 0]
    if work_mask.ndim != 3:
        raise ValueError("CMK LaMa expects MASK as [B,H,W].")
    if tuple(work_mask.shape[-2:]) != tuple(work_image.shape[1:3]):
        work_mask = F.interpolate(
            work_mask.unsqueeze(1),
            size=tuple(work_image.shape[1:3]),
            mode="nearest",
        ).squeeze(1)
    if work_mask.shape[0] == 1 and work_image.shape[0] > 1:
        work_mask = work_mask.expand(work_image.shape[0], -1, -1)

    image_bchw = work_image.permute(0, 3, 1, 2)
    mask_bchw = (work_mask > 0.01).float().unsqueeze(1)
    height, width = image_bchw.shape[-2:]
    pad_h = (-height) % 8
    pad_w = (-width) % 8
    if pad_h or pad_w:
        image_bchw = F.pad(image_bchw, (0, pad_w, 0, pad_h), mode="reflect")
        mask_bchw = F.pad(mask_bchw, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    with torch.inference_mode():
        prediction = _load_lama()(image_bchw, mask_bchw)
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
    if isinstance(prediction, dict):
        prediction = prediction.get("inpainted")
        if prediction is None:
            prediction = prediction.get("output")
    if not isinstance(prediction, torch.Tensor):
        raise RuntimeError("CMK LaMa returned no image tensor.")

    prediction = prediction[..., :height, :width].clamp(0.0, 1.0)
    source = work_image.permute(0, 3, 1, 2)
    alpha = mask_bchw[..., :height, :width]
    result = source * (1.0 - alpha) + prediction * alpha
    return result.permute(0, 2, 3, 1).to(device=image.device, dtype=image.dtype)
