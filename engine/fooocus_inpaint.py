from __future__ import annotations

"""CMK-native Fooocus inpaint support.

Implements the model-loading and patching logic required by
``CMK Sampler Prepare SDXL -Pipe-`` without calling external custom nodes.

Algorithm adapted from Acly/comfyui-inpaint-nodes (GPL-3.0):
https://github.com/Acly/comfyui-inpaint-nodes
"""

import os
from typing import Any

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.utils
import folder_paths
import node_helpers
from comfy.model_base import BaseModel
from comfy.model_management import cast_to_device
from comfy.model_patcher import ModelPatcher
from torch import Tensor


INPAINT_CATEGORY = "inpaint"
_INPAINT_EXTENSIONS = {".safetensors", ".pt", ".pth", ".ckpt", ".patch"}


def ensure_inpaint_model_folder() -> None:
    """Register ``models/inpaint`` when no other node pack has done so."""
    try:
        existing = folder_paths.get_folder_paths(INPAINT_CATEGORY)
        if existing:
            return
    except Exception:
        pass

    path = os.path.join(folder_paths.models_dir, INPAINT_CATEGORY)
    os.makedirs(path, exist_ok=True)
    folder_paths.add_model_folder_path(INPAINT_CATEGORY, path, is_default=True)
    try:
        current = folder_paths.folder_names_and_paths[INPAINT_CATEGORY]
        if isinstance(current, tuple) and len(current) >= 2:
            folder_paths.folder_names_and_paths[INPAINT_CATEGORY] = (current[0], _INPAINT_EXTENSIONS)
    except Exception:
        pass


class InpaintHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Parameter(torch.empty(size=(320, 5, 3, 3), device="cpu"))

    def __call__(self, x: Tensor) -> Tensor:
        x = F.pad(x, (1, 1, 1, 1), "replicate")
        return F.conv2d(x, weight=self.head)


class InpaintBlockPatch:
    def __init__(self) -> None:
        self.inpaint_head_feature: Tensor | None = None
        self._inpaint_block: Tensor | None = None

    def __call__(self, h: Tensor, transformer_options: dict[str, Any]) -> Tensor:
        if transformer_options["block"][1] == 0:
            if self._inpaint_block is None or self._inpaint_block.shape != h.shape:
                if self.inpaint_head_feature is None:
                    raise RuntimeError("Fooocus inpaint head feature is missing")
                batch = h.shape[0] // self.inpaint_head_feature.shape[0]
                self._inpaint_block = self.inpaint_head_feature.to(h).repeat(batch, 1, 1, 1)
            h = h + self._inpaint_block
        return h


def _load_fooocus_patch(lora: dict[str, Tensor], to_load: dict[str, str]) -> dict[str, tuple[str, Tensor]]:
    patch_dict: dict[str, tuple[str, Tensor]] = {}
    loaded_keys: set[str] = set()
    for key in to_load.values():
        value = lora.get(key)
        if value is not None:
            patch_dict[key] = ("fooocus", value)
            loaded_keys.add(key)

    not_loaded = sum(1 for key in lora if key not in loaded_keys)
    if not_loaded > 0:
        print(
            f"[CMK Fooocus Inpaint] {len(loaded_keys)} LoRA keys loaded, "
            f"{not_loaded} remaining keys not found in model."
        )
    return patch_dict


if not hasattr(comfy.lora, "calculate_weight") and hasattr(ModelPatcher, "calculate_weight"):
    raise RuntimeError("CMK Fooocus Inpaint requires ComfyUI v0.1.1 or later")

_ORIGINAL_CALCULATE_WEIGHT = comfy.lora.calculate_weight
_CALCULATE_WEIGHT_PATCHED = False


def _calculate_weight_patched(
    patches,
    weight,
    key,
    intermediate_dtype=torch.float32,
    original_weights=None,
):
    remaining = []
    for patch in patches:
        alpha = patch[0]
        value = patch[1]
        is_fooocus = isinstance(value, tuple) and len(value) == 2 and value[0] == "fooocus"
        if not is_fooocus:
            remaining.append(patch)
            continue

        if alpha == 0.0:
            continue

        value = value[1]
        w1 = cast_to_device(value[0], weight.device, torch.float32)
        if w1.shape != weight.shape:
            print(
                f"[CMK Fooocus Inpaint] Shape mismatch {key}, weight not merged "
                f"({w1.shape} != {weight.shape})"
            )
            continue

        w_min = cast_to_device(value[1], weight.device, torch.float32)
        w_max = cast_to_device(value[2], weight.device, torch.float32)
        w1 = (w1 / 255.0) * (w_max - w_min) + w_min
        weight += alpha * cast_to_device(w1, weight.device, weight.dtype)

    if remaining:
        try:
            return _ORIGINAL_CALCULATE_WEIGHT(
                remaining,
                weight,
                key,
                intermediate_dtype,
                original_weights=original_weights,
            )
        except TypeError:
            return _ORIGINAL_CALCULATE_WEIGHT(remaining, weight, key, intermediate_dtype)
    return weight


def _inject_calculate_weight_patch() -> None:
    global _CALCULATE_WEIGHT_PATCHED
    if not _CALCULATE_WEIGHT_PATCHED:
        print("[CMK Fooocus Inpaint] Injecting Fooocus weight handler")
        comfy.lora.calculate_weight = _calculate_weight_patched
        _CALCULATE_WEIGHT_PATCHED = True


def _resolve_inpaint_file(name: str) -> str:
    ensure_inpaint_model_folder()
    cleaned = str(name or "").strip().replace("\\", "/")
    if not cleaned:
        raise ValueError("Fooocus inpaint filename is empty")
    if os.path.isabs(cleaned) and os.path.isfile(cleaned):
        return cleaned

    direct = folder_paths.get_full_path(INPAINT_CATEGORY, cleaned)
    if direct:
        return direct

    wanted = os.path.basename(cleaned).lower()
    matches: list[str] = []
    for candidate in folder_paths.get_filename_list(INPAINT_CATEGORY):
        if os.path.basename(candidate).lower() == wanted:
            resolved = folder_paths.get_full_path(INPAINT_CATEGORY, candidate)
            if resolved:
                matches.append(resolved)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(f"Fooocus inpaint filename is ambiguous: {name}")
    raise FileNotFoundError(
        f"Fooocus inpaint file not found: {name}. "
        f"Expected below {os.path.join(folder_paths.models_dir, INPAINT_CATEGORY)}"
    )


def load_fooocus_inpaint(head: str, patch: str) -> tuple[InpaintHead, dict[str, Tensor]]:
    head_file = _resolve_inpaint_file(head)
    patch_file = _resolve_inpaint_file(patch)

    model = InpaintHead()
    state_dict = torch.load(head_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    lora = comfy.utils.load_torch_file(patch_file, safe_load=True)
    return model, lora


def apply_fooocus_inpaint(
    model: ModelPatcher,
    patch: tuple[InpaintHead, dict[str, Tensor]],
    latent: dict[str, Any],
) -> ModelPatcher:
    if not isinstance(latent, dict) or "samples" not in latent or "noise_mask" not in latent:
        raise ValueError("Fooocus Inpaint requires latent with 'samples' and 'noise_mask'")

    base_model: BaseModel = model.model
    latent_pixels = base_model.process_latent_in(latent["samples"])
    noise_mask = latent["noise_mask"].round()
    latent_mask = F.max_pool2d(noise_mask, (8, 8)).round().to(latent_pixels)

    inpaint_head_model, inpaint_lora = patch
    feed = torch.cat([latent_mask, latent_pixels], dim=1)
    inpaint_head_model.to(device=feed.device, dtype=feed.dtype)

    block_patch = InpaintBlockPatch()
    block_patch.inpaint_head_feature = inpaint_head_model(feed)

    lora_keys = comfy.lora.model_lora_keys_unet(model.model, {})
    lora_keys.update({key: key for key in base_model.state_dict().keys()})
    loaded_lora = _load_fooocus_patch(inpaint_lora, lora_keys)

    patched_model = model.clone()
    patched_model.set_model_input_block_patch(block_patch)
    patched = patched_model.add_patches(loaded_lora, 1.0)
    not_patched_count = sum(1 for key in loaded_lora if key not in patched)
    if not_patched_count > 0:
        print(f"[CMK Fooocus Inpaint] Failed to patch {not_patched_count} keys")

    _inject_calculate_weight_patch()
    return patched_model


class CMKFooocusInpaintPipeline:
    """Native CMK wrapper for the complete Fooocus inpaint preparation chain.

    The mask is always retained internally for Fooocus model patching.  The
    ``noise_mask`` switch controls only whether the returned sampler latent
    contains a noise mask.
    """

    def prepare(
        self,
        model: ModelPatcher,
        positive,
        negative,
        vae,
        image: Tensor,
        mask: Tensor,
        *,
        head: str = "fooocus_inpaint_head.pth",
        patch: str = "inpaint_v25.fooocus.patch",
        noise_mask: bool = False,
    ):
        if model is None or not hasattr(model, "clone"):
            raise TypeError("CMK Fooocus Inpaint requires a valid ComfyUI MODEL")
        if vae is None:
            raise ValueError("CMK Fooocus Inpaint requires VAE")
        if image is None or mask is None:
            raise ValueError("CMK Fooocus Inpaint requires image and mask")

        # Native equivalent of ComfyUI InpaintModelConditioning.
        x = (image.shape[1] // 8) * 8
        y = (image.shape[2] // 8) * 8
        resized_mask = F.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(image.shape[1], image.shape[2]),
            mode="bilinear",
        )

        original_pixels = image
        pixels = original_pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            resized_mask = resized_mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]
            original_pixels = original_pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]

        keep = (1.0 - resized_mask.round()).squeeze(1)
        for channel in range(3):
            pixels[:, :, :, channel] -= 0.5
            pixels[:, :, :, channel] *= keep
            pixels[:, :, :, channel] += 0.5

        concat_latent = vae.encode(pixels)
        original_latent = vae.encode(original_pixels)

        positive_out = node_helpers.conditioning_set_values(
            positive,
            {"concat_latent_image": concat_latent, "concat_mask": resized_mask},
        )
        negative_out = node_helpers.conditioning_set_values(
            negative,
            {"concat_latent_image": concat_latent, "concat_mask": resized_mask},
        )

        # Fooocus model patch and KSampler require two distinct latents:
        #
        # - patch_latent uses the masked/neutral concat latent. This prevents
        #   the Fooocus inpaint head from seeing and reconstructing the original
        #   content underneath the mask.
        # - sampler_latent keeps the original latent and optionally carries the
        #   noise mask so unmasked regions remain protected during sampling.
        patch_latent = {
            "samples": concat_latent,
            "noise_mask": resized_mask.round(),
        }
        inpaint_patch = load_fooocus_inpaint(head, patch)
        patched_model = apply_fooocus_inpaint(model, inpaint_patch, patch_latent)

        sampler_latent = {"samples": original_latent}
        if noise_mask:
            sampler_latent["noise_mask"] = resized_mask.round()

        return patched_model, positive_out, negative_out, sampler_latent
