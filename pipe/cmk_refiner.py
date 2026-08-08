from __future__ import annotations

import torch.nn.functional as F

from .cmk_final_preview import send_final_preview
from ..utils.cmk_timing import cmk_timed


class CMKRefinerPipe:
    """Execute the prepared refiner and return comparison and refined images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"REFINER": ("CMK_REFINER_PIPE",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE 1ST PASS", "IMAGE REFINED")
    FUNCTION = "run"
    CATEGORY = "CMK/Developer/Pipe/Execute"

    @staticmethod
    def _required(refiner_pipe, key):
        value = refiner_pipe.get(key)
        if value is None:
            raise ValueError(f"CMK Refiner -Pipe-: REFINER['{key}'] is missing")
        return value

    def run(self, REFINER):
        if REFINER is None:
            raise ValueError("CMK Refiner -Pipe-: REFINER is missing")
        if REFINER.get("inpaint_process_mode") == "remove":
            remove_image = REFINER.get("remove_result_image")
            if remove_image is not None:
                send_final_preview(remove_image)
                return (remove_image, remove_image)

        model = self._required(REFINER, "refiner_model")
        positive = self._required(REFINER, "refiner_conditioning_pos")
        negative = self._required(REFINER, "refiner_conditioning_neg")
        latent = self._required(REFINER, "refiner_latent_image")
        vae = self._required(REFINER, "refiner_vae")

        seed = int(REFINER.get("refiner_seed", REFINER.get("seed", 0)))
        steps = int(REFINER.get("refiner_steps", 25))
        cfg = float(REFINER.get("refiner_cfg", 4.8))
        sampler = REFINER.get("refiner_sampler", "euler")
        scheduler = REFINER.get("refiner_scheduler", "simple")
        start_at_step = int(REFINER.get("refiner_start_at_step", int(steps * 0.8)))
        end_at_step = int(REFINER.get("refiner_end_at_step", steps))

        try:
            from nodes import KSamplerAdvanced, VAEDecode
        except Exception as exc:
            raise RuntimeError(f"CMK Refiner -Pipe-: required ComfyUI nodes unavailable: {exc}") from exc

        with cmk_timed("20 REFINER SOURCE VAE DECODE"):
            source_decoded = VAEDecode().decode(vae, latent)
        source_image = source_decoded[0] if isinstance(source_decoded, (tuple, list)) else source_decoded
        if REFINER.get("inpaint_process_mode") == "remove":
            # Preserve the prompt-free first-pass reconstruction and composite
            # only its soft generation area over the untouched source. This
            # prevents a synthetic mask fill from surviving at the hand-drawn
            # edge while retaining the original colour outside the mask.
            original = REFINER.get("inpaint_source_image")
            mask = REFINER.get("mask")
            if original is not None and mask is not None:
                if original.shape[1:3] != source_image.shape[1:3]:
                    original = F.interpolate(
                        original.movedim(-1, 1),
                        size=source_image.shape[1:3],
                        mode="bilinear",
                        align_corners=False,
                    ).movedim(1, -1)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                if mask.ndim == 4:
                    mask = mask[:, 0] if mask.shape[1] == 1 else mask[..., 0]
                soft = F.interpolate(
                    mask.float().unsqueeze(1),
                    size=source_image.shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1).clamp(0.0, 1.0).unsqueeze(-1)
                source_image = source_image * soft + original.to(source_image) * (1.0 - soft)
            # A second diffusion pass with unrelated Refiner prompts can
            # recreate the object that Remove deliberately discarded.
            send_final_preview(source_image)
            return (source_image, source_image)

        with cmk_timed("20 REFINER SAMPLE", f"steps {start_at_step}-{end_at_step}"):
            sampled = KSamplerAdvanced().sample(
                model,
                "enable",
                seed,
                steps,
                cfg,
                sampler,
                scheduler,
                positive,
                negative,
                latent,
                start_at_step,
                end_at_step,
                "disable",
            )
        samples = sampled[0] if isinstance(sampled, (tuple, list)) else sampled

        with cmk_timed("20 REFINER RESULT VAE DECODE"):
            decoded = VAEDecode().decode(vae, samples)
        refined_image = decoded[0] if isinstance(decoded, (tuple, list)) else decoded

        send_final_preview(refined_image)
        return (source_image, refined_image)
