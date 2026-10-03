from __future__ import annotations

from .cmk_final_preview import send_final_preview
from .cmk_visual import empty_visual, register_provider
from ..utils.cmk_timing import cmk_timed
from ..utils.cmk_sampling_warnings import ignore_torchsde_boundary_rounding


class CMKRefinerPipe:
    """Execute the prepared refiner and return comparison and refined images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"REFINER": ("CMK_REFINER_PIPE",)},
            "optional": {"VISUAL": ("CMK_VISUAL_PIPE",)},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "CMK_VISUAL_PIPE")
    RETURN_NAMES = ("IMAGE 1ST PASS", "IMAGE REFINED", "VISUAL")
    FUNCTION = "run"
    CATEGORY = "CMK/Developer/Pipe/Execute"

    @staticmethod
    def _required(refiner_pipe, key):
        value = refiner_pipe.get(key)
        if value is None:
            raise ValueError(f"CMK Refiner -Pipe-: REFINER['{key}'] is missing")
        return value

    def run(self, REFINER, VISUAL=None, unique_id=None):
        if REFINER is None:
            raise ValueError("CMK Refiner -Pipe-: REFINER is missing")
        visual = empty_visual() if VISUAL is None else VISUAL

        if REFINER.get("refiner_prepare_bypassed") or not REFINER.get("refiner_global_enable", True):
            latent = self._required(REFINER, "refiner_latent_image")
            vae = self._required(REFINER, "refiner_vae")
            try:
                from nodes import VAEDecode
            except Exception as exc:
                raise RuntimeError(f"CMK Refiner -Pipe-: VAE Decode unavailable: {exc}") from exc
            with cmk_timed("20 REFINER BYPASS VAE DECODE", "base VAE | sampling skipped"):
                decoded = VAEDecode().decode(vae, latent)
            image = decoded[0] if isinstance(decoded, (tuple, list)) else decoded
            send_final_preview(image)
            return (image, image, visual)

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

        source_image = REFINER.get("refiner_source_image")
        if source_image is None:
            with cmk_timed("20 REFINER SOURCE VAE DECODE"):
                source_decoded = VAEDecode().decode(vae, latent)
            source_image = source_decoded[0] if isinstance(source_decoded, (tuple, list)) else source_decoded
        with cmk_timed("20 REFINER SAMPLE", f"steps {start_at_step}-{end_at_step}"):
            with ignore_torchsde_boundary_rounding():
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
        visual = register_provider(
            visual,
            module_instance_id=unique_id or "refiner",
            module_type="CMKRefinerPipe",
            module_label="2nd-Pass SDXL",
            sequence=20,
            channels={"before": source_image, "after": refined_image},
            status="completed",
            live_node_id=unique_id,
            branch="sdxl",
            stage_key="sdxl.refiner",
        )
        if bool(REFINER.get("unload_models_after_use", True)):
            from .loaders.checkpoint_vae_loader import unload_all_phase_models

            unload_all_phase_models("20 Refiner complete")
        return (source_image, refined_image, visual)
