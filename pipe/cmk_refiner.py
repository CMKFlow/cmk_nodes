from __future__ import annotations


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

        source_decoded = VAEDecode().decode(vae, latent)
        source_image = source_decoded[0] if isinstance(source_decoded, (tuple, list)) else source_decoded

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

        decoded = VAEDecode().decode(vae, samples)
        refined_image = decoded[0] if isinstance(decoded, (tuple, list)) else decoded

        return (source_image, refined_image)
