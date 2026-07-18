from ..cmk_common import SAMPLERS, SCHEDULERS


class CMKPipeSetSampler:
    """First-pass sampler settings with integrated smart ControlNet bypass."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("CMK_PIPE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "conditioning_pos": ("CONDITIONING",),
                "conditioning_neg": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "steps_1st_pass": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler": (SAMPLERS,),
                "scheduler": (SCHEDULERS,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}),
            },
            "optional": {
                "model_patched": ("MODEL", {"forceInput": True}),
                "opt_control_net": ("CONTROL_NET", {"forceInput": True}),
                "opt_controlnet_image": ("IMAGE", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CMK_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "set_sampler"
    CATEGORY = 'CMK/Developer/Pipe/Set'

    @staticmethod
    def _apply_controlnet_or_bypass(
        conditioning_pos,
        conditioning_neg,
        control_net,
        controlnet_image,
        vae,
        cn_strength,
        cn_start_percent,
        cn_end_percent,
    ):
        if control_net is None:
            return conditioning_pos, conditioning_neg, False, "ControlNet bypass | missing opt_control_net"
        if controlnet_image is None:
            return conditioning_pos, conditioning_neg, False, "ControlNet bypass | missing opt_controlnet_image"
        if vae is None:
            return conditioning_pos, conditioning_neg, False, "ControlNet bypass | vae missing"
        if conditioning_pos is None:
            return conditioning_pos, conditioning_neg, False, "ControlNet bypass | conditioning_pos missing"
        if conditioning_neg is None:
            return conditioning_pos, conditioning_neg, False, "ControlNet bypass | conditioning_neg missing"
        if cn_strength <= 0.0:
            return conditioning_pos, conditioning_neg, False, "ControlNet bypass | cn_strength <= 0"
        if cn_end_percent <= cn_start_percent:
            return conditioning_pos, conditioning_neg, False, "ControlNet bypass | cn_end_percent <= cn_start_percent"

        try:
            from nodes import ControlNetApplyAdvanced
        except Exception as exc:
            return conditioning_pos, conditioning_neg, False, f"ControlNet bypass | ControlNetApplyAdvanced unavailable: {exc}"

        try:
            result = ControlNetApplyAdvanced().apply_controlnet(
                conditioning_pos,
                conditioning_neg,
                control_net,
                controlnet_image,
                cn_strength,
                cn_start_percent,
                cn_end_percent,
                vae,
            )
        except TypeError:
            result = ControlNetApplyAdvanced().apply_controlnet(
                conditioning_pos,
                conditioning_neg,
                control_net,
                controlnet_image,
                cn_strength,
                cn_start_percent,
                cn_end_percent,
            )

        return (
            result[0],
            result[1],
            True,
            f"ControlNet applied | strength={cn_strength:.3f} | start={cn_start_percent:.3f} | end={cn_end_percent:.3f}",
        )

    def set_sampler(
        self,
        pipe,
        model,
        clip,
        vae,
        conditioning_pos,
        conditioning_neg,
        latent_image,
        steps_1st_pass,
        cfg,
        sampler,
        scheduler,
        seed,
        model_patched=None,
        opt_control_net=None,
        opt_controlnet_image=None,
    ):
        cn_strength = 1.0
        cn_start_percent = 0.0
        cn_end_percent = 1.0

        conditioning_pos, conditioning_neg, cn_applied, cn_log = self._apply_controlnet_or_bypass(
            conditioning_pos,
            conditioning_neg,
            opt_control_net,
            opt_controlnet_image,
            vae,
            cn_strength,
            cn_start_percent,
            cn_end_percent,
        )

        new_pipe = dict(pipe)
        # Store the clean checkpoint model and the sampler-specific patched model separately.
        # model         = clean checkpoint model
        # model_patched = patched sampler model
        new_pipe["model"] = model
        new_pipe["model_patched"] = model_patched if model_patched is not None else model
        new_pipe["clip"] = clip
        new_pipe["vae"] = vae
        new_pipe["conditioning_pos"] = conditioning_pos
        new_pipe["conditioning_neg"] = conditioning_neg
        new_pipe["latent_image"] = latent_image
        new_pipe["latent_original"] = latent_image
        new_pipe["steps_1st_pass"] = steps_1st_pass
        new_pipe["steps"] = steps_1st_pass
        new_pipe["cfg"] = cfg
        new_pipe["sampler"] = sampler
        new_pipe["scheduler"] = scheduler
        new_pipe["seed"] = seed
        new_pipe["boolean_controlnet_enable"] = bool(cn_applied)
        new_pipe["controlnet_strength"] = cn_strength
        new_pipe["controlnet_start_percent"] = cn_start_percent
        new_pipe["controlnet_end_percent"] = cn_end_percent
        new_pipe["controlnet_log"] = cn_log
        return (new_pipe,)


class CMKPipePeekKSampler:
    """KSampler peek node for the current CMK Prepare -> Execute path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("CMK_PIPE",)}}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "FLOAT", SAMPLERS, SCHEDULERS)
    RETURN_NAMES = ("model_patched", "conditioning_pos", "conditioning_neg", "latent_image", "seed", "steps", "cfg", "sampler", "scheduler")
    FUNCTION = "peek_ksampler"
    CATEGORY = 'CMK/Developer/Pipe/Peek'

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return False

    def peek_ksampler(self, pipe):
        if pipe is None:
            raise ValueError("CMK Pipe Peek KSampler: pipe is missing")

        model = pipe.get("model_patched") or pipe.get("model")
        conditioning_pos = pipe.get("conditioning_pos")
        conditioning_neg = pipe.get("conditioning_neg")
        latent = pipe.get("latent_1st_pass", pipe.get("latent_image"))

        missing = []
        if model is None:
            missing.append("model_patched/model")
        if conditioning_pos is None:
            missing.append("conditioning_pos")
        if conditioning_neg is None:
            missing.append("conditioning_neg")
        if latent is None:
            missing.append("latent_image")
        if missing:
            raise ValueError("CMK Pipe Peek KSampler: missing sampler context: " + ", ".join(missing))

        return (
            model,
            conditioning_pos,
            conditioning_neg,
            latent,
            pipe.get("seed"),
            pipe.get("steps_1st_pass", pipe.get("steps")),
            pipe.get("cfg"),
            pipe.get("sampler"),
            pipe.get("scheduler"),
        )


class CMKPipePeekKSamplerRefinerSource:
    """Minimal handoff from main pipe to refiner_pipe creation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("CMK_PIPE",)}}

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING", "STRING", "STRING", "LORA_STACK")
    RETURN_NAMES = ("latent_image", "seed", "steps_1st_pass", "prompt_pos", "prompt_neg", "active_loras", "lora_stack")
    FUNCTION = "peek_ksampler_refiner_source"
    CATEGORY = 'CMK/Developer/Pipe/Peek'

    def peek_ksampler_refiner_source(self, pipe):
        return (
            pipe.get("latent_1st_pass", pipe.get("latent_image")),
            pipe.get("seed"),
            pipe.get("steps_1st_pass", pipe.get("steps")),
            pipe.get("prompt_pos", ""),
            pipe.get("prompt_neg", ""),
            pipe.get("active_loras", ""),
            pipe.get("lora_stack"),
        )


class CMKPipeSetKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("CMK_PIPE",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("CMK_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "set_ksampler"
    CATEGORY = 'CMK/Developer/Pipe/Set'

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return False

    def set_ksampler(self, pipe, latent):
        new_pipe = dict(pipe)
        new_pipe["latent"] = latent
        new_pipe["latent_image"] = latent
        new_pipe["latent_1st_pass"] = latent
        return (new_pipe,)


class CMKKSamplerPipe:
    """Execute-node for the prepared first-pass sampler context.

    Contract: SAMPLER -> SAMPLED. It consumes only the isolated sampler
    working pipe created by CMK Sampler Prepare SDXL -Pipe- and returns a
    distinct completed sampler payload for the Refiner module. PROCESS and
    LOG are intentionally not routed through this compute node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"SAMPLER": ("CMK_SAMPLER_PIPE",)}}

    RETURN_TYPES = ("CMK_SAMPLED_PIPE",)
    RETURN_NAMES = ("SAMPLED",)
    FUNCTION = "run"
    CATEGORY = "CMK/Developer/Pipe/Execute"

    @staticmethod
    def _require(pipe, key, label=None):
        value = pipe.get(key)
        if value is None:
            raise ValueError(f"CMK KSampler -Pipe-: pipe['{label or key}'] is missing")
        return value

    @staticmethod
    def _normalize_conditioning(value, label):
        # Classic node output wrapper: tuple(conditioning,). CONDITIONING itself
        # is a list and must not be flattened.
        if isinstance(value, tuple):
            value = value[0] if value else None
        if callable(value):
            raise TypeError(f"CMK KSampler -Pipe-: pipe['{label}'] is callable, not CONDITIONING")
        if not isinstance(value, list):
            raise TypeError(f"CMK KSampler -Pipe-: pipe['{label}'] is not CONDITIONING (type={type(value).__name__})")
        return value

    def run(self, SAMPLER):
        pipe = SAMPLER
        if pipe is None:
            raise ValueError("CMK KSampler -Pipe-: pipe is missing")

        model = pipe.get("model_patched") or pipe.get("model")
        if model is None:
            raise ValueError("CMK KSampler -Pipe-: pipe['model_patched/model'] is missing")

        positive = self._normalize_conditioning(self._require(pipe, "conditioning_pos"), "conditioning_pos")
        negative = self._normalize_conditioning(self._require(pipe, "conditioning_neg"), "conditioning_neg")
        latent_image = pipe.get("latent_1st_pass") or pipe.get("latent_image")
        if latent_image is None:
            raise ValueError("CMK KSampler -Pipe-: pipe['latent_image'] is missing")

        seed = int(pipe.get("seed", 0))
        steps = int(pipe.get("steps_1st_pass", pipe.get("steps", 20)))
        cfg = float(pipe.get("cfg", 5.0))
        sampler_name = pipe.get("sampler", "euler_ancestral")
        scheduler = pipe.get("scheduler", "karras")
        denoise = float(pipe.get("denoise", 1.0))

        try:
            from nodes import KSampler
        except Exception as exc:
            raise RuntimeError(f"CMK KSampler -Pipe-: ComfyUI KSampler unavailable: {exc}") from exc

        result = KSampler().sample(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise,
        )
        samples = result[0] if isinstance(result, (tuple, list)) else result

        new_pipe = dict(pipe)
        new_pipe["samples"] = samples
        new_pipe["latent"] = samples
        new_pipe["latent_image"] = samples
        new_pipe["latent_1st_pass"] = samples
        new_pipe["ksampler_log"] = (
            "CMK KSampler -Pipe- | "
            f"seed={seed} | steps={steps} | cfg={cfg} | sampler={sampler_name} | "
            f"scheduler={scheduler} | denoise={denoise}"
        )
        return (new_pipe,)
