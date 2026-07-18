from ..cmk_common import SAMPLERS, SCHEDULERS


class CMKPipeSetRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("CMK_PIPE",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CMK_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "set_refiner"
    CATEGORY = 'CMK/Developer/Pipe/Set'

    def set_refiner(self, pipe, image):
        new_pipe = dict(pipe)
        new_pipe["image_refiner"] = image
        return (new_pipe,)


class CMKPipePeekPreprocessRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("CMK_PIPE",)}}

    RETURN_TYPES = ("IMAGE", "LATENT", "VAE", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("image", "latent_image", "vae", "boolean_detailer_enable", "boolean_face_enable")
    FUNCTION = "peek_preprocess_refiner"
    CATEGORY = 'CMK/Developer/Pipe/Peek'

    def peek_preprocess_refiner(self, pipe):
        return (
            pipe.get("image"),
            pipe.get("latent_1st_pass", pipe.get("latent_image")),
            pipe.get("vae"),
            pipe.get("boolean_detailer_enable", False),
            pipe.get("boolean_face_enable", False),
        )


class CMKPipeCreateRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "conditioning_pos": ("CONDITIONING",),
                "conditioning_neg": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}),
                "steps_1st_pass": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler": (SAMPLERS,),
                "scheduler": (SCHEDULERS,),
                "start_percent": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            },
            "optional": {
                "prompt_pos": ("STRING", {"forceInput": True}),
                "prompt_neg": ("STRING", {"forceInput": True}),
                "active_loras": ("STRING", {"forceInput": True}),
                "lora_stack": ("LORA_STACK", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CMK_REFINER_PIPE",)
    RETURN_NAMES = ("refiner_pipe",)
    FUNCTION = "create_refiner"
    CATEGORY = 'CMK/Developer/Pipe/Create'

    def create_refiner(self, model=None, vae=None, conditioning_pos=None, conditioning_neg=None, latent_image=None, seed=None, steps_1st_pass=None, steps=None, cfg=None, sampler=None, scheduler=None, start_percent=None, prompt_pos=None, prompt_neg=None, active_loras=None, lora_stack=None, **kwargs):
        model = model if model is not None else kwargs.get("refiner_model")
        vae = vae if vae is not None else kwargs.get("refiner_vae")
        conditioning_pos = conditioning_pos if conditioning_pos is not None else kwargs.get("refiner_conditioning_pos")
        conditioning_neg = conditioning_neg if conditioning_neg is not None else kwargs.get("refiner_conditioning_neg")
        latent_image = latent_image if latent_image is not None else kwargs.get("refiner_latent_image")
        seed = seed if seed is not None else kwargs.get("refiner_seed")
        steps_1st_pass = steps_1st_pass if steps_1st_pass is not None else kwargs.get("refiner_steps_1st_pass", kwargs.get("steps_1st_pass", 20))
        steps = steps if steps is not None else kwargs.get("refiner_steps", 20)
        cfg = cfg if cfg is not None else kwargs.get("refiner_cfg", 7.0)
        sampler = sampler if sampler is not None else kwargs.get("refiner_sampler")
        scheduler = scheduler if scheduler is not None else kwargs.get("refiner_scheduler")
        start_percent = start_percent if start_percent is not None else kwargs.get("refiner_start_percent", 80.0)
        prompt_pos = prompt_pos if prompt_pos is not None else kwargs.get("refiner_prompt_pos", kwargs.get("prompt_pos", ""))
        prompt_neg = prompt_neg if prompt_neg is not None else kwargs.get("refiner_prompt_neg", kwargs.get("prompt_neg", ""))
        active_loras = active_loras if active_loras is not None else kwargs.get("refiner_active_loras", kwargs.get("active_loras", ""))
        lora_stack = lora_stack if lora_stack is not None else kwargs.get("refiner_lora_stack", kwargs.get("lora_stack"))

        return ({
            "refiner_model": model,
            "refiner_vae": vae,
            "refiner_conditioning_pos": conditioning_pos,
            "refiner_conditioning_neg": conditioning_neg,
            "refiner_latent_image": latent_image,
            "refiner_seed": seed,
            "refiner_steps_1st_pass": steps_1st_pass,
            "refiner_steps": steps,
            "refiner_cfg": cfg,
            "refiner_sampler": sampler,
            "refiner_scheduler": scheduler,
            "refiner_start_percent": start_percent,
            "refiner_prompt_pos": prompt_pos,
            "refiner_prompt_neg": prompt_neg,
            "refiner_active_loras": active_loras,
            "refiner_lora_stack": lora_stack,
        },)


class CMKPipePeekRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refiner_pipe": ("CMK_REFINER_PIPE",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "INT", "INT", "FLOAT", SAMPLERS, SCHEDULERS, "INT", "INT")
    RETURN_NAMES = ("model", "conditioning_pos", "conditioning_neg", "latent_image", "vae", "seed", "steps", "cfg", "sampler", "scheduler", "start_step", "end_step")
    FUNCTION = "peek_refiner"
    CATEGORY = 'CMK/Developer/Pipe/Peek'

    def peek_refiner(self, refiner_pipe):
        refiner_steps = refiner_pipe.get("refiner_steps", 20)
        refiner_start_percent = refiner_pipe.get("refiner_start_percent", 80.0)
        start_step = int(refiner_steps * refiner_start_percent / 100.0)
        end_step = refiner_steps

        return (
            refiner_pipe.get("refiner_model"),
            refiner_pipe.get("refiner_conditioning_pos"),
            refiner_pipe.get("refiner_conditioning_neg"),
            refiner_pipe.get("refiner_latent_image"),
            refiner_pipe.get("refiner_vae"),
            refiner_pipe.get("refiner_seed"),
            refiner_steps,
            refiner_pipe.get("refiner_cfg"),
            refiner_pipe.get("refiner_sampler"),
            refiner_pipe.get("refiner_scheduler"),
            start_step,
            end_step,
        )
