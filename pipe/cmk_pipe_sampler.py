from ..cmk_common import SAMPLERS, SCHEDULERS
from .cmk_final_preview import send_final_preview
from .cmk_visual import empty_visual, register_provider
from ..utils.cmk_timing import cmk_timed
from ..utils.cmk_sampling_warnings import ignore_torchsde_boundary_rounding


def _unload_completed_controlnet(pipe):
    """Unload ControlNet/model-patch weights after their sampler terminates."""
    zit_patch_keys = (
        "zit_controlnet_model_patch",
        "zit_inpaint_model_patch",
    )
    zit_controlnet_patch = pipe.get("zit_controlnet_model_patch")
    zit_inpaint_patch = pipe.get("zit_inpaint_model_patch")
    has_zit_patch = (
        zit_controlnet_patch is not None or zit_inpaint_patch is not None
    )
    if not bool(pipe.get("boolean_controlnet_enable", False)) and not has_zit_patch:
        return "NOT ACTIVE"
    if not bool(pipe.get("unload_models_after_use", True)):
        return "KEPT LOADED"

    import gc
    import comfy.model_management

    models = []
    conditioning_controls = []
    for key in (
        "conditioning_pos", "conditioning_neg",
        "conditioning_identity_pos", "conditioning_identity_neg",
        "positive", "negative",
    ):
        conditioning = pipe.get(key)
        if not isinstance(conditioning, (list, tuple)):
            continue
        for entry in conditioning:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            metadata = entry[1]
            if not isinstance(metadata, dict):
                continue
            attached = metadata.get("control")
            if attached is not None:
                conditioning_controls.append(attached)

    control_net = pipe.get("control_net")
    for attached_control in [control_net, *conditioning_controls]:
        if attached_control is not None:
            try:
                models.extend(attached_control.get_models())
            except Exception:
                pass
            try:
                attached_control.cleanup()
            except Exception:
                pass

    if control_net is not None:
        pipe.pop("control_net", None)

    # ControlNetApplyAdvanced stores copies in conditioning metadata. Module 10
    # is their terminal consumer; remove those references from the cached
    # sampler result before collecting the offloaded model.
    released_conditioning_refs = 0
    for key in (
        "conditioning_pos", "conditioning_neg",
        "conditioning_identity_pos", "conditioning_identity_neg",
        "positive", "negative",
    ):
        conditioning = pipe.get(key)
        if not isinstance(conditioning, (list, tuple)):
            continue
        for entry in conditioning:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            metadata = entry[1]
            if not isinstance(metadata, dict):
                continue
            if metadata.pop("control", None) is not None:
                released_conditioning_refs += 1
            metadata.pop("control_apply_to_uncond", None)

    zit_patches = [
        patch
        for patch in (zit_controlnet_patch, zit_inpaint_patch)
        if patch is not None
    ]
    models.extend(zit_patches)

    unique_models = []
    seen = set()
    for model in models:
        identity = getattr(model, "clone_base_uuid", None) or id(model)
        if identity in seen:
            continue
        seen.add(identity)
        unique_models.append(model)

    for model in unique_models:
        comfy.model_management.unload_model_and_clones(
            model, unload_additional_models=False
        )

    released_patch_refs = 0
    if zit_patches:
        # ZIT ControlNet is installed as transformer patches on two successive
        # clones (ControlNet apply -> AuraFlow sampling). Merely offloading the
        # additional model leaves both cached sampler objects holding its full
        # patch, encoded hint and temporary tensors in unified RAM. Module 10 is
        # terminal for these patches, so remove only the ControlNet callbacks;
        # the ZIT base model and unrelated model options stay intact.
        for patcher in (pipe.get("model"), pipe.get("model_patched")):
            options = getattr(patcher, "model_options", None)
            if not isinstance(options, dict):
                continue
            transformer = options.get("transformer_options")
            if not isinstance(transformer, dict):
                continue
            patches = transformer.get("patches")
            if not isinstance(patches, dict):
                continue
            for name in ("double_block", "noise_refiner"):
                entries = patches.pop(name, None)
                if isinstance(entries, (list, tuple)):
                    released_patch_refs += len(entries)
                elif entries is not None:
                    released_patch_refs += 1
            if not patches:
                transformer.pop("patches", None)
        # Mutate the cached prepare result as well as preventing propagation to
        # SAMPLED. This is intentional: the patch has completed its only job.
        for key in zit_patch_keys:
            pipe.pop(key, None)

    if unique_models:
        model_count = len(unique_models)
        # Drop the helper's own final references before collecting; otherwise
        # the 3 GB ZIT patch survives until after this function returns.
        models.clear()
        unique_models.clear()
        zit_patches.clear()
        zit_controlnet_patch = None
        zit_inpaint_patch = None
        control_net = None
        try:
            del model
        except UnboundLocalError:
            pass
        gc.collect()
        comfy.model_management.soft_empty_cache(force=True)
        details = []
        if released_patch_refs:
            details.append(f"{released_patch_refs} patch refs released")
        if released_conditioning_refs:
            details.append(f"{released_conditioning_refs} conditioning refs released")
        detail_text = "; ".join(details) if details else "references released"
        status = f"UNLOADED ({model_count} model; {detail_text})"
        print(f"[CMK ControlNet] {status}")
        return status
    return "NO SEPARATE MODEL"


def _cleanup_interrupted_sampling(pipe):
    """Best-effort cleanup that must not replace the sampler's primary error."""
    try:
        _unload_completed_controlnet(pipe)
    except Exception as exc:
        print(
            f"[CMK ControlNet] INTERRUPTED CLEANUP FAILED: {exc}",
            flush=True,
        )

    family = str(pipe.get("model_family", "")).strip().lower()
    if family != "z_image_turbo" or not bool(
        pipe.get("unload_models_after_use", True)
    ):
        return
    try:
        from .loaders.z_image_turbo_loader import invalidate_zit_resource_cache

        invalidate_zit_resource_cache("10 ZIT interrupted")
    except Exception as exc:
        print(
            f"[CMK Model Lifecycle] 10 ZIT interrupted: CLEANUP FAILED: {exc}",
            flush=True,
        )


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
        return {"required": {"SAMPLED": ("CMK_SAMPLED_PIPE",)}}

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING", "STRING", "STRING", "LORA_STACK")
    RETURN_NAMES = ("latent_image", "seed", "steps_1st_pass", "prompt_pos", "prompt_neg", "active_loras", "lora_stack")
    FUNCTION = "peek_ksampler_refiner_source"
    CATEGORY = 'CMK/Developer/Pipe/Peek'

    def peek_ksampler_refiner_source(self, SAMPLED):
        if not isinstance(SAMPLED, dict):
            raise TypeError("CMK Sampler Refiner Source: SAMPLED must be a CMK sampled pipe")
        return (
            SAMPLED.get("latent_1st_pass", SAMPLED.get("latent_image")),
            SAMPLED.get("seed"),
            SAMPLED.get("steps_1st_pass", SAMPLED.get("steps")),
            SAMPLED.get("prompt_pos", ""),
            SAMPLED.get("prompt_neg", ""),
            SAMPLED.get("active_loras", ""),
            SAMPLED.get("lora_stack"),
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
        return {
            "required": {"SAMPLER": ("CMK_SAMPLER_PIPE",)},
            "optional": {"VISUAL": ("CMK_VISUAL_PIPE",)},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CMK_SAMPLED_PIPE", "CMK_VISUAL_PIPE")
    RETURN_NAMES = ("SAMPLED", "VISUAL")
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

    def run(self, SAMPLER, VISUAL=None, unique_id=None):
        completed = False
        try:
            result = self._run(SAMPLER, VISUAL, unique_id)
            completed = True
            return result
        finally:
            if not completed and isinstance(SAMPLER, dict):
                _cleanup_interrupted_sampling(SAMPLER)

    def _run(self, SAMPLER, VISUAL=None, unique_id=None):
        pipe = SAMPLER
        if pipe is None:
            raise ValueError("CMK KSampler -Pipe-: pipe is missing")
        visual = empty_visual() if VISUAL is None else VISUAL

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
        family = str(pipe.get("model_family", "sdxl")).strip().lower()

        if pipe.get("instantid_reference_latent_mode", False):
            new_pipe = dict(pipe)
            new_pipe["samples"] = latent_image
            new_pipe["latent"] = latent_image
            new_pipe["latent_image"] = latent_image
            new_pipe["latent_1st_pass"] = latent_image
            new_pipe.pop("instantid_keypoints_latent", None)
            new_pipe["ksampler_log"] = (
                "CMK KSampler -Pipe- | BYPASSED | "
                "InstantID reference latent is encoded and sampled in module 15"
            )
            return (new_pipe, visual)

        if pipe.get("instantid_enabled", False) and int(pipe.get("instantid_end_at_step", 2)) == 0:
            new_pipe = dict(pipe)
            new_pipe["samples"] = latent_image
            new_pipe["latent"] = latent_image
            new_pipe["latent_image"] = latent_image
            new_pipe["latent_1st_pass"] = latent_image
            new_pipe["instantid_zero_pass"] = True
            new_pipe.pop("instantid_keypoints_latent", None)
            new_pipe["ksampler_log"] = (
                "CMK KSampler -Pipe- | BYPASSED | InstantID zero-pass test"
            )
            return (new_pipe, visual)

        try:
            from nodes import KSampler
        except Exception as exc:
            raise RuntimeError(f"CMK KSampler -Pipe-: ComfyUI KSampler unavailable: {exc}") from exc

        if pipe.get("hybrid_mode", False) and family != "z_image_turbo":
            import math
            import comfy.sample
            import comfy.utils
            import latent_preview

            handoff_percent = min(
                95, max(80, int(pipe.get("hybrid_sdxl_handoff", 90)))
            )
            handoff_step = min(
                steps - 1,
                max(1, int(math.ceil(steps * handoff_percent / 100.0))),
            )
            latent_tensor = latent_image["samples"]
            latent_tensor = comfy.sample.fix_empty_latent_channels(
                model,
                latent_tensor,
                latent_image.get("downscale_ratio_spacial"),
                latent_image.get("downscale_ratio_temporal"),
            )
            noise = comfy.sample.prepare_noise(
                latent_tensor, seed, latent_image.get("batch_index")
            )
            x0_output = {}
            preview_callback = latent_preview.prepare_callback(
                model, steps, x0_output
            )
            with cmk_timed(
                "10 SDXL HYBRID HANDOFF",
                f"steps 0-{handoff_step}/{steps} | {handoff_percent}%",
            ):
                with ignore_torchsde_boundary_rounding():
                    comfy.sample.sample(
                        model,
                        noise,
                        steps,
                        cfg,
                        sampler_name,
                        scheduler,
                        positive,
                        negative,
                        latent_tensor,
                        denoise=denoise,
                        start_step=0,
                        last_step=handoff_step,
                        force_full_denoise=False,
                        noise_mask=latent_image.get("noise_mask"),
                        callback=preview_callback,
                        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                        seed=seed,
                    )
            if "x0" not in x0_output:
                raise RuntimeError(
                    "CMK KSampler -Pipe-: HYBRID handoff produced no clean x0 prediction"
                )
            clean = latent_image.copy()
            clean.pop("downscale_ratio_spacial", None)
            clean.pop("downscale_ratio_temporal", None)
            clean["samples"] = model.model.process_latent_out(
                x0_output["x0"].cpu()
            )
            samples = clean
        elif pipe.get("instantid_enabled", False):
            import comfy.sample
            import comfy.utils
            import latent_preview

            end_at_step = int(pipe.get("instantid_end_at_step", 2))
            latent_tensor = latent_image["samples"]
            latent_tensor = comfy.sample.fix_empty_latent_channels(
                model,
                latent_tensor,
                latent_image.get("downscale_ratio_spacial"),
                latent_image.get("downscale_ratio_temporal"),
            )
            batch_inds = latent_image.get("batch_index")
            noise = comfy.sample.prepare_noise(latent_tensor, seed, batch_inds)
            x0_output = {}
            preview_callback = latent_preview.prepare_callback(model, steps, x0_output)
            with cmk_timed("10 KSAMPLER SAMPLE", f"steps 0-{end_at_step}/{steps} | leftover noise"):
                with ignore_torchsde_boundary_rounding():
                    sampled_tensor = comfy.sample.sample(
                        model,
                        noise,
                        steps,
                        cfg,
                        sampler_name,
                        scheduler,
                        positive,
                        negative,
                        latent_tensor,
                        denoise=denoise,
                        start_step=0,
                        last_step=end_at_step,
                        force_full_denoise=False,
                        noise_mask=latent_image.get("noise_mask"),
                        callback=preview_callback,
                        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                        seed=seed,
                    )
            samples = latent_image.copy()
            samples.pop("downscale_ratio_spacial", None)
            samples.pop("downscale_ratio_temporal", None)
            samples["samples"] = sampled_tensor
            if "x0" in x0_output:
                keypoints_latent = latent_image.copy()
                keypoints_latent.pop("downscale_ratio_spacial", None)
                keypoints_latent.pop("downscale_ratio_temporal", None)
                keypoints_latent["samples"] = model.model.process_latent_out(
                    x0_output["x0"].cpu()
                )
            else:
                keypoints_latent = None

            # Fooocus/Inpaint latents carry a noise mask and an inpaint model
            # contract that cannot safely cross into the ordinary InstantID
            # continuation. Turn the first-pass clean x0 estimate into pixels
            # and encode those pixels again. VAEEncode returns a fresh latent
            # without noise_mask or other Inpaint-specific latent metadata.
            if pipe.get("boolean_inpaint_mode", False):
                if keypoints_latent is None:
                    raise RuntimeError(
                        "CMK KSampler -Pipe-: InstantID Inpaint bridge requires the first-pass x0 estimate"
                    )
                vae = pipe.get("vae")
                if vae is None:
                    raise ValueError(
                        "CMK KSampler -Pipe-: InstantID Inpaint bridge requires pipe['vae']"
                    )
                try:
                    from nodes import VAEDecode, VAEEncode

                    with cmk_timed("10 INSTANTID INPAINT BRIDGE", "x0 decode -> clean VAE encode"):
                        decoded = VAEDecode().decode(vae, keypoints_latent)
                        bridge_image = decoded[0] if isinstance(decoded, (tuple, list)) else decoded
                        encoded = VAEEncode().encode(vae, bridge_image)
                        encoded_latent = encoded[0] if isinstance(encoded, (tuple, list)) else encoded
                    if not isinstance(encoded_latent, dict) or "samples" not in encoded_latent:
                        raise TypeError("VAEEncode returned no LATENT samples")
                    samples = {"samples": encoded_latent["samples"]}
                    keypoints_latent = samples
                    instantid_inpaint_bridge = True
                except Exception as exc:
                    raise RuntimeError(
                        f"CMK KSampler -Pipe-: InstantID Inpaint bridge failed: {exc}"
                    ) from exc
            else:
                instantid_inpaint_bridge = False
        elif pipe.get("suppress_sampler_preview", False):
            # Mirrors ComfyUI's common_ksampler but deliberately omits the
            # latent-preview callback. The final decoded PreviewImage remains.
            import comfy.sample
            import comfy.utils

            latent_tensor = latent_image["samples"]
            latent_tensor = comfy.sample.fix_empty_latent_channels(
                model,
                latent_tensor,
                latent_image.get("downscale_ratio_spacial"),
                latent_image.get("downscale_ratio_temporal"),
            )
            batch_inds = latent_image.get("batch_index")
            noise = comfy.sample.prepare_noise(latent_tensor, seed, batch_inds)
            with cmk_timed("10 KSAMPLER SAMPLE", f"{steps} steps"):
                sampled_tensor = comfy.sample.sample(
                    model,
                    noise,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    latent_tensor,
                    denoise=denoise,
                    noise_mask=latent_image.get("noise_mask"),
                    callback=None,
                    disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                    seed=seed,
                )
            samples = latent_image.copy()
            samples.pop("downscale_ratio_spacial", None)
            samples.pop("downscale_ratio_temporal", None)
            samples["samples"] = sampled_tensor
        else:
            with cmk_timed("10 KSAMPLER SAMPLE", f"{steps} steps"):
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

        controlnet_unload_status = _unload_completed_controlnet(pipe)

        new_pipe = dict(pipe)
        new_pipe["samples"] = samples
        new_pipe["latent"] = samples
        new_pipe["latent_image"] = samples
        new_pipe["latent_1st_pass"] = samples
        new_pipe["controlnet_model_status"] = controlnet_unload_status
        if pipe.get("hybrid_mode", False) and family != "z_image_turbo":
            new_pipe["hybrid_x0_latent"] = samples
            new_pipe["hybrid_handoff_complete"] = True
        if pipe.get("instantid_enabled", False) and keypoints_latent is not None:
            new_pipe["instantid_keypoints_latent"] = keypoints_latent
            new_pipe["instantid_inpaint_bridge"] = bool(instantid_inpaint_bridge)
        new_pipe["ksampler_log"] = (
            "CMK KSampler -Pipe- | "
            f"seed={seed} | steps={steps} | cfg={cfg} | sampler={sampler_name} | "
            f"scheduler={scheduler} | denoise={denoise} | "
            f"controlnet_model={controlnet_unload_status} | "
            f"instantid_inpaint_bridge={bool(pipe.get('instantid_enabled', False) and instantid_inpaint_bridge)}"
        )
        vae = pipe.get("vae")
        if vae is None:
            raise ValueError("CMK KSampler -Pipe-: pipe['vae'] is missing")
        visual_latent = new_pipe.get("instantid_keypoints_latent")
        if visual_latent is None:
            visual_latent = samples
        try:
            from nodes import VAEDecode
        except Exception as exc:
            raise RuntimeError(f"CMK KSampler -Pipe-: VAE Decode unavailable: {exc}") from exc
        with cmk_timed("10 KSAMPLER VISUAL VAE DECODE"):
            decoded = VAEDecode().decode(vae, visual_latent)
        image = decoded[0] if isinstance(decoded, (tuple, list)) else decoded
        new_pipe["image_1st_pass"] = image

        family = str(pipe.get("model_family", "sdxl")).strip().lower()
        hybrid_mode = bool(pipe.get("hybrid_mode", False))
        if family == "z_image_turbo":
            new_pipe["image"] = image
            if hybrid_mode:
                before = pipe.get("hybrid_source_image")
                after = image
                if before is not None and bool(pipe.get("hybrid_masked_finish", False)):
                    finish_mask = pipe.get("hybrid_finish_mask")
                    if finish_mask is not None:
                        from .cmk_z_image_turbo import _hybrid_masked_composite

                        after = _hybrid_masked_composite(before, after, finish_mask)
                label = "2nd-Pass ZIT"
                branch = "hybrid"
                stage_key = "hybrid.second_pass"
                channels = (
                    {"before": before, "after": after}
                    if before is not None
                    else {"result": after}
                )
            else:
                label = "Sampling ZIT"
                branch = "z_image_turbo"
                stage_key = "z_image_turbo.sampling"
                channels = {"result": image}
        else:
            label = "1st-Pass SDXL"
            branch = "sdxl"
            stage_key = "sdxl.first_pass"
            channels = {"result": image}
        send_final_preview(image)
        visual = register_provider(
            visual,
            module_instance_id=unique_id or "ksampler",
            module_type="CMKKSamplerPipe",
            module_label=label,
            sequence=10,
            channels=channels,
            status="completed",
            live_node_id=unique_id,
            branch=branch,
            stage_key=stage_key,
        )
        if (
            bool(pipe.get("unload_models_after_use", True))
            and family == "sdxl"
            and not hybrid_mode
            and not bool(pipe.get("instantid_enabled", False))
        ):
            from .loaders.checkpoint_vae_loader import unload_all_phase_models

            new_pipe["generation_model_status"] = unload_all_phase_models(
                "10 SDXL complete"
            )
        return (new_pipe, visual)
