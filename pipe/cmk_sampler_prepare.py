from __future__ import annotations

import os
import inspect
from typing import Any

from ..cmk_common import SAMPLERS, SCHEDULERS
from ..loader.cmk_lora_text_loader import CMKLoRATextLoader, _resolve_lora_path
from ..engine.fooocus_inpaint import CMKFooocusInpaintPipeline
from ..engine.context_reference import CMKContextReferenceLatentMask
from .cmk_log_pipe import cmk_add_block, cmk_format_loras
from .cmk_pipe_sampler import CMKPipeSetSampler
from ..utils.cmk_diagnostic import make_diagnostic_payload


SAMPLING_MODES = ["eps", "v_prediction", "lcm"]
CMK_FIXED_SEED_WIDGET = {"default": 1565304366, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}


def _get_node_class(*names: str):
    """Resolve a ComfyUI node class by mapping key or attribute name at runtime."""
    try:
        import nodes  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"ComfyUI nodes module unavailable: {exc}") from exc

    mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    for name in names:
        cls = mappings.get(name)
        if cls is not None:
            return cls
    for name in names:
        cls = getattr(nodes, name, None)
        if cls is not None:
            return cls
    raise RuntimeError(f"Required ComfyUI node not found: {' / '.join(names)}")


def _call_node(names: tuple[str, ...], *args):
    cls = _get_node_class(*names)
    fn_name = getattr(cls, "FUNCTION", None)
    if not fn_name:
        raise RuntimeError(f"Resolved node has no FUNCTION: {' / '.join(names)}")
    instance = cls()
    fn = getattr(instance, fn_name)
    return fn(*args)


def _call_node_kwargs(names: tuple[str, ...], **kwargs):
    """Call a ComfyUI node by keyword while tolerating minor signature variants."""
    cls = _get_node_class(*names)
    fn_name = getattr(cls, "FUNCTION", None)
    if not fn_name:
        raise RuntimeError(f"Resolved node has no FUNCTION: {' / '.join(names)}")
    instance = cls()
    fn = getattr(instance, fn_name)
    try:
        sig = inspect.signature(fn)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**accepted)
    except (TypeError, ValueError):
        # Some custom nodes hide or alter signatures. Fall back to positional
        # ordering based on the node INPUT_TYPES declaration.
        input_types = cls.INPUT_TYPES() if hasattr(cls, "INPUT_TYPES") else {}
        order = []
        for group in ("required", "optional"):
            group_def = input_types.get(group, {}) or {}
            order.extend(list(group_def.keys()))
        args = [kwargs[k] for k in order if k in kwargs]
        return fn(*args)


def _unwrap_node_output(value):
    """Return the real payload from a ComfyUI node call without destroying CONDITIONING.

    Important: CONDITIONING is itself a Python list. Older helper code treated
    every list as a multi-output wrapper and accidentally reduced a valid
    CONDITIONING to its first internal element. In newer ComfyUI builds this can
    even leak callable objects such as dict.values into the pipe.
    """
    current = value
    for _ in range(6):
        # Classic Comfy node call: outputs are returned as tuple(output_0, ...).
        # Tuple is a wrapper. List is NOT a safe wrapper because CONDITIONING is a list.
        if isinstance(current, tuple):
            current = current[0] if current else None
            continue

        # Some wrappers expose dict payloads. Only unwrap known output keys.
        if isinstance(current, dict):
            for key in ("result", "results", "output", "outputs", "value", "data"):
                if key in current:
                    current = current[key]
                    break
            else:
                return current
            continue

        # Generic NodeOutput-like wrappers. Never accept callable attributes
        # such as dict.values; those are methods, not payloads.
        unwrapped = False
        for attr in ("result", "results", "output", "outputs", "value", "data"):
            if hasattr(current, attr):
                try:
                    candidate = getattr(current, attr)
                except Exception:
                    continue
                if callable(candidate):
                    continue
                current = candidate
                unwrapped = True
                break
        if unwrapped:
            continue

        if current.__class__.__name__ == "NodeOutput":
            try:
                items = list(current)
                current = items[0] if items else None
                continue
            except Exception:
                pass

        return current
    return current


def _maybe_invert_controlnet_hint(image, enabled):
    if image is None or not bool(enabled):
        return image
    try:
        import torch
        if isinstance(image, torch.Tensor):
            return torch.clamp(1.0 - image, 0.0, 1.0)
    except Exception:
        pass
    try:
        return 1.0 - image
    except Exception:
        return image


def _apply_controlnet_from_pipe(pipe, conditioning_pos, conditioning_neg, vae):
    """Apply a prepared ControlNet from the CMK pipe to both conditionings.

    CMK ControlNet Prepare -Pipe- only prepares/stores the model and hint image.
    The sampler prepare stage is responsible for attaching that ControlNet to
    the final conditioning after all SDXL/Inpaint/Context-Reference work.
    """
    enabled = bool(pipe.get("boolean_controlnet_enable", False))
    control_net = pipe.get("control_net")
    controlnet_image = _maybe_invert_controlnet_hint(
        pipe.get("controlnet_image"),
        pipe.get("controlnet_invert_hint", False),
    )

    strength = _float(pipe.get("controlnet_strength"), 0.60)
    start_percent = _float(pipe.get("controlnet_start_percent"), 0.0)
    end_percent = _float(pipe.get("controlnet_end_percent"), 1.0)

    if not enabled:
        return conditioning_pos, conditioning_neg, False, "ControlNet disabled"
    if control_net is None:
        return conditioning_pos, conditioning_neg, False, "ControlNet bypass | control_net missing"
    if controlnet_image is None:
        return conditioning_pos, conditioning_neg, False, "ControlNet bypass | controlnet_image missing"
    if strength <= 0.0:
        return conditioning_pos, conditioning_neg, False, "ControlNet bypass | strength <= 0"
    if end_percent <= start_percent:
        return conditioning_pos, conditioning_neg, False, "ControlNet bypass | end <= start"

    cls = _get_node_class("ControlNetApplyAdvanced")
    instance = cls()
    fn_name = getattr(cls, "FUNCTION", "apply_controlnet")
    fn = getattr(instance, fn_name)

    try:
        result = fn(
            conditioning_pos, conditioning_neg, control_net, controlnet_image,
            strength, start_percent, end_percent, vae,
        )
    except TypeError:
        result = fn(
            conditioning_pos, conditioning_neg, control_net, controlnet_image,
            strength, start_percent, end_percent,
        )

    if isinstance(result, dict) and "result" in result:
        result = result["result"]
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise TypeError("CMK Sampler Prepare SDXL -Pipe-: ControlNetApplyAdvanced returned invalid output")

    pos = _validate_conditioning(result[0], "conditioning_pos(controlnet)")
    neg = _validate_conditioning(result[1], "conditioning_neg(controlnet)")
    log = (
        "ControlNet applied | "
        f"strength={strength:.3f} | start={start_percent:.3f} | end={end_percent:.3f}"
    )
    return pos, neg, True, log


def _is_conditioning(value: Any) -> bool:
    """Best-effort validation for ComfyUI CONDITIONING payloads."""
    if not isinstance(value, list):
        return False
    if len(value) == 0:
        return True
    first = value[0]
    return isinstance(first, (list, tuple)) and len(first) >= 1


def _validate_conditioning(value: Any, label: str):
    value = _unwrap_node_output(value)
    if callable(value):
        raise TypeError(f"CMK Sampler Prepare SDXL -Pipe-: {label} is callable, not CONDITIONING")
    if not _is_conditioning(value):
        raise TypeError(
            f"CMK Sampler Prepare SDXL -Pipe-: {label} is not valid CONDITIONING "
            f"(type={type(value).__name__})"
        )
    return value

def _first(result):
    return _unwrap_node_output(result)


def _safe_lora_list():
    try:
        import folder_paths  # type: ignore
        names = folder_paths.get_filename_list("loras")
        return names if names else ["None"]
    except Exception:
        return ["None"]


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


class CMKSamplerPrepareSDXLPipe:
    """Prepare the first-pass SDXL sampler context from a CMK pipe.

    Public contract: MODEL + PROCESS + IMAGE + LOG -> SAMPLER + LOG + diagnostic.
    MODEL is read-only. PROCESS is copied into an isolated sampler working pipe.
    IMAGE is consumed as input only and must be forwarded separately at subgraph level.
    """

    @classmethod
    def INPUT_TYPES(cls):
        available_loras = _safe_lora_list()
        loras = ["None"] + [
            item for item in available_loras
            if str(item).strip().lower() != "none"
        ]
        default_lora = "refiner/Hyper-SDXL-1step-lora.safetensors"
        if default_lora in loras:
            # Preserve the established 1st-pass default while allowing an
            # explicit neutral selection.
            default_lora_value = default_lora
        else:
            default_lora_value = "None"

        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE",),
                "PROCESS": ("CMK_PIPE",),
                "IMAGE": ("IMAGE",),

                # CMK UI order. Advanced widgets remain at their functional
                # position and are collapsed by the frontend extension.
                "stop_at_clip_layer": ("INT", {"default": -2, "min": -24, "max": 0, "step": 1, "advanced": True}),
                "fooocus_patch": ("STRING", {"default": "inpaint_v25.fooocus.patch", "advanced": True}),
                "fooocus_head": ("STRING", {"default": "fooocus_inpaint_head.pth", "advanced": True}),

                "lora_name": (loras, {"default": default_lora_value}),
                "strength_model": ("FLOAT", {"default": 1.00, "min": -20.0, "max": 20.0, "step": 0.01, "advanced": True}),
                "strength_clip": ("FLOAT", {"default": 1.00, "min": -20.0, "max": 20.0, "step": 0.01, "advanced": True}),

                "freeu_enabled": ("BOOLEAN", {"default": True}),
                "freeu_b1": ("FLOAT", {"default": 1.30, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True}),
                "freeu_b2": ("FLOAT", {"default": 1.40, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True}),
                "freeu_s1": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True}),
                "freeu_s2": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 10.0, "step": 0.01, "advanced": True}),

                "steps_1st_pass": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler": (SAMPLERS, {"default": "euler_ancestral"} if "euler_ancestral" in SAMPLERS else {}),
                "sampling": (SAMPLING_MODES, {"default": "lcm", "advanced": True}),
                "zsnr": ("BOOLEAN", {"default": False, "advanced": True}),
                "pag_scale": ("FLOAT", {"default": 2.50, "min": 0.0, "max": 20.0, "step": 0.05, "advanced": True}),
                "scheduler": (SCHEDULERS, {"default": "karras"} if "karras" in SCHEDULERS else {}),
                "seed": ("INT", dict(CMK_FIXED_SEED_WIDGET)),

                "inpaint_noise_mask": ("BOOLEAN", {"default": False}),
                "context_reference_enabled": ("BOOLEAN", {"default": True}),
                "context_reference_expand": ("INT", {"default": 3, "min": -64, "max": 64, "step": 1, "advanced": True}),
                "context_reference_blur": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 64.0, "step": 0.1, "advanced": True}),
            },
            "optional": {
                "LOG": ("CMK_LOG_PIPE",),
            },
        }

    RETURN_TYPES = ("CMK_SAMPLER_PIPE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("SAMPLER", "LOG", "diagnostic")
    FUNCTION = "prepare"
    CATEGORY = "CMK/Developer/Pipe/Prepare"


    def _clip_set_last_layer(self, clip, stop_at_clip_layer: int):
        return _first(_call_node(("CLIPSetLastLayer",), clip, stop_at_clip_layer))

    def _apply_single_lora(self, model, clip, lora_name: str, strength_model: float, strength_clip: float):
        """Apply the local LoRA through CMK's path resolver.

        This deliberately accepts nested relative paths, stems without extension,
        LoraManager display names, and absolute paths exactly like
        ``CMK LoRA Text Loader``.
        """
        name = str(lora_name or "").strip()
        if not name or name.lower() == "none":
            return model, clip, ""
        if float(strength_model) == 0.0 and float(strength_clip) == 0.0:
            return model, clip, ""

        import comfy.sd  # type: ignore
        import comfy.utils  # type: ignore

        path = _resolve_lora_path(name)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        model, clip = comfy.sd.load_lora_for_models(
            model,
            clip,
            lora,
            float(strength_model),
            float(strength_clip),
        )
        normalized_name = name.replace("\\", "/")
        return model, clip, f"<lora:{normalized_name}:{float(strength_model):g}:{float(strength_clip):g}>"

    def _apply_pag(self, model, pag_scale: float):
        if float(pag_scale) <= 0.0:
            return model
        return _first(_call_node(("PerturbedAttentionGuidance",), model, float(pag_scale)))

    def _apply_sampling(self, model, sampling: str, zsnr: bool):
        return _first(_call_node(("ModelSamplingDiscrete",), model, sampling, bool(zsnr)))

    def _apply_freeu(self, model, enabled: bool, b1: float, b2: float, s1: float, s2: float):
        if not bool(enabled):
            return model
        return _first(_call_node(("FreeU_V2",), model, float(b1), float(b2), float(s1), float(s2)))

    def _encode_sdxl_plus(self, clip, width: int, height: int, size_cond_factor: int, text: str):
        # Preferred path: ComfyUI Essentials node used by the original subgraph.
        try:
            return _first(_call_node(("CLIPTextEncodeSDXL+",), clip, int(width), int(height), int(size_cond_factor), text))
        except Exception as exc_plus:
            # Fallback for installations without the + node. This keeps the node
            # usable on plain ComfyUI, but CMK SDXL defaults are based on the + node.
            try:
                return _first(
                    _call_node(
                        ("CLIPTextEncodeSDXL",),
                        clip,
                        int(width),
                        int(height),
                        0,
                        0,
                        int(width),
                        int(height),
                        text,
                        text,
                    )
                )
            except Exception as exc_builtin:
                raise RuntimeError(
                    "SDXL conditioning encode failed. Required node 'CLIPTextEncodeSDXL+' "
                    f"is unavailable or incompatible. Plus error: {exc_plus}. Builtin fallback error: {exc_builtin}"
                ) from exc_builtin


    def _vae_encode(self, vae, image):
        if vae is None or image is None:
            return None, "latent skipped | vae/image missing"
        try:
            result = _call_node(("VAEEncode",), vae, image)
            return _first(result), "latent encoded | VAEEncode"
        except Exception as exc:
            return None, f"latent skipped | VAEEncode failed: {exc}"

    def _vae_encode_for_inpaint(self, vae, image, mask, grow_mask_by: int):
        if vae is None or image is None or mask is None:
            return None, "inpaint latent skipped | vae/image/mask missing"
        try:
            result = _call_node(("VAEEncodeForInpaint",), vae, image, mask, int(grow_mask_by))
            return _first(result), f"inpaint latent encoded | grow_mask_by={int(grow_mask_by)}"
        except Exception as exc:
            return None, f"inpaint latent skipped | VAEEncodeForInpaint failed: {exc}"

    def _empty_latent(self, width: int, height: int, batch_size: int = 1):
        """Create the native ComfyUI EmptyLatentImage used by NORMAL generation."""
        result = _call_node(("EmptyLatentImage",), int(width), int(height), max(1, int(batch_size)))
        latent = _first(result)
        if not isinstance(latent, dict) or "samples" not in latent:
            raise TypeError(
                "CMK Sampler Prepare SDXL -Pipe-: EmptyLatentImage returned invalid LATENT "
                f"(type={type(latent).__name__})"
            )
        return latent

    def prepare(
        self,
        MODEL,
        PROCESS,
        IMAGE,
        stop_at_clip_layer,
        fooocus_patch,
        fooocus_head,
        lora_name,
        strength_model,
        strength_clip,
        freeu_enabled,
        freeu_b1,
        freeu_b2,
        freeu_s1,
        freeu_s2,
        steps_1st_pass,
        cfg,
        sampler,
        sampling,
        zsnr,
        pag_scale,
        scheduler,
        seed,
        inpaint_noise_mask,
        context_reference_enabled,
        context_reference_expand,
        context_reference_blur,
        LOG=None,
    ):
        if PROCESS is None:
            raise ValueError("CMK Sampler Prepare SDXL -Pipe-: PROCESS is missing")
        if not isinstance(MODEL, dict):
            raise TypeError("CMK Sampler Prepare SDXL -Pipe-: MODEL must be a CMK_MODEL_PIPE dictionary")

        # Internal CMK defaults deliberately removed from the public UI.
        size_cond_factor = 4
        grow_mask_by = 6
        context_reference_mask_only = True

        model = MODEL.get("model")
        clip = MODEL.get("clip")
        vae = MODEL.get("vae")
        if model is None:
            raise ValueError("CMK Sampler Prepare SDXL -Pipe-: MODEL['model'] is missing")
        if clip is None:
            raise ValueError("CMK Sampler Prepare SDXL -Pipe-: MODEL['clip'] is missing")
        if vae is None:
            raise ValueError("CMK Sampler Prepare SDXL -Pipe-: MODEL['vae'] is missing")

        # MODEL is read-only. PROCESS receives references to the clean shared
        # resources; all sampler-specific patches remain local to PROCESS.
        pipe = dict(PROCESS)
        # IMAGE is transported exclusively through the dedicated IMAGE cable.
        # Remove legacy image payloads so they cannot propagate through PROCESS.
        pipe.pop("image", None)
        pipe.pop("image_original", None)
        pipe["model"] = model
        pipe["model_base"] = model
        pipe["clip"] = clip
        pipe["clip_base"] = clip
        pipe["vae"] = vae
        pipe["model_pipe_metadata"] = {
            key: MODEL.get(key)
            for key in ("ckpt_name", "vae_name", "checkpoint_vae", "vae_source", "model_pipe_source")
            if key in MODEL
        }

        width = _int(pipe.get("width", pipe.get("target_width")), 1024)
        height = _int(pipe.get("height", pipe.get("target_height")), 1024)
        prompt_pos = _clean_text(pipe.get("prompt_pos"), "")
        prompt_neg = _clean_text(pipe.get("prompt_neg"), "")
        lora_syntax = _clean_text(pipe.get("active_loras"), "")
        lora_stack = pipe.get("lora_stack")
        inpaint_mode = bool(pipe.get("boolean_inpaint_mode", False))

        # Build the shared MODEL/CLIP base before selecting the complete
        # NORMAL or INPAINT sampling state.
        prepared_clip = self._clip_set_last_layer(clip, _int(stop_at_clip_layer, -2))
        prepared_model = model
        loaded_stack_loras = ""
        loaded_single_lora = ""

        if inpaint_mode:
            # 2) Global/module LoRA stack from pipe
            prepared_model, prepared_clip, _, loaded_stack_loras = CMKLoRATextLoader().load_loras(
                prepared_model,
                prepared_clip,
                lora_syntax,
                lora_stack,
            )
            prepared_model = _unwrap_node_output(prepared_model)
            prepared_clip = _unwrap_node_output(prepared_clip)

            # 3) Local explicit LoRA from original subgraph defaults
            prepared_model, prepared_clip, loaded_single_lora = self._apply_single_lora(
                prepared_model,
                prepared_clip,
                lora_name,
                _float(strength_model, 1.0),
                _float(strength_clip, 1.0),
            )

            # 4) Model modifiers
            prepared_model = _unwrap_node_output(prepared_model)
            prepared_clip = _unwrap_node_output(prepared_clip)
            if prepared_model is None or not hasattr(prepared_model, "clone"):
                raise TypeError(
                    "CMK Sampler Prepare SDXL -Pipe-: prepared model is not a valid ComfyUI MODEL "
                    f"before model modifiers (type={type(prepared_model).__name__})"
                )
            prepared_model = _unwrap_node_output(self._apply_pag(prepared_model, _float(pag_scale, 2.5)))
            prepared_model = _unwrap_node_output(self._apply_sampling(prepared_model, str(sampling), bool(zsnr)))
            prepared_model = _unwrap_node_output(self._apply_freeu(
                prepared_model,
                bool(freeu_enabled),
                _float(freeu_b1, 1.3),
                _float(freeu_b2, 1.4),
                _float(freeu_s1, 0.9),
                _float(freeu_s2, 0.2),
            ))
        else:
            # NORMAL generation — restored production chain:
            # SOURCE LoRAs -> local LoRA -> PAG -> ModelSamplingDiscrete -> FreeU.
            # CMKLoRATextLoader centrally merges and deduplicates stack/syntax.
            prepared_model, prepared_clip, _, loaded_stack_loras = CMKLoRATextLoader().load_loras(
                prepared_model,
                prepared_clip,
                lora_syntax,
                lora_stack,
            )
            prepared_model = _unwrap_node_output(prepared_model)
            prepared_clip = _unwrap_node_output(prepared_clip)

            prepared_model, prepared_clip, loaded_single_lora = self._apply_single_lora(
                prepared_model,
                prepared_clip,
                lora_name,
                _float(strength_model, 1.0),
                _float(strength_clip, 1.0),
            )
            prepared_model = _unwrap_node_output(prepared_model)
            prepared_clip = _unwrap_node_output(prepared_clip)

            prepared_model = _unwrap_node_output(
                self._apply_pag(
                    prepared_model,
                    _float(pag_scale, 2.5),
                )
            )
            prepared_model = _unwrap_node_output(
                self._apply_sampling(
                    prepared_model,
                    str(sampling),
                    bool(zsnr),
                )
            )
            prepared_model = _unwrap_node_output(
                self._apply_freeu(
                    prepared_model,
                    bool(freeu_enabled),
                    _float(freeu_b1, 1.30),
                    _float(freeu_b2, 1.40),
                    _float(freeu_s1, 0.90),
                    _float(freeu_s2, 0.20),
                )
            )

        # 5) SDXL conditioning
        conditioning_pos = _validate_conditioning(
            self._encode_sdxl_plus(prepared_clip, width, height, _int(size_cond_factor, 4), prompt_pos),
            "conditioning_pos",
        )
        conditioning_neg = _validate_conditioning(
            self._encode_sdxl_plus(prepared_clip, width, height, _int(size_cond_factor, 4), prompt_neg),
            "conditioning_neg",
        )

        # 6) Reconstruct the original two complete branches. NORMAL keeps the
        # conditioned/model-modified base state and receives a native empty
        # latent. INPAINT derives its own model, conditioning and latent from
        # that same base state. Only the final state is selected.
        image = IMAGE
        mask = pipe.get("mask")

        try:
            batch_size = int(image.shape[0]) if image is not None and getattr(image, "shape", None) is not None else 1
        except Exception:
            batch_size = 1

        normal_model = prepared_model
        normal_conditioning_pos = conditioning_pos
        normal_conditioning_neg = conditioning_neg
        normal_latent = self._empty_latent(width, height, batch_size)

        inpaint_model = normal_model
        inpaint_conditioning_pos = normal_conditioning_pos
        inpaint_conditioning_neg = normal_conditioning_neg
        inpaint_latent = None
        inpaint_mask = mask

        fooocus_log = "fooocus inpaint disabled"
        context_reference_active = False
        context_reference_log = "context reference disabled"

        if inpaint_mode:
            if vae is None or image is None or mask is None:
                raise ValueError(
                    "CMK Sampler Prepare SDXL -Pipe-: INPAINT=ON requires MODEL['vae'], IMAGE and PROCESS['mask']"
                )

            inpaint_model, inpaint_conditioning_pos, inpaint_conditioning_neg, inpaint_latent = (
                CMKFooocusInpaintPipeline().prepare(
                    normal_model,
                    normal_conditioning_pos,
                    normal_conditioning_neg,
                    vae,
                    image,
                    mask,
                    head=str(fooocus_head),
                    patch=str(fooocus_patch),
                    noise_mask=bool(inpaint_noise_mask),
                )
            )
            inpaint_model = _unwrap_node_output(inpaint_model)
            inpaint_conditioning_pos = _validate_conditioning(
                inpaint_conditioning_pos, "conditioning_pos(inpaint)"
            )
            inpaint_conditioning_neg = _validate_conditioning(
                inpaint_conditioning_neg, "conditioning_neg(inpaint)"
            )
            if inpaint_model is None or not hasattr(inpaint_model, "clone"):
                raise TypeError(
                    "CMK Sampler Prepare SDXL -Pipe-: Fooocus Apply returned invalid MODEL "
                    f"(type={type(inpaint_model).__name__})"
                )
            if inpaint_latent is None:
                raise ValueError(
                    "CMK Sampler Prepare SDXL -Pipe-: Fooocus Inpaint returned no LATENT"
                )
            fooocus_log = f"fooocus inpaint applied | head={fooocus_head} | patch={fooocus_patch}"

            # Context Reference exists only inside the INPAINT branch.
            context_reference_active = bool(context_reference_enabled)
            if context_reference_active:
                inpaint_conditioning_pos, inpaint_latent, inpaint_mask = (
                    CMKContextReferenceLatentMask().prepare(
                        inpaint_conditioning_pos,
                        inpaint_latent,
                        mask,
                        expand=_int(context_reference_expand, 3),
                        blur=_float(context_reference_blur, 5.0),
                        mask_only=bool(context_reference_mask_only),
                    )
                )
                inpaint_conditioning_pos = _validate_conditioning(
                    inpaint_conditioning_pos, "conditioning_pos(context_reference)"
                )
                context_reference_log = (
                    "context reference applied | "
                    f"expand={_int(context_reference_expand, 3)} | "
                    f"blur={_float(context_reference_blur, 5.0)} | "
                    f"mask_only={bool(context_reference_mask_only)}"
                )
        elif bool(context_reference_enabled):
            context_reference_log = "context reference forced OFF | INPAINT=OFF"

        # 7) Final Model + Conditioning + Latent switch, matching the old
        # standalone subgraphs. No partially patched state crosses branches.
        if inpaint_mode:
            selected_model = inpaint_model
            selected_conditioning_pos = inpaint_conditioning_pos
            selected_conditioning_neg = inpaint_conditioning_neg
            selected_latent = inpaint_latent
            selected_mask = inpaint_mask
            latent_log = f"latent selected | INPAINT branch | noise_mask={bool(inpaint_noise_mask)}"
        else:
            selected_model = normal_model
            selected_conditioning_pos = normal_conditioning_pos
            selected_conditioning_neg = normal_conditioning_neg
            selected_latent = normal_latent
            selected_mask = mask
            latent_log = "latent selected | NORMAL branch | EmptyLatentImage"

        # 8) ControlNet is attached only after the final branch selection, so it
        # affects exactly the conditioning that reaches the KSampler.
        if inpaint_mode:
            # Keep the reconstructed INPAINT branch unchanged.
            selected_conditioning_pos, selected_conditioning_neg, controlnet_applied, controlnet_log = (
                _apply_controlnet_from_pipe(
                    pipe, selected_conditioning_pos, selected_conditioning_neg, vae
                )
            )
        else:
            # DIAGNOSTIC: use the former CMK Pipe Set Sampler ControlNet path
            # verbatim. This calls native nodes.ControlNetApplyAdvanced directly.
            controlnet_hint = _maybe_invert_controlnet_hint(
                pipe.get("controlnet_image"),
                pipe.get("controlnet_invert_hint", False),
            )
            controlnet_strength = _float(pipe.get("controlnet_strength"), 0.60)
            controlnet_start = _float(pipe.get("controlnet_start_percent"), 0.00)
            controlnet_end = _float(pipe.get("controlnet_end_percent"), 1.00)

            selected_conditioning_pos, selected_conditioning_neg, controlnet_applied, controlnet_log = (
                CMKPipeSetSampler._apply_controlnet_or_bypass(
                    selected_conditioning_pos,
                    selected_conditioning_neg,
                    pipe.get("control_net"),
                    controlnet_hint,
                    vae,
                    controlnet_strength,
                    controlnet_start,
                    controlnet_end,
                )
            )
            controlnet_log = (
                f"INVERT HINT={'ON' if pipe.get('controlnet_invert_hint', False) else 'OFF'} | "
                + str(controlnet_log)
            )

        prepared_model = selected_model
        conditioning_pos = selected_conditioning_pos
        conditioning_neg = selected_conditioning_neg
        latent_image = selected_latent
        output_mask = selected_mask

        new_pipe = dict(pipe)
        new_pipe.pop("image", None)
        new_pipe.pop("image_original", None)
        new_pipe["checkpoint_name"] = str(MODEL.get("ckpt_name", ""))
        new_pipe["vae_name"] = str(MODEL.get("vae_name", ""))
        new_pipe["checkpoint_vae"] = bool(MODEL.get("checkpoint_vae", False))
        new_pipe["model"] = model
        new_pipe["model_base"] = model
        new_pipe["model_patched"] = prepared_model
        new_pipe["clip"] = clip
        new_pipe["clip_base"] = clip
        new_pipe["clip_patched"] = prepared_clip
        new_pipe["vae"] = vae
        new_pipe["conditioning_pos"] = conditioning_pos
        new_pipe["conditioning_neg"] = conditioning_neg
        new_pipe["latent_image"] = latent_image
        new_pipe["mask"] = output_mask
        new_pipe["latent_original"] = latent_image
        new_pipe["steps_1st_pass"] = _int(steps_1st_pass, 20)
        new_pipe["steps"] = _int(steps_1st_pass, 20)
        new_pipe["cfg"] = _float(cfg, 5.0)
        new_pipe["sampler"] = sampler
        new_pipe["scheduler"] = scheduler
        new_pipe["seed"] = _int(seed, 0)
        new_pipe["sdxl_width"] = width
        new_pipe["sdxl_height"] = height
        new_pipe["size_cond_factor"] = _int(size_cond_factor, 4)
        new_pipe["grow_mask_by"] = _int(grow_mask_by, 6)
        new_pipe["fooocus_inpaint_enabled"] = inpaint_mode
        new_pipe["boolean_inpaint_mode"] = inpaint_mode
        new_pipe["fooocus_head"] = str(fooocus_head)
        new_pipe["fooocus_patch"] = str(fooocus_patch)
        new_pipe["inpaint_noise_mask"] = bool(inpaint_noise_mask)
        new_pipe["context_reference_enabled"] = bool(context_reference_enabled)
        new_pipe["context_reference_active"] = bool(context_reference_active)
        new_pipe["context_reference_expand"] = _int(context_reference_expand, 3)
        new_pipe["context_reference_blur"] = _float(context_reference_blur, 5.0)
        new_pipe["context_reference_mask_only"] = bool(context_reference_mask_only)
        new_pipe["context_reference_mask"] = output_mask if context_reference_active else None
        new_pipe["boolean_controlnet_enable"] = bool(controlnet_applied)
        new_pipe["controlnet_applied"] = bool(controlnet_applied)
        new_pipe["controlnet_log"] = controlnet_log
        new_pipe["controlnet_strength"] = _float(pipe.get("controlnet_strength"), 1.0)
        new_pipe["controlnet_start_percent"] = _float(pipe.get("controlnet_start_percent"), 0.0)
        new_pipe["controlnet_end_percent"] = _float(pipe.get("controlnet_end_percent"), 1.0)
        new_pipe["stop_at_clip_layer"] = _int(stop_at_clip_layer, -2)
        new_pipe["sampling"] = str(sampling)
        new_pipe["zsnr"] = bool(zsnr)
        new_pipe["pag_scale"] = _float(pag_scale, 2.5)
        new_pipe["freeu_enabled"] = bool(freeu_enabled)
        new_pipe["freeu_b1"] = _float(freeu_b1, 1.3)
        new_pipe["freeu_b2"] = _float(freeu_b2, 1.4)
        new_pipe["freeu_s1"] = _float(freeu_s1, 0.9)
        new_pipe["freeu_s2"] = _float(freeu_s2, 0.2)
        new_pipe["normal_diagnostic_minimal"] = False
        new_pipe["normal_restore_stage"] = "FULL_PROCESS_RESTORE" if not inpaint_mode else "INPAINT_FULL"

        loaded_loras = " ".join(x for x in [loaded_stack_loras, loaded_single_lora] if x).strip()
        new_pipe["sampler_prepare_loras"] = loaded_loras
        new_pipe["sampler_prepare_log"] = (
            "CMK Sampler Prepare SDXL -Pipe- | "
            f"checkpoint={MODEL.get('ckpt_name', '')} | vae={MODEL.get('vae_name', '')} | "
            f"order={'MODEL > CLIPSetLastLayer > SOURCE LoRAs > LOCAL LoRA > PAG > ModelSamplingDiscrete > FreeU > SDXL+ pos/neg > EmptyLatentImage > ControlNet Apply' if not inpaint_mode else 'MODEL > CLIPSetLastLayer > CMKLoRATextLoader > LoraLoader > PAG > ModelSamplingDiscrete > FreeU_V2 > SDXL+ pos/neg > INPAINT > ControlNet'} | "
            f"{width}x{height} | steps={new_pipe['steps_1st_pass']} | cfg={new_pipe['cfg']} | "
            f"sampler={sampler} | scheduler={scheduler} | sampling={sampling} | "
            f"clip_layer={new_pipe['stop_at_clip_layer']} | size_cond_factor={new_pipe['size_cond_factor']} | "
            f"seed={new_pipe['seed']} | seed_mode=fixed | "
            f"latent={'ok' if latent_image is not None else 'missing'} | {fooocus_log} | {context_reference_log} | {controlnet_log}"
        )
        new_pipe["sampler_prepare_status"] = {
            "model": model is not None,
            "model_patched": prepared_model is not None,
            "clip": prepared_clip is not None,
            "vae": vae is not None,
            "conditioning_pos": conditioning_pos is not None,
            "conditioning_neg": conditioning_neg is not None,
            "latent_image": latent_image is not None,
            "latent_log": latent_log,
            "fooocus_log": fooocus_log,
            "fooocus_inpaint_enabled": inpaint_mode,
            "boolean_inpaint_mode": inpaint_mode,
            "context_reference_active": bool(context_reference_active),
            "context_reference_log": context_reference_log,
            "loaded_loras": loaded_loras,
        }
        lora_source = "SOURCE + LOCAL" if loaded_single_lora else "SOURCE"
        log_lines = [
            "STATUS          : PREPARED",
            "MODEL SOURCE    : MODEL",
            f"CHECKPOINT      : {MODEL.get('ckpt_name', '')}",
            f"VAE             : {MODEL.get('vae_name', '')}",
            f"SIZE            : {width} × {height}",
            f"STEPS           : {new_pipe['steps_1st_pass']}",
            f"CFG             : {new_pipe['cfg']}",
            f"SAMPLER         : {sampler}",
            f"SCHEDULER       : {scheduler}",
            f"SEED            : {new_pipe['seed']}",
            f"FOOOCUS INPAINT : {'PREPARED' if inpaint_mode else 'DISABLED'}",
            f"CONTEXT REF.    : {'PREPARED' if context_reference_active else 'DISABLED'}",
            f"CONTROLNET      : {'PREPARED' if controlnet_applied else 'DISABLED'}",
            "PROMPT SOURCE   : SOURCE",
            f"LORA SOURCE     : {lora_source}",
        ]
        if loaded_single_lora:
            log_lines.extend(["", "LOCAL LORAS:", cmk_format_loras(loaded_single_lora)])
        log_pipe = cmk_add_block(LOG, "Sampler Prepare", 40, log_lines, True)

        diagnostic = make_diagnostic_payload(
            title="Sampler Prepare",
            node="CMK Sampler Prepare SDXL -Pipe-",
            previews=[IMAGE] if IMAGE is not None else [],
            summary=f"{width}x{height} | {new_pipe['steps_1st_pass']} steps | CFG {new_pipe['cfg']}",
            details=new_pipe["sampler_prepare_log"],
            mode="inpaint" if inpaint_mode else "generate",
            metadata={
                "checkpoint": MODEL.get("ckpt_name", ""),
                "vae": MODEL.get("vae_name", ""),
                "sampler": sampler,
                "scheduler": scheduler,
                "seed": new_pipe["seed"],
            },
        )
        return (new_pipe, log_pipe, diagnostic)
