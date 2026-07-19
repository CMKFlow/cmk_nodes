from __future__ import annotations

import re

import torch
import torch.nn.functional as F
import nodes


_LORA_PATTERN = re.compile(r"<lora:([^:>]+):([^:>]+)(?::([^:>]+))?>", re.IGNORECASE)


def _image_from(value):
    if isinstance(value, torch.Tensor) and value.ndim == 4:
        return value
    return None


class CMKImageCompare:
    """Dependency-free two-image preview used inside published CMK flows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"image_a": ("IMAGE",), "image_b": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "compare"
    CATEGORY = "CMK/Internal"
    OUTPUT_NODE = True

    def compare(self, image_a=None, image_b=None):
        # Published CMK subgraphs consistently connect the processed RESULT to
        # A and the unprocessed SOURCE to B (matching the former comparer).
        selected = _image_from(image_a)
        if selected is None:
            selected = _image_from(image_b)
        if selected is None:
            selected = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        source = _image_from(image_b)
        if source is None:
            source = selected
        source_info = nodes.PreviewImage().save_images(source[:1])["ui"]["images"][0]
        result_info = nodes.PreviewImage().save_images(selected[:1])["ui"]["images"][0]
        return {
            "ui": {
                # Keep both descriptors in the standard field so the surrounding
                # subgraph can offer the same press-and-hold comparison.
                "images": [source_info, result_info],
                "cmk_compare_images": [source_info, result_info],
            },
            "result": (selected,),
        }


class CMKSEGSPreview:
    """Render CMK SEGS crops without requiring Impact Pack's preview node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"segs": ("SEGS",)},
            "optional": {
                "fallback_image_opt": ("IMAGE",),
                "alpha_mode": ("BOOLEAN", {"default": True}),
                "min_alpha": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preview"
    CATEGORY = "CMK/Internal"
    OUTPUT_NODE = True

    def preview(self, segs, alpha_mode=True, min_alpha=0.2, fallback_image_opt=None):
        crops = []
        if isinstance(segs, tuple) and len(segs) == 2:
            for seg in segs[1]:
                crop = _image_from(getattr(seg, "cropped_image", None))
                if crop is None:
                    continue
                crop = crop[:1].float()
                if bool(alpha_mode):
                    mask = getattr(seg, "cropped_mask", None)
                    if isinstance(mask, torch.Tensor):
                        if mask.ndim == 2:
                            mask = mask.unsqueeze(0)
                        if mask.ndim == 3:
                            mask = mask.unsqueeze(-1)
                        mask = F.interpolate(
                            mask.movedim(-1, 1).float(),
                            size=crop.shape[1:3],
                            mode="bilinear",
                            align_corners=False,
                        ).movedim(1, -1)
                        crop = crop * mask.clamp(float(min_alpha), 1.0)
                crop = F.interpolate(
                    crop.movedim(-1, 1), size=(256, 256), mode="bilinear", align_corners=False
                ).movedim(1, -1)
                crops.append(crop)
        if crops:
            image = torch.cat(crops, dim=0)
        else:
            image = _image_from(fallback_image_opt)
            if image is None:
                image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        return nodes.PreviewImage().save_images(image)


class CMKLoRAStackBuilder:
    """Build the lightweight LORA_STACK consumed by CMK's LoRA loader."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_syntax": ("STRING", {"default": "", "multiline": True}),
                "trigger_words": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {"lora_stack": ("LORA_STACK",)},
        }

    RETURN_TYPES = ("LORA_STACK", "STRING", "STRING")
    RETURN_NAMES = ("lora_stack", "trigger_words", "active_loras")
    FUNCTION = "build"
    CATEGORY = "CMK/Toolbox/Model & LoRA"

    def build(self, lora_syntax="", trigger_words="", lora_stack=None):
        stack = list(lora_stack or [])
        active = []
        for name, model_strength, clip_strength in _LORA_PATTERN.findall(str(lora_syntax or "")):
            try:
                model_value = float(model_strength)
            except Exception:
                model_value = 1.0
            try:
                clip_value = float(clip_strength) if clip_strength else model_value
            except Exception:
                clip_value = model_value
            stack.append((name.strip(), model_value, clip_value))
            active.append(name.strip())
        return (stack, str(trigger_words or "").strip(), ", ".join(active))


class CMKPromptConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt1": ("STRING", {"forceInput": True}),
                "prompt2": ("STRING", {"forceInput": True}),
                "separator": ("STRING", {"default": ", "}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "concat"
    CATEGORY = "CMK/Internal"

    def concat(self, prompt1="", prompt2="", separator=", "):
        values = [str(value).strip() for value in (prompt1, prompt2) if str(value).strip()]
        return (str(separator).join(values),)


class CMKTriggerWordsFilter:
    """Compatibility-preserving CMK replacement for the former toggle widget."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_mode": ("BOOLEAN", {"default": True}),
                "default_active": ("BOOLEAN", {"default": True}),
                "allow_strength_adjustment": ("BOOLEAN", {"default": False}),
            },
            "optional": {"trigger_words": ("STRING", {"forceInput": True})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_trigger_words",)
    FUNCTION = "filter"
    CATEGORY = "CMK/Internal"

    def filter(self, group_mode=True, default_active=True, allow_strength_adjustment=False, trigger_words=""):
        del group_mode, allow_strength_adjustment
        return (str(trigger_words or "").strip() if bool(default_active) else "",)


class CMKStringDual:
    """Two multiline prompt fields with matching positive/negative outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "txt_pos": ("STRING", {"default": "", "multiline": True}),
                "txt_neg": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("txt_pos", "txt_neg")
    FUNCTION = "emit"
    CATEGORY = "CMK/Toolbox/Text"

    def emit(self, txt_pos="", txt_neg=""):
        return (str(txt_pos or ""), str(txt_neg or ""))
