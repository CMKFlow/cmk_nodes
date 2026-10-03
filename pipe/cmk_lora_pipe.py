from __future__ import annotations

from typing import Any


def _pack_lora_bundle(
    family: str,
    lora_stack: Any = None,
    active_loras: str = "",
    trigger_words: str = "",
) -> tuple[dict[str, Any]]:
    return ({
        "family": family,
        "lora_stack": lora_stack,
        "active_loras": str(active_loras or "").strip(),
        "trigger_words": str(trigger_words or "").strip(),
    },)


class _CMKLoRAPackBase:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_stack": ("LORA_STACK", {"forceInput": True}),
                "active_loras": ("STRING", {"forceInput": True}),
                "trigger_words": ("STRING", {"forceInput": True}),
            }
        }

    FUNCTION = "pack"
    CATEGORY = "CMK/Internal"


class CMKLoRASDXLPackPipe(_CMKLoRAPackBase):
    RETURN_TYPES = ("CMK_LORA_SDXL_PIPE",)
    RETURN_NAMES = ("LoRA SDXL",)

    def pack(self, lora_stack=None, active_loras="", trigger_words=""):
        return _pack_lora_bundle("sdxl", lora_stack, active_loras, trigger_words)


class CMKLoRAZITPackPipe(_CMKLoRAPackBase):
    RETURN_TYPES = ("CMK_LORA_ZIT_PIPE",)
    RETURN_NAMES = ("LoRA ZIT",)

    def pack(self, lora_stack=None, active_loras="", trigger_words=""):
        return _pack_lora_bundle("z_image_turbo", lora_stack, active_loras, trigger_words)
