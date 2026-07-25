from .cmk_log_pipe import cmk_add_block, cmk_bool, cmk_clean_text
from ..utils.cmk_diagnostic import make_diagnostic_payload
from comfy.utils import common_upscale



RESOLUTION_PRESETS = [
    "SDXL 1024x1024",
    "SDXL 1152x832",
    "SDXL 832x1152",
    "SDXL 1216x832",
    "SDXL 832x1216",
    "SDXL 1344x768",
    "SDXL 768x1344",
    "SD15 512x512",
    "SD15 768x512",
    "SD15 512x768",
]

UPSCALE_METHODS = ["lanczos", "bicubic", "bilinear", "nearest"]
DEVICES = ["cpu", "mps", "cuda"]
MASKED_AREA_FILL = [
    "neutral",
    "lama",
    "telea",
    "navier-stokes",
    "original",
    "black",
    "white",
    "noise",
]
def parse_resolution(resolution, fallback_width=1024, fallback_height=1024):
    text = str(resolution or "").strip()
    token = text.split()[-1] if text else ""
    if "x" not in token.lower():
        return fallback_width, fallback_height
    left, right = token.lower().split("x", 1)
    try:
        return int(left), int(right)
    except Exception:
        return fallback_width, fallback_height


def get_image_size(image):
    try:
        return int(image.shape[2]), int(image.shape[1])
    except Exception:
        return None, None


def resize_image_tensor(image, width, height, upscale_method):
    if image is None:
        return image
    current_width, current_height = get_image_size(image)
    if current_width == width and current_height == height:
        return image
    samples = image.movedim(-1, 1)
    resized = common_upscale(samples, width, height, upscale_method, "disabled")
    return resized.movedim(1, -1)


def resize_mask_tensor(mask, width, height):
    if mask is None:
        return mask
    original_dim = len(mask.shape)
    if original_dim == 2:
        samples = mask.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:
        samples = mask.unsqueeze(1)
    elif original_dim == 4:
        samples = mask.movedim(-1, 1) if mask.shape[-1] == 1 else mask
    else:
        return mask

    if int(samples.shape[-1]) == width and int(samples.shape[-2]) == height:
        return mask

    resized = common_upscale(samples, width, height, "nearest-exact", "disabled")
    if original_dim == 2:
        return resized.squeeze(0).squeeze(0)
    if original_dim == 3:
        return resized.squeeze(1)
    if original_dim == 4 and mask.shape[-1] == 1:
        return resized.movedim(1, -1)
    return resized


def apply_mask_fill(image, mask, fill_mode: str, seed: int = 0):
    """Return the exact IMAGE payload with CMK's selected mask fill applied."""
    if image is None or mask is None:
        return image

    import torch
    import torch.nn.functional as F

    if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return image

    fill_mode = str(fill_mode or "original").strip().lower()
    if fill_mode == "original":
        return image

    work_mask = mask.float()
    if work_mask.ndim == 2:
        work_mask = work_mask.unsqueeze(0)
    if work_mask.ndim == 4:
        if work_mask.shape[-1] == 1:
            work_mask = work_mask[..., 0]
        elif work_mask.shape[1] == 1:
            work_mask = work_mask[:, 0]
    if work_mask.ndim != 3:
        return image

    if tuple(work_mask.shape[-2:]) != tuple(image.shape[1:3]):
        work_mask = F.interpolate(
            work_mask.unsqueeze(1),
            size=tuple(image.shape[1:3]),
            mode="nearest",
        ).squeeze(1)
    if work_mask.shape[0] == 1 and image.shape[0] > 1:
        work_mask = work_mask.expand(image.shape[0], -1, -1)

    alpha = work_mask.clamp(0.0, 1.0).unsqueeze(-1).to(
        device=image.device,
        dtype=image.dtype,
    )
    if fill_mode == "lama":
        from ..engine.lama_inpaint import lama_inpaint_tensor

        return lama_inpaint_tensor(image, work_mask)
    if fill_mode in {"telea", "navier-stokes"}:
        import cv2
        import numpy as np

        algorithm = (
            cv2.INPAINT_TELEA
            if fill_mode == "telea"
            else cv2.INPAINT_NS
        )
        filled_frames = []
        for index in range(int(image.shape[0])):
            source = (
                image[index]
                .detach()
                .float()
                .clamp(0.0, 1.0)
                .cpu()
                .numpy()
            )
            mask_index = min(index, int(alpha.shape[0]) - 1)
            mask_np = (
                alpha[mask_index, ..., 0]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            source_u8 = np.rint(source * 255.0).astype(np.uint8)
            mask_u8 = np.where(mask_np > 0.01, 255, 0).astype(np.uint8)
            filled_u8 = cv2.inpaint(source_u8, mask_u8, 3.0, algorithm)
            filled_frames.append(
                torch.from_numpy(filled_u8.astype(np.float32) / 255.0)
            )
        fill = torch.stack(filled_frames, dim=0).to(
            device=image.device,
            dtype=image.dtype,
        )
    elif fill_mode == "black":
        fill = torch.zeros_like(image)
    elif fill_mode == "white":
        fill = torch.ones_like(image)
    elif fill_mode == "noise":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
        fill = torch.rand(
            tuple(image.shape),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(device=image.device, dtype=image.dtype)
    else:
        # Neutral preserves scene luminance without retaining masked content.
        keep = 1.0 - alpha
        count = keep.sum(dim=(1, 2), keepdim=True)
        measured = (image * keep).sum(dim=(1, 2), keepdim=True) / count.clamp_min(1.0)
        mean = torch.where(count > 0.0, measured, torch.full_like(measured, 0.5))
        fill = mean.expand_as(image)

    return image * (1.0 - alpha) + fill * alpha




def mask_to_preview_rgb(mask):
    """Convert a ComfyUI MASK tensor to a first-frame RGB uint8 diagnostic image."""
    if mask is None:
        return None
    try:
        import torch
        if isinstance(mask, torch.Tensor):
            arr = mask.detach().cpu().numpy()
        else:
            import numpy as np
            arr = np.asarray(mask)
    except Exception:
        import numpy as np
        arr = np.asarray(mask)

    import numpy as np
    if arr.ndim == 4:
        # [B,H,W,C] or [B,C,H,W]; use first image and squeeze singleton channel.
        arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
    elif arr.ndim == 3:
        # Comfy MASK is commonly [B,H,W].
        arr = arr[0]
    if arr.ndim != 2:
        return None
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if arr.size and float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    gray = (arr * 255.0).round().astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)

def build_image_log_block(
    resolution,
    width,
    height,
    boolean_inpaint_mode,
    outpaint_on,
    swap_dimensions,
    upscale_method,
    device,
    mask_fill_holes,
    fill_masked_area,
    active_loras,
    prompt_pos,
    prompt_neg,
    inpaint_process_mode,
):
    lines = [
        f"SDXL PRESET     : {resolution}",
        f"PROCESS SIZE    : {width} × {height}",
        f"INPAINT MODE    : {cmk_bool(boolean_inpaint_mode)}",
        f"PROCESS MODE    : {str(inpaint_process_mode).upper()}",
        f"OUTPAINT        : {cmk_bool(outpaint_on)}",
        f"SWAP DIMENSIONS : {cmk_bool(swap_dimensions)}",
        f"UPSCALE METHOD  : {upscale_method}",
        f"IMAGE DEVICE    : {str(device).upper()}",
        f"MASK FILL HOLES : {cmk_bool(mask_fill_holes)}",
        f"MASKED AREA     : {fill_masked_area}",
    ]

    loras = cmk_clean_text(active_loras)
    if loras:
        lines.extend(["", "LORA SYNTAX:"])
        lines.extend(loras.splitlines())

    pos = cmk_clean_text(prompt_pos)
    if pos:
        lines.extend(["", "POSITIVE PROMPT:"])
        lines.extend(pos.splitlines())

    neg = cmk_clean_text(prompt_neg)
    if neg:
        lines.extend(["", "NEGATIVE PROMPT:"])
        lines.extend(neg.splitlines())

    return lines


class CMKPipeCreateImage:
    DESCRIPTION = (
        "CMK FLOW START. Creates the authoritative PROCESS, IMAGE and LOG lines. "
        "Continue with 'CMK Flow · 05 ControlNet (optional)' or connect PROCESS, "
        "IMAGE and LOG directly to 'CMK Flow · 10 KSampler 1st Pass'."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "PROMPT POS": ("STRING", {"default": "", "multiline": True, "tooltip": "Positive prompt for the complete Flow."}),
                "PROMPT NEG": ("STRING", {"default": "", "multiline": True, "tooltip": "Negative prompt for the complete Flow."}),
                "INPAINT_MODE": (
                    ["Text2Image", "Inpaint"],
                    {
                        "default": "Text2Image",
                        "tooltip": (
                            "Text2Image creates a new image. Inpaint uses IMAGE and MASK "
                            "and reveals the task-specific inpaint settings."
                        ),
                    },
                ),
                "resolution": (RESOLUTION_PRESETS, {"default": "SDXL 1152x832"}),
                "swap_dimensions": ("BOOLEAN", {"default": False}),
                "upscale_method": (UPSCALE_METHODS, {"default": "lanczos"}),
                "device": (DEVICES, {"default": "cpu"}),
            },
            "optional": {
                "PROCESS": (
                    "CMK_PIPE",
                    {
                        "tooltip": (
                            "Optional incoming PROCESS from an upstream CMK Flow input module. "
                            "Create Image preserves it and applies its authoritative image settings."
                        ),
                    },
                ),
                "IMAGE": ("IMAGE", {"tooltip": "Required only when INPAINT_MODE is enabled."}),
                "MASK": ("MASK", {"tooltip": "Required only when INPAINT_MODE is enabled."}),
                "FILENAME STRING": ("STRING", {"forceInput": True, "default": "", "tooltip": "Required only when INPAINT_MODE is enabled; used by logging and project output."}),
                "LOG": (
                    "CMK_LOG_PIPE",
                    {
                        "tooltip": (
                            "Optional incoming LOG from an upstream CMK Flow input module. "
                            "Create Image appends its image preparation block."
                        ),
                    },
                ),
                "lora_stack": ("LORA_STACK", {"tooltip": "Connect 'CMK Flow · 02 LoRA Stack'."}),
                "lora_syntax": (
                    "STRING",
                    {
                        "forceInput": True,
                        "default": "",
                        "multiline": True,
                    },
                ),
                # Text2Image deliberately hides these widgets. They therefore
                # must be optional in ComfyUI's prompt contract; create_image
                # supplies the same defaults when the UI omits them.
                "outpaint_on": ("BOOLEAN", {"default": False}),
                "mask_fill_holes": ("BOOLEAN", {"default": False}),
                "fill_masked_area": (MASKED_AREA_FILL, {"default": "neutral"}),
                "process_mode": (
                    ["Custom", "Replace Object", "Remove Object", "Extend Image"],
                    {
                        "default": "Custom",
                        "tooltip": (
                            "Selects a guided inpaint preset; it does not perform semantic object recognition. "
                            "Custom keeps the Sampler Advanced values. Replace Object uses noise fill, denoise 1.00, "
                            "noise mask ON, context reference ON and outpaint OFF; user prompt and LoRAs remain active. "
                            "Remove Object uses local LaMa "
                            "for prompt-free object removal; diffusion, existing prompts and all LoRAs are bypassed. "
                            "Extend Image uses "
                            "Navier-Stokes fill, denoise 1.00, noise mask ON and context reference ON; an outpaint "
                            "mask/canvas is still required. Guided modes override fill_masked_area."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("CMK_PIPE", "IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("PROCESS", "IMAGE", "LOG", "diagnostic")
    OUTPUT_TOOLTIPS = (
        "Continue to CMK Flow · 05 ControlNet (optional) or CMK Flow · 10 KSampler 1st Pass.",
        "Authoritative image; route it beside PROCESS and LOG to the next Flow module.",
        "Structured Flow log; route it beside PROCESS and IMAGE to the next Flow module.",
        "Optional diagnostic information for troubleshooting.",
    )
    FUNCTION = "create_image"
    CATEGORY = 'CMK/Flow/Input'

    def create_image(self, **inputs):
        incoming_process = inputs.get("PROCESS")
        incoming_log = inputs.get("LOG")
        image = inputs.get("IMAGE")
        mask = inputs.get("MASK")
        filename_string = str(inputs.get("FILENAME STRING", "") or "")
        lora_stack = inputs.get("lora_stack")
        lora_syntax = inputs.get("lora_syntax", "") or ""
        prompt_pos = inputs.get("PROMPT POS", "") or ""
        prompt_neg = inputs.get("PROMPT NEG", "") or ""
        raw_mode = inputs.get("INPAINT_MODE", "Text2Image")
        INPAINT_MODE = (
            bool(raw_mode)
            if isinstance(raw_mode, bool)
            else str(raw_mode or "Text2Image").strip().lower() == "inpaint"
        )
        process_mode = inputs.get("process_mode", "Custom")
        resolution = inputs.get("resolution", "SDXL 1152x832")
        swap_dimensions = inputs.get("swap_dimensions", False)
        upscale_method = inputs.get("upscale_method", "lanczos")
        device = inputs.get("device", "cpu")
        outpaint_on = inputs.get("outpaint_on", False)
        mask_fill_holes = inputs.get("mask_fill_holes", False)
        fill_masked_area = inputs.get("fill_masked_area", "neutral")
        mode_key = str(process_mode or "Custom").strip().lower()
        inpaint_process_mode = {
            "custom": "custom",
            "replace": "replace",
            "replace object": "replace",
            "remove": "remove",
            "remove object": "remove",
            "extend": "extend",
            "extend image": "extend",
        }.get(mode_key, "custom")
        guided_fill_modes = {
            "replace": "noise",
            "remove": "lama",
            "extend": "navier-stokes",
        }
        effective_fill_mode = (
            guided_fill_modes.get(inpaint_process_mode)
            if bool(INPAINT_MODE) and inpaint_process_mode != "custom"
            else str(fill_masked_area or "neutral").strip().lower()
        )
        source_prompt_pos = prompt_pos
        source_prompt_neg = prompt_neg
        source_lora_syntax = lora_syntax
        source_lora_stack = lora_stack
        remove_isolated = bool(INPAINT_MODE) and inpaint_process_mode == "remove"
        if remove_isolated:
            # Remove Object is a complete CMK task, not another prompt/LoRA
            # variation. Existing workflow styling must not recreate the
            # masked subject.
            prompt_pos = ""
            prompt_neg = ""
            lora_syntax = ""
            lora_stack = None
            outpaint_on = False
        elif bool(INPAINT_MODE) and inpaint_process_mode == "replace":
            # Replacement happens inside the existing canvas. Noise removes
            # the semantic silhouette of the old object before diffusion.
            outpaint_on = False

        if bool(INPAINT_MODE):
            missing = []
            if image is None:
                missing.append("IMAGE")
            if mask is None:
                missing.append("MASK")
            if not filename_string:
                missing.append("FILENAME STRING")
            if missing:
                raise ValueError(
                    "CMK Flow · Create Image: INPAINT_MODE requires "
                    + ", ".join(missing)
                    + "."
                )

        width, height = parse_resolution(resolution)
        if bool(swap_dimensions):
            width, height = height, width

        source_width, source_height = get_image_size(image)

        # This node prepares only the dedicated IMAGE cable and image-related
        # process metadata. The sampler owns LATENT creation and the final
        # NORMAL/INPAINT branch selection.
        image_resized = resize_image_tensor(image, width, height, upscale_method)
        mask_process = resize_mask_tensor(mask, width, height)
        image_out = (
            apply_mask_fill(image_resized, mask_process, effective_fill_mode, seed=0)
            if bool(INPAINT_MODE)
            else image_resized
        )

        pipe = dict(incoming_process) if isinstance(incoming_process, dict) else {}
        pipe.update({
            "mask": mask_process,
            "mask_original": mask,
            "width": width,
            "height": height,
            "source_width": source_width,
            "source_height": source_height,
            "target_width": width,
            "target_height": height,
            "resolution": resolution,
            "swap_dimensions": swap_dimensions,
            "upscale_method": upscale_method,
            "device": device,
            "outpaint_on": outpaint_on,
            "mask_fill_holes": mask_fill_holes,
            "fill_masked_area": effective_fill_mode,
            "mask_fill_applied": bool(INPAINT_MODE and image_out is not None and mask_process is not None),
            "mask_fill_seed": 0,
            "filename_string": filename_string,
            "file_name": filename_string,
            "prompt_pos": prompt_pos,
            "prompt_neg": prompt_neg,
            "lora_syntax": lora_syntax,
            # Compatibility field for Prepare nodes not yet migrated to lora_syntax.
            "active_loras": lora_syntax,
            "lora_stack": lora_stack,
            "source_prompt_pos": source_prompt_pos,
            "source_prompt_neg": source_prompt_neg,
            "source_lora_syntax": source_lora_syntax,
            "source_lora_stack": source_lora_stack,
            "remove_isolated": remove_isolated,
            "remove_result_image": image_out if remove_isolated else None,
            "boolean_inpaint_mode": INPAINT_MODE,
            "inpaint_process_mode": inpaint_process_mode,
            "control_net": None,
            "controlnet_image": None,
        })

        log_lines = build_image_log_block(
            resolution=resolution,
            width=width,
            height=height,
            boolean_inpaint_mode=INPAINT_MODE,
            outpaint_on=outpaint_on,
            swap_dimensions=swap_dimensions,
            upscale_method=upscale_method,
            device=device,
            mask_fill_holes=mask_fill_holes,
            fill_masked_area=effective_fill_mode,
            active_loras=lora_syntax,
            prompt_pos=prompt_pos,
            prompt_neg=prompt_neg,
            inpaint_process_mode=inpaint_process_mode,
        )
        if filename_string:
            log_lines.insert(0, f"FILE NAME       : {filename_string}")
        if remove_isolated:
            log_lines.extend(
                [
                    "",
                    "REMOVE ENGINE   : LaMa",
                    "DIFFUSION       : Bypassed",
                    "SOURCE PROMPTS  : Ignored",
                    "SOURCE LORAS    : Ignored",
                ]
            )
        log_pipe = cmk_add_block(
            incoming_log,
            "Image",
            10,
            log_lines,
            True,
        )
        log_pipe.update(
            {
                "filename_string": filename_string,
                "file_name": filename_string,
                "prompt_pos": prompt_pos,
                "prompt_neg": prompt_neg,
            }
        )

        summary = "\n".join(log_lines)
        diagnostic_previews = [image_resized]
        diagnostic_stages = []
        if image_resized is not None:
            diagnostic_stages.append(
                {
                    "title": "01 Source",
                    "subtitle": "Resized input",
                    "image": image_resized,
                }
            )
        if bool(INPAINT_MODE) and image_out is not None:
            diagnostic_previews.append(image_out)
            diagnostic_stages.append(
                {
                    "title": "02 Mask Fill",
                    "subtitle": effective_fill_mode,
                    "image": image_out,
                }
            )

        diagnostic = make_diagnostic_payload(
            title="Pipe Create Image -Pipe-",
            node="CMK Pipe Create Image -Pipe-",
            previews=diagnostic_previews,
            stages=diagnostic_stages,
            summary=summary,
            details=summary,
            mode="Create",
            metadata={
                "resolution": resolution,
                "source_width": source_width,
                "source_height": source_height,
                "target_width": width,
                "target_height": height,
                "inpaint_mode": bool(INPAINT_MODE),
                "inpaint_process_mode": inpaint_process_mode,
                "outpaint_on": bool(outpaint_on),
                "swap_dimensions": bool(swap_dimensions),
                "upscale_method": upscale_method,
                "device": device,
            },
        )

        return (pipe, image_out, log_pipe, diagnostic)


class CMKPipePeekPreprocessImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("CMK_PIPE",)}}

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "INT",
        "INT",
        "STRING",
        "STRING",
        "STRING",
        "LORA_STACK",
        "BOOLEAN",
        "CONTROL_NET",
        "IMAGE",
    )
    RETURN_NAMES = (
        "image",
        "mask",
        "width",
        "height",
        "prompt_pos",
        "prompt_neg",
        "active_loras",
        "lora_stack",
        "boolean_inpaint_mode",
        "control_net",
        "controlnet_image",
    )
    FUNCTION = "peek_preprocess_image"
    CATEGORY = 'CMK/Developer/Pipe/Peek'

    def peek_preprocess_image(self, pipe):
        return (
            pipe.get("image"),
            pipe.get("mask"),
            pipe.get("width"),
            pipe.get("height"),
            pipe.get("prompt_pos"),
            pipe.get("prompt_neg"),
            pipe.get("active_loras"),
            pipe.get("lora_stack"),
            pipe.get("boolean_inpaint_mode"),
            pipe.get("control_net"),
            pipe.get("controlnet_image"),
        )


class CMKPipePeekControlNetSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("CMK_PIPE",)}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("base_image", "mask")
    FUNCTION = "peek_controlnet_source"
    CATEGORY = 'CMK/Developer/Pipe/Peek'

    def peek_controlnet_source(self, pipe):
        return (
            pipe.get("image"),
            pipe.get("mask"),
        )
