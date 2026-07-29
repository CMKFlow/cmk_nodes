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
RESIZE_MODES = ["Fit", "Crop", "Stretch"]
CROP_POSITIONS = ["Center", "Top", "Bottom", "Left", "Right"]
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


def _normalize_mask_bhw(mask):
    """Normalize the supported ComfyUI MASK layouts to [B,H,W]."""
    if mask is None:
        return None
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 3:
        return mask
    if mask.ndim == 4 and mask.shape[-1] == 1:
        return mask[..., 0]
    if mask.ndim == 4 and mask.shape[1] == 1:
        return mask[:, 0]
    raise ValueError(f"Unsupported MASK shape: {tuple(mask.shape)}")


def _position_offset(space, crop_position, axis):
    position = str(crop_position or "Center").strip().lower()
    if axis == "x" and position == "left":
        return 0
    if axis == "x" and position == "right":
        return space
    if axis == "y" and position == "top":
        return 0
    if axis == "y" and position == "bottom":
        return space
    return space // 2


def prepare_image_and_mask(
    image,
    mask,
    width,
    height,
    upscale_method,
    resize_mode="Fit",
    crop_position="Center",
):
    """Apply one shared transform to IMAGE and MASK.

    Returns the prepared image, transformed source mask and a mask covering
    canvas pixels which did not originate in the source image.
    """
    if image is None:
        return None, resize_mask_tensor(mask, width, height), None

    import torch

    source_width, source_height = get_image_size(image)
    mode = str(resize_mode or "Fit").strip().lower()
    if mode == "stretch":
        return (
            resize_image_tensor(image, width, height, upscale_method),
            resize_mask_tensor(mask, width, height),
            torch.zeros(
                (int(image.shape[0]), height, width),
                device=image.device,
                dtype=image.dtype,
            ),
        )

    scale = (
        max(width / source_width, height / source_height)
        if mode == "crop"
        else min(width / source_width, height / source_height)
    )
    scaled_width = max(1, int(round(source_width * scale)))
    scaled_height = max(1, int(round(source_height * scale)))
    scaled_image = resize_image_tensor(
        image, scaled_width, scaled_height, upscale_method
    )
    scaled_mask = resize_mask_tensor(mask, scaled_width, scaled_height)
    scaled_mask = _normalize_mask_bhw(scaled_mask) if scaled_mask is not None else None

    if mode == "crop":
        left = _position_offset(scaled_width - width, crop_position, "x")
        top = _position_offset(scaled_height - height, crop_position, "y")
        prepared_image = scaled_image[:, top:top + height, left:left + width, :]
        prepared_mask = (
            scaled_mask[:, top:top + height, left:left + width]
            if scaled_mask is not None
            else None
        )
        uncovered = torch.zeros(
            (int(image.shape[0]), height, width),
            device=image.device,
            dtype=image.dtype,
        )
        return prepared_image, prepared_mask, uncovered

    # Fit: preserve the complete source and expose the unused canvas as mask.
    left = _position_offset(width - scaled_width, crop_position, "x")
    top = _position_offset(height - scaled_height, crop_position, "y")
    prepared_image = torch.zeros(
        (int(image.shape[0]), height, width, int(image.shape[-1])),
        device=image.device,
        dtype=image.dtype,
    )
    prepared_image[:, top:top + scaled_height, left:left + scaled_width, :] = scaled_image
    prepared_mask = None
    if scaled_mask is not None:
        prepared_mask = torch.zeros(
            (int(scaled_mask.shape[0]), height, width),
            device=scaled_mask.device,
            dtype=scaled_mask.dtype,
        )
        prepared_mask[:, top:top + scaled_height, left:left + scaled_width] = scaled_mask
    uncovered = torch.ones(
        (int(image.shape[0]), height, width),
        device=image.device,
        dtype=image.dtype,
    )
    uncovered[:, top:top + scaled_height, left:left + scaled_width] = 0
    return prepared_image, prepared_mask, uncovered


def expand_mask_tensor(mask, amount):
    """Grow a ComfyUI mask by ``amount`` image pixels on every side."""
    if mask is None or int(amount) <= 0:
        return mask

    import torch.nn.functional as F

    original_dim = mask.ndim
    original_channel_last = original_dim == 4 and mask.shape[-1] == 1
    original_channel_first = original_dim == 4 and mask.shape[1] == 1
    work_mask = _normalize_mask_bhw(mask).float()
    radius = int(amount)
    expanded = F.max_pool2d(
        work_mask.unsqueeze(1),
        kernel_size=radius * 2 + 1,
        stride=1,
        padding=radius,
    ).squeeze(1)
    expanded = expanded.to(device=mask.device, dtype=mask.dtype)

    if original_dim == 2:
        return expanded[0]
    if original_channel_last:
        return expanded.unsqueeze(-1)
    if original_channel_first:
        return expanded.unsqueeze(1)
    return expanded


def feather_mask_tensor(mask, radius):
    """Return a soft-edged copy while preserving the original mask layout."""
    if mask is None or int(radius) <= 0:
        return mask

    import torch.nn.functional as F

    original_dim = mask.ndim
    original_channel_last = original_dim == 4 and mask.shape[-1] == 1
    original_channel_first = original_dim == 4 and mask.shape[1] == 1
    work_mask = _normalize_mask_bhw(mask).float().unsqueeze(1)
    radius = int(radius)
    work_mask = F.pad(
        work_mask,
        (radius, radius, radius, radius),
        mode="replicate",
    )
    feathered = F.avg_pool2d(
        work_mask,
        kernel_size=radius * 2 + 1,
        stride=1,
    ).squeeze(1)
    feathered = feathered.clamp(0.0, 1.0).to(
        device=mask.device,
        dtype=mask.dtype,
    )

    if original_dim == 2:
        return feathered[0]
    if original_channel_last:
        return feathered.unsqueeze(-1)
    if original_channel_first:
        return feathered.unsqueeze(1)
    return feathered


def fill_mask_holes(mask):
    """Fill background regions fully enclosed by a mask."""
    if mask is None:
        return mask

    import cv2
    import numpy as np
    import torch

    if not isinstance(mask, torch.Tensor):
        return mask

    original_shape = tuple(mask.shape)
    work_mask = mask.float()
    if work_mask.ndim == 2:
        work_mask = work_mask.unsqueeze(0)
    elif work_mask.ndim == 4:
        if work_mask.shape[-1] == 1:
            work_mask = work_mask[..., 0]
        elif work_mask.shape[1] == 1:
            work_mask = work_mask[:, 0]
    if work_mask.ndim != 3:
        return mask

    filled_masks = []
    for frame in work_mask:
        source = frame.detach().cpu().numpy()
        foreground = np.where(source > 0.5, 1, 0).astype(np.uint8)
        background = 1 - foreground

        # Padding supplies a guaranteed exterior seed even when the mask
        # touches every edge. Anything not reached from that seed is a hole.
        exterior = np.pad(background, 1, mode="constant", constant_values=1)
        flood_mask = np.zeros(
            (exterior.shape[0] + 2, exterior.shape[1] + 2),
            dtype=np.uint8,
        )
        cv2.floodFill(exterior, flood_mask, (0, 0), 0)
        holes = exterior[1:-1, 1:-1]
        filled = np.maximum(source, holes.astype(source.dtype))
        filled_masks.append(torch.from_numpy(filled))

    result = torch.stack(filled_masks, dim=0).to(
        device=mask.device,
        dtype=mask.dtype,
    )
    if len(original_shape) == 2:
        return result[0]
    if len(original_shape) == 4 and original_shape[-1] == 1:
        return result.unsqueeze(-1)
    if len(original_shape) == 4 and original_shape[1] == 1:
        return result.unsqueeze(1)
    return result


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


def image_node_preview(image):
    """Return a native, uncaptioned ComfyUI preview for an IMAGE tensor."""
    if image is None:
        return None
    try:
        from nodes import PreviewImage

        payload = PreviewImage().save_images(image)
        if isinstance(payload, dict):
            return payload.get("ui")
    except Exception:
        # A UI preview must never make the image pipeline fail.
        pass
    return None


def build_image_log_block(
    resolution,
    width,
    height,
    boolean_inpaint_mode,
    outpaint_on,
    outpaint_overlap,
    swap_dimensions,
    resize_mode,
    crop_position,
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
        f"OUTPAINT OVERLAP: {int(outpaint_overlap)} px",
        f"SWAP DIMENSIONS : {cmk_bool(swap_dimensions)}",
        f"RESIZE MODE     : {resize_mode}",
        f"CROP POSITION   : {crop_position}",
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
                "device": (DEVICES, {"default": "cpu", "advanced": True}),
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
                "MASK": (
                    "MASK",
                    {
                        "tooltip": (
                            "Required for Inpaint except Extend Image. Extend Image generates "
                            "the uncovered Fit-canvas mask and merges an optional input mask."
                        ),
                    },
                ),
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
                        "label": "ACTIVE LORAS",
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
                            "Extend Image uses Fit to create its outpaint canvas and mask, Navier-Stokes fill, "
                            "denoise 1.00, noise mask ON and context reference ON. "
                            "Guided modes override fill_masked_area."
                        ),
                    },
                ),
                # Appended after the legacy widget sequence so existing saved
                # workflows keep their positional widget values.
                "resize_mode": (
                    RESIZE_MODES,
                    {
                        "default": "Fit",
                        "tooltip": (
                            "Fit preserves the complete image and masks the uncovered canvas. "
                            "Crop fills the target without distortion. Stretch changes the aspect ratio."
                        ),
                    },
                ),
                "crop_position": (
                    CROP_POSITIONS,
                    {
                        "default": "Center",
                        "tooltip": "Anchors the image when Fit or Crop leaves an offset on one axis.",
                    },
                ),
                "outpaint_overlap": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "advanced": True,
                        "tooltip": (
                            "Grows an active Outpaint mask inward over the source image. "
                            "This overlap gives diffusion enough context to remove hard canvas seams."
                        ),
                    },
                ),
                "opt_prompt_pos": (
                    "STRING",
                    {
                        "forceInput": True,
                        "label": "ADDITIONAL PROMPT",
                        "tooltip": (
                            "Optional additional positive prompt. When connected, "
                            "it is appended after PROMPT POS."
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
        prompt_pos_primary = inputs.get("PROMPT POS", "") or ""
        opt_prompt_pos = inputs.get("opt_prompt_pos", "") or ""
        prompt_pos = "\n".join(
            part for part in (
                str(prompt_pos_primary).strip(),
                str(opt_prompt_pos).strip(),
            )
            if part
        )
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
        resize_mode = inputs.get("resize_mode", "Fit")
        crop_position = inputs.get("crop_position", "Center")
        upscale_method = inputs.get("upscale_method", "lanczos")
        device = inputs.get("device", "cpu")
        outpaint_on = inputs.get("outpaint_on", False)
        outpaint_overlap = max(0, min(256, int(inputs.get("outpaint_overlap", 32) or 0)))
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
        requested_resize_mode = resize_mode
        if bool(INPAINT_MODE) and inpaint_process_mode == "extend":
            # Extending requires uncovered canvas. Make the guided preset a
            # complete outpainting operation even when an older workflow has
            # Crop or Stretch serialized.
            resize_mode = "Fit"
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
        elif bool(INPAINT_MODE) and inpaint_process_mode == "extend":
            outpaint_on = True

        if bool(INPAINT_MODE):
            missing = []
            if image is None:
                missing.append("IMAGE")
            # Extend Image creates its outpaint mask from the Fit canvas.
            # An optional input mask is transformed and merged with it.
            if mask is None and inpaint_process_mode != "extend":
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
        image_resized, mask_process, uncovered_mask = prepare_image_and_mask(
            image,
            mask,
            width,
            height,
            upscale_method,
            resize_mode=resize_mode,
            crop_position=crop_position,
        )
        if bool(INPAINT_MODE) and uncovered_mask is not None:
            mask_process = (
                uncovered_mask
                if mask_process is None
                else mask_process.to(
                    device=uncovered_mask.device,
                    dtype=uncovered_mask.dtype,
                ).maximum(uncovered_mask)
            )
        if bool(INPAINT_MODE) and bool(mask_fill_holes):
            mask_process = fill_mask_holes(mask_process)
        if bool(INPAINT_MODE) and bool(outpaint_on):
            mask_process = expand_mask_tensor(mask_process, outpaint_overlap)
        fill_mask_process = mask_process
        if (
            bool(INPAINT_MODE)
            and bool(outpaint_on)
            and effective_fill_mode in {"noise", "neutral", "black", "white"}
        ):
            # Keep the authoritative generation mask fully expanded, but
            # cross-fade synthetic fills inside that overlap. Otherwise their
            # binary edge remains visible even though diffusion has context.
            fill_mask_process = feather_mask_tensor(
                mask_process,
                min(32, max(1, outpaint_overlap // 2)),
            )
        image_out = (
            apply_mask_fill(image_resized, fill_mask_process, effective_fill_mode, seed=0)
            if bool(INPAINT_MODE)
            else image_resized
        )

        pipe = dict(incoming_process) if isinstance(incoming_process, dict) else {}
        pipe.update({
            "mask": mask_process,
            "mask_original": mask,
            "mask_fill": fill_mask_process,
            "width": width,
            "height": height,
            "source_width": source_width,
            "source_height": source_height,
            "target_width": width,
            "target_height": height,
            "resolution": resolution,
            "swap_dimensions": swap_dimensions,
            "resize_mode": resize_mode,
            "requested_resize_mode": requested_resize_mode,
            "crop_position": crop_position,
            "uncovered_mask": uncovered_mask,
            "upscale_method": upscale_method,
            "device": device,
            "outpaint_on": outpaint_on,
            "outpaint_overlap": outpaint_overlap,
            "mask_fill_holes": mask_fill_holes,
            "fill_masked_area": effective_fill_mode,
            "mask_fill_applied": bool(INPAINT_MODE and image_out is not None and mask_process is not None),
            "mask_fill_seed": 0,
            "filename_string": filename_string,
            "file_name": filename_string,
            "prompt_pos": prompt_pos,
            "prompt_pos_primary": prompt_pos_primary,
            "opt_prompt_pos": opt_prompt_pos,
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
            outpaint_overlap=outpaint_overlap,
            swap_dimensions=swap_dimensions,
            resize_mode=resize_mode,
            crop_position=crop_position,
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
                "outpaint_overlap": outpaint_overlap,
                "swap_dimensions": bool(swap_dimensions),
                "resize_mode": resize_mode,
                "crop_position": crop_position,
                "upscale_method": upscale_method,
                "device": device,
            },
        )

        result = (pipe, image_out, log_pipe, diagnostic)
        preview_ui = image_node_preview(image_out)
        if preview_ui is None:
            return result
        return {"ui": preview_ui, "result": result}


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
