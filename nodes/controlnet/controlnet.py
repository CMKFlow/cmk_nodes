import numpy as np
import torch

from ...utils.cmk_diagnostic import make_diagnostic_payload
from ...pipe.cmk_log_pipe import cmk_add_block, cmk_bool
from ...pipe.cmk_pipe_image import mask_to_preview_rgb


CONTROLNET_PREPROCESSORS_FALLBACK = [
    "none",
    "canny",
    "lineart",
    "lineart_coarse",
    "lineart_realistic",
    "softedge_hed",
    "softedge_hedsafe",
    "softedge_pidinet",
    "depth_midas",
    "depth_zoe",
    "openpose",
    "dwpose",
    "normalbae",
    "tile",
    "scribble",
    "mlsd",
    "shuffle",
]


CONTROLNET_IMAGE_SOURCES = [
    "Base Image",
    "Reference Image",
]


def _get_input_files():
    """Return image files from ComfyUI's input directory.

    This deliberately does NOT use the LoadImage `image_upload` widget flag.
    That flag gives the convenient raw-image frontend preview, but CMK
    ControlNet Prepare should preview only the processed ControlNet Image.
    """
    try:
        import os
        import folder_paths
    except Exception:
        return []

    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

    try:
        input_dir = folder_paths.get_input_directory()
    except Exception:
        input_dir = None

    if input_dir and os.path.isdir(input_dir):
        files = []
        try:
            for root, _dirs, filenames in os.walk(input_dir):
                for filename in filenames:
                    if os.path.splitext(filename)[1].lower() not in image_exts:
                        continue
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, input_dir).replace(os.sep, "/")
                    files.append(rel_path)
            return sorted(files, key=str.lower)
        except Exception:
            pass

    try:
        files = folder_paths.get_filename_list("input")
        return sorted(
            [str(f) for f in files if os.path.splitext(str(f))[1].lower() in image_exts],
            key=str.lower,
        )
    except Exception:
        return []


def _get_controlnet_models():
    try:
        import folder_paths
        return folder_paths.get_filename_list("controlnet")
    except Exception:
        return []


def _load_image_from_input(filename):
    if not filename:
        return None

    try:
        import folder_paths
        import numpy as np
        import torch
        from PIL import Image, ImageOps, ImageSequence
    except Exception:
        return None

    try:
        image_path = folder_paths.get_annotated_filepath(filename)
        img = Image.open(image_path)
    except Exception:
        return None

    output_images = []
    try:
        for frame in ImageSequence.Iterator(img):
            frame = ImageOps.exif_transpose(frame)
            if frame.mode == "I":
                frame = frame.point(lambda i: i * (1 / 255))
            image = frame.convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(image_np)[None,])
    except Exception:
        return None

    if not output_images:
        return None

    if len(output_images) == 1:
        return output_images[0]

    try:
        return torch.cat(output_images, dim=0)
    except Exception:
        return output_images[0]


def _apply_mask_to_image(image, mask):
    if image is None or mask is None:
        return image

    try:
        import torch
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1)
        mask = mask.to(device=image.device, dtype=image.dtype)
        if mask.shape[0] == 1 and image.shape[0] > 1:
            mask = mask.repeat(image.shape[0], 1, 1, 1)
        return image * mask
    except Exception:
        return image


def _get_aio_preprocessors():
    return CONTROLNET_PREPROCESSORS_FALLBACK


def _controlnet_preprocessor_input():
    preprocessors = _get_aio_preprocessors()
    default = "LineArtPreprocessor"

    if default in preprocessors:
        return (preprocessors, {"default": default})

    for candidate in preprocessors:
        if str(candidate).lower() == default.lower():
            return (preprocessors, {"default": candidate})

    # Defensive fallback: keep the requested CMK default available even when
    # ControlNet Aux is not loaded while ComfyUI scans custom nodes.
    return ([default] + [p for p in preprocessors if p != default], {"default": default})


def _run_aio_preprocessor(image, preprocessor, resolution, extra_kwargs):
    del resolution, extra_kwargs
    if image is None:
        return None, "CMK preprocessor bypass | image missing"
    if not preprocessor or str(preprocessor).lower() == "none":
        return image, "CMK preprocessor bypass | preprocessor=none"
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        return image, "CMK preprocessor bypass | invalid IMAGE tensor"

    mode = str(preprocessor).strip().lower()
    outputs = []
    try:
        import cv2

        for item in image:
            rgb = np.clip(item.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if mode == "canny":
                processed = cv2.Canny(gray, 100, 200)
            elif mode in {"tile", "shuffle"}:
                processed = rgb
            elif mode.startswith("depth"):
                processed = 255 - cv2.GaussianBlur(gray, (0, 0), 5.0)
            elif mode == "normalbae":
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                nz = np.full_like(gx, 255.0)
                normal = np.stack((-gx, -gy, nz), axis=-1)
                normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
                processed = np.clip((normal * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            else:
                smooth = cv2.bilateralFilter(gray, 7, 40, 40)
                processed = cv2.Canny(smooth, 48, 144)
                if mode == "scribble":
                    processed = np.where(processed > 0, 255, 0).astype(np.uint8)
            if processed.ndim == 2:
                processed = np.repeat(processed[..., None], 3, axis=2)
            outputs.append(torch.from_numpy(processed.astype(np.float32) / 255.0).unsqueeze(0))
        result = torch.cat(outputs, dim=0).to(device=image.device, dtype=image.dtype)
        return result, f"CMK native preprocessor applied | {preprocessor}"
    except Exception as exc:
        return image, f"CMK preprocessor bypass | failed: {exc}"


def _load_controlnet_model(controlnet_model):
    if not controlnet_model:
        return None, "ControlNet model missing"

    try:
        from nodes import ControlNetLoader
        result = ControlNetLoader().load_controlnet(controlnet_model)
        return result[0], f"ControlNet loaded | {controlnet_model}"
    except Exception as exc:
        return None, f"ControlNet load failed | {exc}"


def _tensor_image_to_temp_ui(image, prefix="cmk_controlnet_image"):
    """Save first IMAGE tensor batch item to ComfyUI temp and return UI image metadata."""
    if image is None:
        return []
    try:
        import os
        import hashlib
        import numpy as np
        from PIL import Image
        import folder_paths

        img = image
        if hasattr(img, "detach"):
            img = img.detach().cpu()
        if hasattr(img, "numpy"):
            arr = img.numpy()
        else:
            arr = np.asarray(img)

        # Comfy IMAGE is usually BHWC float 0..1.
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            # Defensive CHW -> HWC conversion for non-standard preprocessors.
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.ndim != 3:
            return []
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 4:
            pil = Image.fromarray(arr, mode="RGBA")
        else:
            pil = Image.fromarray(arr[..., :3])

        digest = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
        filename = f"{prefix}_{digest}.png"
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        pil.save(os.path.join(temp_dir, filename))
        return [{"filename": filename, "subfolder": "", "type": "temp"}]
    except Exception:
        return []


def _empty_controlnet_ui_placeholder():
    """Return a transparent temp image used only to clear/neutralize the UI preview.

    The result outputs still stay None when ControlNet is disabled or when no
    processed ControlNet Image exists. This placeholder is deliberately not a
    data output; it is only UI metadata, so downstream nodes cannot accidentally
    consume a dummy image.
    """
    try:
        import os
        from PIL import Image
        import folder_paths

        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        filename = "cmk_controlnet_empty_preview.png"
        path = os.path.join(temp_dir, filename)
        if not os.path.exists(path):
            Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(path)
        return [{"filename": filename, "subfolder": "", "type": "temp"}]
    except Exception:
        return []


def _build_controlnet_log_lines(
    *,
    use_controlnet,
    controlnet_model,
    image_source,
    reference_image,
    apply_mask,
    preprocessor,
    resolution,
    enabled,
    log,
):
    if not enabled:
        return ["STATUS          : DISABLED"]
    return [
        "STATUS          : PREPARED",
        f"MODEL           : {controlnet_model}",
        f"IMAGE SOURCE    : {image_source}",
        f"REFERENCE       : {reference_image or '-'}",
        f"APPLY MASK      : {cmk_bool(apply_mask)}",
        f"PREPROCESSOR    : {preprocessor}",
        f"RESOLUTION      : {resolution}",
    ]


def _build_controlnet_diagnostic(
    *,
    title,
    node,
    use_controlnet,
    controlnet_model,
    image_source,
    reference_image,
    apply_mask,
    preprocessor,
    resolution,
    base_image,
    mask,
    controlnet_image,
    enabled,
    log,
):
    log_lines = _build_controlnet_log_lines(
        use_controlnet=use_controlnet,
        controlnet_model=controlnet_model,
        image_source=image_source,
        reference_image=reference_image,
        apply_mask=apply_mask,
        preprocessor=preprocessor,
        resolution=resolution,
        enabled=enabled,
        log=log,
    )
    previews = []
    if base_image is not None:
        previews.append(base_image)
    mask_preview = mask_to_preview_rgb(mask) if mask is not None else None
    if mask_preview is not None:
        previews.append(mask_preview)
    if controlnet_image is not None:
        previews.append(controlnet_image)

    summary = "\n".join(log_lines)
    warnings = []
    if not enabled:
        warnings.append(str(log))

    return make_diagnostic_payload(
        title=title,
        node=node,
        previews=previews,
        summary=summary,
        details=summary,
        mode="ControlNet",
        metadata={
            "use_controlnet": bool(use_controlnet),
            "enabled": bool(enabled),
            "controlnet_model": controlnet_model,
            "image_source": image_source,
            "reference_image": reference_image,
            "apply_mask": bool(apply_mask),
            "preprocessor": preprocessor,
            "resolution": resolution,
        },
        warnings=warnings,
    )


class CMKControlNetPrepare:
    """Prepare a ControlNet model and its processed ControlNet Image.

    This node owns the whole ControlNet preparation step:
    - optional ControlNet activation
    - ControlNet model loading
    - base/reference image selection
    - optional mask application
    - optional AIO Aux preprocessing

    When USE CONTROLNET is False, the node does not load the model and does not
    run preprocessing. It returns None outputs so downstream CMK smart-bypass
    logic can treat the module as absent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_files = _get_input_files()
        if not input_files:
            input_files = [""]

        controlnet_models = _get_controlnet_models()
        if not controlnet_models:
            controlnet_models = [""]

        return {
            "required": {
                "USE CONTROLNET": ("BOOLEAN", {"default": False}),
                "controlnet_model": (controlnet_models,),
                "image_source": (CONTROLNET_IMAGE_SOURCES, {"default": "Base Image"}),
                "reference_image": ("STRING", {"default": ""}),
                "apply_mask": ("BOOLEAN", {"default": False}),
                "preprocessor": _controlnet_preprocessor_input(),
                "resolution": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
            },
            "optional": {
                "opt_base_image": ("IMAGE", {"forceInput": True}),
                "opt_mask": ("MASK", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CONTROL_NET", "IMAGE", "BOOLEAN", "STRING", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("control_net", "controlnet_image", "controlnet_enabled", "controlnet_prepare_log", "diagnostic")
    FUNCTION = "prepare_controlnet"
    CATEGORY = "CMK/Toolbox/ControlNet"
    OUTPUT_NODE = True

    @staticmethod
    def _result(control_net, controlnet_image, enabled, log, diagnostic):
        if enabled and controlnet_image is not None:
            ui_images = _tensor_image_to_temp_ui(controlnet_image)
        else:
            ui_images = _empty_controlnet_ui_placeholder()

        return {
            "ui": {"images": ui_images},
            "result": (control_net, controlnet_image, enabled, log, diagnostic),
        }

    @staticmethod
    def _prepare_core(
        use_controlnet,
        controlnet_model,
        image_source,
        reference_image,
        apply_mask,
        preprocessor,
        resolution,
        base_image=None,
        mask=None,
    ):
        if not bool(use_controlnet):
            return None, None, False, "ControlNet disabled | USE CONTROLNET=False"

        if image_source == "Base Image":
            source_image = base_image
            source_log = "Base Image"
        else:
            source_image = _load_image_from_input(reference_image)
            source_log = f"Reference Image | {reference_image}"

        if source_image is None:
            return None, None, False, f"ControlNet bypass | image missing | {source_log}"

        if bool(apply_mask):
            source_image = _apply_mask_to_image(source_image, mask)
            mask_log = "Mask applied" if mask is not None else "Mask requested but missing"
        else:
            mask_log = "Mask not applied"

        controlnet_image, aio_log = _run_aio_preprocessor(
            source_image,
            preprocessor,
            resolution,
            extra_kwargs={},
        )

        control_net, model_log = _load_controlnet_model(controlnet_model)
        if control_net is None:
            return (
                None,
                controlnet_image,
                False,
                f"ControlNet bypass | {model_log} | {source_log} | {mask_log} | {aio_log}",
            )

        log = (
            f"ControlNet prepared | model={controlnet_model} | source={source_log} | "
            f"{mask_log} | {aio_log}"
        )
        return control_net, controlnet_image, True, log

    def prepare_controlnet(
        self,
        controlnet_model,
        image_source,
        reference_image,
        apply_mask,
        preprocessor,
        resolution,
        opt_base_image=None,
        opt_mask=None,
        **kwargs,
    ):
        use_controlnet = kwargs.get("USE CONTROLNET", False)
        control_net, controlnet_image, enabled, log = self._prepare_core(
            use_controlnet,
            controlnet_model,
            image_source,
            reference_image,
            apply_mask,
            preprocessor,
            resolution,
            base_image=opt_base_image,
            mask=opt_mask,
        )

        if image_source == "Base Image":
            diagnostic_source_image = opt_base_image
        else:
            diagnostic_source_image = _load_image_from_input(reference_image)

        diagnostic = _build_controlnet_diagnostic(
            title="ControlNet Prepare",
            node="CMK ControlNet Prepare",
            use_controlnet=use_controlnet,
            controlnet_model=controlnet_model,
            image_source=image_source,
            reference_image=reference_image,
            apply_mask=apply_mask,
            preprocessor=preprocessor,
            resolution=resolution,
            base_image=diagnostic_source_image,
            mask=opt_mask,
            controlnet_image=controlnet_image,
            enabled=enabled,
            log=log,
        )

        return self._result(control_net, controlnet_image, enabled, log, diagnostic)


class CMKPipeSetControlNet:
    """Store ControlNet references in the CMK pipe without touching sampler state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("CMK_PIPE",),
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CMK_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "set_controlnet"
    CATEGORY = 'CMK/Developer/Pipe/Set'

    def set_controlnet(self, pipe, control_net, image):
        new_pipe = dict(pipe)
        new_pipe["control_net"] = control_net
        new_pipe["controlnet_image"] = image
        return (new_pipe,)


class CMKSmartPipeApplyControlNet:
    """Legacy standalone ControlNet applier.

    New workflows should prefer CMK Pipe Set Sampler, which contains the same
    smart ControlNet bypass logic directly inside the sampler setup.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("CMK_PIPE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "opt_control_net": ("CONTROL_NET", {"forceInput": True}),
                "opt_controlnet_image": ("IMAGE", {"forceInput": True}),
                "opt_vae": ("VAE", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CMK_PIPE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("pipe", "controlnet_applied", "controlnet_log")
    FUNCTION = "apply_controlnet"
    CATEGORY = "CMK/Developer/Legacy"

    @staticmethod
    def _bypass(pipe, reason):
        new_pipe = dict(pipe)
        new_pipe["boolean_controlnet_enable"] = False
        new_pipe["controlnet_log"] = f"ControlNet bypass | {reason}"
        return (new_pipe, False, new_pipe["controlnet_log"])

    def apply_controlnet(
        self,
        pipe,
        strength,
        start_percent,
        end_percent,
        opt_control_net=None,
        opt_controlnet_image=None,
        opt_vae=None,
    ):
        missing = []
        if opt_control_net is None:
            missing.append("opt_control_net")
        if opt_controlnet_image is None:
            missing.append("opt_controlnet_image")
        if opt_vae is None:
            missing.append("opt_vae")

        if missing:
            return self._bypass(pipe, "missing " + ", ".join(missing))

        conditioning_pos = pipe.get("conditioning_pos")
        conditioning_neg = pipe.get("conditioning_neg")

        if conditioning_pos is None:
            return self._bypass(pipe, "pipe conditioning_pos missing")
        if conditioning_neg is None:
            return self._bypass(pipe, "pipe conditioning_neg missing")

        if strength <= 0.0:
            return self._bypass(pipe, "strength <= 0")

        if end_percent <= start_percent:
            return self._bypass(pipe, "end_percent <= start_percent")

        try:
            from nodes import ControlNetApplyAdvanced
        except Exception as exc:
            return self._bypass(pipe, f"ControlNetApplyAdvanced unavailable: {exc}")

        try:
            result = ControlNetApplyAdvanced().apply_controlnet(
                conditioning_pos,
                conditioning_neg,
                opt_control_net,
                opt_controlnet_image,
                strength,
                start_percent,
                end_percent,
                opt_vae,
            )
        except TypeError:
            result = ControlNetApplyAdvanced().apply_controlnet(
                conditioning_pos,
                conditioning_neg,
                opt_control_net,
                opt_controlnet_image,
                strength,
                start_percent,
                end_percent,
            )

        new_pos, new_neg = result[0], result[1]

        new_pipe = dict(pipe)
        new_pipe["conditioning_pos"] = new_pos
        new_pipe["conditioning_neg"] = new_neg
        new_pipe["boolean_controlnet_enable"] = True
        new_pipe["controlnet_strength"] = strength
        new_pipe["controlnet_start_percent"] = start_percent
        new_pipe["controlnet_end_percent"] = end_percent
        new_pipe["controlnet_log"] = (
            f"ControlNet applied | strength={strength:.3f} | "
            f"start={start_percent:.3f} | end={end_percent:.3f}"
        )

        return (new_pipe, True, new_pipe["controlnet_log"])
