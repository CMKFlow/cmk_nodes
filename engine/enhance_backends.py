from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from ..models.model_manager import resolve_face_restore_model


def _providers() -> List[str]:
    # Stable default for macOS/Apple Silicon and CPU-only ComfyUI installs.
    return ["CPUExecutionProvider"]


def _clip_rgb(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


@lru_cache(maxsize=2)
def _get_session(model_name: str):
    import onnxruntime as ort

    model_path = resolve_face_restore_model(model_name)
    return ort.InferenceSession(model_path, providers=_providers())


def _first_name(items) -> str:
    return str(items[0].name)


def _model_size(session, fallback: int = 512) -> int:
    try:
        shape = session.get_inputs()[0].shape
        h = shape[2]
        w = shape[3]
        if isinstance(h, int) and isinstance(w, int) and h == w and h > 0:
            return int(h)
    except Exception:
        pass
    return int(fallback)


def _to_nchw_minus1_to_1(image_rgb: np.ndarray) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    return np.transpose(image, (2, 0, 1))[None, ...].astype(np.float32)


def _from_model_output(output: np.ndarray) -> np.ndarray | None:
    pred = np.asarray(output)
    if pred.ndim == 4:
        pred = pred[0]
    if pred.ndim == 3 and pred.shape[0] in (1, 3):
        pred = np.transpose(pred, (1, 2, 0))
    if pred.ndim != 3:
        return None
    if pred.shape[2] == 1:
        pred = np.repeat(pred, 3, axis=2)
    elif pred.shape[2] > 3:
        pred = pred[:, :, :3]

    pred_f = pred.astype(np.float32)
    finite = np.isfinite(pred_f)
    if not bool(finite.all()):
        pred_f = np.where(finite, pred_f, 0.0)

    pmin = float(np.min(pred_f))
    pmax = float(np.max(pred_f))

    # Face-restore ONNX variants are usually either [-1, 1], [0, 1]
    # or already [0, 255]. Handle all three without exposing a UI switch.
    if pmin >= -1.5 and pmax <= 1.5 and pmin < -0.05:
        pred_f = (np.clip(pred_f, -1.0, 1.0) + 1.0) * 127.5
    elif pmin >= -0.05 and pmax <= 1.5:
        pred_f = np.clip(pred_f, 0.0, 1.0) * 255.0
    else:
        pred_f = np.clip(pred_f, 0.0, 255.0)

    return _clip_rgb(pred_f)


class GPENEnhancerBackend:
    """Internal GPEN aligned-face enhancer."""

    name = "gpen_bfr_512"
    model_name = "GPEN-BFR-512.onnx"

    def enhance_aligned(self, aligned_rgb: np.ndarray) -> np.ndarray:
        aligned = _clip_rgb(aligned_rgb)
        if aligned.ndim != 3 or aligned.shape[2] != 3:
            return aligned

        original_height, original_width = aligned.shape[:2]

        try:
            import cv2

            session = _get_session(self.model_name)
            input_name = _first_name(session.get_inputs())
            output_name = _first_name(session.get_outputs())
            size = _model_size(session, fallback=512)

            if aligned.shape[0] != size or aligned.shape[1] != size:
                model_input = cv2.resize(aligned, (size, size), interpolation=cv2.INTER_LANCZOS4)
            else:
                model_input = aligned

            blob = _to_nchw_minus1_to_1(model_input)
            pred = session.run([output_name], {input_name: blob})[0]
            enhanced = _from_model_output(pred)
            if enhanced is None:
                return aligned

            if enhanced.shape[0] != size or enhanced.shape[1] != size:
                enhanced = cv2.resize(enhanced, (size, size), interpolation=cv2.INTER_LANCZOS4)

            model_f = model_input.astype(np.float32)
            enh_f = enhanced.astype(np.float32)

            # GPEN is trained as a generative face restorer. On a 128px
            # INSwapper patch it may invent high-contrast details at 512px
            # (glasses, eyelashes, teeth) which become staircase artefacts when
            # projected back. Keep GPEN conservative and protect structures
            # already present in the swap result.
            gray = cv2.cvtColor(model_input, cv2.COLOR_RGB2GRAY).astype(np.float32)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            gradient = cv2.magnitude(grad_x, grad_y)
            edge_protection = np.clip((gradient - 18.0) / 72.0, 0.0, 1.0)
            edge_protection = cv2.GaussianBlur(
                edge_protection,
                (0, 0),
                sigmaX=1.2,
                sigmaY=1.2,
            )[..., None]

            # In smooth skin regions GPEN may contribute up to 42%; at strong
            # edges the contribution falls to 12%. This preserves glasses,
            # eyebrows, eye contours and teeth instead of redrawing them.
            gpen_weight = 0.42 * (1.0 - edge_protection) + 0.12 * edge_protection
            mixed_f = model_f * (1.0 - gpen_weight) + enh_f * gpen_weight

            # Suppress only newly generated excessive high-frequency detail.
            # The limiter compares the mixed result with its local low-pass
            # version and caps detail amplitude relative to the original patch.
            model_blur = cv2.GaussianBlur(model_f, (0, 0), 0.85)
            mixed_blur = cv2.GaussianBlur(mixed_f, (0, 0), 0.85)
            original_detail = model_f - model_blur
            mixed_detail = mixed_f - mixed_blur
            detail_limit = np.abs(original_detail) * 1.35 + 5.0
            limited_detail = np.clip(mixed_detail, -detail_limit, detail_limit)
            mixed = _clip_rgb(mixed_blur + limited_detail)

            # The affine matrix returned by INSwapper belongs to the original
            # aligned crop size (normally 128×128). INTER_AREA is required for
            # the 512→128 reduction; Lanczos can turn GPEN's synthetic edges
            # into ringing and saw-tooth contours.
            if mixed.shape[0] != original_height or mixed.shape[1] != original_width:
                downscaling = (
                    original_width < mixed.shape[1]
                    or original_height < mixed.shape[0]
                )
                mixed = cv2.resize(
                    mixed,
                    (original_width, original_height),
                    interpolation=(
                        cv2.INTER_AREA
                        if downscaling
                        else cv2.INTER_CUBIC
                    ),
                )
            return _clip_rgb(mixed)
        except Exception as exc:
            raise RuntimeError(f"GPEN enhancement failed: {exc}") from exc

    def enhance(self, *, image_rgb: np.ndarray, target_face) -> np.ndarray:
        return _clip_rgb(image_rgb)


_CODEFORMER_IMPORTS = (
    "basicsr.archs.codeformer_arch",
    "facelib.archs.codeformer_arch",
    "comfy_extras.chainner_models.codeformer_arch",
    "comfy_extras.chainner_models.codeformer",
)


@lru_cache(maxsize=1)
def _get_codeformer_class():
    import importlib

    errors: list[str] = []
    for module_name in _CODEFORMER_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            candidate = getattr(module, "CodeFormer", None)
            if candidate is not None:
                return candidate
            errors.append(f"{module_name}: CodeFormer class missing")
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    detail = "; ".join(errors[-2:]) if errors else "no compatible module found"
    raise ImportError(f"CodeFormer architecture not available: {detail}")


@lru_cache(maxsize=1)
def codeformer_availability() -> tuple[bool, str]:
    try:
        _get_codeformer_class()
        resolve_face_restore_model("codeformer-v0.1.0.pth")
        return True, "available"
    except Exception as exc:
        return False, str(exc)


def get_available_enhancer_modes() -> list[str]:
    modes = ["Off", "GPEN"]
    available, _ = codeformer_availability()
    if available:
        modes.append("CodeFormer")
    return modes


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


@lru_cache(maxsize=1)
def _get_codeformer_model():
    import torch

    cls = _get_codeformer_class()
    model_path = resolve_face_restore_model("codeformer-v0.1.0.pth")
    model = cls(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    )

    checkpoint = torch.load(model_path, map_location="cpu")
    state = _state_dict_from_checkpoint(checkpoint)
    if not isinstance(state, dict):
        raise RuntimeError("Invalid CodeFormer checkpoint.")

    model.load_state_dict(state, strict=False)
    model.eval()
    model.to("cpu")
    return model


class CodeFormerEnhancerBackend:
    """Internal CodeFormer aligned-face enhancer.

    It runs on the aligned face when the CodeFormer Python architecture
    and model are available. Missing or failing dependencies raise an explicit
    error before any result is returned.
    """

    name = "codeformer_v0_1_0"
    model_size = 512
    fidelity = 0.72

    def enhance_aligned(self, aligned_rgb: np.ndarray) -> np.ndarray:
        aligned = _clip_rgb(aligned_rgb)
        if aligned.ndim != 3 or aligned.shape[2] != 3:
            return aligned

        original_height, original_width = aligned.shape[:2]

        try:
            import cv2
            import torch

            if aligned.shape[0] != self.model_size or aligned.shape[1] != self.model_size:
                model_input = cv2.resize(
                    aligned,
                    (self.model_size, self.model_size),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            else:
                model_input = aligned

            tensor = _to_nchw_minus1_to_1(model_input)
            x = torch.from_numpy(tensor).to("cpu")
            model = _get_codeformer_model()

            with torch.no_grad():
                output = None
                call_variants = (
                    {"w": self.fidelity, "adain": True},
                    {"w": self.fidelity},
                    {"fidelity_weight": self.fidelity, "adain": True},
                    {"fidelity_weight": self.fidelity},
                    {},
                )
                last_exc: Exception | None = None
                for kwargs in call_variants:
                    try:
                        output = model(x, **kwargs)
                        break
                    except TypeError as exc:
                        last_exc = exc
                if output is None:
                    raise RuntimeError(f"Unsupported CodeFormer forward signature: {last_exc}")

            if isinstance(output, (tuple, list)):
                output = output[0]
            if hasattr(output, "detach"):
                output = output.detach().cpu().numpy()

            enhanced = _from_model_output(output)
            if enhanced is None:
                return aligned
            if enhanced.shape[0] != self.model_size or enhanced.shape[1] != self.model_size:
                enhanced = cv2.resize(
                    enhanced,
                    (self.model_size, self.model_size),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            base_f = model_input.astype(np.float32)
            enh_f = enhanced.astype(np.float32)
            # CodeFormer can alter identity more strongly than GPEN. Keep it
            # visible, but blend conservatively for swap stability.
            mixed = _clip_rgb(base_f * 0.28 + enh_f * 0.72)

            if mixed.shape[0] != original_height or mixed.shape[1] != original_width:
                mixed = cv2.resize(
                    mixed,
                    (original_width, original_height),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            return _clip_rgb(mixed)

        except Exception as exc:
            raise RuntimeError(f"CodeFormer enhancement failed: {exc}") from exc

    def enhance(self, *, image_rgb: np.ndarray, target_face) -> np.ndarray:
        return _clip_rgb(image_rgb)


class DefaultEnhancerBackend:
    """Internal enhancer chain.

    GPEN remains the robust first pass. CodeFormer is layered on top when the
    local ComfyUI environment exposes its architecture and checkpoint. No UI
    switch is exposed; failed enhancers fall back silently to the last valid
    patch.
    """

    name = "gpen_bfr_512_plus_codeformer"

    def __init__(self) -> None:
        self.gpen = GPENEnhancerBackend()
        self.codeformer = CodeFormerEnhancerBackend()

    def enhance_aligned(self, aligned_rgb: np.ndarray) -> np.ndarray:
        out = self.gpen.enhance_aligned(aligned_rgb)
        out = self.codeformer.enhance_aligned(out)
        return _clip_rgb(out)

    def enhance(self, *, image_rgb: np.ndarray, target_face) -> np.ndarray:
        return _clip_rgb(image_rgb)


@lru_cache(maxsize=1)
def get_default_enhancer_backend() -> GPENEnhancerBackend:
    return GPENEnhancerBackend()


class NoEnhancerBackend:
    name = "off"

    def enhance_aligned(self, aligned_rgb: np.ndarray) -> np.ndarray:
        return _clip_rgb(aligned_rgb)

    def enhance(self, *, image_rgb: np.ndarray, target_face) -> np.ndarray:
        return _clip_rgb(image_rgb)


@lru_cache(maxsize=3)
def validate_enhancer_mode(mode: str) -> str:
    value = str(mode or "Off").strip().lower()
    if value == "off":
        return "Off"
    if value == "gpen":
        try:
            _get_session(GPENEnhancerBackend.model_name)
        except Exception as exc:
            raise RuntimeError(f"GPEN is unavailable: {exc}") from exc
        return "GPEN"
    if value == "codeformer":
        available, reason = codeformer_availability()
        if not available:
            raise RuntimeError(
                "CodeFormer is unavailable in this ComfyUI environment: "
                f"{reason}. Select GPEN or Off."
            )
        try:
            _get_codeformer_model()
        except Exception as exc:
            raise RuntimeError(f"CodeFormer is unavailable: {exc}") from exc
        return "CodeFormer"
    raise ValueError(f"Unsupported face enhancer: {mode}")


@lru_cache(maxsize=3)
def get_enhancer_backend(mode: str):
    canonical = validate_enhancer_mode(mode)
    if canonical == "Off":
        return NoEnhancerBackend()
    if canonical == "CodeFormer":
        return CodeFormerEnhancerBackend()
    return GPENEnhancerBackend()
