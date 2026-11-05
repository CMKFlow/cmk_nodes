"""Deterministic substitutions for expensive engine calls (tier T1 only).

Only engine-level callables *below* the node orchestration are replaced; node
classes (INPUT_TYPES, RETURN_TYPES, FUNCTION, check_lazy_status, IS_CHANGED,
NOT_IDEMPOTENT) are never touched.  Every seam keeps three separate counters:

    original_calls     – the original callable really ran
    substituted_calls  – the deterministic replacement ran
    stopped_calls      – execution reached the seam and was stopped on purpose

Lifecycle side effects that are not engine work (e.g. unload_all_models) are
counted separately in ``watched_calls`` and still run unchanged.

Tripwires guard every real model-loading entry point: if anything tries to
load model weights that no seam covers, the run fails with SeamStop instead
of loading a model.
"""

from __future__ import annotations

import functools
import inspect
import sys
from collections import defaultdict
from types import SimpleNamespace


class SeamStop(RuntimeError):
    """Raised when execution reaches a stop seam or a model-loading tripwire."""


class Seams:
    def __init__(self):
        self.original_calls = defaultdict(int)
        self.substituted_calls = defaultdict(int)
        self.stopped_calls = defaultdict(int)
        self.watched_calls = defaultdict(int)
        self.tripped = []
        self._restore = []

    # -- generic installation -------------------------------------------
    def _install(self, owner, name, wrapper_factory):
        raw = inspect.getattr_static(owner, name)
        target = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
        wrapper = functools.wraps(target)(wrapper_factory(target))
        replacement = wrapper
        if isinstance(raw, staticmethod):
            replacement = staticmethod(wrapper)
        elif isinstance(raw, classmethod):
            replacement = classmethod(wrapper)
        self._restore.append((owner, name, raw))
        setattr(owner, name, replacement)
        return target, replacement

    def _rebind_everywhere(self, original, replacement):
        """Replace a module-level function in every module that imported it."""
        for module in list(sys.modules.values()):
            namespace = getattr(module, "__dict__", None)
            if not namespace:
                continue
            for key, value in list(namespace.items()):
                if value is original:
                    self._restore.append((module, key, value))
                    setattr(module, key, replacement)

    def substitute(self, owner, name, seam, replacement, rebind=False):
        def factory(target):
            def wrapper(*args, **kwargs):
                self.substituted_calls[seam] += 1
                return replacement(*args, **kwargs)
            return wrapper
        target, installed = self._install(owner, name, factory)
        if rebind:
            self._rebind_everywhere(target, installed)

    def stop(self, owner, name, seam, rebind=False):
        def factory(target):
            def wrapper(*args, **kwargs):
                self.stopped_calls[seam] += 1
                raise SeamStop(f"stop seam reached: {seam}")
            return wrapper
        target, installed = self._install(owner, name, factory)
        if rebind:
            self._rebind_everywhere(target, installed)

    def observe(self, owner, name, seam, rebind=False):
        def factory(target):
            def wrapper(*args, **kwargs):
                self.original_calls[seam] += 1
                return target(*args, **kwargs)
            return wrapper
        target, installed = self._install(owner, name, factory)
        if rebind:
            self._rebind_everywhere(target, installed)

    def watch(self, owner, name, seam, rebind=False):
        def factory(target):
            def wrapper(*args, **kwargs):
                self.watched_calls[seam] += 1
                return target(*args, **kwargs)
            return wrapper
        target, installed = self._install(owner, name, factory)
        if rebind:
            self._rebind_everywhere(target, installed)

    def tripwire(self, owner, name, label):
        if owner is None or not hasattr(owner, name):
            return

        def factory(target):
            def wrapper(*args, **kwargs):
                self.tripped.append(label)
                raise SeamStop(f"model-loading tripwire: {label}")
            return wrapper
        self._install(owner, name, factory)

    def uninstall(self):
        while self._restore:
            owner, name, raw = self._restore.pop()
            setattr(owner, name, raw)

    def reset_counters(self):
        for counter in (self.original_calls, self.substituted_calls, self.stopped_calls, self.watched_calls):
            counter.clear()
        self.tripped.clear()

    def snapshot(self) -> dict:
        return {
            "original_calls": dict(self.original_calls),
            "substituted_calls": dict(self.substituted_calls),
            "stopped_calls": dict(self.stopped_calls),
            "watched_calls": dict(self.watched_calls),
            "tripped": list(self.tripped),
        }


# -- deterministic replacements -------------------------------------------
FAKE_MODEL_METADATA_KEYS = ("ckpt_name", "vae_name", "checkpoint_vae", "vae_source", "model_pipe_source")


class _FakeLatentFormat:
    """Stands in for model.model (BaseModel) where the Hybrid handoff converts x0."""

    def process_latent_out(self, latent):
        return latent


def fake_checkpoint_resources(ckpt_name, vae_name, checkpoint_vae):
    """Same shape as loader.checkpoint_vae_loader.load_checkpoint_vae_resources."""
    model = SimpleNamespace(cmk_s1_fake="model", cmk_s1_family="sdxl", ckpt_name=str(ckpt_name),
                            model=_FakeLatentFormat())
    clip = SimpleNamespace(cmk_s1_fake="clip", ckpt_name=str(ckpt_name))
    vae = SimpleNamespace(cmk_s1_fake="vae", ckpt_name=str(ckpt_name))
    metadata = {
        "ckpt_name": str(ckpt_name),
        "vae_name": "internal" if bool(checkpoint_vae) else str(vae_name),
        "checkpoint_vae": bool(checkpoint_vae),
        "vae_source": "checkpoint" if bool(checkpoint_vae) else "external",
        "model_pipe_source": "CMK Checkpoint VAE Loader -Pipe-",
    }
    return model, clip, vae, metadata


def _center_box(image_rgb):
    height, width = image_rgb.shape[:2]
    return width // 4, height // 4, (3 * width) // 4, (3 * height) // 4


FAKE_FACES = 1  # scenarios set 0 to model "no detection"


def fake_detect_image(self, image_rgb, settings):
    """One deterministic face in the image centre (shape of _face_to_dict)."""
    import numpy as np

    if FAKE_FACES == 0:
        return []

    x1, y1, x2, y2 = _center_box(image_rgb)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    kps = np.array([[cx - 20, cy - 20], [cx + 20, cy - 20], [cx, cy], [cx - 15, cy + 20], [cx + 15, cy + 20]], dtype=np.float32)
    raw = SimpleNamespace(bbox=np.array([x1, y1, x2, y2], dtype=np.float32), kps=kps, det_score=0.99)
    return [{"index": 0, "bbox": raw.bbox, "kps": kps, "det_score": 0.99, "raw": raw}]


def fake_swap_selected_with_mask(self, target_rgb, source_rgb, source_selected_face, target_selected_face, settings, *, content_guard_enabled=True):
    """Invert the centre box; return (result, paste_mask) like the real engine."""
    import numpy as np

    result = target_rgb.copy()
    x1, y1, x2, y2 = _center_box(target_rgb)
    result[y1:y2, x1:x2] = 255 - result[y1:y2, x1:x2]
    mask = np.zeros(target_rgb.shape[:2], dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return result, mask


class _FakeUpscaleModel:
    scale = 2


def fake_load_upscale_model(self, model_name):
    return _FakeUpscaleModel()


def fake_upscale(self, image, model):
    import torch

    scale = int(getattr(model, "scale", 2))
    out = image.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)
    return torch.clamp(out, 0.0, 1.0)


# -- standard seam sets ----------------------------------------------------
def install_tripwires(seams: Seams) -> None:
    import comfy.sd
    import comfy.utils

    seams.tripwire(comfy.utils, "load_torch_file", "comfy.utils.load_torch_file")
    seams.tripwire(comfy.sd, "load_checkpoint_guess_config", "comfy.sd.load_checkpoint_guess_config")
    seams.tripwire(comfy.sd, "load_diffusion_model", "comfy.sd.load_diffusion_model")
    try:
        import onnxruntime
        seams.tripwire(onnxruntime, "InferenceSession", "onnxruntime.InferenceSession")
    except ImportError:
        pass
    try:
        import insightface.app
        seams.tripwire(insightface.app, "FaceAnalysis", "insightface.app.FaceAnalysis")
    except ImportError:
        pass


def fake_sam_load(self, *args, **kwargs):
    return (SimpleNamespace(cmk_s1_fake="sam"),)


def fake_clip_set_last_layer(self, clip, stop_at_clip_layer):
    return clip


def fake_apply_single_lora(self, model, clip, lora_name, strength_model, strength_clip):
    return model, clip, ""


def fake_model_patch(self, model, *args, **kwargs):
    return model


def fake_clip_encode(clip, text):
    import torch

    return [[torch.zeros((1, 1, 8)), {"cmk_s1_fake_text": str(text)}]]


def fake_smart_detailer_run(self, image, **kwargs):
    """CMK_SmartDetailer.run without detector or sampler: one SEG in the
    image centre whose pixels are inverted (same tuple shape as the real run)."""
    import torch

    from cmk_s1_fake_detailer import CMKSEG, make_diagnostic_payload  # injected module, see install

    _batch, height, width, _ = image.shape
    x1, y1, x2, y2 = width // 4, height // 4, (3 * width) // 4, (3 * height) // 4
    result = image.clone()
    result[:, y1:y2, x1:x2, :] = 1.0 - result[:, y1:y2, x1:x2, :]
    seg = CMKSEG(
        cropped_image=result[:, y1:y2, x1:x2, :].clone(),
        cropped_mask=torch.ones((y2 - y1, x2 - x1)),
        confidence=0.99,
        crop_region=(x1, y1, x2, y2),
        bbox=(x1, y1, x2, y2),
        label="face",
    )
    segs = ((width, height), [seg])
    diagnostic = make_diagnostic_payload(title="Smart Detailer (S1 fake)", node="CMK Smart Detailer",
                                         previews=[image, result], summary="fake detailer run")
    return result, segs, segs, "fake detailer run", True, diagnostic


def fake_face_process_run(self, image, **kwargs):
    """CMK_FaceProcess.run without detector, restorer or sampler (D5, K5-B):
    one face SEG in the image centre, pixels inverted; parent tuple shape
    (image, image_proceed, segs_detected, segs_processed, log, enabled, diagnostic)."""
    result, segs, _segs, _log, _enabled, _diagnostic = fake_smart_detailer_run(self, image)
    from cmk_s1_fake_detailer import make_diagnostic_payload  # injected module, see install

    diagnostic = make_diagnostic_payload(title="FaceProcess (S1 fake)", node="CMK FaceProcess",
                                         previews=[image, result], summary="fake face process run")
    return result, result, segs, segs, "fake face process run", True, diagnostic


def fake_face_rebuild_run(self, VISUAL=None, unique_id=None, **values):
    """_rebuild of CMKInstantIDFaceRebuildSDXL / ...AdvancedSDXL without
    InstantID, detector or sampler (F4).  Keeps the contract: bypass mirrors the
    real early return; an active run resolves MODEL like the real
    CMKInstantIDFaceDetailerSDXL.detail (counted), inverts one band per active
    slot, appends a LOG block, registers the module VISUAL provider and returns
    a diagnostic."""
    from cmk_s1_fake_detailer import cmk_add_block, empty_visual, make_diagnostic_payload, register_provider, resolve

    model, process, image, log = values["MODEL"], values["PROCESS"], values["IMAGE"], values["LOG"]
    visual = empty_visual() if VISUAL is None else VISUAL
    active = [index for index in range(1, 4) if values.get(f"FACE {index} ENABLE", index == 1)]
    if not values["FACEREBUILD ENABLE"] or not active:
        diagnostic = make_diagnostic_payload(title="FaceRebuild (S1 fake) · Bypass", node="CMK FaceRebuild",
                                             previews=[image], mode="Bypass")
        return model, process, image, log, visual, diagnostic
    resolve(model)
    result = image.clone()
    _batch, height, width, _ = result.shape
    for slot in active:
        y1, y2 = height * slot // 8, height * (slot + 3) // 8
        result[:, y1:y2, width // 4:3 * width // 4, :] = 1.0 - result[:, y1:y2, width // 4:3 * width // 4, :]
    log = cmk_add_block(log, "FaceRebuild (S1 fake)", 25, [f"SLOTS : {active}"], True)
    diagnostic = make_diagnostic_payload(title="FaceRebuild (S1 fake)", node="CMK FaceRebuild",
                                         previews=[image, result], summary=f"slots {active}")
    visual = register_provider(visual, module_instance_id=unique_id, module_type=type(self).__name__,
                               module_label="FaceRebuild", sequence=25, channels={"before": image, "after": result},
                               status="completed", live_node_id=unique_id, branch="sdxl", stage_key="sdxl.facerebuild")
    return model, process, result, log, visual, diagnostic


def install_postprocess_seams(seams: Seams, runtime) -> None:
    """E9 (K-5) / D5 (K5-B) / F4: run the active Detailer, FaceProcess and
    FaceRebuild without weights.  The checkpoint is already substituted; Prepare's SAM load, CLIP
    layer, LoRA, model patches and CLIP encode and the compute runs (detector,
    restorer, sampler) are replaced by deterministic fakes.
    resolve_postprocess_model stays real and is counted (original_calls)."""
    import types

    native = runtime.cmk_module("engine.native_detailer")
    diagnostic = runtime.cmk_module("utils.cmk_diagnostic")
    module = types.ModuleType("cmk_s1_fake_detailer")
    module.CMKSEG = native.CMKSEG
    module.make_diagnostic_payload = diagnostic.make_diagnostic_payload
    module.cmk_add_block = runtime.cmk_module("pipe.cmk_log_pipe").cmk_add_block
    visual = runtime.cmk_module("pipe.cmk_visual")
    module.empty_visual, module.register_provider = visual.empty_visual, visual.register_provider
    module.resolve = lambda model: runtime.cmk_module("pipe.loaders.checkpoint_vae_loader").resolve_postprocess_model(model)
    sys.modules["cmk_s1_fake_detailer"] = module

    seams.substitute(native.CMKSAMLoader, "load", "prepare.sam_load", fake_sam_load)
    helper = runtime.cmk_module("pipe.cmk_sampler_prepare").CMKSamplerPrepareSDXLPipe
    seams.substitute(helper, "_clip_set_last_layer", "prepare.clip_layer", fake_clip_set_last_layer)
    seams.substitute(helper, "_apply_single_lora", "prepare.lora", fake_apply_single_lora)
    for name in ("_apply_pag", "_apply_sampling", "_apply_freeu"):
        seams.substitute(helper, name, "prepare.model_patch", fake_model_patch)
    prepare = runtime.cmk_module("pipe.cmk_detailer_prepare").CMKDetailerPreparePipe
    seams.substitute(prepare, "_encode", "prepare.clip_encode", fake_clip_encode)
    detailer = runtime.cmk_module("nodes.image.smart_detailer").CMK_SmartDetailer
    seams.substitute(detailer, "run", "detailer.run", fake_smart_detailer_run)
    face_prepare = runtime.cmk_module("pipe.cmk_faceprocess_prepare").CMKFaceProcessPreparePipe
    seams.substitute(face_prepare, "_encode", "prepare.clip_encode", fake_clip_encode)
    face_process = runtime.cmk_module("nodes.swap.face_process").CMK_FaceProcess
    seams.substitute(face_process, "run", "faceprocess.run", fake_face_process_run)
    face_rebuild = runtime.cmk_module("nodes.instantid_face_rebuild_advanced")
    for cls in (face_rebuild.CMKInstantIDFaceRebuildSDXL, face_rebuild.CMKInstantIDFaceRebuildAdvancedSDXL):
        seams.substitute(cls, "_rebuild", "facerebuild.run", fake_face_rebuild_run)


# -- generation (START HERE -> SDXL first pass -> Hybrid ZIT) ---------------
# Sampling calls per family of the run in progress (reset by reset_generation_calls).
GENERATION_CALLS = {"sdxl": 0, "zit": 0}


def reset_generation_calls() -> None:
    GENERATION_CALLS.update({"sdxl": 0, "zit": 0})


def _fake_conditioning(text=""):
    import torch

    return [[torch.zeros((1, 1, 8)), {"cmk_s1_fake_text": str(text)[:40]}]]


def _fake_latent(channels, width, height, batch=1):
    import torch

    return {"samples": torch.zeros((max(1, int(batch)), channels, max(1, int(height) // 8), max(1, int(width) // 8)))}


def fake_vae_decode(self, vae, samples):
    """Deterministic latent -> IMAGE (B, 8h, 8w, 3) without a VAE."""
    import torch

    latent = samples["samples"] if isinstance(samples, dict) else samples
    value = torch.sigmoid(latent.float().mean(dim=1, keepdim=True))
    image = value.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
    return (image.permute(0, 2, 3, 1).repeat(1, 1, 1, 3).contiguous(),)


def fake_vae_encode(vae, image, channels=16):
    import torch

    pixels = image.float().mean(dim=-1)[:, None, ::8, ::8]
    return {"samples": pixels.repeat(1, channels, 1, 1)}


def _fake_model(family, name):
    return SimpleNamespace(cmk_s1_fake="model", cmk_s1_family=family, ckpt_name=str(name), model=_FakeLatentFormat())


def fake_call_node(names, *args):
    """cmk_sampler_prepare._call_node without weights: CLIP, latents, VAE and
    model patches of the SDXL and ZIT generation paths."""
    name = names[0]
    if name in {"CLIPTextEncodeSDXL", "CLIPTextEncode"}:
        return (_fake_conditioning(args[-1] if args else ""),)
    if name in {"ConditioningZeroOut", "ConditioningCombine", "ConditioningSetAreaPercentage",
                "ConditioningSetTimestepRange"}:
        return (args[0],)
    if name == "EmptyLatentImage":
        return (_fake_latent(4, args[0], args[1], args[2] if len(args) > 2 else 1),)
    if name == "EmptySD3LatentImage":
        return (_fake_latent(16, args[0], args[1], args[2] if len(args) > 2 else 1),)
    if name == "VAEDecode":
        return fake_vae_decode(None, args[0], args[1])
    if name == "VAEEncode":
        return (fake_vae_encode(args[0], args[1]),)
    if name == "SetLatentNoiseMask":
        return (args[0],)
    if name in {"CLIPSetLastLayer", "PerturbedAttentionGuidance", "ModelSamplingDiscrete", "FreeU_V2",
                "ModelSamplingAuraFlow"}:
        return (args[0],)
    raise RuntimeError(f"S1 generation seam: unexpected ComfyUI node {name}")


def fake_call_node_kwargs(names, **kwargs):
    name = names[0]
    if name == "UNETLoader":
        return (_fake_model("zit", kwargs.get("unet_name", "zit")),)
    if name == "CLIPLoader":
        return (SimpleNamespace(cmk_s1_fake="clip", ckpt_name=str(kwargs.get("clip_name", "clip"))),)
    if name == "VAELoader":
        return (SimpleNamespace(cmk_s1_fake="vae", ckpt_name=str(kwargs.get("vae_name", "vae"))),)
    if name in {"CLIPTextEncode", "CLIPTextEncodeSDXL"}:
        return (_fake_conditioning(kwargs.get("text", "")),)
    raise RuntimeError(f"S1 generation seam: unexpected ComfyUI node {name} {sorted(kwargs)}")


def _family(model):
    return getattr(model, "cmk_s1_family", "sdxl")


def fake_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                denoise=1.0, callback=None, seed=None, **kwargs):
    """comfy.sample.sample: counts per family, returns a seed-determined latent
    and feeds it as clean x0 to the preview callback (Hybrid handoff)."""
    import torch

    GENERATION_CALLS[_family(model)] += 1
    generator = torch.Generator().manual_seed(int(seed or 0))
    result = torch.randn(latent_image.shape, generator=generator)
    if callback is not None:
        last = int(kwargs.get("last_step") or steps)
        callback(last - 1, result, result, steps)
    return result


def fake_ksampler_sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                         latent_image, denoise=1.0):
    latent = latent_image["samples"]
    samples = dict(latent_image)
    samples["samples"] = fake_sample(model, None, steps, cfg, sampler_name, scheduler, positive, negative,
                                     latent, denoise=denoise, seed=seed)
    return (samples,)


def fake_prepare_callback(model, steps, x0_output_dict=None):
    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0
    return callback


def install_generation_seams(seams: Seams, runtime) -> None:
    """Run START HERE -> SDXL first pass -> Hybrid ZIT without weights.  The
    nodes and their orchestration stay real; ComfyUI node calls, sampling,
    latent preview and VAE decode are deterministic fakes.  Sampling calls are
    counted per family in GENERATION_CALLS."""
    import comfy.sample
    import latent_preview
    import nodes

    prepare = runtime.cmk_module("pipe.cmk_sampler_prepare")
    seams.substitute(prepare, "_call_node", "generation.call_node", fake_call_node, rebind=True)
    seams.substitute(prepare, "_call_node_kwargs", "generation.call_node_kwargs", fake_call_node_kwargs, rebind=True)
    seams.substitute(comfy.sample, "fix_empty_latent_channels", "generation.fix_latent",
                     lambda model, latent, *args, **kwargs: latent)
    seams.substitute(comfy.sample, "sample", "generation.sample", fake_sample)
    seams.substitute(latent_preview, "prepare_callback", "generation.preview_callback", fake_prepare_callback)
    seams.substitute(nodes.KSampler, "sample", "generation.ksampler", fake_ksampler_sample)
    seams.substitute(nodes.VAEDecode, "decode", "generation.vae_decode", fake_vae_decode)
    loader = runtime.cmk_module("pipe.loaders.checkpoint_vae_loader")
    seams.substitute(loader, "_load_checkpoint_clip", "checkpoint.load_clip",
                     lambda ckpt_name: SimpleNamespace(cmk_s1_fake="clip", ckpt_name=str(ckpt_name)), rebind=True)


def install_standard(seams: Seams, runtime, *, stop_prepare: bool = True, fake_postprocess: bool = False) -> None:
    """Substitute checkpoint, FaceSwap and upscaler engines; stop Prepare's
    model materialisation and SAM loading (CLIP/LoRA cannot be faked
    meaningfully without real weights) -- unless ``fake_postprocess`` installs
    the model-free Detailer / FaceProcess seams (E9, D5)."""
    leaf = runtime.cmk_module("loader.checkpoint_vae_loader")
    seams.substitute(leaf, "load_checkpoint_vae_resources", "checkpoint.load_resources",
                     fake_checkpoint_resources, rebind=True)

    detector = runtime.cmk_module("engine.detector_engine")
    seams.substitute(detector.CMKDetectorEngine, "detect_image", "faceswap.detect", fake_detect_image)
    swap = runtime.cmk_module("engine.swap_selected_engine")
    seams.substitute(swap.CMKSelectedSwapEngine, "swap_selected_with_mask", "faceswap.swap",
                     fake_swap_selected_with_mask)

    upscaler = runtime.cmk_module("nodes.image.smart_upscale")
    seams.substitute(upscaler.CMK_SmartUpscaler, "load_upscale_model", "upscale.load_model", fake_load_upscale_model)
    seams.substitute(upscaler.CMK_SmartUpscaler, "upscale", "upscale.run", fake_upscale)

    resolver = runtime.cmk_module("pipe.loaders.checkpoint_vae_loader")
    if fake_postprocess:
        seams.observe(resolver, "resolve_postprocess_model", "prepare.resolve_postprocess_model", rebind=True)
        install_postprocess_seams(seams, runtime)
    elif stop_prepare:
        seams.stop(resolver, "resolve_postprocess_model", "prepare.resolve_postprocess_model", rebind=True)
        native = runtime.cmk_module("engine.native_detailer")
        seams.stop(native.CMKSAMLoader, "load", "prepare.sam_load")
    else:
        seams.observe(resolver, "resolve_postprocess_model", "prepare.resolve_postprocess_model", rebind=True)
    import comfy.model_management

    seams.watch(comfy.model_management, "unload_all_models", "lifecycle.unload_all_models", rebind=True)
    install_tripwires(seams)
