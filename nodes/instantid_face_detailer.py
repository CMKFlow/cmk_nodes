from __future__ import annotations

import os

from ..cmk_common import SAMPLERS, SCHEDULERS
from ..pipe.cmk_log_pipe import cmk_add_block
from ..pipe.instantid.upstream import CMK_INSTANTID_CLASSES
from ..utils.cmk_diagnostic import make_diagnostic_payload
from ..utils.cmk_sampling_warnings import ignore_torchsde_boundary_rounding
from ..utils.cmk_translation import translate_prompt


INSTANTID_PROVIDERS = ["CoreML", "CPU", "CUDA", "ROCM"]


def _filename_list(category):
    try:
        import folder_paths

        return list(folder_paths.get_filename_list(category))
    except Exception:
        return []


def _instantid_models():
    models = _filename_list("instantid")
    if models:
        return models
    try:
        import folder_paths

        directory = os.path.join(folder_paths.models_dir, "instantid")
        extensions = tuple(str(ext).lower() for ext in folder_paths.supported_pt_extensions)
        return sorted(
            name for name in os.listdir(directory) if name.lower().endswith(extensions)
        ) if os.path.isdir(directory) else []
    except Exception:
        return []


def _result(value):
    if isinstance(value, dict) and "result" in value:
        value = value["result"]
    return value if isinstance(value, (tuple, list)) else (value,)


class CMKInstantIDFaceDetailerSDXL:
    """Local InstantID Img2Img detailer; deliberately independent of module 15."""

    CATEGORY = "CMK/Toolbox/Face"
    FUNCTION = "detail"
    RETURN_TYPES = ("CMK_MODEL_PIPE", "CMK_PROCESS_SDXL", "IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG", "diagnostic")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE",),
                "PROCESS": ("CMK_PROCESS_SDXL",),
                "TARGET ROI": ("IMAGE",),
                "SOURCE FACE": ("IMAGE",),
                "LOG": ("CMK_LOG_PIPE",),
                "TOTAL STEPS": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "SAMPLING START": ("INT", {
                    "default": 6, "min": 0, "max": 199, "step": 1,
                    "tooltip": "Absolute Img2Img sampling entry. Earlier values permit more anatomical change.",
                }),
                "IDENTITY STRENGTH": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "POSE STRENGTH": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "instantid_model": ((_instantid_models() or [""]), {"advanced": True}),
                "controlnet_model": ((_filename_list("controlnet") or [""]), {"advanced": True}),
                "provider": (INSTANTID_PROVIDERS, {"default": "CoreML", "advanced": True}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 10.0, "step": 0.1, "advanced": True}),
                "identity_noise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True}),
                "sampler": (SAMPLERS, {"default": "euler_ancestral"} if "euler_ancestral" in SAMPLERS else {}),
                "scheduler": (SCHEDULERS, {"default": "karras"} if "karras" in SCHEDULERS else {}),
            }
        }

    def detail(self, **inputs):
        model_pipe = inputs["MODEL"]
        process = inputs["PROCESS"]
        target_roi = inputs["TARGET ROI"]
        source_face = inputs["SOURCE FACE"]
        log_pipe = inputs["LOG"]
        if not isinstance(model_pipe, dict) or not isinstance(process, dict):
            raise TypeError("CMK InstantID Face Detailer requires CMK MODEL and PROCESS pipes")
        if str(process.get("model_family", "sdxl")).strip().lower() != "sdxl":
            raise ValueError("CMK InstantID Face Detailer supports SDXL only")
        if not bool(process.get("face_rebuild_enabled", True)):
            lines = [
                "Workflow          : INSTANTID FACE REBUILD · IMG2IMG DETAILER",
                "Status            : BYPASSED",
                "Sampling          : skipped",
            ]
            log_out = cmk_add_block(log_pipe, "InstantID Face Detailer SDXL", 45, lines, True)
            diagnostic = make_diagnostic_payload(
                title="InstantID Face Rebuild · Detailer Bypass",
                node="CMK InstantID Face Detailer SDXL",
                stages=[{"title": "02 Bypass", "subtitle": "sampling skipped", "image": target_roi}],
                previews=[target_roi],
                summary="\n".join(lines),
                details="\n".join(lines),
                mode="Bypass",
                metadata={"enabled": False, "bypassed": True},
            )
            return model_pipe, process, target_roi, log_out, diagnostic
        model = model_pipe.get("model")
        clip = model_pipe.get("clip")
        vae = model_pipe.get("vae")
        missing = [name for name, value in (("model", model), ("clip", clip), ("vae", vae)) if value is None]
        if missing:
            raise ValueError("CMK InstantID Face Detailer missing " + ", ".join(missing))

        final_steps = max(1, int(inputs["TOTAL STEPS"]))
        start_step = max(0, min(final_steps - 1, int(inputs["SAMPLING START"])))
        instantid_name = str(inputs["instantid_model"])
        controlnet_name = str(inputs["controlnet_model"])
        if not instantid_name or not controlnet_name:
            raise ValueError("CMK InstantID Face Detailer requires InstantID and ControlNet models")

        from nodes import CLIPTextEncode, ControlNetLoader, KSamplerAdvanced, VAEDecode, VAEEncode

        positive_text = translate_prompt(str(process.get("prompt_pos", "") or "")).text
        negative_text = translate_prompt(str(process.get("prompt_neg", "") or "")).text
        positive = _result(CLIPTextEncode().encode(clip, positive_text))[0]
        negative = _result(CLIPTextEncode().encode(clip, negative_text))[0]
        target_latent = _result(VAEEncode().encode(vae, target_roi))[0]

        instantid_loader = CMK_INSTANTID_CLASSES["CMKInternalInstantIDModelLoader"]()
        faceanalysis_loader = CMK_INSTANTID_CLASSES["CMKInternalInstantIDFaceAnalysis"]()
        apply_advanced = CMK_INSTANTID_CLASSES["CMKInternalApplyInstantIDAdvanced"]()
        instantid = _result(instantid_loader.load_model(instantid_name))[0]
        faceanalysis = _result(faceanalysis_loader.load_insight_face(str(inputs["provider"])))[0]
        control_net = _result(ControlNetLoader().load_controlnet(controlnet_name))[0]
        patched_model, positive_out, negative_out = _result(apply_advanced.apply_instantid(
            instantid=instantid,
            insightface=faceanalysis,
            control_net=control_net,
            image=source_face,
            image_kps=target_roi,
            image_kps_is_hint=False,
            model=model,
            positive=positive,
            negative=negative,
            ip_weight=float(inputs["IDENTITY STRENGTH"]),
            cn_strength=float(inputs["POSE STRENGTH"]),
            weight=float(inputs["IDENTITY STRENGTH"]),
            start_at=0.0,
            end_at=1.0,
            noise=float(inputs["identity_noise"]),
            combine_embeds="average",
            target_selection="Largest",
        ))[:3]

        seed = int(process.get("seed", 0) or 0)
        with ignore_torchsde_boundary_rounding():
            sampled = _result(KSamplerAdvanced().sample(
                model=patched_model,
                add_noise="enable",
                noise_seed=seed,
                steps=final_steps,
                cfg=float(inputs["cfg"]),
                sampler_name=str(inputs["sampler"]),
                scheduler=str(inputs["scheduler"]),
                positive=positive_out,
                negative=negative_out,
                latent_image=target_latent,
                start_at_step=start_step,
                end_at_step=final_steps,
                return_with_leftover_noise="disable",
            ))[0]
        image = _result(VAEDecode().decode(vae, sampled))[0]
        lines = [
            "Workflow          : INSTANTID FACE REBUILD · IMG2IMG DETAILER",
            "Inpaint           : disabled",
            "First Pass        : none",
            "KPS source        : unchanged Target ROI",
            "Latent source     : VAE-encoded Target ROI",
            f"Sampling schedule : {start_step}-{final_steps}/{final_steps} | add_noise=enable",
            f"Identity strength : {float(inputs['IDENTITY STRENGTH']):.2f}",
            f"Pose strength     : {float(inputs['POSE STRENGTH']):.2f}",
            f"Identity noise    : {float(inputs['identity_noise']):.2f}",
            f"CFG               : {float(inputs['cfg']):.2f}",
            f"Method            : {inputs['sampler']} | {inputs['scheduler']}",
        ]
        log_out = cmk_add_block(log_pipe, "InstantID Face Detailer SDXL", 45, lines, True)
        diagnostic = make_diagnostic_payload(
            title="InstantID Face Rebuild · Img2Img Detailer",
            node="CMK InstantID Face Detailer SDXL",
            stages=[
                {"title": "01 Target ROI", "subtitle": "direct KPS + latent source", "image": target_roi},
                {"title": "02 Rebuilt ROI", "subtitle": f"steps {start_step}-{final_steps}", "image": image},
            ],
            previews=[image],
            summary="\n".join(lines),
            details="\n".join(lines),
            mode="Img2Img Detailer",
            metadata={
                "inpaint": False,
                "first_pass": False,
                "sampling_start": start_step,
                "steps": final_steps,
                "cfg": float(inputs["cfg"]),
                "kps_source": "target_roi",
            },
        )
        return model_pipe, process, image, log_out, diagnostic
