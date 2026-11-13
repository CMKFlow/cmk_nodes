"""FaceProcess 2.8 Standard (F2): deferred Prepare and Multi-Model execution.

CMK FaceProcess Prepare Multi-Model -Pipe- builds a FACE specification only:
the operating mode, the authoritative image and the arguments a branch needs
to prepare itself.  It never loads a model, SAM or conditioning, and it
requests a model source only for the mode that needs it:

    NONE     IMAGE + LOG                    (complete bypass, also no Refine)
    RESTORE  IMAGE + LOG                    (no diffusion model, no PROCESS)
    SDXL     IMAGE + LOG + PROCESS + MODEL SDXL
    ZIT      IMAGE + LOG + PROCESS + MODEL ZIT

CMK FaceProcess Multi-Model -Pipe- materialises on a branch-cache MISS only:
RESTORE builds a model-free FACE pipe; SDXL runs the unchanged legacy
FaceProcess preparation (after the guarded ZIT release); ZIT runs the
Z-Image Turbo preparation of CMK Detailer Prepare Multi-Model.  The execution
itself is the unchanged legacy FaceProcess path (``process_mode`` restore or
detailer).  Models exist only while the branch runs; no output holds them.

Legacy CMK FaceProcess Prepare -Pipe- / CMK FaceProcess -Pipe- are unchanged.
"""

from __future__ import annotations

from .cmk_detailer_prepare_multimodel import CMKDetailerPrepareMultiModelPipe, ZIT_SAMPLING_ENTRY
from .cmk_faceprocess import CMKFaceProcessPipe
from .cmk_faceprocess_mode import FACEPROCESS_MODES, faceprocess_mode
from .cmk_faceprocess_prepare import CMKFaceProcessPreparePipe
from .cmk_log_pipe import cmk_add_block, cmk_block_to_string
from .cmk_persistent_cache import DISABLED_REVISION, build_node_fingerprint
from ..utils.cmk_diagnostic import make_diagnostic_payload

PREPARE_CLASS = "CMKFaceProcessPrepareMultiModelPipe"
EXECUTE_CLASS = "CMKFaceProcessMultiModelPipe"
PREPARE_LABEL = "CMK FaceProcess Prepare Multi-Model -Pipe-"
EXECUTE_LABEL = "CMK FaceProcess Multi-Model -Pipe-"
FACE_MULTIMODEL_PIPE = "CMK_FACE_MULTIMODEL_PIPE"
FACE_CONTRACT = "faceprocess_multimodel_v1"
MODE_LABELS = {"none": "NONE", "restore": "RESTORE", "sdxl": "SDXL", "z_image_turbo": "ZIT"}

# Prepare inputs by the mode that reads them.
SDXL_SETTINGS = (
    "use_1st_pass_sampling", "lora_name", "strength_model", "strength_clip", "prompt_neg",
    "steps", "cfg", "sampler", "scheduler", "stop_at_clip_layer", "pag_scale", "sampling",
    "zsnr", "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2",
)
ZIT_SETTINGS = ("zit_steps", "zit_cfg", "zit_sampler", "zit_scheduler", "zit_model_shift")
DIFFUSION_SHARED = ("sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler",
                    "use_lora_from_1st_pass", "prompt_pos")
TRANSPORT = ("LOG",)

# Execute inputs by the path that reads them.
RESTORE_INPUTS = ("restore_model", "restore_facedetection", "restore_visibility", "restore_codeformer_weight")
DETAIL_INPUTS = ("detail_guide_size", "detail_guide_size_for", "detail_max_size", "detail_denoise",
                 "detail_noise_mask", "detail_force_inpaint", "detail_paste_feather")


def prepare_excluded_inputs(mode):
    """Prepare inputs outside a mode's identity (never part of its branch key)."""
    if mode == "restore":
        return (*TRANSPORT, "MODEL ZIT", "MODEL SDXL", "PROCESS", *DIFFUSION_SHARED, *SDXL_SETTINGS, *ZIT_SETTINGS)
    if mode == "sdxl":
        return (*TRANSPORT, "MODEL ZIT", *ZIT_SETTINGS)
    if mode == "z_image_turbo":
        return (*TRANSPORT, "MODEL SDXL", *SDXL_SETTINGS)
    return (*TRANSPORT,)


def execute_excluded_inputs(mode):
    transport = ("opt_log", "VISUAL", "opt_diagnostic", "FACE")
    if mode == "restore":
        return (*transport, *DETAIL_INPUTS)
    if mode in ("sdxl", "z_image_turbo"):
        return (*transport, *RESTORE_INPUTS)
    return transport


def _prepare_link(prompt, unique_id):
    node = prompt.get(str(unique_id)) if isinstance(prompt, dict) else None
    link = ((node or {}).get("inputs") or {}).get("FACE") if isinstance(node, dict) else None
    if not (isinstance(link, (list, tuple)) and len(link) == 2):
        return None, None
    prepare = prompt.get(str(link[0]))
    if not isinstance(prepare, dict) or prepare.get("class_type") != PREPARE_CLASS:
        return None, None
    return str(link[0]), prepare


def branch_mode(prompt, unique_id):
    """Mode of the Prepare feeding this branch: an id, or ``None`` if unresolvable.

    A Mode Enable control arrives as its literal selection (prompt inline
    handler); a selection that is still a link cannot be resolved statically.
    """
    _prepare_id, prepare = _prepare_link(prompt, unique_id)
    if prepare is None:
        return None
    value = (prepare.get("inputs") or {}).get("process_mode")
    return faceprocess_mode(value) if isinstance(value, str) else None


class CMKFaceProcessPrepareMultiModelPipe:
    """Deferred FACE preparation for RESTORE, SDXL or Z-Image Turbo (2.8.0).

    Public contract:
        MODEL ZIT (opt) + MODEL SDXL + PROCESS + IMAGE + LOG -> FACE + LOG + diagnostic
    """

    @classmethod
    def INPUT_TYPES(cls):
        legacy = CMKFaceProcessPreparePipe.INPUT_TYPES()["required"]
        detailer = CMKDetailerPrepareMultiModelPipe.INPUT_TYPES()["required"]
        required = {
            "MODEL SDXL": legacy["MODEL"],
            "PROCESS": legacy["PROCESS"],
            "IMAGE": legacy["IMAGE"],
            "process_mode": (FACEPROCESS_MODES, {"default": "None"}),
        }
        required.update({name: legacy[name] for name in (
            "sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler", "use_lora_from_1st_pass",
            "use_1st_pass_sampling", "lora_name", "strength_model", "strength_clip", "prompt_pos", "prompt_neg",
            "steps", "cfg", "sampler", "scheduler",
        )})
        required.update({name: detailer[name] for name in ZIT_SETTINGS})
        required.update({name: legacy[name] for name in (
            "stop_at_clip_layer", "pag_scale", "sampling", "zsnr",
            "freeu_enabled", "freeu_b1", "freeu_b2", "freeu_s1", "freeu_s2",
        )})
        return {
            "required": required,
            "optional": {
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
                "MODEL ZIT": ("CMK_MODEL_PIPE", {"lazy": True}),
            },
        }

    RETURN_TYPES = (FACE_MULTIMODEL_PIPE, "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("FACE", "LOG", "diagnostic")
    FUNCTION = "prepare"
    CATEGORY = "CMK/Flow/PostProcess"

    def check_lazy_status(self, IMAGE=None, PROCESS=None, process_mode="None", **kwargs):
        needed = []
        if IMAGE is None:
            needed.append("IMAGE")
        # An unconnected optional input is absent from kwargs and never requested.
        if "LOG" in kwargs and kwargs["LOG"] is None:
            needed.append("LOG")
        mode = faceprocess_mode(process_mode)
        if mode in ("sdxl", "z_image_turbo"):
            if PROCESS is None:
                needed.append("PROCESS")
            name = "MODEL SDXL" if mode == "sdxl" else "MODEL ZIT"
            if name in kwargs and kwargs[name] is None:
                needed.append(name)
        return needed

    def prepare(self, IMAGE, process_mode, PROCESS=None, LOG=None, **settings):
        if IMAGE is None:
            raise ValueError(f"{PREPARE_LABEL}: IMAGE is missing")
        mode = faceprocess_mode(process_mode)
        face = {
            "face_contract": FACE_CONTRACT,
            "face_deferred": True,
            "face_mode": mode,
            "face_process_mode": str(process_mode),
            "face_image": IMAGE,
            "face_global_enable": mode != "none",
            "boolean_face_enable": mode != "none",
        }
        model_source = "NOT REQUESTED"
        if mode in ("sdxl", "z_image_turbo"):
            if not isinstance(PROCESS, dict):
                raise TypeError(f"{PREPARE_LABEL}: PROCESS must be a CMK process pipe")
            if PROCESS.get("result_contract") != "family_neutral":
                raise ValueError(f"{PREPARE_LABEL} requires the PostProcess Boundary")
            name = "MODEL SDXL" if mode == "sdxl" else "MODEL ZIT"
            model = settings.pop("MODEL SDXL", None) if mode == "sdxl" else settings.pop("MODEL ZIT", None)
            if not isinstance(model, dict):
                raise ValueError(f"{PREPARE_LABEL}: MODE {MODE_LABELS[mode]} requires {name} from CMK PostProcess Model.")
            settings.pop("MODEL SDXL", None)
            settings.pop("MODEL ZIT", None)
            face.update({
                "source_pipe": dict(PROCESS),
                "_source_pipe": dict(PROCESS),
                "face_seed": int(PROCESS.get("seed", 0) or 0),
                "face_model_spec": model,
                "face_prepare_settings": dict(settings),
            })
            model_source = f"{name} · DEFERRED (branch MISS)"
        lines = [
            f"STATUS          : {'NONE · BYPASS' if mode == 'none' else 'PREPARED · DEFERRED'}",
            f"MODE            : {MODE_LABELS[mode]}",
            f"MODEL SOURCE    : {model_source}",
            "SAM / LORA      : NOT LOADED",
            "CONDITIONING    : NOT CREATED",
        ]
        log_pipe = cmk_add_block(LOG, "FaceProcess Prepare", 80, lines, True)
        diagnostic = make_diagnostic_payload(
            title="FaceProcess Prepare",
            node=PREPARE_LABEL,
            previews=[IMAGE],
            summary=f"mode={MODE_LABELS[mode]} | deferred, nothing loaded",
            details="\n".join(lines),
            mode=MODE_LABELS[mode].lower(),
            metadata={"mode": mode, "model_requested": mode in ("sdxl", "z_image_turbo"),
                      "models_loaded": False, "sam_loaded": False, "conditioning_created": False},
        )
        return (face, log_pipe, diagnostic)


class CMKFaceProcessMultiModelPipe(CMKFaceProcessPipe):
    """FaceProcess execution for CMK FaceProcess Prepare Multi-Model (2.8.0).

    The operating mode comes from the Prepare; this node has no own enable or
    mode widget.  Cache entries use their own schema and never meet the legacy
    CMK FaceProcess -Pipe- entries.
    """

    _CACHE_SCHEMA = "cmk_faceprocess_multimodel_branch_v1"
    _ZIT_TEXT_ENCODER = "released after FaceProcess Prepare"

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        required = {name: value for name, value in spec["required"].items() if name not in ("enable", "process_mode")}
        required["FACE"] = (FACE_MULTIMODEL_PIPE, {"lazy": True})
        return {**spec, "required": required}

    # ---- cache identity -------------------------------------------------
    @classmethod
    def dependency_cache_key(cls, prompt, unique_id):
        mode = branch_mode(prompt, unique_id)
        prepare_id, prepare = _prepare_link(prompt, unique_id)
        extra = {"mode": mode or "unknown"}
        if prepare is not None:
            extra["prepare"], _detail = build_node_fingerprint(
                prompt,
                prepare_id,
                (PREPARE_CLASS,),
                "cmk_faceprocess_multimodel_prepare_v1",
                exclude_inputs=prepare_excluded_inputs(mode),
            )
        return build_node_fingerprint(
            prompt,
            unique_id,
            (EXECUTE_CLASS,),
            cls._CACHE_SCHEMA,
            exclude_inputs=execute_excluded_inputs(mode) if prepare is not None else ("opt_log", "VISUAL", "opt_diagnostic"),
            include_node_identity=True,
            extra_identity=extra,
        )

    @classmethod
    def dependency_revision(cls, prompt, unique_id, cache_key):
        # NONE never runs and stores nothing, like a switched-off branch.
        if branch_mode(prompt, unique_id) == "none":
            return DISABLED_REVISION
        return super().dependency_revision(prompt, unique_id, cache_key)

    def check_lazy_status(self, FACE=None, prompt=None, unique_id=None, **kwargs):
        if branch_mode(prompt, unique_id) == "none":
            # The bypass only needs the image of the (model-free) FACE spec.
            return ["FACE"] if FACE is None else []
        return super().check_lazy_status(FACE=FACE, prompt=prompt, unique_id=unique_id, **kwargs)

    # ---- execution ------------------------------------------------------
    def run_pipe(self, FACE=None, select_face="Largest", opt_log=None, VISUAL=None, opt_diagnostic=None,
                 prompt=None, unique_id=None, **kwargs):
        from .cmk_persistent_cache import pickle_available

        mode = branch_mode(prompt, unique_id)
        if mode is None and isinstance(FACE, dict):
            mode = FACE.get("face_mode")
        if mode == "none":
            return self._bypass(FACE, VISUAL, unique_id, opt_log, opt_diagnostic)
        transport = dict(select_face=select_face, opt_log=opt_log, VISUAL=VISUAL, opt_diagnostic=opt_diagnostic,
                         prompt=prompt, unique_id=unique_id, **kwargs)
        cache_key, _detail = self.dependency_cache_key(prompt, unique_id)
        if cache_key and pickle_available(self._CACHE_SCOPE, cache_key):
            # Branch HIT: the legacy path loads the payload; FACE is not read.
            return super().run_pipe(FACE=None, enable=True, process_mode="restore", **transport)
        face_pipe = self._materialize_face(FACE, prompt=prompt)
        process_mode = "restore" if face_pipe.get("face_mode") == "restore" else "detailer"
        return super().run_pipe(FACE=face_pipe, enable=True, process_mode=process_mode, **transport)

    def _bypass(self, FACE, VISUAL, unique_id, opt_log, opt_diagnostic):
        image = FACE.get("face_image") if isinstance(FACE, dict) else None
        if image is None:
            raise ValueError(f"{EXECUTE_LABEL}: FACE carries no image")
        empty = self._empty_segs(image)
        lines = ["STATUS          : NONE · BYPASS", "FACE DETECTION  : SKIPPED", "PROCESSING      : SKIPPED",
                 "REFINE          : SKIPPED", "CACHE           : SKIPPED", "RESULT          : PASSTHROUGH"]
        diagnostic = make_diagnostic_payload(
            title="FaceProcess -Pipe-", node=EXECUTE_LABEL, previews=[image], summary="NONE · bypass",
            details="\n".join(lines), mode="none / passthrough", metadata={"mode": "none"},
        )
        return self._merge_transport(
            image, empty, empty, False, cmk_block_to_string("FaceProcess", 90, lines, True), diagnostic,
            before=image, visual=VISUAL, unique_id=unique_id, opt_log=opt_log, opt_diagnostic=opt_diagnostic,
        )

    def _materialize_face(self, face, prompt=None):
        """FACE spec -> FACE pipe of the legacy execution, on a branch MISS only."""
        if not isinstance(face, dict) or face.get("face_contract") != FACE_CONTRACT:
            raise ValueError(f"{EXECUTE_LABEL}: FACE must come from {PREPARE_LABEL}")
        mode = face.get("face_mode")
        image = face.get("face_image")
        if mode == "restore":
            # Restore needs no diffusion model, conditioning or SAM.
            return {"face_mode": "restore", "face_image": image, "face_global_enable": True,
                    "boolean_face_enable": True, "face_sam_model": None, "face_seed": 0}
        settings = dict(face.get("face_prepare_settings") or {})
        process = face.get("source_pipe")
        if mode == "sdxl":
            from .loaders.z_image_turbo_loader import release_zit_resources

            # SDXL is about to be materialised: drop CMK's own ZIT references
            # unless another consumer still uses them (no-op when empty).
            release_zit_resources("ZIT -> SDXL FaceProcess")
            arguments = {name: settings[name] for name in (
                "sam_model_name", "sam_device_mode", "use_prompt_lora_from_sampler", "use_lora_from_1st_pass",
                *SDXL_SETTINGS, "prompt_pos",
            )}
            pipe = CMKFaceProcessPreparePipe().prepare(
                MODEL=face.get("face_model_spec"), PROCESS=process, IMAGE=image, face_global_enable=True, **arguments,
            )[0]
            return {**pipe, "face_mode": "sdxl"}
        if mode == "z_image_turbo":
            detailer = CMKDetailerPrepareMultiModelPipe().prepare(
                PROCESS=process,
                IMAGE=image,
                model_family="Z-Image Turbo",
                sam_model_name=settings["sam_model_name"],
                sam_device_mode=settings["sam_device_mode"],
                use_prompt_lora_from_sampler=settings["use_prompt_lora_from_sampler"],
                use_lora_from_1st_pass=settings["use_lora_from_1st_pass"],
                use_1st_pass_sampling=False,
                lora_name="None",
                strength_model=1.0,
                strength_clip=1.0,
                prompt_pos=settings["prompt_pos"],
                prompt_neg="",
                steps=settings["steps"],
                cfg=settings["cfg"],
                sampling_entry=95.0,
                zit_steps=settings["zit_steps"],
                zit_cfg=settings["zit_cfg"],
                zit_sampling_entry=ZIT_SAMPLING_ENTRY,
                sampler=settings["sampler"],
                scheduler=settings["scheduler"],
                zit_sampler=settings["zit_sampler"],
                zit_scheduler=settings["zit_scheduler"],
                stop_at_clip_layer=settings["stop_at_clip_layer"],
                pag_scale=settings["pag_scale"],
                sampling=settings["sampling"],
                zsnr=settings["zsnr"],
                zit_model_shift=settings["zit_model_shift"],
                freeu_enabled=settings["freeu_enabled"],
                freeu_b1=settings["freeu_b1"],
                freeu_b2=settings["freeu_b2"],
                freeu_s1=settings["freeu_s1"],
                freeu_s2=settings["freeu_s2"],
                **{"MODEL ZIT": face.get("face_model_spec")},
            )[0]
            return {
                "face_mode": "z_image_turbo",
                "source_pipe": detailer.get("source_pipe"),
                "_source_pipe": detailer.get("source_pipe"),
                "face_model": detailer.get("detailer_model"),
                # The sampling engine never reads CLIP; the text encoder is
                # released after encoding, the BASIC_PIPE check needs a value.
                "face_clip": self._ZIT_TEXT_ENCODER,
                "face_vae": detailer.get("detailer_vae"),
                "face_conditioning_pos": detailer.get("detailer_conditioning_pos"),
                "face_conditioning_neg": detailer.get("detailer_conditioning_neg"),
                "face_image": image,
                "face_seed": detailer.get("detailer_seed", 0),
                "face_global_enable": True,
                "boolean_face_enable": True,
                "face_steps": detailer.get("detailer_steps"),
                "face_cfg": detailer.get("detailer_cfg"),
                "face_sampler": detailer.get("detailer_sampler"),
                "face_scheduler": detailer.get("detailer_scheduler"),
                "face_sam_model": detailer.get("detailer_sam_model"),
                "face_model_identity": detailer.get("detailer_model_identity"),
                "face_zit_model_shift": detailer.get("detailer_zit_model_shift"),
            }
        raise ValueError(f"{EXECUTE_LABEL}: unsupported FACE mode {mode!r}")
