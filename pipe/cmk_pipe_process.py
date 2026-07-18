from ..cmk_common import SAMPLERS, SCHEDULERS, optional_bool
from ..engine.validation import validate_contract, validate_values
import folder_paths
from ..engine.detailer_limits import clamp_detailer_denoise


def _face_detector_rank(name):
    text = str(name or "").lower()
    rank = 100
    if any(token in text for token in ("face", "person", "head", "portrait")):
        rank = 0
    if text.startswith("segm/"):
        rank += 10
    if any(token in text for token in ("breast", "hand", "feet", "foot", "pose", "clothes", "clothing")):
        rank += 50
    return (rank, text)


def _sort_face_detectors(models):
    return sorted(models, key=_face_detector_rank)


def _preferred_face_detector(models):
    models = [m for m in models if m not in (None, "none")]
    return _sort_face_detectors(models)[0] if models else None


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _detailer_source_image(pipe):
    return _first_not_none(
        pipe.get("image_refiner"),
        pipe.get("image_1st_pass"),
        pipe.get("image"),
        pipe.get("image_original"),
    )


def _face_source_image(pipe):
    return _first_not_none(
        pipe.get("image_detailer"),
        pipe.get("image_refiner"),
        pipe.get("image_1st_pass"),
        pipe.get("image"),
        pipe.get("image_original"),
    )



class CMKPipeCreateDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "conditioning_pos": ("CONDITIONING",),
                "conditioning_neg": ("CONDITIONING",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}),
                "detailer_enable": ("BOOLEAN", {"default": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler": (SAMPLERS,),
                "scheduler": (SCHEDULERS,),
            },
            "optional": {
                "sam_model": ("SAM_MODEL", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CMK_DETAILER_PIPE",)
    RETURN_NAMES = ("detailer_pipe",)
    FUNCTION = "create_detailer"
    CATEGORY = 'CMK/Developer/Pipe/Create'

    def create_detailer(self, model, clip, vae, conditioning_pos, conditioning_neg, image, seed, detailer_enable, steps, cfg, sampler, scheduler, sam_model=None):
        seed = 0 if seed is None else int(seed)
        steps = 20 if steps is None else int(steps)
        return ({
            "detailer_model": model,
            "detailer_clip": clip,
            "detailer_vae": vae,
            "detailer_conditioning_pos": conditioning_pos,
            "detailer_conditioning_neg": conditioning_neg,
            "detailer_image": image,
            "detailer_seed": seed,
            "boolean_detailer_enable": bool(detailer_enable),
            "detailer_steps": steps,
            "detailer_cfg": cfg,
            "detailer_sampler": sampler,
            "detailer_scheduler": scheduler,
            "detailer_sam_model": sam_model,
        },)


class CMKPipeCreateFaceProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "conditioning_pos": ("CONDITIONING",),
                "conditioning_neg": ("CONDITIONING",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler": (SAMPLERS,),
                "scheduler": (SCHEDULERS,),
            },
            "optional": {
                "sam_model": ("SAM_MODEL", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CMK_FACE_PIPE",)
    RETURN_NAMES = ("face_pipe",)
    FUNCTION = "create_face_process"
    CATEGORY = 'CMK/Developer/Pipe/Create'

    def create_face_process(self, model, clip, vae, conditioning_pos, conditioning_neg, image, seed,
                            steps=20, cfg=8.0, sampler=None, scheduler=None, sam_model=None):
        seed = 0 if seed is None else int(seed)
        steps = 20 if steps is None else int(steps)
        validate_values(
            node_name="CMK Pipe Create Face Process",
            contract_name="Face Process",
            values={
                "model": model,
                "clip": clip,
                "vae": vae,
                "conditioning_pos": conditioning_pos,
                "conditioning_neg": conditioning_neg,
                "image": image,
                "seed": seed,
            },
            required=["model", "clip", "vae", "conditioning_pos", "conditioning_neg", "image", "seed"],
        )
        bboxs = ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        segms = ["segm/" + x for x in folder_paths.get_filename_list("ultralytics_segm")]
        facerestore_models = ["none"] + folder_paths.get_filename_list("facerestore_models")

        default_detect_model = _preferred_face_detector(bboxs + segms)
        default_restore_model = facerestore_models[1] if len(facerestore_models) > 1 else "none"
        default_sampler = sampler if sampler is not None else (SAMPLERS[0] if SAMPLERS else "euler")
        default_scheduler = scheduler if scheduler is not None else (SCHEDULERS[0] if SCHEDULERS else "simple")

        return ({
            "face_model": model,
            "face_clip": clip,
            "face_vae": vae,
            "face_conditioning_pos": conditioning_pos,
            "face_conditioning_neg": conditioning_neg,
            "face_image": image,
            "face_seed": seed,
            # Runtime enable is handled by CMK Face Process, not by the pipe.
            # Keeping this True avoids dirtying upstream pipe branches when Face is toggled.
            "boolean_face_enable": True,
            "face_process_enable": True,

            # Defaults. Runtime mode and detailed face parameters are intentionally not exposed in the Create node.
            # They remain present in the face_pipe so CMK Face Process can execute safely.
            "face_detect_model": default_detect_model,
            "face_detect_bbox_threshold": 0.5,
            "face_detect_bbox_dilation": 0,
            "face_detect_crop_factor": 3.0,
            "face_detect_drop_size": 10,
            "face_restore_model": default_restore_model,
            "face_restore_facedetection": "retinaface_resnet50",
            "face_restore_visibility": 1.0,
            "face_restore_codeformer_weight": 0.5,
            "face_select_face_selection": "all",
            "face_select_sort_by": "area",
            "face_select_reverse_order": False,
            "face_select_take_start": 0,
            "face_select_take_count": 1,
            "face_detail_guide_size": 512,
            "face_detail_guide_size_for": True,
            "face_detail_max_size": 768,
            "face_steps": steps,
            "face_cfg": cfg,
            "face_sampler": default_sampler,
            "face_scheduler": default_scheduler,
            "face_detail_denoise": 0.5,
            "face_detail_noise_mask": True,
            "face_detail_force_inpaint": True,
            "face_detail_paste_feather": 20,
            "face_sam_model": sam_model,
        },)


class CMKPipePeekPreprocessDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("CMK_PIPE",)}}

    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE", "INT", "STRING", "STRING", "STRING", "LORA_STACK")
    RETURN_NAMES = (
        "image",
        "model",
        "clip",
        "vae",
        "opt_seed",
        "opt_prompt_pos",
        "opt_prompt_neg",
        "opt_active_loras",
        "opt_lora_stack",
    )
    FUNCTION = "peek_preprocess_detailer"
    CATEGORY = 'CMK/Developer/Pipe/Peek'

    def peek_preprocess_detailer(self, pipe):
        contract = {
            "image": _detailer_source_image(pipe) if isinstance(pipe, dict) else None,
            "model": pipe.get("model") if isinstance(pipe, dict) else None,
            "clip": pipe.get("clip") if isinstance(pipe, dict) else None,
            "vae": pipe.get("vae") if isinstance(pipe, dict) else None,
            "seed": pipe.get("seed") if isinstance(pipe, dict) else None,
            "prompt_pos": pipe.get("prompt_pos", "") if isinstance(pipe, dict) else "",
            "prompt_neg": pipe.get("prompt_neg", "") if isinstance(pipe, dict) else "",
            "active_loras": pipe.get("active_loras", "") if isinstance(pipe, dict) else "",
            "lora_stack": pipe.get("lora_stack") if isinstance(pipe, dict) else None,
        }
        validate_contract(
            node_name="CMK Pipe Peek Preprocess Detailer",
            contract_name="Preprocess Detailer",
            data=contract,
            required=["image", "model", "clip", "vae"],
        )
        return (
            contract["image"],
            contract["model"],
            contract["clip"],
            contract["vae"],
            contract["seed"],
            contract["prompt_pos"],
            contract["prompt_neg"],
            contract["active_loras"],
            contract["lora_stack"],
        )


class CMKPipePeekPreprocessFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pipe": ("CMK_PIPE",)}}

    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE", "INT", "STRING", "STRING", "STRING", "LORA_STACK")
    RETURN_NAMES = (
        "image",
        "model",
        "clip",
        "vae",
        "opt_seed",
        "opt_prompt_pos",
        "opt_prompt_neg",
        "opt_active_loras",
        "opt_lora_stack",
    )
    FUNCTION = "peek_preprocess_face"
    CATEGORY = 'CMK/Developer/Pipe/Peek'

    def peek_preprocess_face(self, pipe):
        contract = {
            "image": _face_source_image(pipe) if isinstance(pipe, dict) else None,
            "model": pipe.get("model") if isinstance(pipe, dict) else None,
            "clip": pipe.get("clip") if isinstance(pipe, dict) else None,
            "vae": pipe.get("vae") if isinstance(pipe, dict) else None,
            "seed": pipe.get("seed") if isinstance(pipe, dict) else None,
            "prompt_pos": pipe.get("prompt_pos", "") if isinstance(pipe, dict) else "",
            "prompt_neg": pipe.get("prompt_neg", "") if isinstance(pipe, dict) else "",
            "active_loras": pipe.get("active_loras", "") if isinstance(pipe, dict) else "",
            "lora_stack": pipe.get("lora_stack") if isinstance(pipe, dict) else None,
        }
        validate_contract(
            node_name="CMK Pipe Peek Preprocess Face",
            contract_name="Preprocess Face",
            data=contract,
            required=["image", "model", "clip", "vae"],
        )
        return (
            contract["image"],
            contract["model"],
            contract["clip"],
            contract["vae"],
            contract["seed"],
            contract["prompt_pos"],
            contract["prompt_neg"],
            contract["active_loras"],
            contract["lora_stack"],
        )


class CMKPipeSetDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("CMK_PIPE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "conditioning_pos": ("CONDITIONING",),
                "conditioning_neg": ("CONDITIONING",),
                "detailer_enable": ("BOOLEAN", {"default": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler": (SAMPLERS,),
                "scheduler": (SCHEDULERS,),
            },
            "optional": {
                "sam_model": ("SAM_MODEL",),
            },
        }

    RETURN_TYPES = ("CMK_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "set_detailer"
    CATEGORY = 'CMK/Developer/Pipe/Set'

    def set_detailer(self, pipe, model, clip, conditioning_pos, conditioning_neg, detailer_enable, steps, cfg, sampler, scheduler, sam_model=None):
        new_pipe = dict(pipe)
        # Module-scoped only. Do not overwrite generic main-pipe keys like
        # model/clip/conditioning/steps/cfg/sampler/scheduler/sam_model here;
        # changing a downstream module must not invalidate the 1st-pass context.
        new_pipe["detailer_model"] = model
        new_pipe["detailer_clip"] = clip
        new_pipe["detailer_conditioning_pos"] = conditioning_pos
        new_pipe["detailer_conditioning_neg"] = conditioning_neg
        new_pipe["boolean_detailer_enable"] = bool(detailer_enable)
        new_pipe["detailer_steps"] = steps
        new_pipe["detailer_cfg"] = cfg
        new_pipe["detailer_sampler"] = sampler
        new_pipe["detailer_scheduler"] = scheduler
        new_pipe["detailer_sam_model"] = sam_model
        return (new_pipe,)


class CMKPipeSetSampler2:
    """Legacy alias for old workflows.

    New workflows should use CMK Pipe Set Detailer.
    The behavior is intentionally identical so serialized workflows using
    CMKPipeSetSampler2 continue to load.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return CMKPipeSetDetailer.INPUT_TYPES()

    RETURN_TYPES = ("CMK_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "set_sampler_2"
    CATEGORY = "CMK/Developer/Legacy"

    def set_sampler_2(self, pipe, model, clip, conditioning_pos, conditioning_neg, detailer_enable, steps, cfg, sampler, scheduler, sam_model=None):
        return CMKPipeSetDetailer().set_detailer(
            pipe,
            model,
            clip,
            conditioning_pos,
            conditioning_neg,
            detailer_enable,
            steps,
            cfg,
            sampler,
            scheduler,
            sam_model,
        )


class CMKPipeSetFaceProcess:
    @classmethod
    def INPUT_TYPES(cls):
        bboxs = ["bbox/" + x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        segms = ["segm/" + x for x in folder_paths.get_filename_list("ultralytics_segm")]
        facerestore_models = ["none"] + folder_paths.get_filename_list("facerestore_models")

        return {
            "required": {
                "pipe": ("CMK_PIPE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "conditioning_pos": ("CONDITIONING",),
                "conditioning_neg": ("CONDITIONING",),
                "face_enable": ("BOOLEAN", {"default": True}),
                "process_enable": ("BOOLEAN", {"default": True}),
                "process_mode": (["restore", "detailer"], {"default": "restore"}),

                "detect_model": (bboxs + segms,),
                "detect_bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detect_bbox_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                "detect_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "detect_drop_size": ("INT", {"default": 10, "min": 1, "max": 8192, "step": 1}),

                "restore_model": (facerestore_models,),
                "restore_facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "restore_visibility": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "restore_codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),

                "select_face_selection": (["all", "filter", "largest"], {"default": "all"}),
                "select_sort_by": (["area", "x_position", "y_position", "detection_confidence"], {"default": "area"}),
                "select_reverse_order": ("BOOLEAN", {"default": False}),
                "select_take_start": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "select_take_count": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),

                "detail_guide_size": ("FLOAT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "detail_guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                "detail_max_size": ("FLOAT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
                "detail_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "detail_cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "detail_sampler": (SAMPLERS,),
                "detail_scheduler": (SCHEDULERS,),
                "detail_denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 0.5, "step": 0.01}),
                "detail_noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "detail_force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "detail_paste_feather": ("INT", {"default": 20, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {
                "sam_model": ("SAM_MODEL",),
            },
        }

    RETURN_TYPES = ("CMK_FACE_PIPE",)
    RETURN_NAMES = ("face_pipe",)
    FUNCTION = "set_face_process"
    CATEGORY = 'CMK/Developer/Pipe/Set'

    def set_face_process(self, pipe, model, clip, conditioning_pos, conditioning_neg, face_enable,
                         process_enable=True, process_mode="restore",
                         detect_model=None, detect_bbox_threshold=0.5, detect_bbox_dilation=0, detect_crop_factor=3.0, detect_drop_size=10,
                         restore_model="none", restore_facedetection="retinaface_resnet50", restore_visibility=1.0, restore_codeformer_weight=0.5,
                         select_face_selection="all", select_sort_by="area", select_reverse_order=False, select_take_start=0, select_take_count=1,
                         detail_guide_size=512, detail_guide_size_for=True, detail_max_size=768, detail_steps=20, detail_cfg=8.0,
                         detail_sampler=None, detail_scheduler=None, detail_denoise=0.5, detail_noise_mask=True, detail_force_inpaint=True, detail_paste_feather=20,
                         sam_model=None):
        detail_denoise = clamp_detailer_denoise(detail_denoise)
        source_seed = pipe.get("seed", pipe.get("face_seed", 0))
        source_vae = pipe.get("vae")
        source_image = _face_source_image(pipe)

        face_pipe = CMKPipeCreateFaceProcess().create_face_process(
            model, clip, source_vae, conditioning_pos, conditioning_neg, source_image, source_seed,
            detail_steps, detail_cfg, detail_sampler, detail_scheduler, sam_model,
        )[0]

        face_pipe.update({
            "face_process_enable": bool(process_enable),
            # Runtime mode is now owned by CMK Face Process. Legacy setter keeps
            # this field only as informational metadata; "bypass" is no longer
            # exposed in the UI.
            "face_process_mode": process_mode,
            "face_detect_model": detect_model,
            "face_detect_bbox_threshold": detect_bbox_threshold,
            "face_detect_bbox_dilation": detect_bbox_dilation,
            "face_detect_crop_factor": detect_crop_factor,
            "face_detect_drop_size": detect_drop_size,
            "face_restore_model": restore_model,
            "face_restore_facedetection": restore_facedetection,
            "face_restore_visibility": restore_visibility,
            "face_restore_codeformer_weight": restore_codeformer_weight,
            "face_select_face_selection": select_face_selection,
            "face_select_sort_by": select_sort_by,
            "face_select_reverse_order": select_reverse_order,
            "face_select_take_start": select_take_start,
            "face_select_take_count": select_take_count,
            "face_detail_guide_size": detail_guide_size,
            "face_detail_guide_size_for": detail_guide_size_for,
            "face_detail_max_size": detail_max_size,
            "face_steps": detail_steps,
            "face_cfg": detail_cfg,
            "face_sampler": detail_sampler,
            "face_scheduler": detail_scheduler,
            "face_detail_denoise": detail_denoise,
            "face_detail_noise_mask": detail_noise_mask,
            "face_detail_force_inpaint": detail_force_inpaint,
            "face_detail_paste_feather": detail_paste_feather,
            "face_sam_model": sam_model,
        })
        return (face_pipe,)


class CMKPipePeekDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"detailer_pipe": ("CMK_DETAILER_PIPE",)}}

    RETURN_TYPES = ("BASIC_PIPE", "IMAGE", "INT", "INT", "FLOAT", SAMPLERS, SCHEDULERS, "BOOLEAN", "SAM_MODEL")
    RETURN_NAMES = ("basic_pipe", "image", "seed", "steps", "cfg", "sampler", "scheduler", "boolean_detailer_enable", "sam_model")
    FUNCTION = "peek_detailer"
    CATEGORY = 'CMK/Developer/Pipe/Peek'

    def peek_detailer(self, detailer_pipe):
        basic_pipe = (
            detailer_pipe.get("detailer_model"),
            detailer_pipe.get("detailer_clip"),
            detailer_pipe.get("detailer_vae"),
            detailer_pipe.get("detailer_conditioning_pos"),
            detailer_pipe.get("detailer_conditioning_neg"),
        )

        return (
            basic_pipe,
            detailer_pipe.get("detailer_image"),
            detailer_pipe.get("detailer_seed"),
            detailer_pipe.get("detailer_steps"),
            detailer_pipe.get("detailer_cfg"),
            detailer_pipe.get("detailer_sampler"),
            detailer_pipe.get("detailer_scheduler"),
            detailer_pipe.get("boolean_detailer_enable", False),
            detailer_pipe.get("detailer_sam_model"),
        )


class CMKPipePeekFaceProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"face_pipe": ("CMK_FACE_PIPE",)}}

    RETURN_TYPES = ("BASIC_PIPE", "IMAGE", "INT", "INT", "FLOAT", SAMPLERS, SCHEDULERS, "BOOLEAN", "SAM_MODEL")
    RETURN_NAMES = ("basic_pipe", "image", "seed", "steps", "cfg", "sampler", "scheduler", "boolean_face_enable", "sam_model")
    FUNCTION = "peek_face_process"
    CATEGORY = 'CMK/Developer/Pipe/Peek'

    def peek_face_process(self, face_pipe):
        basic_pipe = (
            face_pipe.get("face_model"),
            face_pipe.get("face_clip"),
            face_pipe.get("face_vae"),
            face_pipe.get("face_conditioning_pos"),
            face_pipe.get("face_conditioning_neg"),
        )

        return (
            basic_pipe,
            face_pipe.get("face_image"),
            face_pipe.get("face_seed"),
            face_pipe.get("face_steps"),
            face_pipe.get("face_cfg"),
            face_pipe.get("face_sampler"),
            face_pipe.get("face_scheduler"),
            face_pipe.get("boolean_face_enable", False),
            face_pipe.get("face_sam_model"),
        )


class CMKPipeSetDetailerResult:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("CMK_PIPE",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CMK_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "set_detailer_result"
    CATEGORY = 'CMK/Developer/Pipe/Set'

    def set_detailer_result(self, pipe, image):
        new_pipe = dict(pipe)
        new_pipe["image_detailer"] = image
        # Final-image staging fallback for workflows without a later face module.
        # Deliberately do not overwrite generic new_pipe["image"].
        new_pipe["image_final"] = image
        return (new_pipe,)


class CMKPipeSetFaceResult:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("CMK_PIPE",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CMK_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "set_face_result"
    CATEGORY = 'CMK/Developer/Pipe/Set'

    def set_face_result(self, pipe, image):
        new_pipe = dict(pipe)
        new_pipe["image_face"] = image
        new_pipe["image_final"] = image
        return (new_pipe,)


class CMKPipeSetFaceProcessResult(CMKPipeSetFaceResult):
    FUNCTION = "set_face_process_result"
    CATEGORY = 'CMK/Developer/Pipe/Set'

    def set_face_process_result(self, pipe, image):
        return self.set_face_result(pipe, image)
