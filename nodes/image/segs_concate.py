from ...pipe.cmk_visual import (
    _upstream_class_nodes,
    empty_visual,
    register_provider,
)
from ...utils.segs_branch_merge import merge_segs_collections, valid_segs
from ...utils.stable_segs import stable_branch_components


_MAX_SEGS_INPUTS = 32


class CMK_SEGSConcate:
    """Merge parallel processed SEGS collections into one authoritative image."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"segs_{index}": ("SEGS",)
            for index in range(2, _MAX_SEGS_INPUTS + 1)
        }
        optional["VISUAL"] = ("CMK_VISUAL_PIPE",)

        return {
            "required": {
                "image": ("IMAGE",),
                "segs": ("SEGS",),
                "feather": (
                    "INT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "advanced": True,
                    },
                ),
                "alpha": (
                    "INT",
                    {
                        "default": 255,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "advanced": True,
                    },
                ),
            },
            "optional": optional,
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "CMK_VISUAL_PIPE")
    RETURN_NAMES = ("IMAGE", "VISUAL")
    FUNCTION = "concate"
    CATEGORY = "CMK/Toolbox/Mask & SEGS"

    def concate(self, image, segs, feather=5, alpha=255, **kwargs):
        if image is None:
            raise ValueError("CMK SEGS CONCAT: image is missing")
        if segs is None:
            raise ValueError("CMK SEGS CONCAT: SEGS 1 is missing")

        segs_inputs = [segs]
        for index in range(2, _MAX_SEGS_INPUTS + 1):
            value = kwargs.get(f"segs_{index}")
            if value is not None:
                segs_inputs.append(value)

        result_image = merge_segs_collections(
            authoritative_image=image,
            segs_collections=segs_inputs,
            feather=int(feather),
            alpha=int(alpha),
        )
        visual = kwargs.get("VISUAL")
        result_visual = empty_visual() if visual is None else visual

        # In an Advanced module, this node is the first point at which all
        # independent branches have become one authoritative image.
        current = (kwargs.get("prompt") or {}).get(str(kwargs.get("unique_id")), {})
        current_inputs = (current.get("inputs", {}) or {}) if isinstance(current, dict) else {}
        unique_text = str(kwargs.get("unique_id") or "")
        scope = unique_text.rsplit(":", 1)[0] if ":" in unique_text else ""
        families = (
            (
                "CMKFaceSwapImagePipe", "CMKSEGSConcateFaceSwap", "FaceSwap",
                40, "result", "result.faceswap.advanced",
            ),
            (
                "CMKFaceProcessPipe", "CMKSEGSConcateFaceProcess", "FaceProcess",
                30, "sdxl", "sdxl.faceprocess.advanced",
            ),
            (
                "CMK_SmartDetailerPipe", "CMKSEGSConcateDetailer", "Detailer",
                23, "sdxl", "sdxl.detailer.advanced",
            ),
        )
        segs_names = (
            "segs",
            *(f"segs_{index}" for index in range(2, _MAX_SEGS_INPUTS + 1)),
        )
        family = None
        live_node_ids = []
        for candidate in families:
            candidate_ids = []
            for name in segs_names:
                for node_id in _upstream_class_nodes(
                    kwargs.get("prompt"), current_inputs.get(name), candidate[0], scope,
                ):
                    if node_id not in candidate_ids:
                        candidate_ids.append(node_id)
            if candidate_ids:
                family = candidate
                live_node_ids = candidate_ids
                break

        def has_content(value):
            if not valid_segs(value):
                return False
            stable = stable_branch_components(value)
            if stable is not None:
                return bool(stable[1].any().item())
            return bool(value[1])

        has_content = any(has_content(value) for value in segs_inputs)
        if family is not None and has_content:
            _, module_type, label, sequence, branch, stage_key = family
            result_visual = register_provider(
                result_visual,
                module_instance_id=kwargs.get("unique_id") or f"{label.lower()}-advanced",
                module_type=module_type,
                module_label=label,
                sequence=sequence,
                channels={"before": image, "after": result_image},
                status="completed",
                live_node_ids=live_node_ids,
                branch=branch,
                stage_key=stage_key,
            )
        return result_image, result_visual


NODE_CLASS_MAPPINGS = {
    "CMK_SEGSConcate": CMK_SEGSConcate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_SEGSConcate": "CMK SEGS CONCAT",
}
