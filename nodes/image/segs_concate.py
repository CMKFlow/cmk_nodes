from ...utils.segs_branch_merge import merge_segs_collections


_MAX_SEGS_INPUTS = 32


class CMK_SEGSConcate:
    """Merge parallel processed SEGS collections into one authoritative image."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"segs_{index}": ("SEGS",)
            for index in range(2, _MAX_SEGS_INPUTS + 1)
        }

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
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
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
        return (result_image,)


NODE_CLASS_MAPPINGS = {
    "CMK_SEGSConcate": CMK_SEGSConcate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_SEGSConcate": "CMK SEGS CONCAT",
}
