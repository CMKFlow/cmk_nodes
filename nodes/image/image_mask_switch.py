class CMK_ImageMaskSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_false": ("IMAGE",),
                "image_true": ("IMAGE",),
                "mask_false": ("MASK",),
                "mask_true": ("MASK",),
                "boolean_value": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("image_output", "mask_output", "boolean")

    FUNCTION = "run"
    CATEGORY = "CMK/Toolbox/Mask & SEGS"

    def run(self, image_false, image_true, mask_false, mask_true, boolean_value):
        if boolean_value:
            return (image_true, mask_true, boolean_value)
        else:
            return (image_false, mask_false, boolean_value)


NODE_CLASS_MAPPINGS = {
    "CMK_ImageMaskSwitch": CMK_ImageMaskSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_ImageMaskSwitch": "CMK Image and Mask Switch",
}
