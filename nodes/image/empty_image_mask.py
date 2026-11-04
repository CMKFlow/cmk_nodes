import torch


class CMK_EmptyImageMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

    FUNCTION = "run"
    CATEGORY = "CMK/Toolbox/Mask & SEGS"

    def run(self, width, height):
        image = torch.zeros((1, height, width, 3), dtype=torch.float32)
        mask = torch.ones((1, height, width), dtype=torch.float32)

        return (image, mask)


NODE_CLASS_MAPPINGS = {
    "CMK_EmptyImageMask": CMK_EmptyImageMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_EmptyImageMask": "CMK Empty Image Mask",
}
