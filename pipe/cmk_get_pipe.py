class CMKGetPipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("CMK_PIPE",),
            },
            "optional": {
                "strict": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "get_pipe"
    CATEGORY = 'CMK/Developer/Pipe/Get'

    def get_pipe(self, pipe, strict=True, debug=False):
        for key in (
            "image_face",
            "image_detailer",
            "image_refiner",
            "image_1st_pass",
        ):
            image = pipe.get(key)
            if image is not None:
                if debug:
                    print(f"[CMK get Pipe] selected {key}")
                return (image,)

        if strict:
            raise Exception("CMK get Pipe: pipe contains no image_face/image_detailer/image_refiner/image_1st_pass")

        if debug:
            print("[CMK get Pipe] no image found; returning None")
        return (None,)


NODE_CLASS_MAPPINGS = {
    "CMKGetPipe": CMKGetPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMKGetPipe": "CMK get Pipe",
}
