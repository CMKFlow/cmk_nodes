import os


class CMK_SaveProjectText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "full_path": ("STRING", {"forceInput": True}),
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_path",)

    FUNCTION = "run"
    CATEGORY = "CMK/Toolbox/I-O"

    def run(self, full_path, text):
        base_path, _ext = os.path.splitext(full_path)
        text_path = base_path + ".txt"

        folder = os.path.dirname(text_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(str(text))

        return (text_path,)


NODE_CLASS_MAPPINGS = {
    "CMK_SaveProjectText": CMK_SaveProjectText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_SaveProjectText": "CMK Save Project Text",
}
