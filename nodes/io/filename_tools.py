import os
import re


class CMK_FilenameBase:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("basename",)

    FUNCTION = "run"
    CATEGORY = "CMK/Toolbox/I-O"

    def run(self, filename):
        # Nur Dateiname ohne Pfad
        name = os.path.basename(filename)

        # Erweiterung entfernen
        name = os.path.splitext(name)[0]

        # " (2)", " (15)" usw. am Ende entfernen
        name = re.sub(r"\s*\(\d+\)$", "", name)

        return (name,)


NODE_CLASS_MAPPINGS = {
    "CMK_FilenameBase": CMK_FilenameBase,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_FilenameBase": "CMK Filename Base",
}
