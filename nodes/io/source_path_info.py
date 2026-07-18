import os


class CMKSourcePathInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("source_file", "source_path")
    FUNCTION = "run"
    CATEGORY = "CMK/Toolbox/I-O"

    def run(self, source_path):
        source_file = os.path.basename(source_path)
        source_dir = os.path.dirname(source_path)
        return (source_file, source_dir)
