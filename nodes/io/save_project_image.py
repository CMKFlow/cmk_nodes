import os
from datetime import datetime

import numpy as np
from PIL import Image
import folder_paths

from ...pipe.cmk_log_pipe import cmk_render_log


class CMK_SaveProjectImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "LOG": ("CMK_LOG_PIPE",),
                "SAVE ENABLED": ("BOOLEAN", {"default": True}),
                "FILENAME PREFIX": ("STRING", {"default": "image"}),
                "OUTPUT FOLDER": ("STRING", {"default": ""}),
                "USE DATE FOLDER": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("FULLPATH",)

    FUNCTION = "run"
    CATEGORY = "CMK/Flow/Finish"
    OUTPUT_NODE = True

    def run(
        self,
        IMAGE,
        LOG,
        **kwargs,
    ):
        save_enabled = bool(kwargs.get("SAVE ENABLED", True))
        filename_prefix = str(kwargs.get("FILENAME PREFIX", "image"))
        output_folder = str(kwargs.get("OUTPUT FOLDER", ""))
        use_date_folder = bool(kwargs.get("USE DATE FOLDER", True))

        if not save_enabled:
            return {
                "ui": {"text": ["SAVE DISABLED"]},
                "result": ("",),
            }

        base_output = folder_paths.get_output_directory()

        parts = []
        if output_folder.strip():
            parts.append(output_folder.strip())

        if use_date_folder:
            parts.append(datetime.now().strftime("%Y-%m-%d"))

        target_folder = os.path.join(base_output, *parts)
        os.makedirs(target_folder, exist_ok=True)

        clean_prefix = filename_prefix.strip() or "image"
        counter = 1
        while True:
            filename = f"{clean_prefix}_{counter:05d}.png"
            full_path = os.path.join(target_folder, filename)
            if not os.path.exists(full_path):
                break
            counter += 1

        img = IMAGE[0].cpu().numpy()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(full_path)

        log_text = cmk_render_log(LOG)
        if log_text:
            text_path = os.path.splitext(full_path)[0] + ".txt"
            with open(text_path, "w", encoding="utf-8") as file:
                file.write(log_text)

        return {
            "ui": {"text": [full_path]},
            "result": (full_path,),
        }


NODE_CLASS_MAPPINGS = {
    "CMK_SaveProjectImage": CMK_SaveProjectImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_SaveProjectImage": "CMK Save Project Image -Pipe-",
}
