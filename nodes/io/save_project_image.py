import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import folder_paths

from ...pipe.cmk_log_pipe import cmk_render_log


class CMK_SaveProjectImage:
    @staticmethod
    def _safe_relative_parts(value):
        """Return clean relative path components confined below ComfyUI/output."""
        text = str(value or "").strip().replace("\\", "/")
        parts = []
        for raw in text.split("/"):
            part = raw.strip()
            if not part or part in {".", ".."}:
                continue
            clean = re.sub(r"[\x00-\x1f<>:\"|?*]", "_", part).strip(" .")
            if clean:
                parts.append(clean)
        return parts

    @classmethod
    def _safe_project_name(cls, value):
        parts = cls._safe_relative_parts(value)
        return "_".join(parts)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("CMK_MODEL_PIPE",),
                "PROCESS": ("CMK_PIPE",),
                "IMAGE": ("IMAGE",),
                "LOG": ("CMK_LOG_PIPE",),
                "SAVE ENABLED": ("BOOLEAN", {"default": True}),
                "FILENAME PREFIX": ("STRING", {"default": "image"}),
                "OUTPUT FOLDER": ("STRING", {"default": ""}),
                "USE DATE FOLDER": ("BOOLEAN", {"default": True}),
                "PROJECT FOLDER": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE", "CMK_PIPE", "IMAGE", "CMK_LOG_PIPE", "STRING")
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG", "FULLPATH")

    FUNCTION = "run"
    CATEGORY = "CMK/Flow/Finish"
    OUTPUT_NODE = True

    def run(
        self,
        MODEL,
        PROCESS,
        IMAGE,
        LOG,
        **kwargs,
    ):
        save_enabled = bool(kwargs.get("SAVE ENABLED", True))
        filename_prefix = str(kwargs.get("FILENAME PREFIX", "image"))
        output_folder = str(kwargs.get("OUTPUT FOLDER", ""))
        use_date_folder = bool(kwargs.get("USE DATE FOLDER", True))
        project_folder = str(kwargs.get("PROJECT FOLDER", ""))

        if not save_enabled:
            return {
                "ui": {"text": ["SAVE DISABLED"]},
                "result": (MODEL, PROCESS, IMAGE, LOG, ""),
            }

        if not isinstance(MODEL, dict):
            raise TypeError("CMK Save Project Image -Pipe-: MODEL must be a CMK model pipe")
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Save Project Image -Pipe-: PROCESS must be a CMK process pipe")

        base_output = Path(folder_paths.get_output_directory()).resolve()

        parts = self._safe_relative_parts(output_folder)
        flow_mode = "Inpaint" if bool(PROCESS.get("boolean_inpaint_mode", False)) else "Text2Image"
        # Older reference workflows sometimes stored the automatic mode folder
        # explicitly in OUTPUT FOLDER. Keep that value compatible without
        # producing Text2Image/Text2Image or Inpaint/Inpaint.
        if not parts or parts[-1].casefold() != flow_mode.casefold():
            parts.append(flow_mode)

        if use_date_folder:
            parts.append(datetime.now().strftime("%Y-%m-%d"))

        clean_project = self._safe_project_name(project_folder)
        if clean_project:
            parts.append(clean_project)

        target_folder = base_output.joinpath(*parts).resolve()
        if target_folder != base_output and base_output not in target_folder.parents:
            raise ValueError("CMK Save Project Image -Pipe-: output path escapes ComfyUI output")
        target_folder.mkdir(parents=True, exist_ok=True)

        clean_prefix = self._safe_project_name(filename_prefix) or "image"
        counter = 1
        while True:
            filename = f"{clean_prefix}_{counter:05d}.png"
            full_path = target_folder / filename
            if not full_path.exists():
                break
            counter += 1

        img = IMAGE[0].cpu().numpy()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(str(full_path))

        log_text = cmk_render_log(LOG)
        if log_text:
            text_path = os.path.splitext(str(full_path))[0] + ".txt"
            with open(text_path, "w", encoding="utf-8") as file:
                file.write(log_text)

        return {
            "ui": {"text": [str(full_path)]},
            "result": (MODEL, PROCESS, IMAGE, LOG, str(full_path)),
        }


NODE_CLASS_MAPPINGS = {
    "CMK_SaveProjectImage": CMK_SaveProjectImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CMK_SaveProjectImage": "CMK Save Project Image -Pipe-",
}
