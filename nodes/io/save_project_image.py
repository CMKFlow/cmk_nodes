import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import folder_paths

from ...pipe.cmk_log_pipe import cmk_render_log
from ...utils.cmk_save_path import save_automatic_folders
from ...utils.cmk_timing import cmk_timed


class _CMKAnyType(str):
    def __ne__(self, other):
        return False


CMK_PROCESS_METADATA_INPUT = _CMKAnyType("*")


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
            "optional": {
                "MODEL (opt)": ("CMK_MODEL_PIPE",),
                # PROCESS is metadata-only here. It may originate from an
                # SDXL, ZIT or already family-neutral result branch.
                "PROCESS": (CMK_PROCESS_METADATA_INPUT,),
                "IMAGE": ("IMAGE",),
                "LOG": ("CMK_LOG_PIPE",),
                "SAVE ENABLED": ("BOOLEAN", {"default": True}),
                "FILENAME PREFIX": ("STRING", {"default": "image"}),
                "OUTPUT FOLDER": ("STRING", {"default": ""}),
                "USE DATE FOLDER": ("BOOLEAN", {"default": True}),
                "PROJECT FOLDER": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE", "CMK_PIPE", "IMAGE", "CMK_LOG_PIPE", "STRING")
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG", "FULLPATH")

    FUNCTION = "run"
    CATEGORY = "CMK/Toolbox/I-O"
    OUTPUT_NODE = True

    def run(
        self,
        PROCESS=None,
        IMAGE=None,
        LOG=None,
        **kwargs,
    ):
        MODEL = kwargs.get("MODEL (opt)")
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

        if MODEL is not None and not isinstance(MODEL, dict):
            raise TypeError("CMK Save Project Image -Pipe-: MODEL must be a CMK model pipe")
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Save Project Image -Pipe-: PROCESS must be a CMK process pipe")

        base_output = Path(folder_paths.get_output_directory()).resolve()

        parts = self._safe_relative_parts(output_folder)
        automatic_parts = save_automatic_folders(PROCESS)
        # Older workflows sometimes stored one or more automatic folders
        # explicitly in OUTPUT FOLDER. Preserve their paths without producing
        # duplicates such as Text2Image/Text2Image or InstantID/InstantID.
        existing = {part.casefold() for part in parts}
        parts.extend(
            part for part in automatic_parts if part.casefold() not in existing
        )

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

        with cmk_timed("90 PNG SAVE", str(full_path)):
            img = IMAGE[0].cpu().numpy()
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(str(full_path))

        log_text = cmk_render_log(LOG)
        if log_text:
            text_path = os.path.splitext(str(full_path))[0] + ".txt"
            with cmk_timed("90 LOG SAVE", str(text_path)):
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
