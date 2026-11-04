from __future__ import annotations

from ...utils.face_set_utils import normalize_face_set, summarize_face_set


class CMKFaceSetInfo:
    """Inspect a CMK_FACE_SET payload without changing it."""

    CATEGORY = "CMK/Toolbox/Face"
    RETURN_TYPES = ("INT", "STRING", "CMK_FACE_SET")
    RETURN_NAMES = ("face_count", "summary", "face_set")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_set": ("CMK_FACE_SET",),
            }
        }

    def run(self, face_set):
        fs = normalize_face_set(face_set)
        return (int(fs.get("total_faces", 0)), summarize_face_set(fs), fs)
