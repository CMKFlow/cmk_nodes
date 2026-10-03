import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "utils" / "cmk_save_path.py"
SPEC = importlib.util.spec_from_file_location("cmk_save_path_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SaveProjectImagePathTests(unittest.TestCase):
    def test_loaded_image_uses_image_processing_base(self):
        self.assertEqual(
            MODULE.save_automatic_folders(
                {
                    "source_model_family": "image",
                    "pipe_origin": "CMK Image Load and Resize -Pipe-",
                }
            ),
            ["ImageProcessing"],
        )

    def test_generation_bases_remain_compatible(self):
        self.assertEqual(MODULE.save_automatic_folders({}), ["Text2Image"])
        self.assertEqual(
            MODULE.save_automatic_folders({"boolean_inpaint_mode": True}),
            ["Inpaint"],
        )

    def test_hybrid_inpaint_uses_inpaint_base_despite_zit_execution_mode(self):
        self.assertEqual(
            MODULE.save_automatic_folders(
                {
                    "model_family": "z_image_turbo",
                    "hybrid_mode": True,
                    "hybrid_inpaint_mode": True,
                    "boolean_inpaint_mode": False,
                    "generation_mode": "inpaint",
                    "samples": object(),
                }
            ),
            ["Inpaint"],
        )

    def test_hybrid_text2image_remains_text2image(self):
        self.assertEqual(
            MODULE.save_automatic_folders(
                {
                    "model_family": "z_image_turbo",
                    "hybrid_mode": True,
                    "hybrid_inpaint_mode": False,
                    "boolean_inpaint_mode": False,
                    "generation_mode": "text2image",
                    "samples": object(),
                }
            ),
            ["Text2Image"],
        )

    def test_direct_sampled_family_uses_the_same_base_as_visualizer_result(self):
        sampled = object()
        self.assertEqual(
            MODULE.save_automatic_folders(
                {
                    "source_model_family": "image",
                    "model_family": "sdxl",
                    "family_active": True,
                    "image_1st_pass": sampled,
                }
            ),
            ["Text2Image"],
        )
        self.assertEqual(
            MODULE.save_automatic_folders(
                {
                    "source_model_family": "image",
                    "model_family": "z_image_turbo",
                    "family_active": True,
                    "samples": sampled,
                    "boolean_inpaint_mode": True,
                }
            ),
            ["Inpaint"],
        )

    def test_mask_detailer_keeps_image_processing_origin(self):
        self.assertEqual(
            MODULE.save_automatic_folders(
                {
                    "type": "CMK_MASK_DETAILER_PROCESS",
                    "source_model_family": "image",
                    "model_family": "sdxl",
                    "image_1st_pass": object(),
                }
            ),
            ["ImageProcessing"],
        )

    def test_applied_stages_are_ordered_and_combined(self):
        self.assertEqual(
            MODULE.save_automatic_folders(
                {
                    "source_model_family": "image",
                    "instantid_applied": True,
                    "faceswap_applied": True,
                    "face_rebuild_applied": True,
                }
            ),
            ["ImageProcessing", "InstantID", "FaceSwap", "FaceRebuild"],
        )

    def test_disabled_stage_markers_do_not_create_folders(self):
        self.assertEqual(
            MODULE.save_automatic_folders(
                {
                    "source_model_family": "image",
                    "instantid_applied": False,
                    "faceswap_applied": False,
                    "face_rebuild_enabled": False,
                }
            ),
            ["ImageProcessing"],
        )

    def test_current_instantid_and_face_rebuild_keys_remain_supported(self):
        self.assertEqual(
            MODULE.save_automatic_folders(
                {
                    "instantid_enabled": True,
                    "face_rebuild_enabled": True,
                }
            ),
            ["Text2Image", "InstantID", "FaceRebuild"],
        )

    def test_faceswap_boundary_does_not_mutate_process(self):
        source = (ROOT / "pipe" / "cmk_module_boundary_cache.py").read_text(
            encoding="utf-8"
        )
        start = source.index("class CMKFaceSwapBoundaryCache:")
        section = source[start:]
        self.assertNotIn("PROCESS", section)


if __name__ == "__main__":
    unittest.main()
