"""Save category of the Visualizer: the real operation decides.

Inpaint is the category for every strategy (Custom, Replace, Remove, Extend) and
for Outpaint.  CMKMaskDetailerFlowGate builds the public PROCESS from a fixed field
list that dropped the Inpaint flags, so a workflow ending in MaskDetailer saved
under Text2Image, even when the module was off.

Model-free: install_generation_seams fakes CLIP, VAE, Fooocus and sampling.
"""

from __future__ import annotations

import unittest
from datetime import date

from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"
SAVE = {"SAVE ENABLED": True}
IMAGE = "cmk_s1_input.png"
STRATEGIES = ("Custom", "Replace Object", "Remove Object", "Extend Image")


def folders(run):
    """Folders below output, without the date folder and the file name."""
    assert run["succeeded"], run["errors"]
    today = date.today().isoformat()
    return [[part for part in name.split("/")[:-1] if part != today] for name in sorted(run["new_files"])]


class SaveCategory(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        run = lambda **kwargs: cls.run_case("hybrid_stop_sequence", **{"stops": [False], "visualizer_inputs": SAVE, **kwargs})
        cls.text2image_zit = run()
        cls.text2image_sdxl = run(start_inputs=[{"model_family": "SDXL"}])
        cls.inpaint_sdxl = run(inpaint_source=IMAGE, start_inputs=[{"model_family": "SDXL"}])
        cls.inoutpaint = run(inpaint_source=IMAGE, start_inputs=[{"model_family": "SDXL", "outpaint_on": True}])
        cls.inoutpaint_zit = run(inpaint_source=IMAGE, start_inputs=[{"outpaint_on": True}], postprocess_tail=True)
        cls.strategies = {
            family: {name: run(inpaint_source=IMAGE, start_inputs=[{"model_family": family, "process_mode": name}])
                     for name in STRATEGIES}
            for family in ("SDXL", "Hybrid")
        }
        cls.custom = run(inpaint_source=IMAGE, postprocess_tail=True,
                         visualizer_inputs={**SAVE, "output folder": "Project/Shots"})
        cls.repeat = run(stops=[False, False], inpaint_source=IMAGE, postprocess_tail=True,
                         mode="ram_pressure", evict_between_runs=True)
        tail = {"postprocess_tail": True, "mask_detailer_tail": True}
        cls.mask_sdxl = run(inpaint_source=IMAGE, start_inputs=[{"model_family": "SDXL", "outpaint_on": True}], **tail)
        cls.mask_zit = run(inpaint_source=IMAGE, start_inputs=[{"outpaint_on": True}], **tail)
        cls.mask_text2image = run(**tail)
        cls.active = cls.run_case("mask_detailer_active_process")

    def test_text2image(self):
        self.assertEqual(folders(self.text2image_zit["runs"][0]), [["Text2Image"]])
        self.assertEqual(folders(self.text2image_sdxl["runs"][0]), [["Text2Image"]])

    def test_inpaint(self):
        self.assertEqual(folders(self.inpaint_sdxl["runs"][0]), [["Inpaint"]])

    def test_inoutpaint(self):
        self.assertEqual(folders(self.inoutpaint["runs"][0]), [["Inpaint"]])

    def test_inoutpaint_with_zit_finish(self):
        self.assertEqual(folders(self.inoutpaint_zit["runs"][0]), [["Inpaint"]])

    def test_every_strategy_is_inpaint(self):
        for family, results in self.strategies.items():
            for name, result in results.items():
                with self.subTest(family=family, strategy=name):
                    self.assertEqual(folders(result["runs"][0]), [["Inpaint"]])

    def test_custom_folder_is_kept(self):
        self.assertEqual(folders(self.custom["runs"][0]), [["Project", "Shots", "Inpaint"]])

    def test_repeat_on_cache_hit_saves_a_new_file_in_the_same_category(self):
        first, second = self.repeat["runs"]
        self.assertEqual(second["sampling"]["zit"], 0)
        self.assertEqual(folders(first), [["Inpaint"]])
        self.assertEqual(folders(second), [["Inpaint"]])
        self.assertNotEqual(first["new_files"], second["new_files"])

    def test_behind_mask_detailer(self):
        self.assertEqual(folders(self.mask_sdxl["runs"][0]), [["Inpaint"]])
        self.assertEqual(folders(self.mask_zit["runs"][0]), [["Inpaint"]])
        self.assertEqual(folders(self.mask_text2image["runs"][0]), [["Text2Image"]])

    def test_active_mask_detailer_keeps_the_operation(self):
        self.assertEqual(self.active, {"inpaint": ["Inpaint"], "hybrid_inpaint": ["Inpaint"],
                                       "text2image": ["Text2Image"]})


class PureRule(unittest.TestCase):
    def test_image_sources_keep_their_category(self):
        from importlib import util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "utils" / "cmk_save_path.py"
        spec = util.spec_from_file_location("cmk_save_path_under_test", path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rule = module.save_base_folder
        self.assertEqual(rule({"source_model_family": "image", "boolean_inpaint_mode": True}), "ImageProcessing")
        self.assertEqual(rule({"pipe_origin": "CMK Image Load and Resize -Pipe-"}), "ImageProcessing")
        for mode in ("custom", "replace", "remove", "extend"):
            self.assertEqual(rule({"model_family": "sdxl", "boolean_inpaint_mode": True,
                                   "inpaint_process_mode": mode}), "Inpaint")
            self.assertEqual(rule({"model_family": "sdxl", "boolean_inpaint_mode": False,
                                   "inpaint_process_mode": mode}), "Text2Image")

    def test_unknown_operation_claims_no_category(self):
        from importlib import util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "utils" / "cmk_save_path.py"
        spec = util.spec_from_file_location("cmk_save_path_under_test", path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIsNone(module.save_base_folder({"model_family": "sdxl"}))
        self.assertEqual(module.save_automatic_folders({"model_family": "sdxl"}), [])
        self.assertEqual(module.save_automatic_folders({"model_family": "sdxl", "instantid_enabled": True}),
                         ["InstantID"])
        self.assertEqual(module.save_automatic_folders({"model_family": "sdxl", "boolean_inpaint_mode": False}),
                         ["Text2Image"])


if __name__ == "__main__":
    unittest.main()
