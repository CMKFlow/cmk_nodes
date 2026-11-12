"""2.7.1 Visualizer POSTPROCESS group: contract of the new PostProcess MODEL ZIT
input, its frontend labels, the slot offset and the unchanged Clean View size.

The visual result (group heading, separation, Clean View pixel identity) is
checked in the real frontend; this file pins the code contracts behind it.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_visualizer_postprocess_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

visual = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_visual")

CMK_TEST_TIER = "T0"
WEB = PACKAGE_ROOT / "web" / "js"

SOCKETS = [
    "MODEL ZIT", "PROCESS ZIT", "IMAGE ZIT", "LOG ZIT", "VISUAL ZIT",
    "MODEL SDXL", "PROCESS SDXL", "IMAGE SDXL", "LOG SDXL", "VISUAL SDXL",
    "POSTPROCESS MODEL ZIT", "MODEL (opt)", "PROCESS", "IMAGE", "LOG", "VISUAL",
]


class VisualizerInputs(unittest.TestCase):
    def setUp(self):
        self.optional = visual.CMKVisualizer.INPUT_TYPES()["optional"]

    def test_socket_order_and_unique_names(self):
        names = list(self.optional)[: len(SOCKETS)]
        self.assertEqual(names, SOCKETS)
        self.assertEqual(len(set(names)), len(names))

    def test_postprocess_model_zit_is_a_lazy_model_pipe(self):
        self.assertEqual(self.optional["POSTPROCESS MODEL ZIT"], ("CMK_MODEL_PIPE", {"lazy": True}))

    def test_existing_contracts_unchanged(self):
        self.assertEqual(
            visual.CMKVisualizer._INPUT_GROUPS["COMMON"],
            ("MODEL (opt)", "PROCESS", "IMAGE", "LOG", "VISUAL"),
        )
        self.assertNotIn("POSTPROCESS MODEL ZIT", sum(visual.CMKVisualizer._INPUT_GROUPS.values(), ()))
        self.assertEqual(self.optional["MODEL (opt)"][1], {"lazy": True})


class VisualizerFrontend(unittest.TestCase):
    def test_labels_and_slot_offset(self):
        source = (WEB / "cmk_visualizer.js").read_text(encoding="utf-8")
        self.assertIn("const VISUALIZER_COMMON_OFFSET = 11;", source)
        self.assertIn('[POSTPROCESS_MODEL_ZIT]: "MODEL ZIT"', source)
        self.assertIn('"MODEL (opt)": "MODEL SDXL"', source)

    def test_group_is_matched_by_input_name_and_hidden_in_clean_view(self):
        source = (WEB / "cmk_family_merge_layout.js").read_text(encoding="utf-8")
        self.assertIn(
            '"POSTPROCESS MODEL ZIT", "MODEL (opt)", "PROCESS", "IMAGE", "LOG", "VISUAL",', source
        )
        self.assertIn('[data-cmk-clean-view="true"] .cmk-postprocess-panel', source)
        self.assertIn(':not([data-cmk-clean-view="true"]) .cmk-postprocess-start', source)
        self.assertNotIn('new Set(["MODEL SDXL", "MODEL"])', source)

    def test_clean_view_compact_size_ignores_the_new_input(self):
        source = (WEB / "cmk_clean_view_prototype.js").read_text(encoding="utf-8")
        self.assertIn('const COMPACT_HEIGHT_EXEMPT_INPUTS = new Set(["POSTPROCESS MODEL ZIT"]);', source)
        self.assertIn("!COMPACT_HEIGHT_EXEMPT_INPUTS.has(slot?.name)", source)


if __name__ == "__main__":
    unittest.main()
