"""CMK Flow · Image Resize v2 (2.8): the unchanged Image Resize with the port
contract MODEL ZIT, MODEL SDXL, PROCESS, IMAGE, LOG, VISUAL.

Processing, bypass and diagnostic are compared with the legacy node; both
MODEL pipes are transport and stay out of the cache key.  The legacy node and
its port contract test are not touched.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_image_resize_v2_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

import torch  # noqa: E402

legacy = importlib.import_module(f"{TEST_PACKAGE}.nodes.image.image_resize")
v2 = importlib.import_module(f"{TEST_PACKAGE}.nodes.image.image_resize_v2")
mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")

CMK_TEST_TIER = "T0"
torch.manual_seed(3)
IMAGE = torch.rand(1, 96, 160, 3)
PROCESS = {"result_contract": "family_neutral", "seed": 1}
LOG = {"type": "CMK_LOG_PIPE", "blocks": []}
MODEL_ZIT = {"model_role": "postprocess", "model_family": "z_image_turbo", "materialized": False}
MODEL_SDXL = {"model_role": "postprocess", "model_family": "sdxl", "materialized": False}
SETTINGS = {"scale_method": "Scale to Max Dimension", "interpolation": "lanczos", "largest_size": 80}


def run(node, enabled=True, **models):
    switch = {"scale_mode": "ENABLE" if enabled else "DISABLE"} if node is v2.CMKImageResizeV2 else {"SCALE ENABLE": enabled}
    return node().resize(PROCESS, IMAGE, LOG, **{**switch, **SETTINGS, **models})


class Ports(unittest.TestCase):
    def test_registration_and_order(self):
        self.assertIs(mappings.NODE_CLASS_MAPPINGS["CMKImageResizeV2"], v2.CMKImageResizeV2)
        self.assertEqual(mappings.NODE_DISPLAY_NAME_MAPPINGS["CMKImageResizeV2"], "CMK Flow · Image Resize v2")
        self.assertEqual(v2.CMKImageResizeV2.RETURN_NAMES,
                         ("MODEL ZIT", "MODEL SDXL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"))
        self.assertEqual(v2.CMKImageResizeV2.RETURN_TYPES[:2], ("CMK_MODEL_PIPE", "CMK_MODEL_PIPE"))
        types_ = v2.CMKImageResizeV2.INPUT_TYPES()
        old = legacy.CMKImageResize.INPUT_TYPES()
        required = dict(old["required"])
        self.assertEqual(list(types_["required"]), [("scale_mode" if n == "SCALE ENABLE" else n) for n in required])
        self.assertEqual(types_["required"]["scale_mode"], (["ENABLE", "DISABLE"], {"default": "ENABLE"}))
        self.assertEqual({k: v for k, v in types_["required"].items() if k != "scale_mode"},
                         {k: v for k, v in required.items() if k != "SCALE ENABLE"})
        self.assertEqual(list(types_["optional"])[:2], ["MODEL ZIT", "MODEL SDXL"])
        self.assertNotIn("MODEL (opt)", types_["optional"])
        self.assertEqual({k: v for k, v in types_["optional"].items() if k not in ("MODEL ZIT", "MODEL SDXL")},
                         {k: v for k, v in old["optional"].items() if k != "MODEL (opt)"})
        self.assertIn("MODEL (opt)", legacy.CMKImageResize.INPUT_TYPES()["optional"], "legacy contract unchanged")

    def test_frontend_orders_the_v2_ports(self):
        source = (PACKAGE_ROOT / "web" / "js" / "cmk_image_resize_ui.js").read_text(encoding="utf-8")
        self.assertIn('const V2_MODEL_INPUTS = ["MODEL ZIT", "MODEL SDXL"];', source)
        self.assertIn("registerModelPipesFirst(nodeData)", source)
        self.assertIn("registerModelInputFirst(nodeData);", source, "the legacy registration stays")


class Processing(unittest.TestCase):
    def test_resize_is_the_legacy_resize(self):
        old = run(legacy.CMKImageResize, **{"MODEL (opt)": MODEL_SDXL})
        new = run(v2.CMKImageResizeV2, **{"MODEL ZIT": MODEL_ZIT, "MODEL SDXL": MODEL_SDXL})
        self.assertIs(new[0], MODEL_ZIT)
        self.assertIs(new[1], MODEL_SDXL)
        self.assertIs(new[2], PROCESS)
        self.assertTrue(torch.equal(new[3], old[2]))
        self.assertEqual(tuple(new[3].shape), (1, 48, 80, 3))
        self.assertEqual(new[4], old[3])
        self.assertEqual(new[6]["node"], old[5]["node"], "same diagnostic")
        self.assertEqual([s["title"] for s in new[6]["stages"]], [s["title"] for s in old[5]["stages"]])
        self.assertEqual(new[5]["providers"][0]["module_type"], old[4]["providers"][0]["module_type"])

    def test_bypass_and_unconnected_models(self):
        new = run(v2.CMKImageResizeV2, enabled=False, **{"MODEL ZIT": MODEL_ZIT, "MODEL SDXL": MODEL_SDXL})
        self.assertIs(new[3], IMAGE, "SCALE ENABLE off: the input image")
        self.assertIs(new[0], MODEL_ZIT)
        empty = run(v2.CMKImageResizeV2)
        self.assertIsNone(empty[0])
        self.assertIsNone(empty[1])

    def test_models_are_transport_in_the_cache_key(self):
        def prompt(**models):
            return {"5": {"class_type": "CMKImageResizeV2", "inputs": {
                "PROCESS": ["1", 1], "IMAGE": ["1", 2], "LOG": ["1", 3], "scale_mode": "ENABLE", **SETTINGS, **models}},
                "1": {"class_type": "CMKImageLoadAndResizePipe", "inputs": {"image": "a.png"}},
                "2": {"class_type": "CMKPostProcessModelSourcePipe", "inputs": {"ckpt_name": "x"}},
                "3": {"class_type": "CMKPostProcessModelSourcePipe", "inputs": {"ckpt_name": "y"}}}

        bare = v2.CMKImageResizeV2._cache_key(prompt(), "5", SETTINGS["scale_method"])[0]
        linked = v2.CMKImageResizeV2._cache_key(prompt(**{"MODEL ZIT": ["2", 1], "MODEL SDXL": ["2", 0]}),
                                                "5", SETTINGS["scale_method"])[0]
        other = v2.CMKImageResizeV2._cache_key(prompt(**{"MODEL ZIT": ["3", 1], "MODEL SDXL": ["3", 0]}),
                                               "5", SETTINGS["scale_method"])[0]
        self.assertTrue(bare)
        self.assertEqual(bare, linked)
        self.assertEqual(linked, other, "another model source does not invalidate the resize")
        bigger = prompt(); bigger["5"]["inputs"]["largest_size"] = 120
        self.assertNotEqual(v2.CMKImageResizeV2._cache_key(bigger, "5", SETTINGS["scale_method"])[0], bare)


if __name__ == "__main__":
    unittest.main()
