from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import comfy.cli_args


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TEST_PACKAGE = "_cmk_content_guard_ui_contract"

comfy.cli_args.args.cpu = True

package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

distribution = importlib.import_module(f"{TEST_PACKAGE}.cmk_distribution")
create_image = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_pipe_image")
image_loader = importlib.import_module(f"{TEST_PACKAGE}.pipe.loaders.cmk_swap_image_loader")
video_loader = importlib.import_module(f"{TEST_PACKAGE}.nodes.video.face_swap_video_loader")
node_mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")


class ContentGuardUIContractTests(unittest.TestCase):
    AUTHORIZED = (
        (create_image.CMKPipeCreateImage, "optional"),
        (image_loader.CMKSwapImageLoaderPipe, "required"),
        (video_loader.CMKFaceSwapVideoLoader, "required"),
    )

    def test_only_three_authorized_baseline_schemas_expose_the_control(self):
        with patch.object(distribution, "DISTRIBUTION_MODE", distribution.BASELINE):
            for node_class, section in self.AUTHORIZED:
                schema = node_class.INPUT_TYPES()
                self.assertIn(distribution.CONTENT_GUARD_PROCESS_KEY, schema[section])
                self.assertIs(schema[section][distribution.CONTENT_GUARD_PROCESS_KEY][1]["default"], True)

            actual = set()
            for name, node_class in node_mappings.NODE_CLASS_MAPPINGS.items():
                schema = node_class.INPUT_TYPES()
                for section in ("required", "optional", "hidden"):
                    if distribution.CONTENT_GUARD_PROCESS_KEY in schema.get(section, {}):
                        actual.add((name, section))
        self.assertEqual(actual, {
            ("CMKPipeCreateImage", "optional"),
            ("CMKSwapImageLoaderPipe", "required"),
            ("CMKFaceSwapVideoLoader", "required"),
        })

    def test_release_schemas_expose_no_authorized_control_and_force_on(self):
        with patch.object(distribution, "DISTRIBUTION_MODE", distribution.RELEASE):
            for node_class, section in self.AUTHORIZED:
                self.assertNotIn(distribution.CONTENT_GUARD_PROCESS_KEY, node_class.INPUT_TYPES()[section])
            for name, node_class in node_mappings.NODE_CLASS_MAPPINGS.items():
                schema = node_class.INPUT_TYPES()
                for section in ("required", "optional", "hidden"):
                    self.assertNotIn(
                        distribution.CONTENT_GUARD_PROCESS_KEY,
                        schema.get(section, {}),
                        f"{name}.{section}",
                    )
            self.assertTrue(distribution.effective_content_guard_enabled({
                distribution.CONTENT_GUARD_PROCESS_KEY: False,
            }))
            self.assertTrue(distribution.content_guard_prompt_state({
                "1": {
                    "class_type": "CMKPipeCreateImage",
                    "inputs": {distribution.CONTENT_GUARD_WIDGET: False},
                }
            }))

    def test_no_production_source_injects_a_functional_guard_input(self):
        forbidden = "_cmk_content_guard_enabled"
        offenders = []
        for pattern in ("*.py", "*.js"):
            for path in PACKAGE_ROOT.rglob(pattern):
                if "tests" in path.parts:
                    continue
                if forbidden in path.read_text(encoding="utf-8"):
                    offenders.append(path.relative_to(PACKAGE_ROOT).as_posix())
        self.assertEqual(offenders, [])

    def test_packaged_workflows_contain_no_guard_widget_serialization(self):
        offenders = []
        for folder in ("subgraphs", "workflows"):
            for path in (PACKAGE_ROOT / folder).rglob("*.json"):
                data = json.loads(path.read_text(encoding="utf-8"))
                text = json.dumps(data, ensure_ascii=False)
                if "_cmk_content_guard_enabled" in text or '"content_guard_enabled"' in text:
                    offenders.append(path.relative_to(PACKAGE_ROOT).as_posix())
        self.assertEqual(offenders, [])

    def test_all_packaged_workflows_remain_valid_json(self):
        for folder in ("subgraphs", "workflows"):
            for path in (PACKAGE_ROOT / folder).rglob("*.json"):
                with self.subTest(path=path.name):
                    json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
