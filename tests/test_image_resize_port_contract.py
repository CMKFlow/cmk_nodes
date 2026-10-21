from __future__ import annotations

import hashlib
import importlib
from importlib.util import find_spec
import json
from pathlib import Path
import sys
import types
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_image_resize_port_contract"
WORKFLOW = PACKAGE_ROOT / "workflows" / "showcase" / "CMK 2.6 · Full Flow .json"
WORKFLOW_SHA256 = "f9966d0e7bf7da6876f1d745dc9eb80e26b9cff5b3ea5590035a28a99b3db3ac"

EXPECTED_LINKS = {
    22541: (8455, 0, 8456, 0, "CMK_MODEL_PIPE", "MODEL", "MODEL (opt)"),
    22542: (8455, 1, 8456, 1, "CMK_MASK_DETAILER_PROCESS", "MASK DETAILER PROCESS", "PROCESS"),
    22543: (8455, 2, 8456, 2, "IMAGE", "IMAGE", "IMAGE"),
    22544: (8455, 3, 8456, 3, "CMK_LOG_PIPE", "LOG", "LOG"),
    22482: (8455, 4, 8456, 4, "CMK_VISUAL_PIPE", "VISUAL", "VISUAL"),
}

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)
image_resize = importlib.import_module(
    f"{TEST_PACKAGE}.nodes.image.image_resize"
)


class ImageResizePortContractTests(unittest.TestCase):
    def test_python_keeps_model_optional(self):
        inputs = image_resize.CMKImageResize.INPUT_TYPES()
        self.assertNotIn("MODEL (opt)", inputs["required"])
        self.assertEqual(inputs["optional"]["MODEL (opt)"], ("CMK_MODEL_PIPE",))

    def test_frontend_orders_ports_during_definition_registration(self):
        source = (PACKAGE_ROOT / "web" / "js" / "cmk_image_resize_ui.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("function registerModelInputFirst(nodeData)", source)
        self.assertIn("registerModelInputFirst(nodeData);", source)
        self.assertIn('"MODEL (opt)": [type, { ...options, isOptional: true }]', source)
        self.assertIn("delete nodeData.input_order;", source)
        self.assertNotIn("moveModelInputFirst", source)
        self.assertNotIn("node.inputs.splice", source)
        self.assertNotIn("node.inputs.unshift", source)

    def test_installed_frontend_hook_precedes_input_creation(self):
        frontend = find_spec("comfyui_frontend_package")
        if frontend is None or frontend.origin is None:
            self.skipTest("installed ComfyUI frontend package is unavailable")
        assets = Path(frontend.origin).parent / "static" / "assets"
        source_map = next(assets.glob("settingStore-*.js.map"))
        data = json.loads(source_map.read_text(encoding="utf-8"))
        sources = dict(zip(data["sources"], data["sourcesContent"]))
        service = sources["../../src/services/litegraphService.ts"]
        migration = sources["../../src/schemas/nodeDef/migration.ts"]
        ordering = sources["../../src/workbench/utils/nodeDefOrderingUtil.ts"]

        registration = service[service.index("async function registerNodeDef"):]
        self.assertLess(
            registration.index("'beforeRegisterNodeDef'"),
            registration.index("new ComfyNodeDefImpl(nodeDefV1)"),
        )
        self.assertLess(
            registration.index("new ComfyNodeDefImpl(nodeDefV1)"),
            registration.index("LiteGraph.registerNodeType(nodeId, node)"),
        )
        self.assertIn("addInputs(this, ComfyNode.nodeData.inputs)", registration)
        self.assertIn("isOptional,\n    ...options", migration)
        self.assertIn("return Object.values(inputs)", ordering)

    def test_full_flow_links_match_registered_port_contract(self):
        payload = WORKFLOW.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), WORKFLOW_SHA256)
        workflow = json.loads(payload)
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}

        for link_id, expected in EXPECTED_LINKS.items():
            source_id, source_slot, target_id, target_slot, link_type, source_name, target_name = expected
            self.assertEqual(
                links[link_id],
                [link_id, source_id, source_slot, target_id, target_slot, link_type],
            )
            self.assertEqual(nodes[source_id]["outputs"][source_slot]["name"], source_name)
            self.assertEqual(nodes[target_id]["inputs"][target_slot]["name"], target_name)
            self.assertEqual(nodes[target_id]["inputs"][target_slot]["type"], {
                "MODEL (opt)": "CMK_MODEL_PIPE",
                "PROCESS": "CMK_RESULT_PROCESS,CMK_MASK_DETAILER_PROCESS",
                "IMAGE": "IMAGE",
                "LOG": "CMK_LOG_PIPE",
                "VISUAL": "CMK_VISUAL_PIPE",
            }[target_name])


if __name__ == "__main__":
    unittest.main()
