import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MERGE_TYPE = "CMKFamilyResultMergePipe"
WIDGET_INPUTS = [
    "postprocess_checkpoint",
    "postprocess_vae",
    "use_checkpoint_vae",
]
ZIT_INPUTS = ["MODEL ZIT", "PROCESS ZIT", "IMAGE ZIT", "LOG ZIT", "VISUAL ZIT"]
SDXL_INPUTS = [
    "MODEL SDXL",
    "PROCESS SDXL",
    "IMAGE SDXL",
    "LOG SDXL",
    "VISUAL SDXL",
]
EXPECTED_INPUTS = WIDGET_INPUTS + ZIT_INPUTS + SDXL_INPUTS

# Reduced from the user-verified three-node Slot-Test workflow. The original
# file was created with frontend 1.52.7 and reproduces the bad retargeting when
# opened with frontend 1.53.6 while live input arrays are reordered.
SLOT_TEST = {
    "nodes": [
        {
            "id": 8348,
            "type": "ZIT sampler subgraph",
            "outputs": [
                {"name": "MODEL", "type": "CMK_MODEL_PIPE"},
                {"name": "PROCESS", "type": "CMK_PROCESS_Z_IMAGE"},
                {"name": "IMAGE", "type": "IMAGE"},
                {"name": "LOG", "type": "CMK_LOG_PIPE"},
                {"name": "VISUAL", "type": "CMK_VISUAL_PIPE"},
            ],
        },
        {
            "id": 8349,
            "type": "SDXL refiner subgraph",
            "outputs": [
                {"name": "MODEL", "type": "CMK_MODEL_PIPE"},
                {"name": "PROCESS", "type": "CMK_PROCESS_SDXL"},
                {"name": "IMAGE REFINED", "type": "IMAGE"},
                {"name": "LOG", "type": "CMK_LOG_PIPE"},
                {"name": "VISUAL", "type": "CMK_VISUAL_PIPE"},
            ],
        },
        {
            "id": 8350,
            "type": MERGE_TYPE,
            "inputs": [{"name": name} for name in EXPECTED_INPUTS],
        },
    ],
    "links": [
        [21942, 8348, 0, 8350, 3, "CMK_MODEL_PIPE"],
        [21943, 8348, 1, 8350, 4, "CMK_PROCESS_Z_IMAGE"],
        [21944, 8348, 2, 8350, 5, "IMAGE"],
        [21945, 8348, 3, 8350, 6, "CMK_LOG_PIPE"],
        [21946, 8348, 4, 8350, 7, "CMK_VISUAL_PIPE"],
        [21947, 8349, 0, 8350, 8, "CMK_MODEL_PIPE"],
        [21948, 8349, 1, 8350, 9, "CMK_PROCESS_SDXL"],
        [21949, 8349, 2, 8350, 10, "IMAGE"],
        [21950, 8349, 3, 8350, 11, "CMK_LOG_PIPE"],
        [21951, 8349, 4, 8350, 12, "CMK_VISUAL_PIPE"],
    ],
}


def _family_for_node(node):
    output_types = {output.get("type") for output in node.get("outputs", [])}
    if "CMK_PROCESS_Z_IMAGE" in output_types:
        return "ZIT"
    if "CMK_PROCESS_SDXL" in output_types:
        return "SDXL"
    return None


def _assert_family_links(test, data, label):
    nodes = {node["id"]: node for node in data["nodes"]}
    merges = [node for node in data["nodes"] if node.get("type") == MERGE_TYPE]
    test.assertGreaterEqual(len(merges), 1, label)
    for merge in merges:
        test.assertEqual(
            EXPECTED_INPUTS,
            [item["name"] for item in merge["inputs"][: len(EXPECTED_INPUTS)]],
            label,
        )
        incoming = [link for link in data["links"] if link[3] == merge["id"]]
        family_links = [link for link in incoming if link[4] >= len(WIDGET_INPUTS)]
        test.assertEqual(10, len(family_links), label)
        for link in family_links:
            source = nodes[link[1]]
            family = _family_for_node(source)
            target_name = merge["inputs"][link[4]]["name"]
            test.assertIsNotNone(family, f"{label}: link {link[0]} has no family")
            test.assertTrue(
                target_name.endswith(f" {family}"),
                f"{label}: link {link[0]} from {family} targets {target_name}",
            )


class FamilyMergeSlotRegressionTests(unittest.TestCase):
    def test_verified_three_node_slot_workflow_keeps_family_assignments(self):
        _assert_family_links(self, SLOT_TEST, "Slot-Test")

    def test_layout_extension_never_mutates_live_input_order_or_link_slots(self):
        source = (ROOT / "web" / "js" / "cmk_family_merge_layout.js").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("inputs.splice", source)
        self.assertNotIn("target_slot =", source)
        self.assertNotIn("arrangeInputs", source)

    def test_python_contract_owns_the_canonical_input_order(self):
        source = (ROOT / "pipe" / "cmk_family_result.py").read_text(encoding="utf-8")
        section = source[
            source.index("class CMKFamilyResultMergePipe"):
            source.index("class _CMKSinglePostProcessBoundary")
        ]
        positions = [section.index(f'"{name}"') for name in ZIT_INPUTS + SDXL_INPUTS]
        self.assertEqual(sorted(positions), positions)

    def test_all_packaged_combined_workflows_keep_semantic_family_links(self):
        paths = []
        for path in sorted((ROOT / "workflows" / "showcase").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if any(node.get("type") == MERGE_TYPE for node in data.get("nodes", [])):
                paths.append((path, data))
        self.assertEqual(7, len(paths))
        for path, data in paths:
            _assert_family_links(self, data, path.name)


if __name__ == "__main__":
    unittest.main()
