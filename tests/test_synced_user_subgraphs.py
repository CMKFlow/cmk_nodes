from __future__ import annotations

import json
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SUBGRAPHS = PACKAGE_ROOT / "subgraphs"

EXPECTED = {
    "CMK Flow · 15A InstantID Regional SDXL.json": {
        "id": "f3c9666b-a0f1-4032-bbec-aff6ac4fe59b",
        "inputs": [
            "MODEL", "PROCESS", "SAMPLED", "LOG", "VISUAL",
            "opt_image_file", "opt_image_file_1",
        ],
        "layout": {
            ("CMKLoadImage", (240, 230)): ((450, 660), 0),
            ("CMKInstantIDRegionalAdvancedSDXLPipe", (1160, 230)): ((520, 660), 1),
            ("CMKInstantIDBoundary", (930, 945)): ((230, 110), 2),
            ("CMKProcessForwardPipe", (1160, 945)): ((235, 30), 3),
            ("CMKProcessEnableFlag", (470, 945)): ((230, 35), 4),
            ("CMKSamplerBypassGate", (700, 945)): ((230, 195), 5),
            ("CMKLoadImage", (700, 230)): ((450, 660), 6),
        },
    },
    "CMK Flow · FaceRebuild SDXL · Advanced.json": {
        "id": "ffd2127b-6fcf-4ea8-ad8c-e259b8653e40",
        "inputs": [
            "MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL",
            "facerebuild_global_enable", "opt_image_file",
            "opt_image_file_1", "opt_image_file_2",
        ],
        "layout": {
            ("CMKLoadImage", (45, 830)): ((245, 362), 0),
            ("CMKLoadImage", (50, 1230)): ((240, 318), 1),
            ("CMKLoadImage", (50, 1630)): ((240, 318), 2),
            ("CMKInstantIDFaceRebuildAdvancedSDXL", (300, 830)): ((450, 1165), 3),
            ("CMKFaceRebuildBoundaryCache", (1165, 2040)): ((230, 90), 4),
            ("CMKResultProcessForwardPipe", (1395, 2040)): ((280, 30), 5),
            ("CMKVisualCompare", (760, 830)): ((1105, 1165), 6),
            ("CMKModuleBypassGate", (935, 2040)): ((230, 235), 7),
            # OD-2: the public MODEL output is forwarded from SG_IN.MODEL.
            ("CMKModelForwardPipe", (1395, 2090)): ((235, 30), 0),
        },
    },
    "CMK Flow · FaceSwap · Advanced.json": {
        "id": "062af015-5929-4d94-86bd-fd060f67ee53",
        "inputs": [
            "MODEL (opt)", "PROCESS", "IMAGE_TARGET", "LOG", "VISUAL",
            "FACESWAP ENABLE", "opt_image_file", "opt_image_file_1",
            "opt_image_file_2",
        ],
        "layout": {
            ("CMKFaceSwapImagePipe", (720, 940)): ((380, 400), 8),
            ("CMKLoadImage", (720, 540)): ((380, 340), 9),
            ("CMKFaceSwapBoundaryCache", (1610, 1400)): ((230, 90), 10),
            ("CMKVisualCompare", (1890, 540)): ((740, 800), 11),
            ("CMKFaceSwapImagePipe", (1110, 940)): ((380, 400), 12),
            ("CMKLoadImage", (1110, 540)): ((380, 340), 1),
            ("CMK_SEGSConcate", (1380, 1400)): ((230, 134), 2),
            ("CMKLoadImage", (1500, 540)): ((380, 340), 3),
            ("CMKFaceSwapImagePipe", (1500, 940)): ((380, 400), 4),
            ("CMKResultUnpackPipe", (920, 1400)): ((230, 90), 5),
            ("CMKResultPackPipe", (1850, 1400)): ((230, 90), 6),
            ("CMKResultProcessForwardPipe", (2090, 1400)): ((270, 30), 7),
            ("CMKModuleBypassGate", (1150, 1400)): ((230, 235), 0),
        },
    },
}


def _definition(name):
    payload = json.loads((SUBGRAPHS / name).read_text(encoding="utf-8"))
    return payload, payload["definitions"]["subgraphs"][0]


class SyncedUserSubgraphTests(unittest.TestCase):
    def test_canonical_ids_interfaces_and_layout_are_preserved(self):
        for name, expected in EXPECTED.items():
            with self.subTest(name=name):
                payload, graph = _definition(name)
                self.assertEqual(graph["id"], expected["id"])
                self.assertEqual(payload["nodes"][0]["type"], expected["id"])
                self.assertEqual(
                    [item["name"] for item in graph["inputs"]],
                    expected["inputs"],
                )
                actual_layout = {
                    (node["type"], tuple(node["pos"])):
                    (tuple(node["size"]), node["order"])
                    for node in graph["nodes"]
                }
                self.assertEqual(actual_layout, expected["layout"])

    def test_internal_links_are_bidirectionally_consistent(self):
        for name in EXPECTED:
            with self.subTest(name=name):
                _, graph = _definition(name)
                nodes = {str(node["id"]): node for node in graph["nodes"]}
                links = {str(link["id"]): link for link in graph["links"]}
                for link in links.values():
                    origin = nodes.get(str(link["origin_id"]))
                    target = nodes.get(str(link["target_id"]))
                    if origin is not None:
                        self.assertIn(
                            link["id"], origin["outputs"][link["origin_slot"]]["links"]
                        )
                    else:
                        self.assertEqual(str(link["origin_id"]), "-10")
                    if target is not None:
                        self.assertEqual(
                            target["inputs"][link["target_slot"]]["link"], link["id"]
                        )
                    else:
                        self.assertEqual(str(link["target_id"]), "-20")

    def test_visual_compare_remains_behind_existing_bypass_gate(self):
        for name in (
            "CMK Flow · FaceRebuild SDXL · Advanced.json",
            "CMK Flow · FaceSwap · Advanced.json",
        ):
            with self.subTest(name=name):
                _, graph = _definition(name)
                nodes = {str(node["id"]): node for node in graph["nodes"]}
                links = {str(link["id"]): link for link in graph["links"]}
                compare = next(
                    node for node in nodes.values()
                    if node["type"] == "CMKVisualCompare"
                )
                visual = next(item for item in compare["inputs"] if item["name"] == "VISUAL")
                origin = nodes[str(links[str(visual["link"])]["origin_id"])]
                self.assertEqual(origin["type"], "CMKModuleBypassGate")

    def test_faceswap_dynamic_segs_inputs_match_approved_definition(self):
        _, graph = _definition("CMK Flow · FaceSwap · Advanced.json")
        segs = next(node for node in graph["nodes"] if node["type"] == "CMK_SEGSConcate")
        self.assertEqual(
            [item["name"] for item in segs["inputs"]],
            ["image", "segs", "segs_2", "segs_3", "segs_4", "VISUAL", "feather", "alpha"],
        )


if __name__ == "__main__":
    unittest.main()
