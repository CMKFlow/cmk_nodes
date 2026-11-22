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
            ("CMKFaceRebuildBoundaryCache", (1090, 2055)): ((230, 130), 4),
            ("CMKResultProcessForwardPipe", (1320, 2055)): ((280, 30), 5),
            ("CMKVisualCompare", (760, 830)): ((1105, 1165), 6),
            ("CMKModuleBypassGate", (860, 2055)): ((230, 235), 7),
            # OD-2: the public MODEL output is forwarded from SG_IN.MODEL.
            ("CMKModelForwardPipe", (1560, 2055)): ((235, 30), 8),
        },
    },
    # 2.8.0 Multi-Model Detailer: the user subgraph taken over unchanged (2.8 package sync).
    "CMK Flow · Detailer.json": {
        "id": "70aa4358-3856-4a78-8304-7fef47a51263",
        "inputs": ["MODEL ZIT", "MODEL_1", "PROCESS_1", "IMAGE_1", "LOG BYPASS", "VISUAL IN", "model_family"],
        "layout": {
            ("CMKDetailerBoundaryCache", (8250, 2910)): ((230, 170), 0),
            ("CMKResultProcessForwardPipe", (8720, 2910)): ((280, 30), 1),
            ("CMKVisualCompare", (8915, 1915)): ((530, 590), 2),
            ("CMKModuleBypassGate", (8485, 2910)): ((230, 285), 3),
            ("CMKModelForwardPipe", (9205, 2910)): ((235, 60), 4),
            ("CMKDetailerPrepareMultiModelPipe", (8005, 1915)): ((450, 590), 5),
            ("CMK_SmartDetailerMultiModelPipe", (8460, 1915)): ((450, 590), 6),
            ("CMKModelForwardPipe", (8970, 2910)): ((235, 30), 7),
            ("CMKDetailerFamilyEnable", (8015, 2910)): ((230, 60), 8),
            ("CMKPreviewRender", (8005, 2540)): ((1435, 330), 9),
        },
    },
    "CMK Flow · Detailer · Advanced.json": {
        "id": "c65f6b01-4f30-4937-8ea6-2b229ef2e076",
        "inputs": [
            "MODEL ZIT", "MODEL SDXL", "PROCESS", "IMAGE", "LOG", "VISUAL",
            "detailer_global_enable",
        ],
        "layout": {
            ("CMKDetailerPrepareAdvancedMultiModelPipe", (8005, 1915)): ((450, 760), 13),
            ("CMKDetailerFamilyEnable", (8465, 1915)): ((450, 110), 0),
            ("CMK_SmartDetailerMultiModelPipe", (8465, 2065)): ((450, 525), 3),
            ("CMKDetailerFamilyEnable", (8925, 1915)): ((450, 110), 1),
            ("CMK_SmartDetailerMultiModelPipe", (8925, 2065)): ((450, 525), 4),
            ("CMKDetailerFamilyEnable", (9385, 1915)): ((450, 110), 2),
            ("CMK_SmartDetailerMultiModelPipe", (9385, 2065)): ((450, 525), 5),
            ("CMKDetailerMergeMultiModel", (8765, 3370)): ((320, 305), 6),
            ("CMKDetailerBoundaryCache", (9000, 3370)): ((190, 170), 7),
            ("CMKModuleBypassGate", (8530, 3370)): ((285, 260), 8),
            ("CMKVisualCompare", (9845, 1915)): ((605, 675), 9),
            ("CMKResultProcessForwardPipe", (9235, 3370)): ((280, 30), 10),
            ("CMKModelForwardPipe", (9485, 3370)): ((225, 30), 11),
            ("CMKModelForwardPipe", (9720, 3370)): ((225, 30), 12),
            ("CMKPreviewRender", (9230, 2985)): ((1220, 320), 14),
            ("CMKPreviewRender", (8005, 2985)): ((1220, 320), 15),
            ("CMKPreviewRender", (8005, 2630)): ((1220, 320), 16),
            ("CMKPreviewRender", (9230, 2630)): ((1220, 320), 17),
            # 2.8 blueprint sync (2026-11-16): Advanced Mode and Diagnostic Gate.
            ("CMKDetailerAdvancedModeEnable", (8005, 3370)): ((270, 82), 18),
            ("CMKModuleDiagnosticGate", (8005, 3445)): ((285, 130), 19),
        },
    },
    # 2.8 package sync: user subgraphs taken over unchanged (ids, ports, layout).
    "CMK Flow · FaceProcess.json": {
        "id": "2f64eba0-47e2-4a50-8505-41736a22ff35",
        "inputs": ["MODEL ZIT", "MODEL_1", "PROCESS_1", "IMAGE_1", "LOG BYPASS", "VISUAL IN", "process_mode"],
        "layout": {
            ("CMKFaceBoundaryCache", (8255, 2910)): ((230, 170), 0),
            ("CMKResultProcessForwardPipe", (8725, 2910)): ((280, 30), 1),
            ("CMKVisualCompare", (8915, 1915)): ((530, 590), 2),
            ("CMKModuleBypassGate", (8490, 2910)): ((230, 285), 3),
            ("CMKModelForwardPipe", (9210, 2910)): ((235, 60), 4),
            ("CMKFaceProcessPrepareMultiModelPipe", (8005, 1915)): ((450, 590), 5),
            ("CMKFaceProcessMultiModelPipe", (8460, 1915)): ((450, 590), 6),
            ("CMKModelForwardPipe", (8975, 2910)): ((235, 30), 7),
            ("CMKFaceProcessModeEnable", (8020, 2910)): ((255, 80), 8),
            ("CMKPreviewRender", (8005, 2540)): ((1440, 325), 9),
        },
    },
    "CMK Flow · FaceProcess · Advanced.json": {
        "id": "ffd0a883-f3c2-4eae-a1be-591cc0cb9255",
        "inputs": ["MODEL ZIT", "MODEL SDXL", "PROCESS", "IMAGE", "LOG", "VISUAL", "faceprocess_global_enable"],
        "layout": {
            ("CMKFaceProcessPrepareAdvancedMultiModelPipe", (8005, 1915)): ((450, 760), 3),
            ("CMKFaceProcessModeEnable", (8465, 1915)): ((450, 110), 0),
            ("CMKFaceProcessMultiModelPipe", (8465, 2065)): ((450, 610), 4),
            ("CMKFaceProcessModeEnable", (8925, 1915)): ((450, 110), 1),
            ("CMKFaceProcessMultiModelPipe", (8925, 2065)): ((450, 610), 5),
            ("CMKFaceProcessModeEnable", (9385, 1915)): ((450, 110), 2),
            ("CMKFaceProcessMultiModelPipe", (9385, 2065)): ((450, 610), 6),
            ("CMKFaceProcessMergeMultiModel", (8785, 3435)): ((320, 305), 7),
            ("CMKFaceBoundaryCache", (9045, 3435)): ((190, 170), 8),
            ("CMKModuleBypassGate", (8550, 3435)): ((285, 260), 9),
            ("CMKVisualCompare", (9845, 1915)): ((725, 760), 10),
            ("CMKResultProcessForwardPipe", (9280, 3435)): ((280, 30), 11),
            ("CMKModelForwardPipe", (9530, 3435)): ((225, 30), 12),
            ("CMKModelForwardPipe", (9765, 3435)): ((225, 30), 13),
            ("CMKPreviewRender", (9290, 3065)): ((1280, 320), 14),
            ("CMKPreviewRender", (8005, 3065)): ((1280, 320), 15),
            ("CMKPreviewRender", (8005, 2710)): ((1280, 320), 16),
            ("CMKPreviewRender", (9290, 2710)): ((1280, 320), 17),
            # 2.8 blueprint sync (2026-11-16): Advanced Mode and Diagnostic Gate.
            ("CMKFaceProcessAdvancedModeEnable", (8025, 3435)): ((270, 82), 18),
            ("CMKModuleDiagnosticGate", (8025, 3510)): ((285, 130), 19),
        },
    },
    "CMK Flow · FaceRebuild SDXL v2.json": {
        "id": "abf9d2d6-e61b-48ea-8a94-ce2aae383478",
        "inputs": ["MODEL ZIT", "MODEL SDXL", "PROCESS", "IMAGE", "LOG", "VISUAL", "facerebuild_mode", "opt_image_file"],
        "layout": {
            ("CMKLoadImage", (-385, 805)): ((450, 720), 0),
            ("CMKInstantIDFaceRebuildSDXLV2", (75, 805)): ((450, 720), 1),
            ("CMKFaceRebuildBoundaryCache", (-55, 1940)): ((230, 130), 2),
            ("CMKResultProcessForwardPipe", (180, 1940)): ((280, 30), 3),
            ("CMKVisualCompare", (535, 805)): ((660, 720), 4),
            ("CMKModuleBypassGate", (-290, 1940)): ((230, 235), 5),
            ("CMKModelForwardPipe", (430, 1940)): ((235, 30), 6),
            ("CMKPreviewRender", (-385, 1565)): ((1580, 320), 7),
            ("CMKFaceRebuildModeEnable", (665, 1940)): ((270, 60), 8),
            ("CMKModelForwardPipe", (900, 1940)): ((225, 30), 9),
        },
    },
    "CMK Flow · FaceRebuild SDXL · Advanced v2.json": {
        "id": "4f31016e-e56d-471b-9094-749efaf79486",
        "inputs": ["MODEL ZIT", "MODEL SDXL", "PROCESS", "IMAGE", "LOG", "VISUAL", "facerebuild_mode", "opt_image_file", "opt_image_file_1", "opt_image_file_2"],
        "layout": {
            ("CMKLoadImage", (45, 825)): ((245, 365), 0),
            ("CMKLoadImage", (45, 1225)): ((245, 365), 1),
            ("CMKLoadImage", (45, 1625)): ((245, 365), 2),
            ("CMKInstantIDFaceRebuildAdvancedSDXLV2", (300, 825)): ((600, 1400), 3),
            ("CMKFaceRebuildBoundaryCache", (520, 2410)): ((230, 130), 4),
            ("CMKResultProcessForwardPipe", (755, 2410)): ((280, 30), 5),
            ("CMKVisualCompare", (760, 825)): ((1105, 1165), 6),
            ("CMKModuleBypassGate", (285, 2410)): ((230, 235), 7),
            ("CMKModelForwardPipe", (1005, 2410)): ((235, 30), 8),
            ("CMKPreviewRender", (45, 2030)): ((1820, 325), 11),
            ("CMKFaceRebuildModeEnable", (1240, 2410)): ((270, 60), 9),
            ("CMKModelForwardPipe", (1475, 2410)): ((225, 30), 10),
        },
    },
    "CMK Flow · FaceSwap v2.json": {
        "id": "2a24e1b4-11e8-4c16-b11a-f6d5527482b2",
        "inputs": ["MODEL ZIT", "MODEL SDXL", "PROCESS", "IMAGE", "LOG", "VISUAL", "faceswap_mode", "opt_image_file"],
        "layout": {
            ("CMKFaceSwapImagePipeV2", (1180, 540)): ((450, 590), 0),
            ("CMKLoadImage", (720, 540)): ((450, 590), 1),
            ("CMKFaceSwapBoundaryCache", (1225, 1545)): ((230, 130), 2),
            ("CMKResultUnpackPipe", (750, 1545)): ((230, 90), 3),
            ("CMKResultProcessForwardPipe", (985, 1545)): ((270, 30), 4),
            ("CMKVisualCompare", (1640, 540)): ((530, 590), 5),
            ("CMKModuleBypassGate", (1460, 1545)): ((230, 235), 6),
            ("CMKPreviewRender", (720, 1170)): ((1450, 320), 7),
            ("CMKFaceSwapModeEnable", (1695, 1545)): ((270, 60), 8),
            ("CMKModelForwardPipe", (1930, 1545)): ((225, 30), 9),
        },
    },
    "CMK Flow · FaceSwap · Advanced v2.json": {
        "id": "2309dab9-b821-4908-8a37-72a28dbc107a",
        "inputs": ["MODEL ZIT", "MODEL SDXL", "PROCESS", "IMAGE", "LOG", "VISUAL", "faceswap_mode", "opt_image_file", "opt_image_file_1", "opt_image_file_2"],
        "layout": {
            ("CMKFaceSwapSlotPipeV2", (720, 940)): ((380, 400), 7),
            ("CMKLoadImage", (720, 540)): ((380, 340), 8),
            ("CMKFaceSwapBoundaryCache", (1705, 1780)): ((230, 90), 9),
            ("CMKVisualCompare", (1890, 540)): ((740, 800), 10),
            ("CMKFaceSwapSlotPipeV2", (1110, 940)): ((380, 400), 11),
            ("CMKLoadImage", (1110, 540)): ((380, 340), 1),
            ("CMK_SEGSConcate", (1185, 1780)): ((230, 134), 2),
            ("CMKLoadImage", (1500, 540)): ((380, 340), 3),
            ("CMKFaceSwapSlotPipeV2", (1500, 940)): ((380, 400), 4),
            ("CMKResultUnpackPipe", (715, 1780)): ((230, 90), 5),
            ("CMKResultProcessForwardPipe", (2175, 1780)): ((270, 30), 6),
            ("CMKModuleBypassGate", (950, 1780)): ((230, 235), 0),
            ("CMKPreviewRender", (720, 1380)): ((1910, 345), 12),
            ("CMKFaceSwapModeEnable", (1940, 1780)): ((270, 60), 13),
            ("CMKModelForwardPipe", (2415, 1780)): ((225, 30), 14),
            ("CMKFaceSwapAdvancedDiagnosticV2", (1420, 1780)): ((270, 130), 15),
        },
    },
    "CMK Flow · MaskDetailer.json": {
        "id": "63419f68-5573-4c90-9619-226b0c413a9c",
        "inputs": ["MODEL ZIT", "MODEL SDXL", "PROCESS", "IMAGE", "LOG", "VISUAL", "maskdetailer_mode"],
        "layout": {
            ("CMKVisualCompare", (9305, 255)): ((750, 810), 1),
            ("CMKMaskDetailerIntake", (8385, 255)): ((450, 420), 2),
            ("CMKMaskDetailerFlowGateV2", (9540, 1470)): ((305, 215), 3),
            ("CMKResultUnpackPipe", (9305, 1470)): ((230, 90), 4),
            ("CMKMaskDetailerPrepareMultiModel", (8385, 710)): ((450, 354), 5),
            ("CMKMaskDetailerProcessMultiModel", (8845, 405)): ((450, 660), 6),
            ("CMKDetailerFamilyEnable", (8845, 255)): ((450, 110), 0),
            ("CMKPreviewRender", (8385, 1105)): ((1670, 320), 7),
            ("CMKModelForwardPipe", (8830, 1470)): ((235, 30), 9),
            ("CMKModelForwardPipe", (9065, 1470)): ((235, 30), 8),
            ("CMKMaskDetailerModeEnable", (8595, 1470)): ((270, 82), 10),
        },
    },
    "CMK Flow · Upscale & Save v2.json": {
        "id": "6a25aa85-d5c8-48a9-87d8-7e9a123c947b",
        "inputs": ["MODEL ZIT", "MODEL SDXL", "PROCESS", "IMAGE", "LOG", "VISUAL", "upscale_save_mode", "FILENAME PREFIX"],
        "layout": {
            ("CMK_SmartUpscalerPipe", (11790, -560)): ((450, 190), 0),
            ("CMKUpscaleSaveBoundaryCache", (12520, 80)): ((230, 90), 1),
            ("CMKModuleBypassGate", (12290, 80)): ((230, 160), 2),
            ("CMKVisualCompare", (12250, -560)): ((530, 590), 3),
            ("CMKSaveEndpoint", (11790, -310)): ((450, 340), 5),
            ("CMKVisualProvider", (12055, 80)): ((230, 250), 4),
            ("CMKModelForwardPipe", (11790, 80)): ((270, 30), 6),
            ("CMKModelForwardPipe", (11790, 125)): ((270, 30), 7),
            ("CMKResultProcessForwardPipe", (11790, 170)): ((270, 30), 8),
            ("CMKUpscaleSaveModeEnable", (11790, 215)): ((270, 60), 9),
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
