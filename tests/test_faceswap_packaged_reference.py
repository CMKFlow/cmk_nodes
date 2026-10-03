import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "CMK Package · face_reference.png"


class FaceSwapPackagedReferenceTests(unittest.TestCase):
    def test_swap_image_loader_exposes_neutral_result_process(self):
        source = (
            ROOT / "pipe" / "loaders" / "cmk_swap_image_loader.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'RETURN_TYPES = (\n        "CMK_RESULT_PROCESS",',
            source,
        )
        self.assertIn('"result_contract": "family_neutral"', source)
        self.assertIn('"source_model_family": "image"', source)

    def test_standalone_swap_loader_is_a_model_free_neutral_result(self):
        loader = (
            ROOT / "pipe" / "loaders" / "cmk_swap_image_loader.py"
        ).read_text(encoding="utf-8")
        unpack = (ROOT / "pipe" / "cmk_family_result.py").read_text(encoding="utf-8")
        self.assertIn('"result_contract": "family_neutral"', loader)
        self.assertIn('"source_model_family": "image"', loader)
        self.assertIn('"CMK Swap Image Loader -Pipe-"', unpack)
        self.assertIn("PROCESS.get(\"pipe_origin\") in neutral_image_origins", unpack)

    def test_faceswap_execute_and_subgraphs_own_log_and_diagnostic_transport(self):
        source = (ROOT / "nodes" / "swap" / "face_swap.py").read_text(encoding="utf-8")
        self.assertIn('"opt_log": ("CMK_LOG_PIPE",)', source)
        self.assertIn('"opt_diagnostic": ("CMK_DIAGNOSTIC",)', source)
        self.assertIn(
            'RETURN_NAMES = ("IMAGE PROCEED", "SEGS PROCESSED", "LOG", "VISUAL", "diagnostic")',
            source,
        )
        self.assertIn("CMKLogConcat().concat(opt_log, log_block)", source)
        self.assertIn("CMKDiagnosticConcat().concat(", source)
        self.assertIn('stage_key="result.faceswap.standard"', source)

        for filename in (
            "CMK Flow · FaceSwap.json",
            "CMK Flow · FaceSwap · Advanced.json",
        ):
            with self.subTest(filename=filename):
                document = json.loads((ROOT / "subgraphs" / filename).read_text(encoding="utf-8"))
                definition = document["definitions"]["subgraphs"][0]
                execute_nodes = [
                    node for node in definition["nodes"]
                    if node["type"] == "CMKFaceSwapImagePipe"
                ]
                self.assertNotIn("CMKLogConcat", {node["type"] for node in definition["nodes"]})
                self.assertNotIn("CMKDiagnosticConcat", {node["type"] for node in definition["nodes"]})
                for execute in execute_nodes:
                    inputs = {item["name"]: item for item in execute["inputs"]}
                    outputs = {item["name"]: item for item in execute["outputs"]}
                    self.assertIn("opt_log", inputs)
                    self.assertIn("VISUAL", inputs)
                    self.assertIn("opt_diagnostic", inputs)
                    self.assertEqual("CMK_LOG_PIPE", outputs["LOG"]["type"])
                    self.assertEqual("CMK_VISUAL_PIPE", outputs["VISUAL"]["type"])

                if filename.endswith("Advanced.json"):
                    links = {link["id"]: link for link in definition["links"]}
                    by_log_link = {
                        next(item for item in node["inputs"] if item["name"] == "opt_log")["link"]: node
                        for node in execute_nodes
                    }
                    first = next(node for node in execute_nodes if next(item for item in node["inputs"] if item["name"] == "opt_diagnostic")["link"] is None)
                    ordered = [first]
                    while len(ordered) < len(execute_nodes):
                        prior_log = next(item for item in ordered[-1]["outputs"] if item["name"] == "LOG")["links"][0]
                        ordered.append(by_log_link[prior_log])
                    for previous, current in zip(ordered, ordered[1:]):
                        current_inputs = {item["name"]: item for item in current["inputs"]}
                        log_link = links[current_inputs["opt_log"]["link"]]
                        diagnostic_link = links[current_inputs["opt_diagnostic"]["link"]]
                        self.assertEqual((previous["id"], 2), (log_link["origin_id"], log_link["origin_slot"]))
                        self.assertEqual((previous["id"], 4), (diagnostic_link["origin_id"], diagnostic_link["origin_slot"]))

    def test_reference_asset_and_loader_stay_inside_package(self):
        backend = (ROOT / "pipe" / "loaders" / "cmk_load_image.py").read_text(
            encoding="utf-8"
        )
        frontend = (
            ROOT / "web" / "js" / "cmk_packaged_faceswap_reference.js"
        ).read_text(encoding="utf-8")
        routes = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('"faceswap_reference.png"', backend)
        self.assertIn('assets" / "references', backend)
        self.assertIn('"faceswap_reference.png"', frontend)
        self.assertIn("`/cmk/reference-assets/${filename}`", frontend)
        self.assertIn('request.path.rstrip("/").endswith("/view")', routes)
        self.assertIn("_PACKAGED_REFERENCES", routes)
        self.assertTrue(
            (ROOT / "assets" / "references" / "faceswap_reference.png").is_file()
        )

    def test_40_subgraphs_use_visual_only_preview_compare(self):
        for filename in (
            "CMK Flow · FaceSwap.json",
            "CMK Flow · FaceSwap · Advanced.json",
        ):
            with self.subTest(filename=filename):
                definition = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )["definitions"]["subgraphs"][0]
                nodes = {node["id"]: node for node in definition["nodes"]}
                compare = next(
                    node for node in nodes.values() if node["type"] == "CMKVisualCompare"
                )
                self.assertEqual(["VISUAL", "enable"], [item["name"] for item in compare["inputs"]])
                self.assertEqual(compare["outputs"], [])
                self.assertNotIn("ImageCompare", {node["type"] for node in nodes.values()})
                self.assertNotIn("CMKImageCompareEnableGate", {node["type"] for node in nodes.values()})
                self.assertNotIn("CMKVisualProvider", {node["type"] for node in nodes.values()})
                packaged_references = {
                    value
                    for node in nodes.values()
                    if node["type"] == "CMKLoadImage"
                    for value in (node.get("widgets_values") or [])
                    if isinstance(value, str)
                    and value.startswith("CMK Package · ")
                }
                self.assertTrue(packaged_references)

    def test_40_bypass_gate_connections_follow_named_contract(self):
        expected = {
            "MODEL BYPASS": ("CMKResultUnpackPipe", "MODEL"),
            "IMAGE BYPASS": ("CMKResultUnpackPipe", "IMAGE"),
            "LOG BYPASS": ("CMKResultUnpackPipe", "LOG"),
            "MODEL ACTIVE": ("CMKFaceSwapBoundaryCache", "MODEL"),
            "IMAGE ACTIVE": ("CMKFaceSwapBoundaryCache", "IMAGE"),
            "LOG ACTIVE": ("CMKFaceSwapBoundaryCache", "LOG"),
            "DIAGNOSTIC ACTIVE": ("CMKFaceSwapBoundaryCache", "diagnostic"),
        }
        for filename in (
            "CMK Flow · FaceSwap.json",
            "CMK Flow · FaceSwap · Advanced.json",
        ):
            with self.subTest(filename=filename):
                definition = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )["definitions"]["subgraphs"][0]
                nodes = {node["id"]: node for node in definition["nodes"]}
                links = {link["id"]: link for link in definition["links"]}
                gate = next(
                    node for node in nodes.values()
                    if node["type"] == "CMKModuleBypassGate"
                )

                for input_name, (origin_type, output_name) in expected.items():
                    gate_input = next(
                        item for item in gate["inputs"] if item["name"] == input_name
                    )
                    link = links[gate_input["link"]]
                    origin = nodes[link["origin_id"]]
                    self.assertEqual(origin_type, origin["type"])
                    self.assertEqual(output_name, origin["outputs"][link["origin_slot"]]["name"])

                enable = next(item for item in gate["inputs"] if item["name"] == "ENABLE")
                enable_link = links[enable["link"]]
                public_enable_slot = next(
                    index for index, item in enumerate(definition["inputs"])
                    if item["name"] == "FACESWAP ENABLE"
                )
                self.assertEqual((-10, public_enable_slot), (
                    enable_link["origin_id"], enable_link["origin_slot"]
                ))
                self.assertEqual("BOOLEAN", enable_link["type"])

    def test_40_outer_dimensions_follow_published_variant_contract(self):
        expected_sizes = {
            "CMK Flow · FaceSwap.json": [300, 190],
            "CMK Flow · FaceSwap · Advanced.json": [300, 190],
        }
        for filename, expected_size in expected_sizes.items():
            with self.subTest(filename=filename):
                document = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )
                outer = document["nodes"][0]
                self.assertEqual(outer["size"], expected_size)
                self.assertEqual(
                    outer["properties"]["cmkCleanViewExpandedSize"], expected_size,
                )

    def test_standard_40_matches_compact_visual_source_contract(self):
        document = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · FaceSwap.json").read_text(
                encoding="utf-8"
            )
        )
        outer = document["nodes"][0]
        definition = document["definitions"]["subgraphs"][0]
        nodes = {node["id"]: node for node in definition["nodes"]}

        self.assertEqual(
            ["MODEL (opt)", "PROCESS", "IMAGE_TARGET", "LOG", "VISUAL", "FACESWAP ENABLE", "opt_image_file"],
            [item["name"] for item in outer["inputs"]],
        )
        self.assertEqual(
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"],
            [item["name"] for item in outer["outputs"]],
        )
        self.assertEqual([True], outer["widgets_values"])
        process = next(
            node for node in nodes.values() if node["type"] == "CMKFaceSwapImagePipe"
        )
        visual = next(item for item in process["outputs"] if item["name"] == "VISUAL")
        self.assertEqual("CMK_VISUAL_PIPE", visual["type"])

        for link in definition["links"]:
            with self.subTest(link=link["id"]):
                origin = nodes.get(link["origin_id"])
                target = nodes.get(link["target_id"])
                if origin is not None:
                    self.assertIn(link["id"], origin["outputs"][link["origin_slot"]].get("links") or [])
                if target is not None:
                    self.assertEqual(link["id"], target["inputs"][link["target_slot"]].get("link"))

    def test_advanced_40_matches_compact_parallel_visual_contract(self):
        document = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · FaceSwap · Advanced.json").read_text(
                encoding="utf-8"
            )
        )
        outer = document["nodes"][0]
        definition = document["definitions"]["subgraphs"][0]
        nodes = {node["id"]: node for node in definition["nodes"]}

        self.assertEqual(
            ["MODEL (opt)", "PROCESS", "IMAGE_TARGET", "LOG", "VISUAL", "FACESWAP ENABLE", "opt_image_file"],
            [item["name"] for item in outer["inputs"]],
        )
        self.assertEqual([True], outer["widgets_values"])
        self.assertEqual(
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"],
            [item["name"] for item in outer["outputs"]],
        )
        concat = next(
            node for node in nodes.values() if node["type"] == "CMK_SEGSConcate"
        )
        visual = next(item for item in concat["outputs"] if item["name"] == "VISUAL")
        self.assertEqual("CMK_VISUAL_PIPE", visual["type"])
        declaration = outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual("result.faceswap.advanced", declaration["stage_key"])

        for link in definition["links"]:
            with self.subTest(link=link["id"]):
                origin = nodes.get(link["origin_id"])
                target = nodes.get(link["target_id"])
                if origin is not None:
                    self.assertIn(link["id"], origin["outputs"][link["origin_slot"]].get("links") or [])
                if target is not None:
                    self.assertEqual(link["id"], target["inputs"][link["target_slot"]].get("link"))

    def test_faceswap_showcases_embed_portable_reference_and_compare(self):
        for filename in (
            "CMK 2.5 · Full Flow .json",
        ):
            with self.subTest(filename=filename):
                document = json.loads(
                    (ROOT / "workflows" / "showcase" / filename).read_text(
                        encoding="utf-8"
                    )
                )
                definition = next(
                    item for item in document["definitions"]["subgraphs"]
                    if "FaceSwap" in item["name"]
                )
                self.assertIn(
                    REFERENCE,
                    {
                        value
                        for node in definition["nodes"]
                        if node["type"] == "CMKLoadImage"
                        for value in (node.get("widgets_values") or [])
                    },
                )
                self.assertIn(
                    "CMKVisualCompare", {node["type"] for node in definition["nodes"]}
                )


if __name__ == "__main__":
    unittest.main()
