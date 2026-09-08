import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "CMK Package · faceswap_reference.png"


class FaceSwapPackagedReferenceTests(unittest.TestCase):
    def test_faceswap_execute_and_subgraphs_own_log_and_diagnostic_transport(self):
        source = (ROOT / "nodes" / "swap" / "face_swap.py").read_text(encoding="utf-8")
        self.assertIn('"opt_log": ("CMK_LOG_PIPE",)', source)
        self.assertIn('"opt_diagnostic": ("CMK_DIAGNOSTIC",)', source)
        self.assertIn('RETURN_NAMES = ("IMAGE PROCEED", "SEGS PROCESSED", "LOG", "diagnostic")', source)
        self.assertIn("CMKLogConcat().concat(opt_log, log_block)", source)
        self.assertIn("CMKDiagnosticConcat().concat(", source)

        for filename in (
            "CMK Flow · 40 FaceSwap.json",
            "CMK Flow · 40 FaceSwap · Advanced.json",
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
                    self.assertIn("opt_diagnostic", inputs)
                    self.assertEqual("CMK_LOG_PIPE", outputs["LOG"]["type"])

                if filename.endswith("Advanced.json"):
                    links = {link["id"]: link for link in definition["links"]}
                    ordered = [next(node for node in execute_nodes if node["id"] == node_id) for node_id in (5239, 5052, 4939)]
                    for previous, current in zip(ordered, ordered[1:]):
                        current_inputs = {item["name"]: item for item in current["inputs"]}
                        log_link = links[current_inputs["opt_log"]["link"]]
                        diagnostic_link = links[current_inputs["opt_diagnostic"]["link"]]
                        self.assertEqual((previous["id"], 2), (log_link["origin_id"], log_link["origin_slot"]))
                        self.assertEqual((previous["id"], 3), (diagnostic_link["origin_id"], diagnostic_link["origin_slot"]))

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

    def test_40_subgraphs_compare_only_after_boundary_cache(self):
        for filename in (
            "CMK Flow · 40 FaceSwap.json",
            "CMK Flow · 40 FaceSwap · Advanced.json",
        ):
            with self.subTest(filename=filename):
                definition = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )["definitions"]["subgraphs"][0]
                nodes = {node["id"]: node for node in definition["nodes"]}
                compare = next(
                    node for node in nodes.values() if node["type"] == "ImageCompare"
                )
                cache = next(
                    node
                    for node in nodes.values()
                    if node["type"] == "CMKFaceSwapBoundaryCache"
                )
                pack = next(
                    node for node in nodes.values() if node["type"] == "CMKResultPackPipe"
                )
                source_link = next(
                    link for link in definition["links"]
                    if link["target_id"] == compare["id"] and link["target_slot"] == 0
                )
                compare_gate = nodes[source_link["origin_id"]]
                self.assertEqual(compare_gate["type"], "CMKImageCompareEnableGate")
                result_link_id = next(
                    item["link"] for item in compare_gate["inputs"]
                    if item["name"] == "IMAGE A"
                )
                gated_result_link = next(
                    link for link in definition["links"] if link["id"] == result_link_id
                )
                result_link = next(
                    link for link in definition["links"]
                    if link["origin_id"] == cache["id"]
                    and link["origin_slot"] == source_link["origin_slot"]
                    and link["target_id"] == pack["id"]
                )
                self.assertEqual(gated_result_link["origin_id"], cache["id"])
                self.assertEqual(result_link["target_id"], pack["id"])
                self.assertEqual(compare["outputs"], [])
                packaged_references = {
                    value
                    for node in nodes.values()
                    if node["type"] == "CMKLoadImage"
                    for value in (node.get("widgets_values") or [])
                    if isinstance(value, str)
                    and value.startswith("CMK Package · ")
                }
                self.assertTrue(packaged_references)

    def test_40_outer_dimensions_follow_published_variant_contract(self):
        expected_sizes = {
            "CMK Flow · 40 FaceSwap.json": [450, 230],
            "CMK Flow · 40 FaceSwap · Advanced.json": [450, 230],
        }
        for filename, expected_size in expected_sizes.items():
            with self.subTest(filename=filename):
                document = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )
                outer = document["nodes"][0]
                self.assertEqual(outer["size"], expected_size)
                self.assertEqual(
                    outer.get("properties", {}).get("cmkOuterSize", expected_size),
                    expected_size,
                )

    def test_standard_40_matches_compact_visual_source_contract(self):
        document = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · 40 FaceSwap.json").read_text(
                encoding="utf-8"
            )
        )
        outer = document["nodes"][0]
        definition = document["definitions"]["subgraphs"][0]
        nodes = {node["id"]: node for node in definition["nodes"]}

        self.assertEqual(
            ["MODEL (opt)", "PROCESS", "IMAGE_TARGET", "LOG", "VISUAL", "FACESWAP ENABLE", "image"],
            [item["name"] for item in outer["inputs"]],
        )
        self.assertEqual("source face", outer["inputs"][-1]["label"])
        self.assertEqual(
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"],
            [item["name"] for item in outer["outputs"]],
        )
        self.assertEqual(["CMK Package · face_reference.png"], outer["widgets_values"])
        provider = next(node for node in nodes.values() if node["type"] == "CMKVisualProvider")
        self.assertEqual(
            ["FaceSwap", 40, "CMKFaceSwapImagePipe", "result", "result.faceswap.standard"],
            provider["widgets_values"],
        )

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
            (ROOT / "subgraphs" / "CMK Flow · 40 FaceSwap · Advanced.json").read_text(
                encoding="utf-8"
            )
        )
        outer = document["nodes"][0]
        definition = document["definitions"]["subgraphs"][0]
        nodes = {node["id"]: node for node in definition["nodes"]}

        self.assertEqual(
            ["MODEL (opt)", "PROCESS", "IMAGE_TARGET", "LOG", "VISUAL", "FACESWAP ENABLE"],
            [item["name"] for item in outer["inputs"]],
        )
        self.assertEqual([], outer["widgets_values"])
        self.assertEqual(
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"],
            [item["name"] for item in outer["outputs"]],
        )
        self.assertEqual([["5239", "GLOBAL ENABLE"]], outer["properties"]["proxyWidgets"])
        provider = next(node for node in nodes.values() if node["type"] == "CMKVisualProvider")
        self.assertEqual(
            ["FaceSwap", 40, "CMKFaceSwapImagePipe", "result", "result.faceswap.advanced"],
            provider["widgets_values"],
        )
        declaration = outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual(["5239", "5052", "4939"], declaration["live_node_ids"])

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
            "CMK - Full Flow.json",
        ):
            with self.subTest(filename=filename):
                document = json.loads(
                    (ROOT / "workflows" / "showcase" / filename).read_text(
                        encoding="utf-8"
                    )
                )
                definition = next(
                    item for item in document["definitions"]["subgraphs"]
                    if "40 FaceSwap" in item["name"]
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
                    "ImageCompare", {node["type"] for node in definition["nodes"]}
                )


if __name__ == "__main__":
    unittest.main()
