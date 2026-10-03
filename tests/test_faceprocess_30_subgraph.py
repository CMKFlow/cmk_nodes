import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "subgraphs" / "CMK Flow · FaceProcess SDXL.json"


class FaceProcess30SubgraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(PATH.read_text(encoding="utf-8"))
        cls.outer = cls.document["nodes"][0]
        cls.definition = cls.document["definitions"]["subgraphs"][0]
        cls.nodes = {node["id"]: node for node in cls.definition["nodes"]}
        cls.links = {link["id"]: link for link in cls.definition["links"]}

    def test_outer_contract_is_compact_and_display_free(self):
        self.assertEqual([300, 190], self.outer["size"])
        self.assertEqual([300, 100], self.outer["properties"]["cmkOuterSize"])
        self.assertEqual([300, 100], self.outer["properties"]["cmkManualSize"])
        self.assertEqual([], self.outer["properties"]["previewExposures"])
        self.assertNotIn("proxyWidgets", self.outer["properties"])
        self.assertEqual(
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "face_global_enable"],
            [item["name"] for item in self.outer["inputs"]],
        )
        self.assertEqual("FACEPROCESS ENABLE", self.outer["inputs"][5]["label"])
        self.assertEqual([True], self.outer["widgets_values"])
        self.assertEqual(
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"],
            [item["name"] for item in self.outer["outputs"]],
        )

    def test_execute_node_owns_log_and_diagnostic_concatenation(self):
        execute = next(node for node in self.nodes.values() if node["type"] == "CMKFaceProcessPipe")
        self.assertEqual("CMKFaceProcessPipe", execute["type"])
        execute_input_names = [item["name"] for item in execute["inputs"]]
        self.assertIn("opt_log", execute_input_names)
        self.assertIn("VISUAL", execute_input_names)
        self.assertIn("opt_diagnostic", execute_input_names)
        self.assertEqual(
            ["LOG", "VISUAL", "diagnostic"],
            [item["name"] for item in execute["outputs"][-3:]],
        )
        self.assertNotIn(
            "CMKLogConcat",
            {node["type"] for node in self.definition["nodes"]},
        )
        self.assertNotIn(
            "CMKDiagnosticConcat",
            {node["type"] for node in self.definition["nodes"]},
        )
        self.assertEqual(
            (execute["id"], execute_input_names.index("opt_log")),
            (self.links[16001]["target_id"], self.links[16001]["target_slot"]),
        )
        self.assertEqual(
            (execute["id"], execute_input_names.index("opt_diagnostic")),
            (self.links[16002]["target_id"], self.links[16002]["target_slot"]),
        )

    def test_visual_transport_is_not_forwarded_to_legacy_parent(self):
        source = (ROOT / "pipe" / "cmk_faceprocess.py").read_text(encoding="utf-8")
        start = source.index("    def run_pipe(")
        signature = source[start:start + 400]
        self.assertIn("VISUAL=None", signature)

        start = source.index("        result = super().run(")
        parent_call = source[start:start + 500]
        self.assertNotIn("VISUAL", parent_call)

    def test_prepare_lora_menu_always_offers_none(self):
        source = (ROOT / "pipe" / "cmk_faceprocess_prepare.py").read_text(encoding="utf-8")
        self.assertIn('loras = ["None"] + [', source)
        self.assertIn('str(name).strip().lower() != "none"', source)

    def test_process_mode_exists_only_on_execute_node(self):
        prepare = next(node for node in self.nodes.values() if node["type"] == "CMKFaceProcessPreparePipe")
        execute = next(node for node in self.nodes.values() if node["type"] == "CMKFaceProcessPipe")
        self.assertNotIn("process_mode", [item["name"] for item in prepare["inputs"]])
        self.assertIn("process_mode", [item["name"] for item in execute["inputs"]])

        declaration = self.outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual("FaceProcess", declaration["label"])
        self.assertEqual("sdxl.faceprocess.standard", declaration["stage_key"])

    def test_process_visual_feeds_output_and_internal_compare(self):
        process = next(node for node in self.definition["nodes"] if node["type"] == "CMKFaceProcessPipe")
        compare = next(node for node in self.definition["nodes"] if node["type"] == "CMKVisualCompare")
        visual_slot = next(index for index, item in enumerate(process["outputs"]) if item["name"] == "VISUAL")
        compare_link = self.links[compare["inputs"][0]["link"]]
        self.assertEqual((process["id"], visual_slot), (compare_link["origin_id"], compare_link["origin_slot"]))
        self.assertEqual([], compare["outputs"])
        self.assertNotIn("CMKVisualProvider", {node["type"] for node in self.definition["nodes"]})
        self.assertNotIn("CMKImageCompareEnableGate", {node["type"] for node in self.definition["nodes"]})
        self.assertNotIn("ImageCompare", {node["type"] for node in self.definition["nodes"]})
        self.assertNotIn("proxyWidgets", self.outer["properties"])

    def test_every_link_is_declared_at_both_endpoints(self):
        for link in self.definition["links"]:
            with self.subTest(link=link["id"]):
                origin = self.nodes.get(link["origin_id"])
                target = self.nodes.get(link["target_id"])
                if origin is not None:
                    self.assertIn(link["id"], origin["outputs"][link["origin_slot"]].get("links") or [])
                if target is not None:
                    self.assertEqual(link["id"], target["inputs"][link["target_slot"]].get("link"))


class FaceProcess30AdvancedCompatibilityTests(unittest.TestCase):
    def test_existing_advanced_branches_follow_the_new_transport_contract(self):
        document = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · FaceProcess SDXL · Advanced.json").read_text(
                encoding="utf-8"
            )
        )
        definition = document["definitions"]["subgraphs"][0]
        nodes = {node["id"]: node for node in definition["nodes"]}
        links = {link["id"]: link for link in definition["links"]}
        execute_nodes = [node for node in definition["nodes"] if node["type"] == "CMKFaceProcessPipe"]
        prepare = next(node for node in definition["nodes"] if node["type"] == "CMKFaceProcessPreparePipe")

        outer = document["nodes"][0]
        self.assertEqual([300, 190], outer["size"])
        self.assertEqual([True], outer["widgets_values"])
        self.assertNotIn("proxyWidgets", outer["properties"])
        self.assertEqual(
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "face_global_enable"],
            [item["name"] for item in outer["inputs"]],
        )
        self.assertEqual("FACEPROCESS ENABLE", outer["inputs"][5]["label"])
        self.assertEqual(
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"],
            [item["name"] for item in outer["outputs"]],
        )

        self.assertNotIn("CMKLogConcat", {node["type"] for node in nodes.values()})
        self.assertNotIn("CMKDiagnosticConcat", {node["type"] for node in nodes.values()})
        self.assertNotIn("process_mode", [item["name"] for item in prepare["inputs"]])
        for execute in execute_nodes:
            input_names = [item["name"] for item in execute["inputs"]]
            self.assertIn("process_mode", input_names)
            self.assertIn("opt_log", input_names)
            self.assertIn("VISUAL", input_names)
            self.assertIn("opt_diagnostic", input_names)
            self.assertEqual(
                ["LOG", "VISUAL", "diagnostic"],
                [item["name"] for item in execute["outputs"][-3:]],
            )

        first = next(
            node for node in execute_nodes
            if links[next(item for item in node["inputs"] if item["name"] == "opt_log")["link"]]["origin_id"]
            not in {item["id"] for item in execute_nodes}
        )
        ordered = [first]
        while len(ordered) < len(execute_nodes):
            log_link = next(item for item in ordered[-1]["outputs"] if item["name"] == "LOG")["links"][0]
            ordered.append(next(node for node in execute_nodes if next(item for item in node["inputs"] if item["name"] == "opt_log")["link"] == log_link))
        for previous, current in zip(ordered, ordered[1:]):
            log_link = links[next(item for item in current["inputs"] if item["name"] == "opt_log")["link"]]
            diagnostic_link = links[next(item for item in current["inputs"] if item["name"] == "opt_diagnostic")["link"]]
            self.assertEqual(previous["id"], log_link["origin_id"])
            self.assertEqual(previous["id"], diagnostic_link["origin_id"])

        concat = next(node for node in nodes.values() if node["type"] == "CMK_SEGSConcate")
        self.assertIn("VISUAL", [item["name"] for item in concat["inputs"]])
        self.assertIn("VISUAL", [item["name"] for item in concat["outputs"]])
        self.assertNotIn("CMKVisualProvider", {node["type"] for node in nodes.values()})
        self.assertNotIn("CMKImageCompareEnableGate", {node["type"] for node in nodes.values()})
        self.assertNotIn("ImageCompare", {node["type"] for node in nodes.values()})
        declaration = outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual(3, len(declaration["live_node_ids"]))

        for link in definition["links"]:
            with self.subTest(link=link["id"]):
                origin = nodes.get(link["origin_id"])
                target = nodes.get(link["target_id"])
                if origin is not None:
                    self.assertIn(link["id"], origin["outputs"][link["origin_slot"]].get("links") or [])
                if target is not None:
                    self.assertEqual(link["id"], target["inputs"][link["target_slot"]].get("link"))


if __name__ == "__main__":
    unittest.main()
