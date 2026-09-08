import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "subgraphs" / "CMK Flow · 30 FaceProcess SDXL.json"


class FaceProcess30SubgraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(PATH.read_text(encoding="utf-8"))
        cls.outer = cls.document["nodes"][0]
        cls.definition = cls.document["definitions"]["subgraphs"][0]
        cls.nodes = {node["id"]: node for node in cls.definition["nodes"]}
        cls.links = {link["id"]: link for link in cls.definition["links"]}

    def test_outer_contract_is_compact_and_display_free(self):
        self.assertEqual([450, 230], self.outer["size"])
        self.assertEqual([450, 230], self.outer["properties"]["cmkOuterSize"])
        self.assertEqual([450, 230], self.outer["properties"]["cmkManualSize"])
        self.assertEqual([], self.outer["properties"]["previewExposures"])
        self.assertNotIn("proxyWidgets", self.outer["properties"])
        self.assertEqual(
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "face_global_enable", "process_mode"],
            [item["name"] for item in self.outer["inputs"]],
        )
        self.assertEqual("FACEPROCESS ENABLE", self.outer["inputs"][5]["label"])
        self.assertEqual("process mode", self.outer["inputs"][6]["label"])
        self.assertEqual([True, "restore"], self.outer["widgets_values"])
        self.assertEqual(
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"],
            [item["name"] for item in self.outer["outputs"]],
        )

    def test_execute_node_owns_log_and_diagnostic_concatenation(self):
        execute = self.nodes[6218]
        self.assertEqual("CMKFaceProcessPipe", execute["type"])
        execute_input_names = [item["name"] for item in execute["inputs"]]
        self.assertIn("opt_log", execute_input_names)
        self.assertIn("opt_diagnostic", execute_input_names)
        self.assertEqual(
            ["LOG", "diagnostic"],
            [item["name"] for item in execute["outputs"][-2:]],
        )
        self.assertNotIn(
            "CMKLogConcat",
            {node["type"] for node in self.definition["nodes"]},
        )
        self.assertNotIn(
            "CMKDiagnosticConcat",
            {node["type"] for node in self.definition["nodes"]},
        )
        self.assertEqual(5229, self.links[16001]["origin_id"])
        self.assertEqual(
            (6218, execute_input_names.index("opt_log")),
            (self.links[16001]["target_id"], self.links[16001]["target_slot"]),
        )
        self.assertEqual(
            (6218, execute_input_names.index("opt_diagnostic")),
            (self.links[16002]["target_id"], self.links[16002]["target_slot"]),
        )

    def test_process_mode_is_exposed_without_changing_live_identity(self):
        mode_input = self.definition["inputs"][6]
        mode_link = self.links[mode_input["linkIds"][0]]
        self.assertEqual((-10, 6), (mode_link["origin_id"], mode_link["origin_slot"]))
        execute = self.nodes[6218]
        process_mode_slot = next(
            index for index, item in enumerate(execute["inputs"])
            if item["name"] == "process_mode"
        )
        self.assertEqual(
            (6218, process_mode_slot),
            (mode_link["target_id"], mode_link["target_slot"]),
        )

        provider = next(
            node for node in self.definition["nodes"] if node["type"] == "CMKVisualProvider"
        )
        self.assertEqual(
            ["FaceProcess", 30, "FaceProcess", "sdxl", "sdxl.faceprocess.standard"],
            provider["widgets_values"],
        )
        declaration = self.outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual("FaceProcess", declaration["label"])
        self.assertEqual("6218", declaration["live_node_id"])
        self.assertEqual(f"cmk-CMKVisualProvider-{provider['id']}", declaration["provider_id"])

    def test_visual_provider_receives_chain_before_after_and_enable(self):
        provider = next(
            node for node in self.definition["nodes"] if node["type"] == "CMKVisualProvider"
        )
        provider_inputs = {item["name"]: item for item in provider["inputs"]}
        self.assertEqual(16006, provider_inputs["VISUAL"]["link"])
        self.assertEqual(16007, provider_inputs["BEFORE"]["link"])
        self.assertEqual(16008, provider_inputs["IMAGE"]["link"])
        self.assertEqual(16009, provider_inputs["AFTER"]["link"])
        self.assertEqual(16010, provider_inputs["enable"]["link"])
        self.assertEqual([16011], provider["outputs"][0]["links"])
        compare = next(node for node in self.definition["nodes"] if node["type"] == "ImageCompare")
        compare_gate = next(
            node for node in self.definition["nodes"] if node["type"] == "CMKImageCompareEnableGate"
        )
        self.assertEqual([], compare["outputs"])
        self.assertEqual(16010, provider_inputs["enable"]["link"])
        self.assertEqual(12689, next(item for item in compare_gate["inputs"] if item["name"] == "ENABLE")["link"])
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
            (ROOT / "subgraphs" / "CMK Flow · 30 FaceProcess SDXL · Advanced.json").read_text(
                encoding="utf-8"
            )
        )
        definition = document["definitions"]["subgraphs"][0]
        nodes = {node["id"]: node for node in definition["nodes"]}
        links = {link["id"]: link for link in definition["links"]}
        execute_nodes = [nodes[node_id] for node_id in (5029, 5030, 5032)]

        outer = document["nodes"][0]
        self.assertEqual([450, 230], outer["size"])
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
        for execute in execute_nodes:
            input_names = [item["name"] for item in execute["inputs"]]
            self.assertIn("opt_log", input_names)
            self.assertIn("opt_diagnostic", input_names)
            self.assertEqual(
                ["LOG", "diagnostic"],
                [item["name"] for item in execute["outputs"][-2:]],
            )

        self.assertEqual((5027, 1, 5029, 21), tuple(links[12164][key] for key in ("origin_id", "origin_slot", "target_id", "target_slot")))
        self.assertEqual((5029, 4, 5030, 21), tuple(links[12166][key] for key in ("origin_id", "origin_slot", "target_id", "target_slot")))
        self.assertEqual((5030, 4, 5032, 21), tuple(links[12168][key] for key in ("origin_id", "origin_slot", "target_id", "target_slot")))
        self.assertEqual((5032, 4, 5011, 3), tuple(links[12170][key] for key in ("origin_id", "origin_slot", "target_id", "target_slot")))

        provider = next(node for node in nodes.values() if node["type"] == "CMKVisualProvider")
        self.assertEqual(
            ["FaceProcess", 30, "FaceProcess", "sdxl", "sdxl.faceprocess.advanced"],
            provider["widgets_values"],
        )
        declaration = outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual(["5029", "5030", "5032"], declaration["live_node_ids"])
        self.assertEqual(f"cmk-CMKVisualProvider-{provider['id']}", declaration["provider_id"])

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
