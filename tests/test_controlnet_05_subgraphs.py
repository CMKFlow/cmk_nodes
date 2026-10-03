import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBGRAPHS = ROOT / "subgraphs"

SPECS = {
    "CMK Flow · 05 ControlNet SDXL.json": (
        "CMKControlNetPreparePipe", "CMKControlNetBypassGate", ["PROCESS"]
    ),
    "CMK Flow · 05 ControlNet ZIT.json": (
        "CMKZITControlNetPreparePipe", "CMKZITControlNetBypassGate", ["PROCESS"]
    ),
    "CMK Flow · 05 ControlNet Combined.json": (
        "CMKCombinedControlNetPreparePipe",
        "CMKCombinedControlNetBypassGate",
        ["PROCESS SDXL", "PROCESS ZIT"],
    ),
}


class ControlNet05SubgraphTests(unittest.TestCase):
    def load(self, name):
        return json.loads((SUBGRAPHS / name).read_text(encoding="utf-8"))

    def assert_reference_layout(self, payload):
        outer = payload["nodes"][0]
        graph = payload["definitions"]["subgraphs"][0]
        self.assertEqual([300, 190], outer["size"])
        self.assertEqual([300, 190], outer["properties"]["cmkCleanViewExpandedSize"])
        self.assertIn(outer["properties"]["cmkManualSize"], ([300, 100], [300, 190]))
        self.assertNotIn("proxyWidgets", outer["properties"])
        enabled = next(item for item in outer["inputs"] if item["name"] == "ENABLE")
        self.assertEqual({"name": "ENABLE"}, enabled["widget"])
        self.assertNotIn("shape", enabled)
        self.assertEqual("#334155", outer["color"])
        self.assertEqual("#1f2937", outer["bgcolor"])
        self.assertEqual([8360, -360, 930], graph["groups"][0]["bounding"][:3])
        self.assertGreaterEqual(graph["groups"][0]["bounding"][3], 660)
        self.assertNotIn("CMK Boundary Cache", {node["type"] for node in graph["nodes"]})
        self.assertNotIn("CMKVisualCompare", {node["type"] for node in graph["nodes"]})

    @staticmethod
    def endpoint_names(graph, link):
        nodes = {node["id"]: node for node in graph["nodes"]}
        if link["origin_id"] == -10:
            source = f"PUBLIC.{graph['inputs'][link['origin_slot']]['name']}"
        else:
            node = nodes[link["origin_id"]]
            source = f"{node['type']}.{node['outputs'][link['origin_slot']]['name']}"
        if link["target_id"] == -20:
            target = f"PUBLIC.{graph['outputs'][link['target_slot']]['name']}"
        else:
            node = nodes[link["target_id"]]
            target = f"{node['type']}.{node['inputs'][link['target_slot']]['name']}"
        return source, target

    def test_every_variant_has_the_exact_named_port_topology(self):
        for filename, (prepare_type, gate_type, processes) in SPECS.items():
            with self.subTest(filename=filename):
                graph = self.load(filename)["definitions"]["subgraphs"][0]
                actual = {self.endpoint_names(graph, link) for link in graph["links"]}
                expected = set()
                for name in [*processes, "IMAGE", "LOG", "VISUAL", "ENABLE"]:
                    expected.add((f"PUBLIC.{name}", f"{prepare_type}.{name}"))
                expected.update({
                    ("CMKLoadImage.IMAGE", f"{prepare_type}.REFERENCE IMAGE INPUT"),
                    ("CMKLoadImage.FILENAME_STRING", f"{prepare_type}.REFERENCE IMAGE NAME"),
                    ("PUBLIC.ENABLE", f"{gate_type}.ENABLE"),
                    (f"{prepare_type}.diagnostic", f"{gate_type}.DIAGNOSTIC ACTIVE"),
                    (f"{gate_type}.diagnostic", "PUBLIC.diagnostic"),
                    (f"{prepare_type}.VISUAL", "PUBLIC.VISUAL"),
                })
                for name in [*processes, "IMAGE", "LOG"]:
                    expected.update({
                        (f"PUBLIC.{name}", f"{gate_type}.{name} BYPASS"),
                        (f"{prepare_type}.{name}", f"{gate_type}.{name} ACTIVE"),
                        (f"{gate_type}.{name}", f"PUBLIC.{name}"),
                    })
                self.assertEqual(expected, actual)

    def test_every_link_is_registered_at_both_endpoints(self):
        for filename in SPECS:
            with self.subTest(filename=filename):
                graph = self.load(filename)["definitions"]["subgraphs"][0]
                nodes = {node["id"]: node for node in graph["nodes"]}
                for link in graph["links"]:
                    if link["origin_id"] == -10:
                        self.assertIn(link["id"], graph["inputs"][link["origin_slot"]]["linkIds"])
                    else:
                        self.assertIn(
                            link["id"],
                            nodes[link["origin_id"]]["outputs"][link["origin_slot"]]["links"],
                        )
                    if link["target_id"] == -20:
                        self.assertIn(link["id"], graph["outputs"][link["target_slot"]]["linkIds"])
                    else:
                        self.assertEqual(
                            link["id"],
                            nodes[link["target_id"]]["inputs"][link["target_slot"]]["link"],
                        )

    def test_zit_contract_and_visual_provider(self):
        payload = self.load("CMK Flow · 05 ControlNet ZIT.json")
        self.assert_reference_layout(payload)
        outer = payload["nodes"][0]
        self.assertEqual(
            ["PROCESS", "IMAGE", "LOG", "VISUAL", "ENABLE"],
            [item["name"] for item in outer["inputs"]],
        )
        self.assertEqual(
            ["PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"],
            [item["name"] for item in outer["outputs"]],
        )
        provider = outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual("zit", provider["branch"])
        self.assertEqual("zit.controlnet", provider["stage_key"])
        self.assertIn("CMKZITControlNetPreparePipe", provider["provider_id"])

    def test_combined_has_only_split_process_ports(self):
        payload = self.load("CMK Flow · 05 ControlNet Combined.json")
        self.assert_reference_layout(payload)
        outer = payload["nodes"][0]
        self.assertEqual(
            ["PROCESS SDXL", "PROCESS ZIT", "IMAGE", "LOG", "VISUAL", "ENABLE"],
            [item["name"] for item in outer["inputs"]],
        )
        self.assertEqual(
            ["PROCESS SDXL", "PROCESS ZIT", "IMAGE", "LOG", "VISUAL", "diagnostic"],
            [item["name"] for item in outer["outputs"]],
        )
        provider = outer["properties"]["cmkVisualProviders"][0]
        self.assertEqual("", provider["branch"])
        self.assertEqual("controlnet", provider["stage_key"])
        metadata = payload["extra"]["CMKFlow"]
        self.assertEqual("96aabb3d-a0c9-4539-86c0-2b2a7abb0712", metadata["variantOf"])
        self.assertEqual("Combined", metadata["variantLabel"])

    def test_all_variants_share_post_sampling_model_unload(self):
        sampler = (ROOT / "pipe" / "cmk_pipe_sampler.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def _unload_completed_controlnet", sampler)
        self.assertIn('pipe.get("control_net")', sampler)
        self.assertIn('pipe.get("zit_controlnet_model_patch")', sampler)
        self.assertIn('new_pipe["controlnet_model_status"]', sampler)


if __name__ == "__main__":
    unittest.main()
