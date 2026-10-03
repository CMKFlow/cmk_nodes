import json
import unittest
from pathlib import Path


class ZImageSubgraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).resolve().parents[1]
            / "subgraphs"
            / "CMK Flow · 10 KSampler Z-Image Turbo.json"
        )
        cls.document = json.loads(path.read_text(encoding="utf-8"))
        cls.definition = cls.document["definitions"]["subgraphs"][0]

    def test_public_contract_rejoins_common_image_flow(self):
        self.assertEqual(
            [item["name"] for item in self.definition["inputs"]],
            ["PROCESS", "IMAGE", "LOG", "VISUAL", "SAMPLED SDXL", "LOG SDXL", "VISUAL SDXL"],
        )
        self.assertEqual(
            [item["name"] for item in self.definition["outputs"]],
            ["MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic"],
        )
        self.assertEqual(
            self.definition["inputs"][0]["type"],
            "CMK_PROCESS_Z_IMAGE",
        )
        self.assertEqual(
            self.definition["outputs"][1]["type"],
            "CMK_PROCESS_Z_IMAGE",
        )

    def test_native_z_image_chain_is_complete(self):
        nodes = {node["type"]: node for node in self.definition["nodes"]}
        self.assertIn("CMKZImageTurboLoaderPipe", nodes)
        self.assertIn("CMKSamplerPrepareZImageTurboPipe", nodes)
        self.assertIn("CMKKSamplerPipe", nodes)
        self.assertIn("CMKZImageTurboFinalizePipe", nodes)

        prepare = nodes["CMKSamplerPrepareZImageTurboPipe"]
        self.assertEqual(
            prepare["widgets_values"],
            [
                1565304366,
                "fixed",
                8,
                "res_multistep",
                "simple",
                1,
                3,
                "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors",
            ],
        )

        image_input = next(
            item for item in prepare["inputs"] if item["name"] == "IMAGE"
        )
        image_link = next(
            link for link in self.definition["links"]
            if link["id"] == image_input["link"]
        )
        bridge = nodes["CMKHybridZITInputPipe"]
        self.assertEqual(image_link["origin_id"], bridge["id"])
        self.assertEqual(bridge["outputs"][image_link["origin_slot"]]["name"], "IMAGE")

    def test_zit_prepare_applies_the_selected_lora_bundle(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "pipe"
            / "cmk_z_image_turbo.py"
        ).read_text(encoding="utf-8")
        self.assertIn("CMKLoRATextLoader().load_loras", source)
        self.assertIn('PROCESS.get("lora_stack")', source)
        self.assertIn('PROCESS.get("lora_syntax", "")', source)
        self.assertIn('"loaded_loras": loaded_loras', source)

    def test_loader_is_hard_gated_by_the_direct_z_process_signal(self):
        loader = next(
            node for node in self.definition["nodes"]
            if node["type"] == "CMKZImageTurboLoaderPipe"
        )
        process_slot = next(
            index for index, item in enumerate(loader["inputs"])
            if item["name"] == "PROCESS"
        )
        process_link = next(
            link for link in self.definition["links"]
            if link["target_id"] == loader["id"]
            and link["target_slot"] == process_slot
        )
        public_process_slot = next(
            index for index, item in enumerate(self.definition["inputs"])
            if item["name"] == "PROCESS"
        )
        self.assertEqual(process_link["origin_id"], -10)
        self.assertEqual(process_link["origin_slot"], public_process_slot)

    def test_every_declared_boundary_link_exists(self):
        links = {link["id"] for link in self.definition["links"]}
        for boundary in self.definition["inputs"] + self.definition["outputs"]:
            self.assertTrue(set(boundary["linkIds"]).issubset(links))

    def test_finalize_has_no_path_around_boundary_cache(self):
        nodes = {node["id"]: node for node in self.definition["nodes"]}
        finalize = next(
            node for node in nodes.values()
            if node["type"] == "CMKZImageTurboFinalizePipe"
        )
        boundary = next(
            node for node in nodes.values()
            if node["type"] == "CMKZImageBoundaryCache"
        )
        gate = next(
            node for node in nodes.values()
            if node["type"] == "CMKFamilyBranchGateZImage"
        )
        for finalize_slot, boundary_slot in ((0, 0), (2, 1), (3, 2), (4, 3)):
            finalize_link = next(
                link for link in self.definition["links"]
                if link["origin_id"] == finalize["id"]
                and link["origin_slot"] == finalize_slot
            )
            self.assertEqual(finalize_link["target_id"], boundary["id"])
            self.assertEqual(finalize_link["target_slot"], boundary_slot)
            boundary_links = [
                link for link in self.definition["links"]
                if link["origin_id"] == boundary["id"]
                and link["origin_slot"] == boundary_slot
            ]
            if finalize_slot == 4:
                self.assertEqual(len(boundary_links), 1)
                self.assertEqual(boundary_links[0]["target_id"], -20)
            else:
                self.assertEqual(len(boundary_links), 1)
                self.assertEqual(boundary_links[0]["target_id"], gate["id"])
        self.assertEqual([], finalize["outputs"][1]["links"])

    def test_confirmed_catalog_entry_is_published_as_beta(self):
        metadata = self.document["extra"]["CMKFlow"]
        self.assertTrue(metadata["published"])
        self.assertEqual(metadata["status"], "STABLE")
        self.assertEqual(metadata["compatibility"], ["Z-Image Turbo"])
        self.assertIn("Experimentelles maskiertes Inpainting ausführen", metadata["features"])
        self.assertEqual(
            metadata["recommendedAfter"],
            [
                {"label": "Active Family Result (optional)", "targetId": "CMKFamilyResultMergePipe"},
                {"label": "FaceSwap", "targetId": "9993a5f9-7cd5-431c-8653-6e187ef9d214"},
                {"label": "Visualizer", "targetId": "CMKVisualizer"},
                {"label": "Upscale & Save", "targetId": "6f8d63a4-7ea5-4c18-9900-2ec3ed33c9b6"},
            ],
        )

    def test_zit_inpaint_ui_exposes_mask_hole_fill(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "web"
            / "js"
            / "cmk_flow_start_guidance_v51.js"
        ).read_text(encoding="utf-8")
        self.assertIn('"z-image-inpaint"', source)
        self.assertIn('ZIT_INPAINT_DEFAULT_SIZE = "768x512"', source)
        self.assertIn('familyResolutionValues(family, inpaint = false)', source)
        self.assertIn('mode === "z-image-inpaint"', source)
        self.assertIn('"MODE · INPAINT EXPERIMENTAL"', source)
        hidden_block = source[
            source.index("const Z_IMAGE_HIDDEN_WIDGETS"):
            source.index("const USER_INPUT_LABELS")
        ]
        self.assertNotIn('"mask_fill_holes"', hidden_block)


if __name__ == "__main__":
    unittest.main()
