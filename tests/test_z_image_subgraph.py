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
            ["PROCESS", "IMAGE", "LOG"],
        )
        self.assertEqual(
            [item["name"] for item in self.definition["outputs"]],
            ["MODEL", "PROCESS", "IMAGE", "LOG", "diagnostic"],
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
        self.assertEqual(image_link["origin_id"], -10)
        self.assertEqual(image_link["origin_slot"], 1)

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
        for slot in range(5):
            finalize_link = next(
                link for link in self.definition["links"]
                if link["origin_id"] == finalize["id"]
                and link["origin_slot"] == slot
            )
            self.assertEqual(finalize_link["target_id"], boundary["id"])
            boundary_link = next(
                link for link in self.definition["links"]
                if link["origin_id"] == boundary["id"]
                and link["origin_slot"] == slot
            )
            self.assertEqual(boundary_link["target_id"], gate["id"])

    def test_confirmed_catalog_entry_is_published_as_beta(self):
        metadata = self.document["extra"]["CMKFlow"]
        self.assertTrue(metadata["published"])
        self.assertEqual(metadata["status"], "BETA")
        self.assertEqual(metadata["compatibility"], ["Z-Image Turbo"])
        self.assertIn("Experimentelles maskiertes Inpaint", metadata["features"])
        self.assertEqual(
            metadata["recommendedAfter"],
            [
                "35 Active Family Result (optional)",
                "40 FaceSwap",
                "90 Upscale & Save",
            ],
        )

    def test_zit_inpaint_ui_exposes_mask_hole_fill(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "web"
            / "js"
            / "cmk_flow_start_guidance_v37.js"
        ).read_text(encoding="utf-8")
        self.assertIn('"z-image-inpaint"', source)
        self.assertIn('ZIT_INPAINT_DEFAULT_SIZE = "768x512"', source)
        self.assertIn('"MODE · INPAINT EXPERIMENTAL"', source)
        hidden_block = source[
            source.index("const Z_IMAGE_HIDDEN_WIDGETS"):
            source.index("const USER_INPUT_LABELS")
        ]
        self.assertNotIn('"mask_fill_holes"', hidden_block)


if __name__ == "__main__":
    unittest.main()
