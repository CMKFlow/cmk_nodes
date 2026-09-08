import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FlowBrowserCatalogTests(unittest.TestCase):
    EXPECTED_FLOWS = {
        "SDXL LoRA Stack",
        "05 ControlNet",
        "05 ControlNet ZIT",
        "05 ControlNet Combined",
        "10 KSampler SDXL 1st Pass",
        "10 KSampler Z-Image Turbo",
        "15 InstantID-Sampler SDXL",
        "20 Refiner SDXL",
        "23 Detailer SDXL",
        "23 Detailer SDXL · Advanced",
        "25 FaceRebuild SDXL",
        "25 FaceRebuild SDXL · Advanced",
        "30 FaceProcess SDXL",
        "30 FaceProcess SDXL · Advanced",
        "40 FaceSwap",
        "40 FaceSwap · Advanced",
        "90 Upscale & Save",
    }
    EXPECTED_REFERENCE_WORKFLOWS = {
        "Flow #01 - CMK Text2Image ZIT.json",
        "Flow #02 - CMK Text2Image ControlNet ZIT.json",
        "Flow #03 - CMK Inpaint ZIT · Experimentell.json",
        "Flow #04 - CMK Text2Image ZIT+SDXL.json",
        "Flow #05 - CMK Text2Image SDXL.json",
        "Flow #06 - CMK Text2Image ControlNet SDXL.json",
        "Flow #07 - CMK Text2Image InstantID SDXL.json",
        "Flow #08 - CMK Text2Image Area Conditioning SDXL .json",
        "Flow #09 - CMK Inpaint SDXL.json",
        "Flow #10 - CMK Inpaint InstantID SDXL.json",
        "Ref #01 - CMK Detailer.json",
        "Ref #02 - CMK FaceRebuild.json",
        "Ref #03 - CMK FaceProcess.json",
        "Ref #04 - CMK FaceSwap.json",
        "CMK - Full Flow.json",
        "CMK - FaceSwap Video.json",
        "CMK: FaceSwap vs FaceRebuild.json",
    }

    def test_image_loaders_default_to_face_reference(self):
        for relative_path in (
            "pipe/loaders/cmk_load_image.py",
            "pipe/loaders/cmk_image_load_resize.py",
        ):
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertLess(
                source.index('"face_reference.png"'),
                source.index('"controlnet_reference.png"'),
                relative_path,
            )


    def test_reference_catalog_contains_only_confirmed_workflows(self):
        showcase = ROOT / "workflows" / "showcase"
        metadata_root = showcase / "metadata"
        workflows = {path.name for path in showcase.glob("*.json")}
        metadata_files = {path.name for path in metadata_root.glob("*.json")}
        self.assertEqual(workflows, self.EXPECTED_REFERENCE_WORKFLOWS)
        self.assertEqual(metadata_files, self.EXPECTED_REFERENCE_WORKFLOWS)

        for filename in sorted(workflows):
            with self.subTest(filename=filename):
                workflow = json.loads((showcase / filename).read_text(encoding="utf-8"))
                self.assertIsInstance(workflow.get("nodes"), list)
                metadata = json.loads(
                    (metadata_root / filename).read_text(encoding="utf-8")
                )
                self.assertTrue(metadata.get("published"))
                for field in (
                    "displayName", "category", "category_en", "description",
                    "description_en", "cmkHighlight", "cmkHighlight_en",
                ):
                    self.assertTrue(metadata.get(field), f"{filename}: {field}")
                self.assertTrue(metadata.get("previews"), filename)
                for preview in metadata["previews"]:
                    asset = ROOT / "web" / preview["src"]
                    self.assertTrue(asset.is_file(), f"{filename}: {preview['src']}")
                    self.assertGreater(asset.stat().st_size, 0)

    def test_technical_reference_directory_keeps_only_video_reference(self):
        reference_root = ROOT / "workflows" / "reference"
        self.assertEqual(
            {path.name for path in reference_root.glob("*.json")},
            {"CMK_FaceSwap_Video_Reference_v2.2.json"},
        )

    def test_all_curated_flow_subgraphs_are_published_with_real_previews(self):
        published = set()
        for path in sorted((ROOT / "subgraphs").glob("CMK Flow · *.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            metadata = document.get("extra", {}).get("CMKFlow", {})
            if not metadata.get("published"):
                continue
            published.add(metadata["displayName"])
            self.assertTrue(metadata.get("previews"), path.name)
            for preview in metadata["previews"]:
                asset = ROOT / "web" / preview["src"]
                self.assertTrue(asset.is_file(), f"{path.name}: {preview['src']}")
                self.assertGreater(asset.stat().st_size, 0)
        self.assertEqual(published, self.EXPECTED_FLOWS)

    def test_faceswap_flows_expose_both_families_and_native_engine(self):
        expected = {"SDXL", "Z-Image Turbo", "CMK Native FaceSwap"}
        for filename in (
            "CMK Flow · 40 FaceSwap.json",
            "CMK Flow · 40 FaceSwap · Advanced.json",
        ):
            with self.subTest(filename=filename):
                metadata = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )["extra"]["CMKFlow"]
                self.assertEqual(set(metadata["compatibility"]), expected)
                self.assertNotIn("ReActor", metadata["compatibility"])

    def test_custom_flow_nodes_reference_existing_previews(self):
        metadata = json.loads(
            (ROOT / "web" / "flow_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]
        self.assertFalse(
            any(
                "LaMa" in str(feature)
                for feature in metadata["CMKPipeCreateImage"]["features"]
            )
        )
        required_catalog_fields = {
            "category", "compatibility", "version", "author", "status"
        }
        for node_name, entry in metadata.items():
            self.assertTrue(
                required_catalog_fields.issubset(entry),
                f"{node_name}: incomplete catalog column metadata",
            )
            self.assertTrue(entry["compatibility"], node_name)
            self.assertNotEqual(entry["version"], "—", node_name)
            self.assertTrue(entry["author"], node_name)
        for node_name, entry in metadata.items():
            for preview in entry.get("previews", []):
                asset = ROOT / "web" / preview["src"]
                self.assertTrue(asset.is_file(), f"{node_name}: {preview['src']}")
                self.assertGreater(asset.stat().st_size, 0)

    def test_toolbox_catalog_previews_are_complete_and_existing(self):
        metadata = json.loads(
            (ROOT / "web" / "toolbox_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]
        preview_entries = {
            node_name: entry["previews"]
            for node_name, entry in metadata.items()
            if entry.get("previews")
        }
        self.assertEqual(len(preview_entries), 39)
        for node_name, previews in preview_entries.items():
            with self.subTest(node_name=node_name):
                for preview in previews:
                    asset = ROOT / "web" / preview["src"]
                    self.assertTrue(asset.is_file(), f"{node_name}: {preview['src']}")
                    self.assertGreater(asset.stat().st_size, 0)

    def test_english_browser_localizes_preview_tab_labels(self):
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('Modul: "Module"', source)
        self.assertIn('Aufbau: "Structure"', source)
        self.assertIn('Wirkung: "Effect"', source)
        self.assertIn('`View ${index + 1}`', source)

        english = json.loads(
            (ROOT / "web" / "browser_content_en.json").read_text(encoding="utf-8")
        )
        create_features = english["flows"]["CMKPipeCreateImage"]["features"]
        self.assertFalse(any("LaMa" in str(feature) for feature in create_features))

    def test_preview_paths_are_encoded_per_segment(self):
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'preview.src.split("/").map(encodeURIComponent).join("/")', source
        )

    def test_flow_browser_uses_custom_node_catalog_metadata(self):
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("metadata.category || nodeDef.category", source)
        self.assertIn('metadata.version || "1.0.0"', source)
        self.assertIn("metadata.author ||", source)
        self.assertIn("metadata.compatibility", source)
        self.assertIn(".cmk-flow-meta-value", source)
        self.assertIn("overflow-wrap: anywhere", source)
        self.assertNotIn(
            ".cmk-flow-meta-value { display: block; margin-top: 3px; overflow: hidden",
            source,
        )

    def test_non_numbered_flow_loaders_follow_module_90(self):
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("CMKImageLoadAndResizePipe: 101", source)
        self.assertIn("CMKLoadImage: 102", source)
        self.assertIn("CMKCheckpointVAELoaderPipe: 103", source)

    def test_faceswap_image_input_is_a_toolbox_node(self):
        source = (ROOT / "pipe" / "loaders" / "cmk_swap_image_loader.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('CATEGORY = "CMK/Toolbox/I-O"', source)
        metadata = json.loads(
            (ROOT / "web" / "toolbox_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]
        self.assertIn("CMKSwapImageLoaderPipe", metadata)

    def test_facerebuild_custom_node_names_do_not_claim_flow_position(self):
        mappings = (ROOT / "cmk_mappings.py").read_text(encoding="utf-8")
        self.assertIn(
            '"CMKInstantIDFaceRebuildSDXL": "CMK FaceRebuild SDXL"',
            mappings,
        )
        self.assertIn(
            '"CMKInstantIDFaceRebuildAdvancedSDXL": '
            '"CMK FaceRebuild SDXL · Advanced"',
            mappings,
        )
        self.assertNotIn(
            '"CMKInstantIDFaceRebuildSDXL": "CMK Flow · 25', mappings
        )
        for filename in (
            "CMK Flow · 25 FaceRebuild SDXL.json",
            "CMK Flow · 25 FaceRebuild SDXL · Advanced.json",
        ):
            document = json.loads((ROOT / "subgraphs" / filename).read_text(encoding="utf-8"))
            self.assertEqual(document["definitions"]["subgraphs"][0]["name"], filename[:-5])
            self.assertEqual(document["extra"]["CMKFlow"]["order"], 25)

    def test_toolbox_includes_functional_subgraph_building_blocks_only(self):
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        for node_type in (
            "CMKDetailerPreparePipe", "CMKFaceProcessPreparePipe",
            "CMKFaceProcessPipe", "CMKFaceSwapImagePipe", "CMKKSamplerPipe",
            "CMKRefinerPrepareSDXLPipe", "CMKRefinerPipe",
            "CMKSamplerPrepareSDXLPipe", "CMKSamplerPrepareZImageTurboPipe",
            "CMKZImageTurboLoaderPipe", "CMKZImageTurboFinalizePipe",
            "CMK_SmartDetailerPipe", "CMK_SmartUpscalerPipe",
        ):
            self.assertIn(f'["{node_type}",', source)
        for implementation_detail in (
            "CMKDetailerBoundaryCache", "CMKFamilyBranchGateSDXL",
            "CMKResultPackPipe", "CMKResultUnpackPipe",
        ):
            self.assertNotIn(f'["{implementation_detail}",', source)

    def test_save_project_image_is_a_toolbox_node(self):
        source = (ROOT / "nodes" / "io" / "save_project_image.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('CATEGORY = "CMK/Toolbox/I-O"', source)

    def test_controlnet_and_loader_entries_use_shared_variant_pages(self):
        metadata = json.loads(
            (ROOT / "web" / "flow_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]
        self.assertEqual(metadata["CMKControlNetPreparePipe"]["displayName"], "05 ControlNet")
        primary_controlnet = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · 05 ControlNet SDXL.json").read_text(
                encoding="utf-8"
            )
        )["extra"]["CMKFlow"]
        self.assertEqual("SDXL", primary_controlnet["variantLabel"])
        self.assertEqual(
            metadata["CMKZITControlNetPreparePipe"]["variantOf"],
            "CMK Flow · 05 ControlNet SDXL",
        )
        self.assertEqual(
            metadata["CMKCombinedControlNetPreparePipe"]["variantOf"],
            "CMK Flow · 05 ControlNet SDXL",
        )
        self.assertEqual(metadata["CMKImageLoadAndResizePipe"]["displayName"], "Loader")
        self.assertEqual(
            metadata["CMKLoadImage"]["variantOf"], "CMK Flow · Image Input"
        )
        self.assertEqual(
            metadata["CMKCheckpointVAELoaderPipe"]["variantOf"],
            "CMK Flow · Image Input",
        )
        lora_stack = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · 02 SDXL LoRA Stack.json").read_text(
                encoding="utf-8"
            )
        )["extra"]["CMKFlow"]
        self.assertEqual(lora_stack["variantOf"], "CMK Flow · Image Input")
        self.assertEqual(lora_stack["variantLabel"], "SDXL LoRA Stack")
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(encoding="utf-8")
        self.assertIn("const allFlows = [...discovered, ...discoverCuratedNodes", source)
        self.assertIn('nodeType === "CMKVisualizer"', source)
        self.assertEqual(metadata["CMKVisualizer"]["category"], "Finish")
        self.assertIn("button.textContent = variant.variantLabel", source)


if __name__ == "__main__":
    unittest.main()
