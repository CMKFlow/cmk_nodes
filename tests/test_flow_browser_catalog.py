import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FlowBrowserCatalogTests(unittest.TestCase):
    EXPECTED_FLOWS = {
        "LoRA Stack · Combined",
        "LoRA Stack · SDXL",
        "LoRA Stack · ZIT",
        "05 ControlNet",
        "05 ControlNet ZIT",
        "05 ControlNet Combined",
        "10 KSampler",
        "10 KSampler Z-Image Turbo",
        "15 InstantID-Sampler SDXL",
        "20 Refiner SDXL",
        "Detailer SDXL",
        "Detailer SDXL · Advanced",
        "FaceRebuild SDXL",
        "FaceRebuild SDXL · Advanced",
        "FaceProcess SDXL",
        "FaceProcess SDXL · Advanced",
        "FaceSwap",
        "FaceSwap · Advanced",
        "Upscale & Save",
        "MaskDetailer SDXL",
    }
    EXPECTED_REFERENCE_WORKFLOWS = {
        "CMK 2.5 · Flow · InOutpaint HYBRID.json",
        "CMK 2.5 · Flow · InOutpaint SDXL.json",
        "CMK 2.5 · Flow · InOutpaint SDXL · HYBRID · ZIT.json",
        "CMK 2.5 · Flow · Inpaint - FaceRebuild HYBRID.json",
        "CMK 2.5 · Flow · Inpaint Z-Image Turbo.json",
        "CMK 2.5 · Flow · InstantID · Hybrid Inpaint SDXL.json",
        "CMK 2.5 · Flow · Text2Image - InstantID SDXL and FaceRebuild HYBRID · ZIT.json",
        "CMK 2.5 · Flow · Text2Image - Regional Conditioning HYBRID.json",
        "CMK 2.5 · Flow · Text2Image - Regional Conditioning SDXL.json",
        "CMK 2.5 · Flow · Text2Image ControlNet HYBRID.json",
        "CMK 2.5 · Flow · Text2Image ControlNet SDXL · HYBRID · ZIT.json",
        "CMK 2.5 · Flow · Text2Image ControlNet Z-Image Turbo.json",
        "CMK 2.5 · Flow · Text2Image HYBRID.json",
        "CMK 2.5 · Flow · Text2Image SDXL.json",
        "CMK 2.5 · Flow · Text2Image SDXL · HYBRID · ZIT.json",
        "CMK 2.5 · Flow · Text2Image Z-Image Turbo.json",
        "CMK 2.5 · Flow · Text2Image-ControlNet SDXL.json",
        "CMK 2.5 · Flow · Text2Image-FaceRebuild HYBRID.json",
        "CMK 2.5 · Flow · Text2Image-FaceRebuild Z-Image Turbo.json",
        "CMK 2.5 · Flow · Text2Image-InstantID SDXL.json",
        "CMK 2.5 · FaceProcess.json",
        "CMK 2.5 · FaceProcess · Modul.json",
        "CMK 2.5 · FaceRebuild.json",
        "CMK 2.5 · FaceRebuild · Modul.json",
        "CMK 2.5 · FaceSwap.json",
        "CMK 2.5 · FaceSwap · Modul.json",
        "CMK 2.5 · MaskDetailer .json",
        "CMK 2.5 · MaskDetailer · Modul.json",
        "CMK 2.5 · SmartDetailer.json",
        "CMK 2.5 · SmartDetailer · Modul.json",
        "FaceSwap vs FaceRebuild.json",
        "Identity Processing Matrix.json",
        "CMK 2.5 · InOutPaint.json",
        "CMK 2.5 · Text2Image.json",
        "CMK 2.5 · Text2Image Identity 1.json",
        "CMK 2.5 · Text2Image Identity 3.json",
        "CMK 2.5 · Full Flow .json",
        "CMK FaceSwap Video.json",
    }
    EXPECTED_SOURCE_PREVIEW_WORKFLOWS = {
        "CMK 2.5 · Flow · Inpaint - FaceRebuild HYBRID.json",
        "CMK 2.5 · Flow · InstantID · Hybrid Inpaint SDXL.json",
        "CMK 2.5 · Flow · Text2Image - InstantID SDXL and FaceRebuild HYBRID · ZIT.json",
        "CMK 2.5 · Flow · Text2Image-FaceRebuild HYBRID.json",
        "CMK 2.5 · Flow · Text2Image-FaceRebuild Z-Image Turbo.json",
        "CMK 2.5 · Flow · Text2Image-InstantID SDXL.json",
    }
    EXPECTED_NON_PREVIEW_ASSETS = {
        "REFERENCES/MODULE WORKFLOWS/CMK 2.5 · FaceRebuild · Modul/portrait_reference_00003.png",
        "REFERENCES/MODULE WORKFLOWS/CMK 2.5 · FaceSwap · Modul/portrait_reference_00003.png",
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
                    "description_en",
                ):
                    self.assertTrue(metadata.get(field), f"{filename}: {field}")
                self.assertEqual(
                    bool(metadata.get("cmkHighlight")),
                    bool(metadata.get("cmkHighlight_en")),
                    f"{filename}: bilingual CMK highlight mismatch",
                )
                self.assertEqual(
                    bool(metadata.get("info")),
                    bool(metadata.get("info_en")),
                    f"{filename}: bilingual info mismatch",
                )
                self.assertTrue(metadata.get("previews"), filename)
                for preview in metadata["previews"]:
                    asset = ROOT / "web" / preview["src"]
                    self.assertTrue(asset.is_file(), f"{filename}: {preview['src']}")
                    self.assertGreater(asset.stat().st_size, 0)

        categories = Counter(
            json.loads((metadata_root / filename).read_text(encoding="utf-8"))["category"]
            for filename in workflows
        )
        self.assertEqual(
            categories,
            {
                "TASK WORKFLOWS": 20,
                "MODULE WORKFLOWS": 10,
                "COMPARISONS": 2,
                "REAL-WORLD WORKFLOWS": 4,
                "SYSTEM WORKFLOW": 1,
                "LEGACY": 1,
            },
        )

        preview_paths = {
            Path(preview["src"]).relative_to("assets/showcase").as_posix()
            for filename in workflows
            for preview in json.loads(
                (metadata_root / filename).read_text(encoding="utf-8")
            )["previews"]
        }
        asset_root = ROOT / "web" / "assets" / "showcase"
        asset_paths = {
            path.relative_to(asset_root).as_posix()
            for path in asset_root.rglob("*.png")
        }
        self.assertTrue(self.EXPECTED_NON_PREVIEW_ASSETS <= asset_paths)
        self.assertEqual(preview_paths, asset_paths - self.EXPECTED_NON_PREVIEW_ASSETS)
        self.assertEqual(len(preview_paths), 107)

        source_preview_workflows = set()
        for filename in workflows:
            previews = json.loads(
                (metadata_root / filename).read_text(encoding="utf-8")
            )["previews"]
            source_previews = [
                preview for preview in previews
                if Path(preview["src"]).name == "portrait_reference_00003.png"
            ]
            if source_previews:
                source_preview_workflows.add(filename)
                self.assertEqual(source_previews, [previews[-1]], filename)
                self.assertEqual(source_previews[0]["label"], "Source")
                self.assertEqual(source_previews[0]["label_en"], "Source")
        self.assertEqual(
            source_preview_workflows,
            self.EXPECTED_SOURCE_PREVIEW_WORKFLOWS,
        )

    def test_module_reference_previews_show_module_before_structure(self):
        metadata_root = ROOT / "workflows" / "showcase" / "metadata"
        for filename in (
            "CMK 2.5 · FaceProcess · Modul.json",
            "CMK 2.5 · FaceRebuild · Modul.json",
            "CMK 2.5 · FaceSwap · Modul.json",
            "CMK 2.5 · MaskDetailer · Modul.json",
            "CMK 2.5 · SmartDetailer · Modul.json",
        ):
            with self.subTest(filename=filename):
                previews = json.loads(
                    (metadata_root / filename).read_text(encoding="utf-8")
                )["previews"]
                self.assertEqual(
                    [preview["label"] for preview in previews[:2]],
                    ["Modul", "Aufbau"],
                )

    def test_inoutpaint_hybrid_preview_navigation_is_grouped_without_remapping(self):
        metadata = json.loads(
            (
                ROOT
                / "workflows"
                / "showcase"
                / "metadata"
                / "CMK 2.5 · Flow · InOutpaint HYBRID.json"
            ).read_text(encoding="utf-8")
        )
        previews = metadata["previews"]
        self.assertEqual(
            [preview["group"] for preview in previews],
            ["InOutpaint"] * 3 + ["Remove"] * 3 + ["Extend"] * 3,
        )
        self.assertEqual(
            [preview["label"] for preview in previews],
            ["Preparation", "1st-Pass SDXL", "2nd-Pass ZIT"] * 3,
        )
        self.assertEqual(
            [Path(preview["src"]).name for preview in previews],
            [
                "CMK 2.5 · Flow · InOutpaint HYBRID - 1 InOutpaint 1 Preparation.png",
                "CMK 2.5 · Flow · InOutpaint HYBRID - 1 InOutpaint 2 1st-Pass SDXL.png",
                "CMK 2.5 · Flow · InOutpaint HYBRID - 1 InOutpaint 3 2nd-Pass ZIT.png",
                "CMK 2.5 · Flow · InOutpaint HYBRID - 2 Remove 1 Preparation.png",
                "CMK 2.5 · Flow · InOutpaint HYBRID - 2 Remove 2 1st-Pass SDXL.png",
                "CMK 2.5 · Flow · InOutpaint HYBRID - 2 Remove 3 2nd-Pass ZIT.png",
                "CMK 2.5 · Flow · InOutpaint HYBRID - 3 Extend 1 Preparation.png",
                "CMK 2.5 · Flow · InOutpaint HYBRID - 3 Extend 2 1st-Pass SDXL.png",
                "CMK 2.5 · Flow · InOutpaint HYBRID - 3 Extend 3 2nd-Pass ZIT.png",
            ],
        )

        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        renderer = source[
            source.index("function renderNodePreview") : source.index("function flowIcon")
        ]
        self.assertIn("flow.previews.some((preview) => preview.group)", renderer)
        self.assertIn('group.className = "cmk-flow-preview-tab-group"', renderer)
        self.assertNotIn("InOutpaint HYBRID", renderer)

    def test_inoutpaint_hybrid_uses_curated_reference_copy(self):
        metadata = json.loads(
            (
                ROOT
                / "workflows"
                / "showcase"
                / "metadata"
                / "CMK 2.5 · Flow · InOutpaint HYBRID.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            metadata["description"],
            "Zeigt InOutpaint, Remove und Extend im HYBRID-Modus – jeweils von der Vorbereitung über den SDXL 1st-Pass bis zum abschließenden ZIT-Pass.",
        )
        self.assertEqual(
            metadata["cmkHighlight"],
            "Die unterschiedlichen Inpaint-Aufgaben werden zentral in 01 START HERE vorbereitet und anschließend durch denselben HYBRID-Prozess geführt.",
        )
        self.assertEqual(
            metadata["description_en"],
            "Shows InOutpaint, Remove, and Extend in HYBRID mode – from preparation through the SDXL 1st pass to the final ZIT pass.",
        )
        self.assertEqual(
            metadata["cmkHighlight_en"],
            "The different inpainting tasks are prepared centrally in 01 START HERE and then run through the same HYBRID process.",
        )

    def test_inoutpaint_family_comparison_omits_cmk_highlight(self):
        metadata = json.loads(
            (
                ROOT
                / "workflows"
                / "showcase"
                / "metadata"
                / "CMK 2.5 · Flow · InOutpaint SDXL · HYBRID · ZIT.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            metadata["description"],
            "Zeigt denselben InOutpaint-Workflow in SDXL, HYBRID und Z-Image Turbo und macht die Unterschiede der drei Generationspfade unmittelbar vergleichbar.",
        )
        self.assertEqual(
            metadata["description_en"],
            "Shows the same InOutpaint workflow in SDXL, HYBRID, and Z-Image Turbo, making the differences between the three generation paths immediately comparable.",
        )
        self.assertEqual(metadata["cmkHighlight"], "")
        self.assertEqual(metadata["cmkHighlight_en"], "")

        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        renderer = source[
            source.index("function renderReferenceBrowser") :
            source.index("function renderBrowser")
        ]
        self.assertIn('highlight.closest("section").hidden = !selected.highlightLabel', renderer)

    def test_zit_inpaint_reference_uses_optional_bilingual_info(self):
        metadata = json.loads(
            (
                ROOT
                / "workflows"
                / "showcase"
                / "metadata"
                / "CMK 2.5 · Flow · Inpaint Z-Image Turbo.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            metadata["description"],
            "Zeigt experimentelles maskiertes Inpainting mit Z-Image Turbo – von der Vorbereitung bis zum ZIT-Sampling.",
        )
        self.assertEqual(
            metadata["description_en"],
            "Shows experimental masked inpainting with Z-Image Turbo—from preparation through ZIT sampling.",
        )
        self.assertEqual(metadata["cmkHighlight"], "")
        self.assertEqual(metadata["cmkHighlight_en"], "")
        self.assertEqual(
            metadata["info"],
            "Z-Image-Turbo-Inpainting befindet sich weiterhin in Entwicklung.",
        )
        self.assertEqual(
            metadata["info_en"],
            "Z-Image Turbo inpainting is still under development.",
        )

        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        renderer = source[
            source.index("function renderReferenceBrowser") :
            source.index("function renderBrowser")
        ]
        self.assertIn('class="cmk-flow-compact-info cmk-showcase-info" hidden', renderer)
        self.assertIn("info.hidden = !selected.infoLabel", renderer)
        endpoint = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('"info": metadata.get("info", "")', endpoint)
        self.assertIn('"info_en": metadata.get("info_en", "")', endpoint)

    def test_multi_preview_reference_labels_follow_compact_grouping_grammar(self):
        metadata_root = ROOT / "workflows" / "showcase" / "metadata"
        expected = {
            "CMK 2.5 · Flow · InOutpaint SDXL · HYBRID · ZIT.json": (
                ["", "", ""], ["SDXL", "HYBRID", "ZIT"]
            ),
            "CMK 2.5 · Flow · InOutpaint SDXL.json": (
                ["InOutpaint"] * 3 + ["Remove"] * 3 + ["Extend"] * 3,
                ["Preparation", "1st-Pass", "2nd-Pass"] * 3,
            ),
            "CMK 2.5 · Flow · Inpaint - FaceRebuild HYBRID.json": (
                [""] * 5,
                ["Preparation", "1st-Pass SDXL", "2nd-Pass ZIT", "FaceRebuild", "Source"],
            ),
            "CMK 2.5 · Flow · Inpaint Z-Image Turbo.json": (
                [""] * 2, ["Preparation", "ZIT-Sampling"]
            ),
            "CMK 2.5 · Flow · InstantID · Hybrid Inpaint SDXL.json": (
                [""] * 5,
                ["Preparation", "1st-Pass SDXL", "Identity", "2nd-Pass SDXL", "Source"],
            ),
            "CMK 2.5 · Flow · Text2Image - InstantID SDXL and FaceRebuild HYBRID · ZIT.json": (
                ["SDXL-HYBRID"] * 4 + ["SDXL ZERO-PASS"] * 3 + ["HYBRID"] * 3 + ["ZIT"] * 2 + [""],
                [
                    "1st-Pass SDXL", "Identity", "2nd-Pass SDXL", "FaceRebuild",
                    "Identity", "2nd-Pass SDXL", "FaceRebuild",
                    "1st-Pass SDXL", "2nd-Pass ZIT", "FaceRebuild",
                    "Sampling ZIT", "FaceRebuild", "Source",
                ],
            ),
            "CMK 2.5 · Flow · Text2Image ControlNet HYBRID.json": (
                [""] * 3, ["ControlNet", "1st-Pass SDXL", "2nd-Pass ZIT"]
            ),
            "CMK 2.5 · Flow · Text2Image ControlNet SDXL · HYBRID · ZIT.json": (
                [""] * 3, ["SDXL", "HYBRID", "ZIT"]
            ),
            "CMK 2.5 · Flow · Text2Image ControlNet Z-Image Turbo.json": (
                [""] * 2, ["ControlNet", "ZIT"]
            ),
            "CMK 2.5 · Flow · Text2Image HYBRID.json": (
                [""] * 2, ["1st-Pass SDXL", "2nd-Pass ZIT"]
            ),
            "CMK 2.5 · Flow · Text2Image SDXL · HYBRID · ZIT.json": (
                [""] * 3, ["SDXL", "HYBRID", "ZIT"]
            ),
            "CMK 2.5 · Flow · Text2Image-ControlNet SDXL.json": (
                [""] * 2, ["ControlNet", "SDXL"]
            ),
            "CMK 2.5 · Flow · Text2Image-FaceRebuild HYBRID.json": (
                [""] * 4, ["1st-Pass SDXL", "2nd-Pass ZIT", "FaceRebuild", "Source"]
            ),
            "CMK 2.5 · Flow · Text2Image-FaceRebuild Z-Image Turbo.json": (
                [""] * 3, ["Sampling", "FaceRebuild", "Source"]
            ),
            "CMK 2.5 · Flow · Text2Image-InstantID SDXL.json": (
                ["HYBRID", "HYBRID", "ZERO-PASS", ""],
                ["1st-Pass SDXL", "Refined", "Refined", "Source"],
            ),
            "CMK 2.5 · Text2Image Identity 1.json": (
                [""] * 3, ["Identity", "FaceRebuild", "FaceSwap"]
            ),
            "CMK 2.5 · Text2Image Identity 3.json": (
                [""] * 3, ["Identity", "FaceRebuild", "FaceSwap"]
            ),
            "CMK 2.5 · Text2Image.json": (
                [""] * 4, ["ControlNet", "1st-Pass SDXL", "2nd-Pass ZIT", "Detailer"]
            ),
        }
        for filename, (groups, labels) in expected.items():
            with self.subTest(filename=filename):
                previews = json.loads(
                    (metadata_root / filename).read_text(encoding="utf-8")
                )["previews"]
                self.assertEqual([preview.get("group", "") for preview in previews], groups)
                self.assertEqual([preview["label"] for preview in previews], labels)
                self.assertEqual([preview["label_en"] for preview in previews], labels)

    def test_grouped_preview_renderer_keeps_source_outside_process_groups(self):
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        renderer = source[
            source.index("function renderNodePreview") : source.index("function flowIcon")
        ]
        self.assertIn("if (grouped && preview.group)", renderer)
        self.assertIn('standalone.className = "cmk-flow-preview-tab-standalone"', renderer)
        self.assertNotIn('label.textContent = currentGroup ||', renderer)

        metadata_root = ROOT / "workflows" / "showcase" / "metadata"
        for filename in self.EXPECTED_SOURCE_PREVIEW_WORKFLOWS:
            previews = json.loads(
                (metadata_root / filename).read_text(encoding="utf-8")
            )["previews"]
            source_preview = previews[-1]
            self.assertEqual(source_preview["label"], "Source")
            self.assertNotIn("group", source_preview)

    def test_module_reference_catalog_keeps_discrete_workflow_before_module(self):
        metadata_root = ROOT / "workflows" / "showcase" / "metadata"
        module_workflows = []
        for path in metadata_root.glob("*.json"):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata["category"] == "MODULE WORKFLOWS":
                module_workflows.append((metadata["order"], path.name))
        self.assertEqual(
            [filename for _, filename in sorted(module_workflows)],
            [
                "CMK 2.5 · FaceProcess.json",
                "CMK 2.5 · FaceProcess · Modul.json",
                "CMK 2.5 · FaceRebuild.json",
                "CMK 2.5 · FaceRebuild · Modul.json",
                "CMK 2.5 · FaceSwap.json",
                "CMK 2.5 · FaceSwap · Modul.json",
                "CMK 2.5 · MaskDetailer .json",
                "CMK 2.5 · MaskDetailer · Modul.json",
                "CMK 2.5 · SmartDetailer.json",
                "CMK 2.5 · SmartDetailer · Modul.json",
            ],
        )

    def test_reference_list_is_grouped_during_rendering_only(self):
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        reference_renderer = source[
            source.index("function renderReferenceBrowser"):
            source.index("function renderBrowser")
        ]
        self.assertIn('header.className = "cmk-reference-group-header"', reference_renderer)
        self.assertIn('groupElement.className = `cmk-reference-group${expanded ? " is-expanded" : ""}`', reference_renderer)
        self.assertIn('header.setAttribute("aria-expanded", String(expanded))', reference_renderer)
        self.assertIn('items.hidden = !expanded', reference_renderer)
        self.assertIn('referenceExpandedCategory = expanded ? null : group.category', reference_renderer)
        self.assertIn('group.references.length', reference_renderer)
        self.assertIn(
            "button.innerHTML = '<span class=\"cmk-flow-item-name\"></span>';",
            reference_renderer,
        )
        self.assertNotIn("cmk-flow-item-category", reference_renderer)

    def test_technical_reference_directory_keeps_only_video_references(self):
        reference_root = ROOT / "workflows" / "reference"
        self.assertEqual(
            {path.name for path in reference_root.glob("*.json")},
            {
                "CMK_FaceSwap_Video_Reference_v2.2.json",
                "CMK_FaceSwap_Video_Project_Template.json",
            },
        )

    def test_all_curated_flow_subgraphs_are_published_with_real_previews(self):
        published = set()
        for path in sorted((ROOT / "subgraphs").glob("*.json")):
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

    def test_flow_showcase_subgraphs_use_prepared_identities(self):
        expected = {
            "CMK Flow · 05 ControlNet Combined.json": "0edf7a0d-3271-4b2e-8344-72d3506bb628",
            "CMK Flow · 05 ControlNet SDXL.json": "96aabb3d-a0c9-4539-86c0-2b2a7abb0712",
            "CMK Flow · 05 ControlNet ZIT.json": "aeec8af2-85bf-4969-8743-9b39bb5483ef",
            "CMK Flow · 10 KSampler SDXL 1st Pass.json": "e371220c-caba-4cfd-ae2b-81e59664d525",
            "CMK Flow · 10 KSampler Z-Image Turbo.json": "92cbb212-4e34-4b9a-8aa7-903d96ce39f2",
            "CMK Flow · 15 InstantID-Sampler SDXL.json": "1c91adfd-df9f-4729-90a1-35dc53915ca6",
            "CMK Flow · 20 Refiner SDXL.json": "79b164c2-6919-4e9f-81aa-c14469d1715f",
            "LoRA Stack · SDXL.json": "29d596c9-48b8-4cf5-aa02-e1c50d125712",
            "LoRA Stack · ZIT.json": "ee158dd0-99a8-40b2-8ea8-142b79368125",
            "LoRA Stack · combined.json": "57da8c95-878c-4e93-a016-5da4c700e9ca",
        }
        for filename, identity in expected.items():
            with self.subTest(filename=filename):
                document = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )
                definition = document["definitions"]["subgraphs"][0]
                self.assertEqual(definition["id"], identity)
                self.assertEqual(document["nodes"][0]["type"], identity)

    def test_flow_showcase_uses_all_51_prepared_previews(self):
        preview_paths = []
        for path in (ROOT / "subgraphs").glob("*.json"):
            metadata = json.loads(path.read_text(encoding="utf-8")).get("extra", {}).get("CMKFlow", {})
            if not metadata.get("published"):
                continue
            previews = metadata.get("previews", [])
            self.assertEqual([preview["label"] for preview in previews], ["Modul", "Aufbau"], path.name)
            preview_paths.extend(preview["src"] for preview in previews)
        node_metadata = json.loads(
            (ROOT / "web" / "flow_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]
        preview_paths.extend(
            preview["src"]
            for entry in node_metadata.values()
            for preview in entry.get("previews", [])
        )
        self.assertEqual(len(preview_paths), 51)
        self.assertEqual(len(set(preview_paths)), 51)
        for preview_path in preview_paths:
            self.assertTrue((ROOT / "web" / preview_path).is_file(), preview_path)

    def test_flow_variants_have_explicit_grouping_and_order(self):
        subgraph_groups = (
            (
                "LoRA Stack · SDXL.json",
                (("LoRA Stack · SDXL.json", "SDXL", 10), ("LoRA Stack · ZIT.json", "Z-Image Turbo", 20), ("LoRA Stack · combined.json", "Combined", 30)),
            ),
            (
                "CMK Flow · 05 ControlNet SDXL.json",
                (("CMK Flow · 05 ControlNet SDXL.json", "SDXL", 10), ("CMK Flow · 05 ControlNet ZIT.json", "Z-Image Turbo", 20), ("CMK Flow · 05 ControlNet Combined.json", "Combined", 30)),
            ),
            (
                "CMK Flow · 10 KSampler SDXL 1st Pass.json",
                (("CMK Flow · 10 KSampler SDXL 1st Pass.json", "SDXL 1st Pass", 10), ("CMK Flow · 10 KSampler Z-Image Turbo.json", "Z-Image Turbo", 20)),
            ),
        )
        for primary_filename, members in subgraph_groups:
            primary = json.loads((ROOT / "subgraphs" / primary_filename).read_text(encoding="utf-8"))
            primary_id = primary["definitions"]["subgraphs"][0]["id"]
            for filename, label, order in members:
                with self.subTest(filename=filename):
                    document = json.loads((ROOT / "subgraphs" / filename).read_text(encoding="utf-8"))
                    metadata = document["extra"]["CMKFlow"]
                    self.assertEqual(metadata["variantLabel"], label)
                    self.assertEqual(metadata["variantOrder"], order)
                    if filename == primary_filename:
                        self.assertNotIn("variantOf", metadata)
                    else:
                        self.assertEqual(metadata["variantOf"], primary_id)

        metadata = json.loads(
            (ROOT / "web" / "flow_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]
        for node_type, variant_of, label, order in (
            ("CMKImageLoadAndResizePipe", None, "For Detailing", 10),
            ("CMKLoadImage", "CMKImageLoadAndResizePipe", "For Inpaint", 20),
            ("CMKCheckpointVAELoaderPipe", "CMKImageLoadAndResizePipe", "Checkpoint & VAE", 30),
            ("CMKSwapImageLoaderPipe", "CMKImageLoadAndResizePipe", "FaceSwap Image Input", 40),
            ("CMKImageFileLoader", "CMKImageLoadAndResizePipe", "Image File Loader", 50),
            ("CMKPostProcessBoundarySDXLPipe", None, "SDXL", 10),
            ("CMKPostProcessBoundaryZITPipe", "CMKPostProcessBoundarySDXLPipe", "Z-Image Turbo", 20),
            ("CMKFamilyResultMergePipe", "CMKPostProcessBoundarySDXLPipe", "Combined", 30),
        ):
            with self.subTest(node_type=node_type):
                self.assertEqual(metadata[node_type]["variantLabel"], label)
                self.assertEqual(metadata[node_type]["variantOrder"], order)
                if node_type in {
                    "CMKPostProcessBoundarySDXLPipe",
                    "CMKPostProcessBoundaryZITPipe",
                    "CMKFamilyResultMergePipe",
                }:
                    self.assertEqual(metadata[node_type]["status"], "STABLE")
                if variant_of:
                    self.assertEqual(metadata[node_type]["variantOf"], variant_of)
                else:
                    self.assertNotIn("variantOf", metadata[node_type])

        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(encoding="utf-8")
        self.assertIn("a.variantOrder - b.variantOrder", source)

    def test_flow_catalog_uses_requested_browser_order(self):
        node_metadata = json.loads(
            (ROOT / "web" / "flow_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]
        expected_nodes = {
            "CMKVisualizer": 10,
            "CMKImageLoadAndResizePipe": 20,
            "CMKPipeCreateImage": 40,
            "CMKRegionalConditioningSDXL": 50,
            "CMKPostProcessBoundarySDXLPipe": 100,
        }
        for node_type, order in expected_nodes.items():
            self.assertEqual(node_metadata[node_type]["browserOrder"], order)

        expected_subgraphs = {
            "LoRA Stack · SDXL.json": 30,
            "CMK Flow · 05 ControlNet SDXL.json": 60,
            "CMK Flow · 10 KSampler SDXL 1st Pass.json": 70,
            "CMK Flow · 15 InstantID-Sampler SDXL.json": 80,
            "CMK Flow · 20 Refiner SDXL.json": 90,
            "CMK Flow · FaceRebuild SDXL.json": 110,
            "CMK Flow · FaceSwap.json": 120,
            "CMK Flow · FaceProcess SDXL.json": 130,
            "CMK Flow · Detailer SDXL.json": 140,
            "CMK Flow · MaskDetailer SDXL.json": 150,
            "CMK Flow · Upscale & Save.json": 160,
        }
        for filename, order in expected_subgraphs.items():
            document = json.loads((ROOT / "subgraphs" / filename).read_text(encoding="utf-8"))
            self.assertEqual(document["extra"]["CMKFlow"]["browserOrder"], order)

        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(encoding="utf-8")
        self.assertIn("a.browserOrder - b.browserOrder", source)

    def test_faceswap_flows_expose_both_families_and_native_engine(self):
        expected = {"SDXL", "Z-Image Turbo", "CMK Native FaceSwap"}
        for filename in (
            "CMK Flow · FaceSwap.json",
            "CMK Flow · FaceSwap · Advanced.json",
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
            'String(path).split("/").map(encodeURIComponent).join("/")', source
        )

    def test_extension_assets_follow_the_installed_directory_name(self):
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('new URL("../", import.meta.url)', source)
        self.assertIn("const NODE_PACK = `custom_nodes.${EXTENSION_DIRECTORY}`", source)
        self.assertIn("extensionAssetUrl(preview.src, PREVIEW_CACHE_VERSION)", source)
        self.assertNotIn("/extensions/cmk_nodes/", source)

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

    def test_catalog_order_is_explicit_and_not_parsed_from_visible_names(self):
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        metadata = json.loads(
            (ROOT / "web" / "flow_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]
        self.assertEqual(1, metadata["CMKPipeCreateImage"]["order"])
        self.assertEqual(2, metadata["CMKRegionalConditioningSDXL"]["order"])
        for node_type in (
            "CMKFamilyResultMergePipe",
            "CMKPostProcessBoundarySDXLPipe",
            "CMKPostProcessBoundaryZITPipe",
        ):
            self.assertNotIn("order", metadata[node_type])
            self.assertEqual("postprocess", metadata[node_type]["catalogGroup"])
        self.assertNotIn("order", metadata["CMKVisualizer"])
        self.assertEqual("terminal", metadata["CMKVisualizer"]["catalogGroup"])
        self.assertIn("order: catalogOrder(metadata, categoryOrder)", source)
        self.assertNotIn("displayName.match", source)
        self.assertIn("String(a.identity).localeCompare(String(b.identity))", source)
        self.assertNotIn("a.name.localeCompare", source)
        self.assertIn("CMKImageLoadAndResizePipe: 101", source)
        self.assertIn("CMKLoadImage: 102", source)
        self.assertIn("CMKCheckpointVAELoaderPipe: 103", source)

    def test_catalog_identity_and_variants_use_technical_ids(self):
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        metadata = json.loads(
            (ROOT / "web" / "flow_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]
        self.assertIn("definitions.find((item) => item.id === rootType)", source)
        self.assertIn("englishContent.flows?.[blueprintId]", source)
        self.assertIn("variant.variantOf === flow.identity", source)
        self.assertEqual(
            "CMKImageLoadAndResizePipe",
            metadata["CMKLoadImage"]["variantOf"],
        )
        self.assertEqual(
            "CMKImageLoadAndResizePipe",
            metadata["CMKCheckpointVAELoaderPipe"]["variantOf"],
        )
        for node_type in (
            "CMKZITControlNetPreparePipe",
            "CMKCombinedControlNetPreparePipe",
        ):
            self.assertEqual("CMKControlNetPreparePipe", metadata[node_type]["variantOf"])
        for node_type in ("CMKSwapImageLoaderPipe", "CMKImageFileLoader"):
            self.assertEqual("CMKImageLoadAndResizePipe", metadata[node_type]["variantOf"])
        self.assertEqual(
            "CMKPostProcessBoundarySDXLPipe",
            metadata["CMKPostProcessBoundaryZITPipe"]["variantOf"],
        )
        self.assertEqual(
            "CMKPostProcessBoundarySDXLPipe",
            metadata["CMKFamilyResultMergePipe"]["variantOf"],
        )

    def test_english_subgraph_content_is_keyed_by_blueprint_id(self):
        english = json.loads(
            (ROOT / "web" / "browser_content_en.json").read_text(encoding="utf-8")
        )["flows"]
        expected_ids = {
            "e371220c-caba-4cfd-ae2b-81e59664d525",
            "92cbb212-4e34-4b9a-8aa7-903d96ce39f2",
            "1c91adfd-df9f-4729-90a1-35dc53915ca6",
            "79b164c2-6919-4e9f-81aa-c14469d1715f",
            "c1b3b220-bc27-479a-ae12-a43e226d96e3",
            "761fe2bd-e305-4adf-898d-54f87d99fd57",
            "0a3f7a10-a21c-49ed-9faa-7e55e7b7a4dd",
            "cec036fc-09d6-4f44-b564-4692ea7051a0",
            "9993a5f9-7cd5-431c-8653-6e187ef9d214",
            "062af015-5929-4d94-86bd-fd060f67ee53",
            "6f8d63a4-7ea5-4c18-9900-2ec3ed33c9b6",
        }
        self.assertTrue(expected_ids.issubset(english))
        self.assertFalse(any(key.startswith("CMK Flow · 10") for key in english))

    def test_every_visible_flow_entry_has_complete_parallel_english_copy(self):
        english = json.loads(
            (ROOT / "web" / "browser_content_en.json").read_text(encoding="utf-8")
        )["flows"]
        node_metadata = json.loads(
            (ROOT / "web" / "flow_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]
        visible_node_ids = {
            "CMKVisualizer",
            "CMKImageLoadAndResizePipe",
            "CMKLoadImage",
            "CMKCheckpointVAELoaderPipe",
            "CMKSwapImageLoaderPipe",
            "CMKImageFileLoader",
            "CMKPipeCreateImage",
            "CMKRegionalConditioningSDXL",
            "CMKPostProcessBoundarySDXLPipe",
            "CMKPostProcessBoundaryZITPipe",
            "CMKFamilyResultMergePipe",
        }
        sources = [(identity, node_metadata[identity]) for identity in visible_node_ids]

        published_subgraph_count = 0
        for path in (ROOT / "subgraphs").glob("*.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            metadata = document.get("extra", {}).get("CMKFlow", {})
            if not metadata.get("published"):
                continue
            identity = document["definitions"]["subgraphs"][0]["id"]
            sources.append((identity, metadata))
            published_subgraph_count += 1

        self.assertEqual(len(visible_node_ids), 11)
        self.assertEqual(published_subgraph_count, 20)
        self.assertEqual(len(sources), 31)
        for identity, german in sources:
            with self.subTest(identity=identity):
                self.assertIn(identity, english)
                localized = english[identity]
                self.assertTrue(localized.get("description"))
                self.assertTrue(localized.get("previewAlt"))
                self.assertIsInstance(localized.get("features"), list)
                self.assertEqual(len(localized["features"]), len(german["features"]))
                self.assertTrue(all(isinstance(item, str) and item for item in localized["features"]))
                self.assertNotEqual(localized["description"], german["description"])
                self.assertEqual(bool(localized.get("info")), bool(german.get("info")))
                if german.get("info"):
                    self.assertNotEqual(localized["info"], german["info"])

    def test_english_flow_copy_matches_current_editorial_scope(self):
        english = json.loads(
            (ROOT / "web" / "browser_content_en.json").read_text(encoding="utf-8")
        )["flows"]
        self.assertEqual(
            english["761fe2bd-e305-4adf-898d-54f87d99fd57"]["description"],
            "Independently detects and segments up to three relevant image areas and refines each one selectively.",
        )
        self.assertEqual(
            english["CMKVisualizer"]["features"],
            ["Display all registered processing stages in one place and optionally upscale and save the final image"],
        )
        self.assertEqual(
            english["92cbb212-4e34-4b9a-8aa7-903d96ce39f2"]["info"],
            "Z-Image Turbo inpainting is still under development.",
        )
        self.assertEqual(
            english["5e25c315-1e30-4fca-af11-db30118b807e"]["info"],
            "When processing starts, the current image is frozen. Further runs use this image until the input changes.",
        )

    def test_recommendation_metadata_keeps_stable_target_ids(self):
        recommendation_count = 0
        metadata_documents = [
            json.loads((ROOT / "web" / "flow_node_metadata.json").read_text(encoding="utf-8"))["nodes"],
        ]
        metadata_documents.extend(
            {
                path.name: json.loads(path.read_text(encoding="utf-8"))
                .get("extra", {})
                .get("CMKFlow", {})
            }
            for path in (ROOT / "subgraphs").glob("*.json")
        )
        for document in metadata_documents:
            for owner, entry in document.items():
                for field in ("recommendedBefore", "recommendedAfter"):
                    for recommendation in entry.get(field, []):
                        if recommendation == "fertige Bildbearbeitung":
                            continue
                        recommendation_count += 1
                        self.assertIsInstance(recommendation, dict, owner)
                        self.assertTrue(recommendation.get("label"), owner)
                        self.assertTrue(recommendation.get("targetId"), owner)
        self.assertGreater(recommendation_count, 150)

        stale_ids = {
            "01f48a49-15c3-43af-be4f-fdf0def34a7a",
            "b9c2355a-8515-430b-81a9-c321f9259049",
            "bf61b724-f289-459a-a557-269b77bdb1d7",
            "026a9a6c-cd11-4c42-8ada-7e12affcb845",
            "9a5b69d3-fb32-4fa2-8d33-f5f25f020dbf",
            "c4eac662-274e-44a7-be11-424fd52cd8b5",
            "43c90cd1-37d3-450b-b6f3-5c0a17a3fe8b",
            "e8686dd2-3662-4879-a1b4-d4618f647050",
            "c3f78046-9abe-43dd-bf26-6b1c15613f2a",
        }
        serialized_metadata = json.dumps(metadata_documents, sort_keys=True)
        self.assertTrue(
            all(stale_id not in serialized_metadata for stale_id in stale_ids)
        )

    def test_postprocess_catalog_uses_groups_instead_of_historic_numbers(self):
        for path in (ROOT / "subgraphs").glob("CMK Flow · *.json"):
            metadata = json.loads(path.read_text(encoding="utf-8")).get("extra", {}).get("CMKFlow", {})
            if metadata.get("catalogGroup") not in {"postprocess", "terminal"}:
                continue
            self.assertNotIn("order", metadata, path.name)
            self.assertNotRegex(metadata.get("displayName", ""), r"^(23|25|30|40|90|95|100)\b")

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
            '"CMKInstantIDFaceRebuildSDXL": "CMK Flow ·', mappings
        )
        for filename in (
            "CMK Flow · FaceRebuild SDXL.json",
            "CMK Flow · FaceRebuild SDXL · Advanced.json",
        ):
            document = json.loads((ROOT / "subgraphs" / filename).read_text(encoding="utf-8"))
            self.assertEqual(document["definitions"]["subgraphs"][0]["name"], filename[:-5])
            self.assertNotIn("order", document["extra"]["CMKFlow"])
            self.assertEqual(
                "postprocess", document["extra"]["CMKFlow"]["catalogGroup"]
            )

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

    def test_flow_insertion_uses_visible_canvas_center_not_stale_mouse(self):
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("function visibleCanvasInsertionPosition(canvas)", source)
        self.assertIn("const position = visibleCanvasInsertionPosition(canvas)", source)
        self.assertNotIn("const position = Array.isArray(mouse)", source)

    def test_browser_refreshes_updated_package_content(self):
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('api.fetchApi(path, { cache: "no-store" })', source)
        self.assertIn('const PREVIEW_CACHE_VERSION = "20261002-cmk25-flow-showcase"', source)
        open_browser = source[source.index("async function openFlowBrowser()") :]
        self.assertIn("browserDataPromise = undefined", "\n".join(open_browser.split("\n", 8)[0:8]))

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
            "CMKControlNetPreparePipe",
        )
        self.assertEqual(
            metadata["CMKCombinedControlNetPreparePipe"]["variantOf"],
            "CMKControlNetPreparePipe",
        )
        self.assertEqual(
            metadata["CMKImageLoadAndResizePipe"]["displayName"],
            "Loader",
        )
        self.assertEqual(
            metadata["CMKLoadImage"]["variantOf"], "CMKImageLoadAndResizePipe"
        )
        self.assertEqual(
            metadata["CMKCheckpointVAELoaderPipe"]["variantOf"],
            "CMKImageLoadAndResizePipe",
        )
        for filename, display_name, family in (
            ("LoRA Stack · SDXL.json", "LoRA Stack · SDXL", "SDXL"),
            ("LoRA Stack · ZIT.json", "LoRA Stack · ZIT", "Z-Image Turbo"),
        ):
            lora_stack = json.loads(
                (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
            )["extra"]["CMKFlow"]
            self.assertEqual(lora_stack["displayName"], display_name)
            self.assertEqual(lora_stack["compatibility"], [family])
        expected_lora_copy = {
            "LoRA Stack · SDXL.json": (
                "Bündelt SDXL-kompatible LoRAs und ihre Triggerwörter für den SDXL-Generationspfad.",
                [
                    "Nimmt ausschließlich SDXL-kompatible LoRAs auf",
                    "Bündelt die zugehörigen Triggerwörter",
                    "Stellt den LoRA Stack für den SDXL-Generationspfad bereit",
                ],
            ),
            "LoRA Stack · ZIT.json": (
                "Bündelt Z-Image-Turbo-kompatible LoRAs und ihre Triggerwörter für den Z-Image-Turbo-Generationspfad.",
                [
                    "Nimmt ausschließlich Z-Image-Turbo-kompatible LoRAs auf",
                    "Bündelt die zugehörigen Triggerwörter",
                    "Stellt den LoRA Stack für den Z-Image-Turbo-Generationspfad bereit",
                ],
            ),
            "LoRA Stack · combined.json": (
                "Bündelt SDXL- und Z-Image-Turbo-kompatible LoRAs und ihre Triggerwörter für beide Generationspfade.",
                [
                    "Nimmt SDXL- und Z-Image-Turbo-kompatible LoRAs auf",
                    "Bündelt die Triggerwörter getrennt für beide Modellfamilien",
                    "Stellt getrennte LoRA Stacks für beide Generationspfade bereit",
                ],
            ),
        }
        for filename, (description, features) in expected_lora_copy.items():
            lora_stack = json.loads(
                (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
            )["extra"]["CMKFlow"]
            self.assertEqual(lora_stack["description"], description)
            self.assertEqual(lora_stack["features"], features)
        source = (ROOT / "web" / "js" / "cmk_flow_browser.js").read_text(encoding="utf-8")
        self.assertIn("const allFlows = [...discovered, ...discoverCuratedNodes", source)
        self.assertIn('nodeType === "CMKVisualizer"', source)
        self.assertIn("nodeMetadata[nodeType]?.flowPublished === true", source)
        self.assertTrue(metadata["CMKImageLoadAndResizePipe"]["flowPublished"])
        self.assertTrue(metadata["CMKSwapImageLoaderPipe"]["flowPublished"])
        self.assertEqual(metadata["CMKVisualizer"]["category"], "Flow")
        self.assertEqual(metadata["CMKVisualizer"]["displayName"], "Visualizer")
        self.assertEqual(
            metadata["CMKPipeCreateImage"]["description"],
            "Zentraler Einstieg für Text2Image und Inpaint. Legt Aufgabe, Bild, Maske und Prompts fest und bereitet den vollständigen Prozess für den weiteren Flow vor.",
        )
        self.assertEqual(
            metadata["CMKPipeCreateImage"]["features"],
            [
                "Text2Image oder Inpaint als Prozess festlegen",
                "Inpaint-Aufgaben wie Replace, Remove, Extend und Custom vorbereiten",
                "Bild, Maske und Prompts zentral für den nachfolgenden Flow bereitstellen",
            ],
        )
        self.assertEqual(
            metadata["CMKRegionalConditioningSDXL"]["description"],
            "Ergänzt das globale SDXL-Conditioning um bis zu drei räumlich begrenzte Promptbereiche.",
        )
        self.assertEqual(
            metadata["CMKRegionalConditioningSDXL"]["features"],
            [
                "Bis zu drei unabhängig steuerbare Regionen verwenden",
                "Standard-Presets für Position und Wirkungsbereich nutzen",
                "Position, Stärke und Wirkungsdauer jeder Region individuell festlegen",
            ],
        )
        self.assertEqual(
            metadata["CMKVisualizer"]["features"],
            [
                "Zeigt die registrierten Bearbeitungsstufen zentral an und kann das finale Bild optional hochskalieren und speichern.",
            ],
        )
        self.assertTrue(metadata["CMKVisualizer"]["hideRecommendations"])
        self.assertEqual(
            metadata["CMKPostProcessBoundarySDXLPipe"]["displayName"],
            "PostProcess Boundary",
        )
        expected_boundary_copy = {
            "CMKPostProcessBoundarySDXLPipe": (
                "Beendet einen reinen SDXL-Generationspfad und stellt einen unabhängigen Arbeitskontext für nachfolgende PostProcess-Module bereit.",
                [
                    "Übergibt das erzeugte Bild an den PostProcess",
                    "Gibt das SDXL-Generationsmodell frei",
                    "Lädt ein separates PostProcess-Modell nur bei Bedarf",
                ],
            ),
            "CMKPostProcessBoundaryZITPipe": (
                "Beendet einen reinen Z-Image-Turbo-Generationspfad und stellt einen unabhängigen Arbeitskontext für nachfolgende PostProcess-Module bereit.",
                [
                    "Übergibt das erzeugte Bild an den PostProcess",
                    "Gibt das ZIT-Generationsmodell frei",
                    "Lädt ein separates PostProcess-Modell nur bei Bedarf",
                ],
            ),
            "CMKFamilyResultMergePipe": (
                "Beendet den aktiven SDXL-, ZIT- oder HYBRID-Generationspfad und stellt einen unabhängigen Arbeitskontext für nachfolgende PostProcess-Module bereit.",
                [
                    "Übernimmt ausschließlich den aktiven Generationszweig",
                    "Übergibt dessen Ergebnis an den PostProcess",
                    "Lädt ein separates PostProcess-Modell nur bei Bedarf",
                ],
            ),
        }
        for node_type, (description, features) in expected_boundary_copy.items():
            self.assertEqual(metadata[node_type]["description"], description)
            self.assertEqual(metadata[node_type]["features"], features)
        flow_renderer = source[
            source.index("function renderFlowBrowser") :
            source.index("function renderToolboxBrowser")
        ]
        self.assertIn('class="cmk-flow-compact-top"', flow_renderer)
        self.assertIn('class="cmk-flow-compact-features"', flow_renderer)
        self.assertIn('class="cmk-flow-compact-info" hidden', flow_renderer)
        self.assertIn(".cmk-flow-compact-info[hidden] { display: none; }", source)
        self.assertIn("selected.features.slice(0, 3)", flow_renderer)
        self.assertIn("if (selected.info)", flow_renderer)
        self.assertIn(
            ".cmk-flow-compact-top.has-variants { grid-template-columns: minmax(260px, 1fr) minmax(300px, 1.2fr); }",
            source,
        )
        self.assertIn(
            ".cmk-flow-compact-top, .cmk-flow-compact-top.has-variants { grid-template-columns: 1fr; }",
            source,
        )
        self.assertNotIn('class="cmk-flow-meta"', flow_renderer)
        self.assertNotIn('class="cmk-flow-interface"', flow_renderer)
        self.assertNotIn('class="cmk-flow-sequence"', flow_renderer)
        self.assertNotIn("cmk-flow-placement", flow_renderer)
        self.assertIn("button.textContent = variant.variantLabel", source)

    def test_controlnet_reference_copy_is_compact_and_user_facing(self):
        expected_features = [
            "Struktur und Bildaufbau eines Referenzbildes auf die Generierung übertragen",
            "Stärke und Wirkungsdauer der Referenzführung festlegen",
            "Aufbereitete ControlNet-Referenz im Visualizer sichtbar machen",
        ]
        expected_descriptions = {
            "CMK Flow · 05 ControlNet SDXL.json":
                "Führt die Bildgenerierung anhand eines Referenzbildes und überträgt dessen räumliche oder strukturelle Vorgaben auf das neue Bild.",
            "CMK Flow · 05 ControlNet ZIT.json":
                "Führt die Bildgenerierung anhand eines Referenzbildes und überträgt dessen räumliche oder strukturelle Vorgaben auf das neue Bild.",
            "CMK Flow · 05 ControlNet Combined.json":
                "Führt die Bildgenerierung anhand eines Referenzbildes und überträgt dessen räumliche oder strukturelle Vorgaben auf das neue Bild.",
        }
        for filename, expected_description in expected_descriptions.items():
            metadata = json.loads(
                (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
            )["extra"]["CMKFlow"]
            self.assertEqual(metadata["features"], expected_features)
            self.assertEqual(metadata["description"], expected_description)
            self.assertEqual(
                metadata["info"],
                "Die Referenzdimension bestimmt die Ausgabeauflösung",
            )
            self.assertLessEqual(len(metadata["features"]), 3)
            self.assertEqual(len(metadata["description"].splitlines()), 1)

    def test_ksampler_copy_is_compact_and_variant_specific(self):
        expected = {
            "CMK Flow · 10 KSampler SDXL 1st Pass.json": {
                "description": "Erzeugt den ersten SDXL-Bilddurchlauf und übernimmt dabei Text2Image sowie die vorbereiteten Inpaint-Aufgaben.",
                "features": [
                    "Ersten SDXL-Bilddurchlauf erzeugen",
                    "Text2Image und vorbereitete Inpaint-Aufgaben ausführen",
                    "Stärke und Einfluss des ersten Durchlaufs steuern",
                ],
                "info": None,
            },
            "CMK Flow · 10 KSampler Z-Image Turbo.json": {
                "description": "Erzeugt Bilder direkt mit Z-Image Turbo und unterstützt zusätzlich experimentelles maskiertes Inpainting.",
                "features": [
                    "Text2Image mit Z-Image Turbo erzeugen",
                    "Schnelles 8-Schritt-Turbo-Sampling verwenden",
                    "Experimentelles maskiertes Inpainting ausführen",
                ],
                "info": "Z-Image-Turbo-Inpainting befindet sich weiterhin in Entwicklung.",
            },
        }
        for filename, expected_metadata in expected.items():
            metadata = json.loads(
                (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
            )["extra"]["CMKFlow"]
            self.assertEqual(metadata["description"], expected_metadata["description"])
            self.assertEqual(metadata["features"], expected_metadata["features"])
            self.assertEqual(metadata.get("info"), expected_metadata["info"])

    def test_instantid_sampler_copy_is_user_facing(self):
        metadata = json.loads(
            (
                ROOT
                / "subgraphs"
                / "CMK Flow · 15 InstantID-Sampler SDXL.json"
            ).read_text(encoding="utf-8")
        )["extra"]["CMKFlow"]
        self.assertEqual(
            metadata["description"],
            "Überträgt die Identität eines Referenzgesichts auf die erzeugte Person und führt den begonnenen SDXL-Bildaufbau unter Identity-Guidance weiter.",
        )
        self.assertEqual(
            metadata["features"],
            [
                "Referenzgesicht als Identitätsvorgabe verwenden",
                "Identität und Pose getrennt gewichten",
                "begonnenen SDXL-Bildaufbau mit InstantID fortführen",
            ],
        )
        self.assertEqual(
            metadata["info"],
            "Für InstantID wird ein Referenzgesicht benötigt.",
        )

    def test_refiner_copy_is_user_facing(self):
        metadata = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · 20 Refiner SDXL.json").read_text(
                encoding="utf-8"
            )
        )["extra"]["CMKFlow"]
        self.assertEqual(
            metadata["features"],
            [
                "SDXL-Ergebnis in einem zweiten Durchlauf verfeinern",
                "Prompt und Sampling-Einstellungen aus dem ersten Durchlauf übernehmen",
                "Feinabstimmung unabhängig von der LoRA-Vererbung ermöglichen",
            ],
        )

    def test_facerebuild_copy_is_variant_specific(self):
        expected = {
            "CMK Flow · FaceRebuild SDXL.json": {
                "description": "Rekonstruiert ein ausgewähltes Zielgesicht neu und setzt den bearbeiteten Bereich gezielt in das ansonsten unveränderte Bild zurück.",
                "features": [
                    "Ein gewünschtes Zielgesicht vollständig neu rekonstruieren",
                    "Die Rekonstruktion auf den Gesichtsbereich begrenzen",
                    "Den bearbeiteten Bereich gezielt in das Ausgangsbild zurückführen",
                ],
                "info": "Den Halsbereich bei Bedarf in die Rekonstruktion einbeziehen",
            },
            "CMK Flow · FaceRebuild SDXL · Advanced.json": {
                "description": "Rekonstruiert bis zu drei Zielgesichter unabhängig voneinander neu und setzt die jeweiligen Bereiche gezielt in das unveränderte Bild zurück.",
                "features": [
                    "Bis zu drei Gesichter separat neu rekonstruieren",
                    "Für jedes Gesicht eigene Einstellungen verwenden",
                    "Die rekonstruierten Bereiche unabhängig in das Ausgangsbild zurückführen",
                ],
                "info": "Die Halsbereiche bei Bedarf in die Rekonstruktion einbeziehen",
            },
        }
        for filename, expected_metadata in expected.items():
            metadata = json.loads(
                (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
            )["extra"]["CMKFlow"]
            self.assertEqual(metadata["description"], expected_metadata["description"])
            self.assertEqual(metadata["features"], expected_metadata["features"])
            self.assertEqual(metadata["info"], expected_metadata["info"])

    def test_faceswap_copy_is_variant_specific(self):
        expected = {
            "CMK Flow · FaceSwap.json": {
                "description": "Ersetzt ein ausgewähltes Gesicht im Zielbild durch die Identität eines Referenzgesichts.",
                "features": [
                    "Ein Zielgesicht durch ein Referenzgesicht ersetzen",
                    "Gesichtsausrichtung und Einpassung automatisch an das Zielbild anpassen",
                    "Das übrige Bild unverändert lassen",
                ],
            },
            "CMK Flow · FaceSwap · Advanced.json": {
                "description": "Führt bis zu drei FaceSwaps unabhängig voneinander in einem Durchlauf aus.",
                "features": [
                    "Bis zu drei Zielgesichter separat ersetzen",
                    "Für jeden Swap eine eigene Referenz verwenden",
                    "Mehrere Gesichter unabhängig in einem einzigen Durchlauf bearbeiten",
                ],
            },
        }
        for filename, expected_metadata in expected.items():
            metadata = json.loads(
                (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
            )["extra"]["CMKFlow"]
            self.assertEqual(metadata["description"], expected_metadata["description"])
            self.assertEqual(metadata["features"], expected_metadata["features"])

    def test_faceprocess_copy_exposes_detailer_and_restore_modes(self):
        expected = {
            "CMK Flow · FaceProcess SDXL.json": (
                "Bearbeitet wahlweise ein ausgewähltes oder alle erkannten Gesichter im Detailer- oder Restore-Modus.",
                [
                    "Ein ausgewähltes oder alle erkannten Gesichter bearbeiten",
                    "Zwischen Detailer und Restore wählen",
                    "Gesichtsbearbeitung lokal ausführen, ohne das übrige Bild neu zu erzeugen",
                ],
            ),
            "CMK Flow · FaceProcess SDXL · Advanced.json": (
                "Bearbeitet bis zu drei ausgewählte Gesichter unabhängig voneinander oder alle erkannten Gesichter im Detailer- oder Restore-Modus.",
                [
                    "Bis zu drei ausgewählte oder alle erkannten Gesichter bearbeiten",
                    "Zwischen Detailer und Restore wählen",
                    "Gesichtsbearbeitung lokal ausführen, ohne das übrige Bild neu zu erzeugen",
                ],
            ),
        }
        for filename, (description, features) in expected.items():
            metadata = json.loads(
                (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
            )["extra"]["CMKFlow"]
            self.assertEqual(metadata["description"], description)
            self.assertEqual(metadata["features"], features)

    def test_detailer_copy_is_variant_specific(self):
        expected = {
            "CMK Flow · Detailer SDXL.json": (
                "Erkennt und segmentiert einen relevanten Bildbereich automatisch anhand der gewählten Modelle und bearbeitet ihn gezielt nach.",
                [
                    "Relevanten Bildbereich automatisch erkennen und segmentieren",
                    "Erkannten Bereich gezielt nachbearbeiten",
                    "Erkennungs- und Segmentierungsmodelle für die Auswahl des Bereichs nutzen",
                ],
            ),
            "CMK Flow · Detailer SDXL · Advanced.json": (
                "Erkennt und segmentiert bis zu drei relevante Bildbereiche unabhängig voneinander und bearbeitet sie gezielt nach.",
                [
                    "Bis zu drei Bildbereiche separat erkennen und segmentieren",
                    "Jeden Bereich unabhängig nachbearbeiten",
                    "Für die Bereiche jeweils eigene Einstellungen verwenden",
                ],
            ),
        }
        for filename, (description, features) in expected.items():
            metadata = json.loads(
                (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
            )["extra"]["CMKFlow"]
            self.assertEqual(metadata["description"], description)
            self.assertEqual(metadata["features"], features)

    def test_mask_detailer_copy_explains_frozen_source(self):
        metadata = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · MaskDetailer SDXL.json").read_text(
                encoding="utf-8"
            )
        )["extra"]["CMKFlow"]
        self.assertEqual(
            metadata["description"],
            "Bearbeitet einen manuell maskierten Bildbereich gezielt und wiederholbar, ohne das übrige Bild neu zu erzeugen.",
        )
        self.assertEqual(
            metadata["features"],
            [
                "Bildbereich manuell per Maske festlegen",
                "Eingefrorenes Ausgangsbild wiederholt mit veränderter Maske bearbeiten",
                "Bearbeitung automatisch an einen neuen Upstream-Zustand binden",
            ],
        )
        self.assertEqual(
            metadata["info"],
            "Beim Start wird das aktuelle Bild eingefroren. Weitere Durchläufe verwenden dieses Bild, bis sich der Eingang ändert.",
        )

    def test_upscale_and_save_copy_is_compact(self):
        metadata = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · Upscale & Save.json").read_text(
                encoding="utf-8"
            )
        )["extra"]["CMKFlow"]
        self.assertEqual(
            metadata["description"],
            "Speichert den aktuellen Bearbeitungsstand.",
        )
        self.assertEqual(metadata["features"], ["Aktuelles Bild speichern"])
        self.assertEqual(metadata["info"], "Upscaling ist optional.")


if __name__ == "__main__":
    unittest.main()
