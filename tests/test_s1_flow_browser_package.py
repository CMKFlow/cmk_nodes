"""Flow Browser catalog, Showcase registry and shipped image references (2.7, Detailer 2.8)."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"
TESTS = Path(__file__).resolve().parent
PACKAGE = TESTS.parent
WEB = PACKAGE / "web"
SHOWCASE = PACKAGE / "workflows" / "showcase"
REFERENCES = PACKAGE / "assets" / "references"

VISIBLE_NODES = {
    "CMKVisualizer", "CMKImageLoadAndResizePipe", "CMKLoadImage", "CMKImageFileLoader", "CMKMultiImageLoader",
    "CMKPipeCreateImage", "CMKRegionalLayoutSDXL", "CMKPostProcessSetupV2Pipe", "CMKImageResizeV2",
}
CATEGORIES = ["REAL-WORLD WORKFLOWS", "TASK WORKFLOWS", "MODULE WORKFLOWS", "SYSTEM WORKFLOW", "COMPARISONS", "LEGACY"]
# Entries listed in the category view; the 2.8 Advanced module workflows are
# published as search-only entries and are not counted here.
COUNTS = {"REAL-WORLD WORKFLOWS": 4, "TASK WORKFLOWS": 13, "MODULE WORKFLOWS": 10, "SYSTEM WORKFLOW": 1, "COMPARISONS": 2, "LEGACY": 1}
SEARCH_ONLY = {
    "CMK 2.8 · Detailer Advanced · Modul.json",
    "CMK 2.8 · FaceProcess Advanced · Modul.json",
    "CMK 2.8 · FaceRebuild Advanced SDXL · Modul.json",
    "CMK 2.8 · FaceSwap Advanced · Modul.json",
}


def metadata():
    return json.loads((WEB / "flow_node_metadata.json").read_text(encoding="utf-8"))["nodes"]


def published_showcase():
    rows = []
    for path in sorted((SHOWCASE / "metadata").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("published"):
            rows.append((path.name, data))
    return sorted(rows, key=lambda row: row[1]["order"])


def listed_showcase():
    return [(name, data) for name, data in published_showcase() if data.get("searchOnly") is not True]


def packaged_names(loader):
    """CMK Package · names a loader resolves (its CMK_PACKAGED_REFERENCES)."""
    source = (PACKAGE / "pipe" / "loaders" / loader).read_text(encoding="utf-8")
    block = re.search(r"\(\n((?:\s+\"[^\"]+\",\n)+)\s*\)", source).group(1)
    return set(re.findall(r'"([^"]+)"', block))


class FlowCatalog(unittest.TestCase):
    def test_visible_set_is_closed(self):
        nodes = metadata()
        self.assertEqual(set(nodes), VISIBLE_NODES)
        self.assertTrue(all(entry.get("flowPublished") is True for entry in nodes.values()))

    def test_loader_variants_in_order(self):
        nodes = metadata()
        order = sorted((entry["variantOrder"], entry["variantLabel"]) for name, entry in nodes.items()
                       if entry.get("variantOf") == "CMKImageLoadAndResizePipe" or name == "CMKImageLoadAndResizePipe")
        self.assertEqual([label for _, label in order], ["For Detailing", "For Inpaint", "Image File Loader", "Multi-File Loader"])

    def test_start_here_has_no_v2_label_and_keeps_its_identity(self):
        entry = metadata()["CMKPipeCreateImage"]
        self.assertEqual(entry["displayName"], "01 START HERE · Create Image")
        self.assertNotIn("v2", entry["description"])
        self.assertEqual(entry["nodeProperties"], {"cmkStartUiVersion": 2})

    def test_regional_layout_sits_between_controlnet_and_ksampler(self):
        order = metadata()["CMKRegionalLayoutSDXL"]["browserOrder"]
        sub = {path.name: json.loads(path.read_text(encoding="utf-8"))["extra"]["CMKFlow"] for path in (PACKAGE / "subgraphs").glob("*.json")}
        self.assertLess(sub["CMK Flow · 05 ControlNet Combined.json"]["browserOrder"], order)
        self.assertLess(order, sub["CMK Flow · 10 KSampler SDXL 1st Pass.json"]["browserOrder"])

    def test_previews_exist(self):
        sources = [p["src"] for entry in metadata().values() for p in entry.get("previews", [])]
        for path in (PACKAGE / "subgraphs").glob("*.json"):
            sources += [p["src"] for p in json.loads(path.read_text(encoding="utf-8"))["extra"]["CMKFlow"].get("previews", [])]
        for source in sources:
            self.assertTrue((WEB / source).is_file(), source)


class CatalogReferences(unittest.TestCase):
    def test_every_recommendation_targets_a_published_entry(self):
        nodes = metadata()
        subgraphs = {}
        for path in (PACKAGE / "subgraphs").glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            subgraphs[data["nodes"][0]["type"]] = data["extra"]["CMKFlow"]
        targets = set(nodes) | set(subgraphs)
        for entry in [*nodes.values(), *subgraphs.values()]:
            for key in ("recommendedBefore", "recommendedAfter"):
                for item in entry.get(key, []):
                    self.assertIn(item["targetId"], targets, item["label"])

    def test_retired_entry_names_are_gone_from_the_texts(self):
        retired = re.compile(r"Active Family Result|PostProcess Boundary (ZIT|SDXL) or Combined")
        english = json.loads((WEB / "browser_content_en.json").read_text(encoding="utf-8"))["flows"]
        sources = [json.dumps(english, ensure_ascii=False), json.dumps(metadata(), ensure_ascii=False)]
        sources += [path.read_text(encoding="utf-8") for path in (PACKAGE / "subgraphs").glob("*.json")]
        for text in sources:
            self.assertIsNone(retired.search(text))


class ShowcaseRegistry(unittest.TestCase):
    def test_categories_and_counts_in_order(self):
        rows = listed_showcase()
        seen = []
        for _, data in rows:
            if data["category"] not in seen:
                seen.append(data["category"])
        self.assertEqual(seen, CATEGORIES)
        for category, count in COUNTS.items():
            self.assertEqual(sum(1 for _, data in rows if data["category"] == category), count, category)

    def test_numeric_prefixes_set_the_order(self):
        for name, data in listed_showcase():
            if data["category"] in ("LEGACY", "COMPARISONS", "SYSTEM WORKFLOW"):
                continue
            folder = Path(data["previews"][0]["src"]).parent.name
            prefix = int(re.match(r"^(\d+)", folder).group(1))
            self.assertEqual(data["order"] % 100, prefix, name)

    def test_workflows_and_previews_exist(self):
        for name, data in published_showcase():
            self.assertTrue((SHOWCASE / name).is_file(), name)
            for preview in data["previews"]:
                self.assertTrue((WEB / preview["src"]).is_file(), preview["src"])

    def test_every_image_reference_is_packaged(self):
        names = {path.name for path in REFERENCES.glob("*.png")}
        for name, data in published_showcase():
            workflow = json.loads((SHOWCASE / name).read_text(encoding="utf-8"))
            nodes = list(workflow.get("nodes", [])) + [node for sub in workflow.get("definitions", {}).get("subgraphs", []) for node in sub.get("nodes", [])]
            for node in nodes:
                if node.get("type") in ("CMKLoadImage", "CMKImageLoadAndResizePipe"):
                    value = (node.get("widgets_values") or [""])[0]
                    if isinstance(value, str) and value:
                        self.assertTrue(value.startswith("CMK Package · ") and value.split(" · ", 1)[1] in names, (name, value))
                if node.get("type") == "CMKSwapImageLoaderPipe":
                    pair = json.loads(node["widgets_values"][5])
                    self.assertLessEqual(set(pair.values()), names, name)
                if node.get("type") == "CMKMultiImageLoader":
                    for slot in node["properties"]["cmkMultiImageLoader"]["slots"]:
                        if slot["file"]:
                            self.assertTrue(slot["file"].split(" · ", 1)[-1] in names, (name, slot["file"]))


class FlowBrowser28Sync(unittest.TestCase):
    """2.8 Flow Browser sync: replacement, search-only entries, package references."""

    def test_identity_2_replaces_identity_3_at_its_position(self):
        rows = {data["order"]: name for name, data in listed_showcase()}
        self.assertEqual(rows[104], "CMK 2.8 · Text2Image · Identity 2.json")
        for retired in ("CMK 2.8 · Text2Image · Identity 3.json", "CMK 2.7 · Text2Image · Identity 3.json"):
            self.assertFalse((SHOWCASE / retired).exists(), retired)
            self.assertFalse((SHOWCASE / "metadata" / retired).exists(), retired)
        sources = [p["src"] for _, data in published_showcase() for p in data["previews"]]
        self.assertFalse([src for src in sources if "Identity 3" in src])

    def test_regional_layout_entry_is_the_2_8_workflow(self):
        rows = {data["order"]: name for name, data in listed_showcase()}
        self.assertEqual(rows[209], "CMK 2.8 · Flow · Text2Image Regional Layout SDXL-ZIT HYBRID.json")
        self.assertFalse((SHOWCASE / "CMK 2.7 · Flow · Text2Image Regional Layout SDXL-ZIT HYBRID.json").exists())

    def test_search_only_entries_are_the_four_advanced_modules(self):
        search_only = {name for name, data in published_showcase() if data.get("searchOnly") is True}
        self.assertEqual(search_only, SEARCH_ONLY)
        for name in SEARCH_ONLY:
            data = json.loads((SHOWCASE / "metadata" / name).read_text(encoding="utf-8"))
            self.assertEqual((data["published"], data["category"]), (True, "MODULE WORKFLOWS"), name)
            self.assertTrue((SHOWCASE / name).is_file(), name)
            self.assertTrue(data["description"] and data["description_en"], name)

    def test_search_only_flag_reaches_the_browser(self):
        backend = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
        route = backend[backend.index('routes.get("/cmk/showcase-workflows")'):backend.index('routes.get("/cmk/reference-assets/')]
        self.assertIn('"searchOnly": metadata.get("searchOnly", False) is True', route)
        javascript = (WEB / "js" / "cmk_flow_browser.js").read_text(encoding="utf-8")
        self.assertIn("filterReferences(references.map(localizeReference), term, language)", javascript)

    def test_every_sidecar_belongs_to_a_workflow(self):
        for path in (SHOWCASE / "metadata").glob("*.json"):
            self.assertTrue((SHOWCASE / path.name).is_file(), path.name)
        referenced = {(WEB / p["src"]).resolve() for path in (SHOWCASE / "metadata").glob("*.json")
                      for p in json.loads(path.read_text(encoding="utf-8")).get("previews", [])}
        shipped = {path.resolve() for path in (WEB / "assets" / "showcase").rglob("*") if path.is_file() and path.name != ".DS_Store"}
        self.assertEqual(shipped - referenced, set(), "showcase assets without a sidecar reference")

    def test_package_references_resolve_through_their_loader(self):
        resolve_load_image = packaged_names("cmk_load_image.py")
        resolve_load_resize = packaged_names("cmk_image_load_resize.py")
        manifest = {entry["package_file"]: entry["sha256"] for entry in json.loads((REFERENCES / "manifest.json").read_text(encoding="utf-8"))["assets"]}
        for name, _ in published_showcase():
            workflow = json.loads((SHOWCASE / name).read_text(encoding="utf-8"))
            nodes = list(workflow.get("nodes", [])) + [node for sub in workflow.get("definitions", {}).get("subgraphs", []) for node in sub.get("nodes", [])]
            for node in nodes:
                values = []
                if node.get("type") in ("CMKLoadImage", "CMKImageFileLoader", "CMKImageLoadAndResizePipe"):
                    values = [(node.get("widgets_values") or [""])[0]]
                elif node.get("type") == "CMKMultiImageLoader":
                    values = [slot["file"] for slot in node["properties"]["cmkMultiImageLoader"]["slots"] if slot["file"]]
                for value in values:
                    if not (isinstance(value, str) and value.startswith("CMK Package · ")):
                        continue
                    filename = value.split(" · ", 1)[1]
                    names = resolve_load_resize if node["type"] == "CMKImageLoadAndResizePipe" else resolve_load_image
                    self.assertIn(filename, names, (name, node["type"], value))
                    path = REFERENCES / filename
                    self.assertTrue(path.is_file(), (name, value))
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest.get(filename), filename)


DETAILER_STANDARD = "70aa4358-3856-4a78-8304-7fef47a51263"
DETAILER_ADVANCED = "c65f6b01-4f30-4937-8ea6-2b229ef2e076"


def subgraph_flow(name):
    return json.loads((PACKAGE / "subgraphs" / name).read_text(encoding="utf-8"))["extra"]["CMKFlow"]


class DetailerPackage(unittest.TestCase):
    """2.8.0: Multi-Model Detailers replace the SDXL Detailers in both catalogs."""

    def test_flows_catalog_publishes_the_new_detailers_only(self):
        standard = subgraph_flow("CMK Flow · Detailer.json")
        advanced = subgraph_flow("CMK Flow · Detailer · Advanced.json")
        self.assertTrue(standard["published"] and advanced["published"])
        self.assertEqual(advanced["variantOf"], DETAILER_STANDARD)
        self.assertEqual((standard["variantLabel"], advanced["variantLabel"]), ("Standard", "Advanced"))
        self.assertEqual(standard["compatibility"], ["SDXL", "Z-Image Turbo"])
        for legacy in ("CMK Flow · Detailer SDXL.json", "CMK Flow · Detailer SDXL · Advanced.json"):
            self.assertIs(subgraph_flow(legacy)["published"], False, legacy)
        english = json.loads((WEB / "browser_content_en.json").read_text(encoding="utf-8"))["flows"]
        for flow_id in (DETAILER_STANDARD, DETAILER_ADVANCED):
            self.assertTrue(english[flow_id]["description"], flow_id)

    def test_module_workflows_301_and_302_are_the_new_detailers(self):
        # Like every module pair: odd = CMK · <module>, even = its 2.8 module workflow.
        rows = {}
        for name, data in listed_showcase():
            rows.setdefault(data["order"], []).append(name)
        self.assertEqual(rows[301], ["CMK · Detailer.json"])
        self.assertEqual(rows[302], ["CMK 2.8 · Detailer · Modul.json"], "the Advanced module workflow is search-only")
        advanced = json.loads((SHOWCASE / "metadata" / "CMK 2.8 · Detailer Advanced · Modul.json").read_text(encoding="utf-8"))
        self.assertEqual((advanced["order"], advanced["published"], advanced["searchOnly"]), (302, True, True))
        # 2.8 package cleanup: the SDXL Detailer showcase workflows are gone.
        for legacy in ("CMK · Detailer SDXL.json", "CMK 2.7 · Detailer SDXL · Modul.json"):
            self.assertFalse((SHOWCASE / legacy).exists(), legacy)

    def test_module_workflows_use_the_new_subgraphs(self):
        for name, subgraph in (("CMK 2.8 · Detailer · Modul.json", "CMK Flow · Detailer"),
                               ("CMK 2.8 · Detailer Advanced · Modul.json", "CMK Flow · Detailer · Advanced")):
            workflow = json.loads((SHOWCASE / name).read_text(encoding="utf-8"))
            definitions = {item["id"]: item["name"] for item in workflow["definitions"]["subgraphs"]}
            used = [definitions[node["type"]] for node in workflow["nodes"] if node["type"] in definitions]
            self.assertEqual(used, [subgraph], name)


class SwapLoaderFallback(ScenarioCase):
    def test_package_files_resolve_on_a_fresh_install(self):
        result = self.run_case("swap_loader_packaged_references")
        self.assertIs(result["validate"], True)
        self.assertIsInstance(result["changed"], str)
        self.assertEqual(result["resolved"], "portrait_reference_00001.png")
        self.assertNotEqual(result["missing"], True)
        self.assertTrue(result["user_file_wins"])


if __name__ == "__main__":
    unittest.main()
