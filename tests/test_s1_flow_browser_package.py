"""Flow Browser catalog, Showcase registry and shipped image references (2.7, Detailer 2.8)."""

from __future__ import annotations

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
# 2.8 showcase sync: the Advanced module workflows are catalogued, not published.
COUNTS = {"REAL-WORLD WORKFLOWS": 4, "TASK WORKFLOWS": 13, "MODULE WORKFLOWS": 10, "SYSTEM WORKFLOW": 1, "COMPARISONS": 2, "LEGACY": 1}


def metadata():
    return json.loads((WEB / "flow_node_metadata.json").read_text(encoding="utf-8"))["nodes"]


def published_showcase():
    rows = []
    for path in sorted((SHOWCASE / "metadata").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("published"):
            rows.append((path.name, data))
    return sorted(rows, key=lambda row: row[1]["order"])


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
        rows = published_showcase()
        seen = []
        for _, data in rows:
            if data["category"] not in seen:
                seen.append(data["category"])
        self.assertEqual(seen, CATEGORIES)
        for category, count in COUNTS.items():
            self.assertEqual(sum(1 for _, data in rows if data["category"] == category), count, category)

    def test_numeric_prefixes_set_the_order(self):
        for name, data in published_showcase():
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
        for name, data in published_showcase():
            rows.setdefault(data["order"], []).append(name)
        self.assertEqual(rows[301], ["CMK · Detailer.json"])
        self.assertEqual(rows[302], ["CMK 2.8 · Detailer · Modul.json"], "the Advanced module workflow is catalogued, not published")
        advanced = json.loads((SHOWCASE / "metadata" / "CMK 2.8 · Detailer Advanced · Modul.json").read_text(encoding="utf-8"))
        self.assertEqual((advanced["order"], advanced["published"]), (302, False))
        published = {name for name, _ in published_showcase()}
        for legacy in ("CMK · Detailer SDXL.json", "CMK 2.7 · Detailer SDXL · Modul.json"):
            self.assertNotIn(legacy, published)
            self.assertTrue((SHOWCASE / legacy).is_file(), legacy)

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
