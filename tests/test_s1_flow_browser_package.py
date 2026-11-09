"""Flow Browser catalog, Showcase registry and shipped image references (2.7)."""

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
    "CMKPipeCreateImage", "CMKRegionalLayoutSDXL", "CMKPostProcessSetupPipe", "CMKImageResize",
}
CATEGORIES = ["REAL-WORLD WORKFLOWS", "TASK WORKFLOWS", "MODULE WORKFLOWS", "SYSTEM WORKFLOW", "COMPARISONS", "LEGACY"]
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
