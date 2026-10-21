from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_statistics_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

statistics = importlib.import_module(f"{TEST_PACKAGE}.utils.cmk_statistics")
persistent = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_persistent_cache")
boundaries = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_module_boundary_cache")
refiner = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_refiner_boundary_cache")


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "package"
        self.temp_root = Path(self.temporary.name) / "temp"
        (self.root / "subgraphs").mkdir(parents=True)
        (self.root / "workflows" / "showcase" / "metadata").mkdir(parents=True)
        (self.root / "web" / "assets" / "showcase").mkdir(parents=True)
        (self.root / "pyproject.toml").write_text(
            '[project]\nname = "cmk-flow"\nversion = "9.8.7"\n', encoding="utf-8"
        )

    def tearDown(self):
        boundaries.reset_module_boundary_cache_state()
        refiner.reset_refiner_boundary_cache_state()
        self.temporary.cleanup()

    def _workflow(self, filename, *, published, category, previews=()):
        workflow = self.root / "workflows" / "showcase" / filename
        workflow.write_text("{}", encoding="utf-8")
        metadata = workflow.parent / "metadata" / filename
        metadata.write_text(json.dumps({
            "published": published,
            "category": category,
            "previews": [{"src": source} for source in previews],
        }), encoding="utf-8")

    def test_package_version_nodes_and_subgraphs_are_dynamic(self):
        class First:
            pass

        class Second:
            pass

        (self.root / "subgraphs" / "one.json").write_text("{}", encoding="utf-8")
        (self.root / "subgraphs" / "two.json").write_text("{}", encoding="utf-8")
        result = statistics.cmk_flow_statistics(
            {"First": First, "FirstAlias": First, "Second": Second}, self.root
        )
        self.assertEqual(result, {
            "version": "9.8.7", "registered_nodes": 2, "subgraphs": 2,
        })

    def test_workflows_visibility_removed_reference_and_categories(self):
        self._workflow("visible.json", published=True, category="TASK WORKFLOWS")
        self._workflow("hidden.json", published=False, category="TASK WORKFLOWS")
        self._workflow("system.json", published=True, category="SYSTEM WORKFLOW")
        # An orphan metadata file represents a removed workflow and must not count.
        removed = self.root / "workflows" / "showcase" / "metadata" / "removed.json"
        removed.write_text(json.dumps({
            "published": True, "category": "LEGACY",
            "displayName": "CMK 2.6 · Flow · Inpaint - FaceRebuild HYBRID",
        }), encoding="utf-8")
        result = statistics.flow_browser_statistics(self.root)
        self.assertEqual((result["total"], result["visible"], result["hidden"]), (3, 2, 1))
        self.assertEqual(result["categories"]["TASK WORKFLOWS"], 2)
        self.assertEqual(result["categories"]["SYSTEM WORKFLOW"], 1)
        self.assertEqual(result["categories"]["LEGACY"], 0)

    def test_showcase_counts_physical_images_once_and_sums_sizes(self):
        image = self.root / "web" / "assets" / "showcase" / "same.png"
        second = image.with_name("other.jpg")
        image.write_bytes(b"12345")
        second.write_bytes(b"1234567")
        self._workflow(
            "one.json", published=True, category="TASK WORKFLOWS",
            previews=("assets/showcase/same.png", "assets/showcase/same.png"),
        )
        self._workflow(
            "two.json", published=False, category="LEGACY",
            previews=("assets/showcase/same.png", "assets/showcase/other.jpg"),
        )
        result = statistics.showcase_statistics(self.root)
        self.assertEqual(result, {"galleries": 2, "screenshots": 2, "bytes": 12})

    def test_cache_uses_registered_scopes_and_is_read_only(self):
        key = "a" * 64
        completion = self.root / "cache" / "generation_completion"
        completion.mkdir(parents=True)
        tensor = completion / f"{key}.safetensors"
        manifest = completion / f"{key}.json"
        tensor.write_bytes(b"tensor")
        manifest.write_bytes(b"{}")
        orphan = completion / ("b" * 64 + ".json")
        orphan.write_bytes(b"orphan")
        branch = self.temp_root / "cmk" / "detailer_branch" / "entry.pkl"
        boundary = self.temp_root / "cmk" / "refiner_boundary" / "entry.json"
        branch.parent.mkdir(parents=True)
        boundary.parent.mkdir(parents=True)
        branch.write_bytes(b"branch")
        boundary.write_bytes(b"boundary")
        tracked = [tensor, manifest, orphan, branch, boundary]
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tracked}
        boundaries._DETAILER_SESSION["keep"] = object()
        refiner._SESSION_STATE["keep"] = object()

        result = statistics.persistent_cache_statistics(self.root, self.temp_root)

        self.assertEqual(result["generation_completion"], {
            "entries": 1, "files": 3, "bytes": 14,
        })
        self.assertEqual(result["branch"], {"entries": 1, "files": 1, "bytes": 6})
        self.assertEqual(result["boundary"], {"entries": 1, "files": 1, "bytes": 8})
        self.assertEqual(result["total"], {"entries": 3, "files": 5, "bytes": 28})
        self.assertEqual(
            before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tracked},
        )
        self.assertIn("keep", boundaries._DETAILER_SESSION)
        self.assertIn("keep", refiner._SESSION_STATE)

    def test_missing_directories_and_symlinks_are_safe(self):
        empty = Path(self.temporary.name) / "empty-package"
        empty.mkdir()
        result = statistics.persistent_cache_statistics(empty, self.temp_root)
        self.assertEqual(result["total"], {"entries": 0, "files": 0, "bytes": 0})

        outside = Path(self.temporary.name) / "outside.png"
        outside.write_bytes(b"do not count")
        link = self.root / "web" / "assets" / "showcase" / "link.png"
        link.symlink_to(outside)
        self._workflow(
            "linked.json", published=True, category="TASK WORKFLOWS",
            previews=("assets/showcase/link.png", "../outside.png"),
        )
        self.assertEqual(
            statistics.showcase_statistics(self.root),
            {"galleries": 0, "screenshots": 0, "bytes": 0},
        )

        cmk_root = self.temp_root / "cmk"
        cmk_root.mkdir(parents=True, exist_ok=True)
        target = Path(self.temporary.name) / "outside-cache"
        target.mkdir()
        (cmk_root / persistent.PERSISTENT_DISK_SCOPES[0]).symlink_to(
            target, target_is_directory=True,
        )
        cache = statistics.persistent_cache_statistics(self.root, self.temp_root)
        self.assertFalse(cache["branch"]["available"])
        self.assertFalse(cache["boundary"]["available"])

    def test_runtime_has_stable_keys_and_does_not_import_device_api(self):
        before = "comfy.model_management" in sys.modules
        result = statistics.runtime_statistics()
        self.assertEqual(set(result), {"comfyui", "python", "pytorch", "compute"})
        self.assertTrue(result["python"])
        if not before:
            self.assertNotIn("comfy.model_management", sys.modules)


class CurrentPackageContractTests(unittest.TestCase):
    def test_current_showcase_reference_counts(self):
        browser = statistics.flow_browser_statistics(PACKAGE_ROOT)
        showcase = statistics.showcase_statistics(PACKAGE_ROOT)
        self.assertEqual((browser["total"], browser["visible"], browser["hidden"]), (37, 35, 2))
        self.assertEqual(browser["categories"], {
            "REAL-WORLD WORKFLOWS": 4,
            "TASK WORKFLOWS": 19,
            "MODULE WORKFLOWS": 10,
            "SYSTEM WORKFLOW": 1,
            "COMPARISONS": 2,
            "LEGACY": 1,
        })
        self.assertEqual(showcase, {
            "galleries": 37, "screenshots": 104, "bytes": 221_221_899,
        })
        self.assertFalse((
            PACKAGE_ROOT / "workflows" / "showcase" /
            "CMK 2.6 · Flow · Inpaint - FaceRebuild HYBRID.json"
        ).exists())

    def test_ui_and_read_only_route_contract(self):
        javascript = (PACKAGE_ROOT / "web/js/cmk_flow_browser.js").read_text(encoding="utf-8")
        backend = (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('class="cmk-cache-open"', javascript)
        self.assertIn('class="cmk-statistics-open"', javascript)
        self.assertIn('class="cmk-translation-open"', javascript)
        header = javascript[javascript.index('<div class="cmk-flow-header-actions">'):]
        self.assertLess(header.index('class="cmk-translation-open"'), header.index('class="cmk-statistics-open"'))
        self.assertLess(header.index('class="cmk-statistics-open"'), header.index('class="cmk-cache-open"'))
        self.assertLess(header.index('class="cmk-cache-open"'), header.index('class="cmk-storage-open"'))
        self.assertLess(header.index('class="cmk-storage-open"'), header.index('class="cmk-about-open"'))
        self.assertIn('language === "de" ? "Statistik" : "Statistics"', javascript)
        self.assertIn('className = "cmk-statistics-kpis"', javascript)
        self.assertIn('className = "cmk-statistics-middle"', javascript)
        self.assertIn('className = "cmk-statistics-table"', javascript)
        self.assertIn('gap: 12px; align-items: stretch;', javascript)
        self.assertIn('@media (max-width: 680px)', javascript)
        self.assertIn('"KiB", "MiB", "GiB", "TiB"', javascript)
        self.assertIn('unavailable: "Nicht verfügbar"', javascript)
        self.assertIn('unavailable: "Unavailable"', javascript)
        self.assertIn('api.fetchApi("/cmk/statistics", { cache: "no-store" })', javascript)
        self.assertIn('routes.get("/cmk/statistics")', backend)
        self.assertNotIn('routes.post("/cmk/statistics")', backend)


if __name__ == "__main__":
    unittest.main()
