"""Ownership of package resources versus test data (CMK Flow 2.8).

Tests must protect current contracts, not keep files alive in the published
package.  These checks keep the boundary explicit:

* every reference image in ``assets/references`` has a production consumer
  (a shipped workflow, subgraph or runtime template, or a production code
  default) - a loader registration or a manifest entry alone is no consumer;
* every registration (Python loader lists, preview middleware, frontend preview
  map, manifest) names an existing package file;
* test-only images live in ``tests/fixtures/references`` and are never
  registered as package references;
* every showcase workflow is a published Flow Browser entry, and tests read
  showcase workflows only by the names of such entries.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

CMK_TEST_TIER = "T0"
TESTS = Path(__file__).resolve().parent
PACKAGE = TESTS.parent
REFERENCES = PACKAGE / "assets" / "references"
FIXTURE_REFERENCES = TESTS / "fixtures" / "references"
SHOWCASE = PACKAGE / "workflows" / "showcase"

# Files whose image literals are registrations, not consumers.
REGISTRATIONS = [
    PACKAGE / "__init__.py",
    PACKAGE / "pipe" / "loaders" / "cmk_load_image.py",
    PACKAGE / "pipe" / "loaders" / "cmk_image_load_resize.py",
    PACKAGE / "web" / "js" / "cmk_packaged_faceswap_reference.js",
]
PNG = re.compile(r'"(?:CMK Package · )?([^"/\\]+\.png)"')


def package_images():
    return {path.name for path in REFERENCES.glob("*.png")}


def shipped_documents():
    folders = (PACKAGE / "workflows", PACKAGE / "subgraphs")
    return [path for folder in folders for path in sorted(folder.rglob("*.json")) if "metadata" not in path.parts]


def production_sources():
    sources = [path for path in PACKAGE.rglob("*.py") if "tests" not in path.relative_to(PACKAGE).parts]
    sources += sorted((PACKAGE / "web" / "js").glob("*.js"))
    return [path for path in sources if path not in REGISTRATIONS]


class ReferenceImageOwnership(unittest.TestCase):
    def test_every_package_image_has_a_production_consumer(self):
        documents = "".join(path.read_text(encoding="utf-8") for path in shipped_documents())
        code = "".join(path.read_text(encoding="utf-8") for path in production_sources())
        for name in sorted(package_images()):
            with self.subTest(image=name):
                in_document = re.search(r'(?<![\w-])' + re.escape(name), documents) is not None
                self.assertTrue(in_document or name in code, f"{name}: no workflow, subgraph, template or code default uses it")

    def test_registrations_name_existing_package_files(self):
        existing = package_images()
        for source in REGISTRATIONS:
            for name in set(PNG.findall(source.read_text(encoding="utf-8"))):
                with self.subTest(source=source.name, image=name):
                    self.assertIn(name, existing)
        manifest = json.loads((REFERENCES / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual({entry["package_file"] for entry in manifest["assets"]}, existing)

    def test_fixture_images_stay_out_of_the_package(self):
        registrations = "".join(source.read_text(encoding="utf-8") for source in REGISTRATIONS)
        for path in FIXTURE_REFERENCES.glob("*.png"):
            with self.subTest(image=path.name):
                self.assertNotIn(path.name, package_images())
                self.assertNotIn(path.name, registrations)


class ShowcaseOwnership(unittest.TestCase):
    def published(self):
        names = set()
        for path in (SHOWCASE / "metadata").glob("*.json"):
            if json.loads(path.read_text(encoding="utf-8")).get("published") is True:
                names.add(path.name)
        return names

    def test_every_showcase_workflow_is_a_published_entry(self):
        shipped = {path.name for path in SHOWCASE.glob("*.json")}
        self.assertEqual(shipped, self.published())

    def test_tests_read_only_published_showcase_workflows(self):
        published = self.published()
        pattern = re.compile(r'(?:"showcase"\s*/\s*"|workflows/showcase/)([^"/]+\.json)"')
        for path in sorted(TESTS.rglob("*.py")) + sorted(TESTS.rglob("*.mjs")):
            if "fixtures" in path.relative_to(TESTS).parts or path.name == Path(__file__).name:
                continue
            for name in pattern.findall(path.read_text(encoding="utf-8")):
                with self.subTest(test=path.name, workflow=name):
                    self.assertIn(name, published)


if __name__ == "__main__":
    unittest.main()
