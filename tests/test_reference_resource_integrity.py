from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_REFERENCES = PACKAGE_ROOT / "tests" / "fixtures" / "references"
# Test-only images (2.8: moved byte-identically out of assets/references).
EXPECTED_FIXTURES = {
    "clipspace-mask-1792499379891.png": (
        832,
        1152,
        6,
        "2d6e2de3df2d0fa0f3e5d94d2d28aeec2ec09cb1061d05a0770a0c4f62c53472",
    ),
    "inpaint_reference2.png": (
        1024,
        1024,
        6,
        "c3ad98c617e8d8dab652a63e413b603d1cc6f4d0fa6ecc0eb5ee6a09da0da7c9",
    ),
}
EXPECTED_REFERENCES = {
    "detailer_reference.png": (
        1152,
        832,
        2,
        "ee75a396ec154d2304277364bd8be58e609118a6664f97ab65920888a44bde9f",
    ),
    # 2.8: the 1664 x 2304 original used by the 2.8 workflows.
    "mask_detailer_reference.png": (
        1664,
        2304,
        2,
        "14444bcfa3b24eb144949957f91811a8eb7ef26efe637a440b4b489674c42bdd",
    ),
}
RETIRED_REFERENCES = {"face_reference2.png", "inpaint_reference3.png"}


class ReferenceResourceIntegrityTests(unittest.TestCase):
    def test_optimized_reference_resources_match_manifest(self):
        manifest = json.loads((
            PACKAGE_ROOT / "assets" / "references" / "manifest.json"
        ).read_text(encoding="utf-8"))
        entries = {item["package_file"]: item for item in manifest["assets"]}
        for filename, (expected_width, expected_height, expected_color_type, expected_sha) in EXPECTED_REFERENCES.items():
            path = PACKAGE_ROOT / "assets" / "references" / filename
            payload = path.read_bytes()
            self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
            width, height, bit_depth, color_type = struct.unpack(">IIBB", payload[16:26])
            self.assertEqual(
                (width, height, bit_depth, color_type),
                (expected_width, expected_height, 8, expected_color_type),
            )
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_sha)
            self.assertEqual(entries[filename]["sha256"], expected_sha)

    def test_manifest_matches_packaged_resources(self):
        reference_root = PACKAGE_ROOT / "assets" / "references"
        manifest = json.loads((reference_root / "manifest.json").read_text(encoding="utf-8"))
        entries = {item["package_file"]: item for item in manifest["assets"]}
        self.assertTrue(RETIRED_REFERENCES.isdisjoint(entries))
        for filename, entry in entries.items():
            path = reference_root / filename
            self.assertTrue(path.is_file(), filename)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])

    def test_fixture_images_are_unchanged_and_not_package_references(self):
        sources = [
            PACKAGE_ROOT / "__init__.py",
            PACKAGE_ROOT / "pipe" / "loaders" / "cmk_load_image.py",
            PACKAGE_ROOT / "pipe" / "loaders" / "cmk_image_load_resize.py",
        ]
        registrations = "".join(source.read_text(encoding="utf-8") for source in sources)
        for filename, (width, height, color_type, digest) in EXPECTED_FIXTURES.items():
            payload = (FIXTURE_REFERENCES / filename).read_bytes()
            self.assertEqual(struct.unpack(">IIBB", payload[16:26]), (width, height, 8, color_type), filename)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest, filename)
            self.assertFalse((PACKAGE_ROOT / "assets" / "references" / filename).exists(), filename)
            self.assertNotIn(filename, registrations)
        for retired in RETIRED_REFERENCES:
            self.assertNotIn(retired, registrations)

    def test_retired_resources_are_absent_from_workflows_and_subgraphs(self):
        for folder in ("workflows", "subgraphs"):
            for path in (PACKAGE_ROOT / folder).rglob("*.json"):
                content = path.read_text(encoding="utf-8")
                for retired in RETIRED_REFERENCES:
                    self.assertNotIn(retired, content, str(path.relative_to(PACKAGE_ROOT)))


if __name__ == "__main__":
    unittest.main()
