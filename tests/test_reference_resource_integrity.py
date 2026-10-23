from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_NAME = "clipspace-mask-1792499379891.png"
VIRTUAL_NAME = f"CMK Package · {RESOURCE_NAME}"
EXPECTED_REFERENCES = {
    "clipspace-mask-1792499379891.png": (
        832,
        1152,
        6,
        "2d6e2de3df2d0fa0f3e5d94d2d28aeec2ec09cb1061d05a0770a0c4f62c53472",
    ),
    "detailer_reference.png": (
        1152,
        832,
        2,
        "ee75a396ec154d2304277364bd8be58e609118a6664f97ab65920888a44bde9f",
    ),
    "mask_detailer_reference.png": (
        832,
        1152,
        2,
        "9878f5bb242f919c592692717e28d85b7fecf96a6f7419199981d1e3501c25f9",
    ),
}
RETIRED_REFERENCES = {"face_reference2.png", "inpaint_reference3.png"}


def _walk_strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


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

    def test_affected_workflows_use_only_the_packaged_original(self):
        expected_occurrences = {
            "CMK 2.6 · InOutPaint.json": 3,
            "CMK 2.6 · Full Flow .json": 2,
        }
        for filename, count in expected_occurrences.items():
            workflow = json.loads((
                PACKAGE_ROOT / "workflows" / "showcase" / filename
            ).read_text(encoding="utf-8"))
            values = list(_walk_strings(workflow))
            self.assertEqual(values.count(VIRTUAL_NAME), count)
            for retired in RETIRED_REFERENCES:
                self.assertNotIn(f"CMK Package · {retired}", values)
            self.assertNotIn(RESOURCE_NAME, values)

    def test_existing_package_resolution_registers_the_resource(self):
        sources = (
            PACKAGE_ROOT / "__init__.py",
            PACKAGE_ROOT / "pipe" / "loaders" / "cmk_load_image.py",
            PACKAGE_ROOT / "pipe" / "loaders" / "cmk_image_load_resize.py",
        )
        for source in sources:
            content = source.read_text(encoding="utf-8")
            self.assertIn(RESOURCE_NAME, content)
            for retired in RETIRED_REFERENCES:
                self.assertNotIn(retired, content)

    def test_retired_resources_are_absent_from_workflows_and_subgraphs(self):
        for folder in ("workflows", "subgraphs"):
            for path in (PACKAGE_ROOT / folder).rglob("*.json"):
                content = path.read_text(encoding="utf-8")
                for retired in RETIRED_REFERENCES:
                    self.assertNotIn(retired, content, str(path.relative_to(PACKAGE_ROOT)))


if __name__ == "__main__":
    unittest.main()
