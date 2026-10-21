from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_NAME = "clipspace-mask-1792499379891.png"
VIRTUAL_NAME = f"CMK Package · {RESOURCE_NAME}"
EXPECTED_SHA256 = "55f064628e2e0c8efbaab2fc20ea8f24dca524e9852fd9a09ddb649fb93094fb"


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
    def test_original_clipspace_resource_is_packaged_byte_identically(self):
        path = PACKAGE_ROOT / "assets" / "references" / RESOURCE_NAME
        payload = path.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), EXPECTED_SHA256)
        self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
        width, height, bit_depth, color_type = struct.unpack(">IIBB", payload[16:26])
        self.assertEqual((width, height, bit_depth, color_type), (1664, 2304, 8, 6))

        manifest = json.loads((
            PACKAGE_ROOT / "assets" / "references" / "manifest.json"
        ).read_text(encoding="utf-8"))
        entry = next(
            item for item in manifest["assets"]
            if item["package_file"] == RESOURCE_NAME
        )
        self.assertEqual(entry["original_file"], RESOURCE_NAME)
        self.assertEqual(entry["sha256"], EXPECTED_SHA256)

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
            self.assertNotIn("CMK Package · inpaint_reference3.png", values)
            self.assertNotIn(RESOURCE_NAME, values)

    def test_existing_package_resolution_registers_the_resource(self):
        sources = (
            PACKAGE_ROOT / "__init__.py",
            PACKAGE_ROOT / "pipe" / "loaders" / "cmk_load_image.py",
            PACKAGE_ROOT / "pipe" / "loaders" / "cmk_image_load_resize.py",
        )
        for source in sources:
            self.assertIn(RESOURCE_NAME, source.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
