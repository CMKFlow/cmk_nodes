"""H-1 regression: boundary diagnostics with shared tensor references.

Exercises the real save/load path (``_save_diagnostic`` / ``_load_diagnostic``)
of the loaded CMK package.  Diagnostics legitimately alias one image from
several places; persistence must be lossless and must not copy more than
safetensors requires.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from harness.comfy_runtime import boot

CMK_TEST_TIER = "T1"
SCOPE = "faceswap_boundary"
TENSOR_REF = "__cmk_diagnostic_tensor__"


def _same(test, decoded, original, path="$"):
    if isinstance(original, (np.ndarray, torch.Tensor)):
        expected = original.numpy() if isinstance(original, torch.Tensor) else original
        test.assertIsInstance(decoded, np.ndarray, path)
        test.assertEqual(decoded.dtype, expected.dtype, path)
        test.assertTrue(np.array_equal(decoded, expected), path)
    elif isinstance(original, dict):
        test.assertEqual(set(decoded), {str(key) for key in original}, path)
        for key, value in original.items():
            _same(test, decoded[str(key)], value, f"{path}.{key}")
    elif isinstance(original, (list, tuple)):
        test.assertEqual(len(decoded), len(original), path)
        for index, (left, right) in enumerate(zip(decoded, original)):
            _same(test, left, right, f"{path}[{index}]")
    else:
        test.assertEqual(decoded, original, path)


def _refs(encoded, path="$"):
    if isinstance(encoded, dict):
        if set(encoded) == {TENSOR_REF}:
            yield path, encoded[TENSOR_REF]
            return
        for key, value in encoded.items():
            yield from _refs(value, f"{path}.{key}")
    elif isinstance(encoded, list):
        for index, value in enumerate(encoded):
            yield from _refs(value, f"{path}[{index}]")


def _payload(**fields):
    base = {"type": "CMK_DIAGNOSTIC", "version": 2, "title": "t", "node": "n", "metadata": {}}
    base.update(fields)
    return base


class DiagnosticSerialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="cmk-s1-diag-"))
        cls.boundary = boot(cls.root).cmk_module("pipe.cmk_module_boundary_cache")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _encode(self, payload):
        tensors = {}
        encoded = self.boundary._encode_diagnostic_value(payload, tensors, [0], {})
        return encoded, tensors

    def _roundtrip(self, payload, key):
        self.boundary._save_diagnostic(SCOPE, key, payload)
        decoded = self.boundary._load_diagnostic(SCOPE, key)
        _same(self, decoded, payload)
        tensor_path, json_path = self.boundary._diagnostic_entry_paths(SCOPE, key)
        return json.loads(json_path.read_text(encoding="utf-8"))

    def test_identical_references_share_one_stored_tensor(self):
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        other = np.full((4, 5, 3), 7, dtype=np.uint8)
        stages = [{"title": "a", "subtitle": "", "image": image}, {"title": "b", "subtitle": "", "image": other}]
        payload = _payload(
            stages=stages,
            preview=[image, other],
            images=[image, other],
            metadata={"diagnostic_groups": [_payload(stages=[dict(s) for s in stages], preview=[image], images=[image])]},
        )
        encoded, tensors = self._encode(payload)
        self.assertEqual(len(tensors), 2)
        keys = dict(_refs(encoded))
        self.assertEqual(
            {keys["$.preview[0]"], keys["$.images[0]"], keys["$.stages[0].image"],
             keys["$.metadata.diagnostic_groups[0].stages[0].image"],
             keys["$.metadata.diagnostic_groups[0].images[0]"]},
            {keys["$.preview[0]"]},
        )
        stored = self._roundtrip(payload, "identical")
        self.assertEqual(dict(_refs(stored)), keys)

    def test_same_tensor_object_from_torch(self):
        image = torch.rand(1, 6, 6, 3)
        payload = _payload(preview=[image], images=[image])
        _, tensors = self._encode(payload)
        self.assertEqual(len(tensors), 1)
        self._roundtrip(payload, "torch_identical")

    def test_disjoint_views_are_stored_without_copy(self):
        base = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
        first, second = base[0], base[1]
        payload = _payload(preview=[first, second], images=[first, second])
        _, tensors = self._encode(payload)
        self.assertEqual(len(tensors), 2)
        stored = list(tensors.values())
        # Non-overlapping views of one storage are accepted by safetensors as is.
        self.assertEqual(stored[0].untyped_storage().data_ptr(), stored[1].untyped_storage().data_ptr())
        self._roundtrip(payload, "disjoint")

    def test_overlapping_views_are_copied_and_lossless(self):
        base = np.arange(10 * 4 * 3, dtype=np.uint8).reshape(10, 4, 3)
        whole, upper, middle = base, base[:6], base[3:8]
        payload = _payload(preview=[whole, upper, middle], images=[middle, upper, whole])
        _, tensors = self._encode(payload)
        self.assertEqual(len(tensors), 3)
        storages = {tensor.untyped_storage().data_ptr() for tensor in tensors.values()}
        self.assertEqual(len(storages), 3)
        self._roundtrip(payload, "overlapping")

    def test_non_contiguous_identical_views_share_one_key(self):
        base = np.arange(6 * 4 * 3, dtype=np.uint8).reshape(6, 4, 3)
        view = base.transpose(1, 0, 2)
        payload = _payload(preview=[view, base], images=[view, base])
        _, tensors = self._encode(payload)
        self.assertEqual(len(tensors), 2)
        self._roundtrip(payload, "non_contiguous")

    def test_nested_diagnostic_groups(self):
        a = np.random.default_rng(1).integers(0, 255, (8, 8, 3), dtype=np.uint8)
        b = a[2:6]
        inner = _payload(stages=[{"title": "x", "subtitle": "", "image": a}], preview=[a], images=[a])
        middle = _payload(
            stages=[{"title": "x", "subtitle": "", "image": a}, {"title": "y", "subtitle": "", "image": b}],
            preview=[a, b], images=[a, b], metadata={"diagnostic_groups": [inner, inner]},
        )
        payload = _payload(
            stages=[dict(stage) for stage in middle["stages"]],
            preview=[a, b], images=[a, b],
            metadata={"diagnostic_groups": [middle, inner], "warnings": ["w"], "count": 2},
        )
        _, tensors = self._encode(payload)
        # ``a`` once, ``b`` (overlaps ``a``) once as a copy.
        self.assertEqual(len(tensors), 2)
        self._roundtrip(payload, "nested")

    def test_text_only_payload_keeps_marker(self):
        payload = _payload(preview=[], images=[], stages=[], summary="none")
        self._roundtrip(payload, "text_only")


if __name__ == "__main__":
    unittest.main()
