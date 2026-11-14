"""FaceProcess 2.8 F4: processing layers and the Refine face selection.

Reconstruction contract: the layers of a branch reproduce its result on the
same input exactly inside the union of the layer regions; outside, the branch
differs from its input only by the 2.7 whole-image quantisation (0 on the
8-bit grid, at most 1/255 off it).  Restore pasteback, detailer pasteback and
refine are real code; detection, InsightFace, the enhancer and the sampler are
stubs.
"""

from __future__ import annotations

import importlib
import pickle
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_faceprocess_layers_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

import numpy as np  # noqa: E402
import torch  # noqa: E402

layers_mod = importlib.import_module(f"{TEST_PACKAGE}.utils.face_layers")
stable = importlib.import_module(f"{TEST_PACKAGE}.utils.stable_segs")
mm = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_faceprocess_multimodel")
engine = importlib.import_module(f"{TEST_PACKAGE}.engine.native_face_restore")
native = importlib.import_module(f"{TEST_PACKAGE}.engine.native_detailer")

CMK_TEST_TIER = "T0"
EXECUTE = mm.CMKFaceProcessMultiModelPipe
H, W = 200, 320
BOXES = [(30, 50, 90, 130), (130, 50, 190, 130), (230, 50, 290, 130)]


def image(on_grid=True, seed=3):
    base = torch.rand(1, H, W, 3, generator=torch.Generator().manual_seed(seed))
    return (base * 255).round() / 255 if on_grid else base


def face(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    angles = np.linspace(0, 2 * np.pi, 106, endpoint=False)
    return SimpleNamespace(
        bbox=np.asarray((x1 + 1, y1 + 1, x2 - 1, y2 - 1), np.float32),
        kps=np.asarray([(x1 + .3 * w, y1 + .4 * h), (x1 + .7 * w, y1 + .4 * h), (x1 + .5 * w, y1 + .55 * h),
                        (x1 + .35 * w, y1 + .75 * h), (x1 + .65 * w, y1 + .75 * h)], np.float32),
        landmark_2d_106=np.stack([x1 + w / 2 + .45 * w * np.cos(angles), y1 + h / 2 + .5 * h * np.sin(angles)], 1).astype(np.float32),
        det_score=0.9)


class Inverting:
    model_size = 512

    def enhance_aligned(self, aligned):
        return 255 - aligned


class StubDetailer:
    """SEGSDetailer stand-in: a different, smaller crop per SEG (exercises the paste resize)."""

    def doit(self, image, segs, **_kwargs):
        items = []
        for seg in segs[1]:
            crop = seg.cropped_image
            half = torch.nn.functional.interpolate(crop.movedim(-1, 1), scale_factor=0.5, mode="bilinear",
                                                   align_corners=False).movedim(1, -1)
            items.append(seg._replace(cropped_image=(1.0 - half).clamp(0, 1)))
        return ((segs[0], items),)


class Branches(unittest.TestCase):
    def setUp(self):
        self.node = EXECUTE()
        self.faces = [face(box) for box in BOXES]
        self.node._insightface_faces = lambda rgb: list(self.faces)
        self.node._detect_segs = self.detect
        self.node._merge_transport = lambda image, selected, segs, enabled, log, diagnostic, **_kw: (
            image, selected, segs, enabled, log, None, diagnostic)
        for owner, name, value in ((mm, "resolve_restore_backend", lambda name: (Inverting(), {"backend": "STUB", "file": "/m/stub.pth"})),
                                   (mm, "SEGSDetailer", StubDetailer)):
            original = getattr(owner, name)
            self.addCleanup(setattr, owner, name, original)
            setattr(owner, name, value)

    def detect(self, image, detect_model, detect_crop_factor=1.2, opt_sam_model=None, **_kwargs):
        items = []
        for box in BOXES:
            x1, y1, x2, y2 = native._crop_region(box, W, H, detect_crop_factor)
            mask = torch.zeros(1, y2 - y1, x2 - x1)
            mask[:, box[1] - y1:box[3] - y1, box[0] - x1:box[2] - x1] = 1.0
            items.append(native.CMKSEG(image[:, y1:y2, x1:x2, :].clone(), mask, 0.9, (x1, y1, x2, y2), box, "face", None, 0))
        return ((W, H), items)

    def branch(self, mode, source, select_face="All Faces", restore_visibility=1.0, **params):
        values = {"detect_model": "bbox/face/face_yolov8m.pt", "refine_mode": "Off", "detect_crop_factor": 1.2,
                  "detail_paste_feather": 20, "detail_denoise": 80.0}
        values.update(params)
        face_pipe = {"face_mode": mode, "face_image": source, "face_global_enable": True, "face_steps": 4, "face_cfg": 1.0,
                     "face_restore_model": "stub.pth", "face_restore_visibility": restore_visibility}
        run = self.node._run_restore if mode == "restore" else self.node._run_detailer
        return run(face_pipe, select_face, values, opt_log=None, VISUAL=None, opt_diagnostic=None, unique_id="e",
                   cache_key=None, cache_detail="")

    def assert_reconstructs(self, source, result, layers, tolerance_outside):
        recomposed = layers_mod.compose_layers(source, layers)
        roi = layers_mod.layers_roi(layers)
        self.assertIsNotNone(roi)
        x1, y1, x2, y2 = roi
        inside = torch.zeros(H, W, dtype=torch.bool)
        inside[y1:y2, x1:x2] = True
        deviation = (recomposed - result).abs().amax(-1)[0]
        self.assertEqual(float(deviation[inside].max()), 0.0, "exact inside the layer regions")
        self.assertLessEqual(float(deviation[~inside].max()) if (~inside).any() else 0.0, tolerance_outside)

    def test_reconstruction_per_path_and_combined(self):
        for mode in ("restore", "z_image_turbo"):
            for refine in ("Off", "Detail", "Sharpen", "Smooth"):
                for on_grid in (True, False):
                    if mode == "z_image_turbo" and refine == "Off" and not on_grid:
                        tolerance = 0.0  # the detailer alone works in float: no quantisation at all
                    else:
                        tolerance = 0.0 if on_grid else 1.0 / 255 + 1e-7
                    with self.subTest(mode=mode, refine=refine, on_grid=on_grid):
                        source = image(on_grid)
                        result, _sel, segs, *_rest = self.branch(mode, source, refine_mode=refine, restore_visibility=0.6)
                        layers = segs.cmk_layers
                        kinds = [layer["kind"] for layer in layers]
                        primary = "restore" if mode == "restore" else "detailer"
                        self.assertEqual(kinds.count(primary), 3)
                        self.assertEqual("refine" in kinds, refine != "Off")
                        self.assertLess(kinds.index(primary), len(kinds) - 1 if refine != "Off" else len(kinds),
                                        "refine follows the primary step")
                        if mode == "restore":
                            self.assertEqual(kinds[0], "quantize")
                        self.assert_reconstructs(source, result, layers, tolerance)

    def test_soft_alpha_edges_are_kept(self):
        _result, _sel, segs, *_rest = self.branch("restore", image(), refine_mode="Smooth")
        for layer in segs.cmk_layers:
            if layer["kind"] in ("restore", "refine"):
                alpha = layer["alpha"]
                self.assertEqual(alpha.dtype, np.float32)
                self.assertTrue(((alpha > 0) & (alpha < alpha.max())).any(), f"{layer['kind']} keeps its soft edge")
        refine = next(layer for layer in segs.cmk_layers if layer["kind"] == "refine")
        self.assertAlmostEqual(float(refine["alpha"].max()), 0.75, places=5)

    def test_refine_reaches_beyond_the_crop_region(self):
        source = image()
        result, _sel, segs, *_rest = self.branch("z_image_turbo", source, refine_mode="Sharpen")
        refine = next(layer for layer in segs.cmk_layers if layer["kind"] == "refine")
        crops = [seg.crop_region for seg in segs[1]]
        rx1, ry1, rx2, ry2 = refine["roi"]
        self.assertTrue(any(rx1 < c[0] or ry1 < c[1] or rx2 > c[2] or ry2 > c[3] for c in crops) or
                        ry1 < min(c[1] for c in crops) or ry2 > max(c[3] for c in crops))
        outside = torch.ones(H, W, dtype=torch.bool)
        for x1, y1, x2, y2 in crops:
            outside[y1:y2, x1:x2] = False
        effect = (result - source).abs().amax(-1)[0]
        self.assertGreater(float(effect[outside].max()), 0.0, "refine changes pixels outside the crop regions")
        self.assert_reconstructs(source, result, segs.cmk_layers, 0.0)

    def test_refine_only_on_selected_faces(self):
        source = image()
        for mode in ("restore", "z_image_turbo"):
            with self.subTest(mode=mode):
                result, selected, segs, *_rest = self.branch(mode, source, select_face="Leftmost", refine_mode="Sharpen")
                self.assertEqual(len(selected[1]), 1)
                refine = next(layer for layer in segs.cmk_layers if layer["kind"] == "refine")
                self.assertLess(refine["roi"][2], BOXES[1][0], "refine region ends before the next face")
                effect = (result - source).abs().amax(-1)[0]
                self.assertEqual(float(effect[:, BOXES[1][0] - 10:].max()), 0.0, "other faces untouched")

    def test_no_refine_without_insightface_assignment(self):
        self.faces = [face(BOXES[0])]  # faces 2 and 3 have no InsightFace face
        source = image()
        result, selected, segs, _enabled, _log, _visual, diagnostic = self.branch("restore", source, refine_mode="Detail")
        self.assertEqual([row["status"] for row in diagnostic["metadata"]["faces"]], ["OK", "UNMATCHED", "UNMATCHED"])
        self.assertEqual(len(selected[1]), 1)
        effect = (result - source).abs().amax(-1)[0]
        self.assertEqual(float(effect[:, BOXES[1][0] - 10:].max()), 0.0, "no refine on unassigned faces")

    def test_no_fallback_detection_without_selection(self):
        calls = []
        self.node._detect_segs = lambda image, **kwargs: (calls.append(1), ((W, H), []))[1]
        source = image()
        for mode in ("restore", "z_image_turbo"):
            calls.clear()
            result, _sel, segs, *_rest = self.branch(mode, source, refine_mode="Sharpen")
            self.assertEqual(len(calls), 1, f"{mode}: one detection, no unfiltered re-detection for refine")
            self.assertTrue(torch.equal(result, source))
            self.assertEqual(segs.cmk_layers, [])

    def test_refine_equals_the_27_refine_stage(self):
        source = image(on_grid=False)
        segs = self.detect(source, "x")
        for mode in ("Detail", "Sharpen", "Smooth"):
            mine, layers, _info = self.node._refine(source, segs, mode)
            reference, _change, _text = self.node._apply_refine_stage(source, segs, mode)
            self.assertTrue(torch.equal(mine, reference), mode)


class Contracts(unittest.TestCase):
    def test_layer_segs_pickle_and_stable_components(self):
        source = image()
        base = stable.make_stable_segs(((W, H), []), source, source)
        layer = layers_mod.refine_layer(0, np.zeros((H, W, 3), np.uint8), np.pad(np.full((10, 10), 0.5, np.float32), 20))
        segs = layers_mod.with_layers(base, [layer])
        loaded = pickle.loads(pickle.dumps(segs))
        self.assertIsInstance(loaded, layers_mod.CMKFaceLayerSEGS)
        self.assertEqual(loaded.cmk_layers[0]["roi"], (20, 20, 30, 30))
        self.assertEqual(loaded.cmk_layer_schema, "faceprocess_layers_v1")
        self.assertIsNotNone(stable.stable_branch_components(loaded), "still a stable branch for existing merges")
        self.assertIs(type(base), stable.CMKStableSEGS, "the Detailer's stable SEGS class is unchanged")

    def test_engine_capture_does_not_change_the_result(self):
        source = image()
        faces = [[face(BOXES[0]), face(BOXES[2])]]
        plain = engine.RestoreFaceAdvanced().execute(image=source, model="x", assigned_faces=faces, enhancer=Inverting())[0]
        captured = []
        with_layers = engine.RestoreFaceAdvanced().execute(image=source, model="x", assigned_faces=faces,
                                                           enhancer=Inverting(), layers=captured)[0]
        self.assertTrue(torch.equal(plain, with_layers))
        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0]["alpha"].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
