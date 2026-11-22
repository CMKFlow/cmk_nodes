"""FaceProcess 2.8 F3: YOLO-led RESTORE.

- assignment rule YOLO -> InsightFace (OK / UNMATCHED / AMBIGUOUS / CONFLICT);
- restore backend resolution without silent fallback (E5);
- the RESTORE branch: only assigned faces are restored and claimed, skipped
  faces are reported separately, configuration errors are explicit;
- the restore engine with assigned face objects equals its own selection path.

No model is loaded: detection, InsightFace and the enhancer are stubs; the
restore pasteback itself is the real CPU code.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_faceprocess_restore_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

import numpy as np  # noqa: E402
import torch  # noqa: E402

matching = importlib.import_module(f"{TEST_PACKAGE}.utils.face_matching")
restore = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_faceprocess_restore")
mm = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_faceprocess_multimodel")
engine = importlib.import_module(f"{TEST_PACKAGE}.engine.native_face_restore")
backends = importlib.import_module(f"{TEST_PACKAGE}.engine.enhance_backends")
native = importlib.import_module(f"{TEST_PACKAGE}.engine.native_detailer")

CMK_TEST_TIER = "T0"
EXECUTE = mm.CMKFaceProcessMultiModelPipe


def face(box, kps=None, landmarks=True):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    points = kps if kps is not None else [(x1 + .3 * w, y1 + .4 * h), (x1 + .7 * w, y1 + .4 * h), (x1 + .5 * w, y1 + .55 * h),
                                          (x1 + .35 * w, y1 + .75 * h), (x1 + .65 * w, y1 + .75 * h)]
    item = SimpleNamespace(bbox=np.asarray(box, np.float32), kps=np.asarray(points, np.float32), det_score=0.9)
    if landmarks:
        angles = np.linspace(0, 2 * np.pi, 106, endpoint=False)
        item.landmark_2d_106 = np.stack([x1 + w / 2 + .45 * w * np.cos(angles), y1 + h / 2 + .5 * h * np.sin(angles)], 1).astype(np.float32)
    return item


class Assignment(unittest.TestCase):
    def test_rule_constants_are_the_calibrated_ones(self):
        self.assertEqual((matching.MATCH_IOU, matching.MATCH_MIN_KEYPOINTS, matching.MATCH_MARGIN), (0.30, 4, 0.20))

    def test_unique_faces(self):
        rows = matching.assign_faces([(10, 10, 50, 60), (100, 10, 140, 60)],
                                     [face((102, 12, 138, 58)), face((11, 12, 49, 58))])
        self.assertEqual([(r["status"], r["face_index"]) for r in rows], [("OK", 1), ("OK", 0)])
        self.assertEqual(rows[0]["keypoints"], 5)

    def test_unmatched(self):
        # no InsightFace face, too little overlap, keypoints outside the YOLO box
        self.assertEqual(matching.assign_faces([(10, 10, 50, 60)], [])[0]["status"], "UNMATCHED")
        self.assertEqual(matching.assign_faces([(10, 10, 50, 60)], [face((40, 50, 90, 110))])[0]["status"], "UNMATCHED")
        outside = face((12, 12, 48, 58), kps=[(0, 0), (1, 1), (2, 2), (30, 30), (31, 31)])
        self.assertEqual(matching.assign_faces([(10, 10, 50, 60)], [outside])[0]["status"], "UNMATCHED")

    def test_ambiguous_within_margin(self):
        rows = matching.assign_faces([(0, 0, 100, 100)], [face((5, 5, 95, 95)), face((10, 10, 90, 90))])
        self.assertEqual(rows[0]["status"], "AMBIGUOUS")
        self.assertIsNone(rows[0]["face_index"])

    def test_clear_leader_is_not_ambiguous(self):
        rows = matching.assign_faces([(0, 0, 100, 100)], [face((2, 2, 98, 98)), face((30, 30, 70, 70))])
        self.assertEqual((rows[0]["status"], rows[0]["face_index"]), ("OK", 0))

    def test_conflict_assigns_neither(self):
        rows = matching.assign_faces([(0, 0, 100, 100), (5, 5, 105, 105)], [face((3, 3, 102, 102))])
        self.assertEqual([r["status"] for r in rows], ["CONFLICT", "CONFLICT"])

    def test_no_substitute_face(self):
        # The only InsightFace face belongs to another position: never used for the selected one.
        rows = matching.assign_faces([(10, 10, 50, 60)], [face((200, 10, 240, 60))])
        self.assertEqual(rows[0]["status"], "UNMATCHED")


class Backends(unittest.TestCase):
    def patch(self, files, codeformer=(True, "available")):
        def resolve(name):
            if name in files:
                return files[name]
            raise FileNotFoundError(name)

        for owner, attr, value in ((restore, "resolve_face_restore_model", resolve),
                                   (backends, "codeformer_availability", lambda: codeformer)):
            original = getattr(owner, attr)
            self.addCleanup(setattr, owner, attr, original)
            setattr(owner, attr, value)

    def test_gfpgan_uses_exactly_the_selected_file(self):
        self.patch({"GFPGANv1.3.pth": "/m/GFPGANv1.3.pth"})
        backend, info = restore.resolve_restore_backend("GFPGANv1.3.pth")
        self.assertIsInstance(backend, backends.GFPGANEnhancerBackend)
        self.assertEqual((backend.model_name, info["backend"], info["file"]), ("GFPGANv1.3.pth", "GFPGAN", "/m/GFPGANv1.3.pth"))

    def test_gpen_only_for_its_file(self):
        self.patch({"GPEN-BFR-512.onnx": "/m/GPEN-BFR-512.onnx", "GPEN-BFR-1024.onnx": "/m/GPEN-BFR-1024.onnx"})
        self.assertIsInstance(restore.resolve_restore_backend("GPEN-BFR-512.onnx")[0], backends.GPENEnhancerBackend)
        with self.assertRaisesRegex(ValueError, "Unsupported restore model: GPEN-BFR-1024.onnx"):
            restore.resolve_restore_backend("GPEN-BFR-1024.onnx")

    def test_codeformer_reference_behaviour_and_file(self):
        self.patch({"codeformer-v0.1.0.pth": "/m/codeformer-v0.1.0.pth", "codeformer_other.pth": "/m/codeformer_other.pth"})
        backend, info = restore.resolve_restore_backend("codeformer-v0.1.0.pth")
        self.assertIsInstance(backend, backends.CodeFormerEnhancerBackend)
        self.assertEqual(info["parameters"], {"fidelity_w": 0.72, "blend_result": 0.72, "blend_input": 0.28})
        with self.assertRaisesRegex(ValueError, "CodeFormer loads only codeformer-v0.1.0.pth"):
            restore.resolve_restore_backend("codeformer_other.pth")

    def test_codeformer_unavailable_is_an_error(self):
        self.patch({"codeformer-v0.1.0.pth": "/m/codeformer-v0.1.0.pth"}, codeformer=(False, "no arch"))
        with self.assertRaisesRegex(ValueError, "CodeFormer is unavailable"):
            restore.resolve_restore_backend("codeformer-v0.1.0.pth")

    def test_no_silent_fallback(self):
        self.patch({"RestoreFormer.pth": "/m/RestoreFormer.pth"})
        for name, message in (("none", "requires a restore model"), ("", "requires a restore model"),
                              ("RestoreFormer.pth", "Unsupported restore model"), ("missing.pth", "not found")):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                restore.resolve_restore_backend(name)

    def test_identity_without_loading(self):
        self.patch({"GFPGANv1.4.pth": __file__})
        identity = restore.backend_identity("GFPGANv1.4.pth")
        self.assertEqual((identity["backend"], identity["file"]), ("GFPGAN", Path(__file__).name))
        self.assertIn("error", restore.backend_identity("none"))


class StubEnhancer:
    model_size = 512

    def enhance_aligned(self, aligned):
        return 255 - aligned


# On the 8-bit grid: the restore converts the whole image to uint8 (unchanged 2.7 behaviour).
IMAGE = (torch.rand(1, 160, 240, 3, generator=torch.Generator().manual_seed(3)) * 255).round() / 255


def seg(box, batch_index=0):
    x1, y1, x2, y2 = box
    return native.CMKSEG(IMAGE[:, y1:y2, x1:x2, :], torch.ones(1, y2 - y1, x2 - x1), 0.9, box, box, "face", None, batch_index)


class RestoreBranch(unittest.TestCase):
    BOXES = [(20, 30, 80, 100), (100, 30, 160, 100), (170, 30, 230, 100)]

    def setUp(self):
        self.calls = {"detect": 0, "restore": []}
        self.node = EXECUTE()
        self.node._detect_segs = self.detect
        self.faces = [face(self.BOXES[0]), face((102, 32, 158, 98)), face((104, 34, 156, 96))]  # 2 and 3: ambiguous near box 2
        self.node._insightface_faces = lambda rgb: list(self.faces)
        # The branch's own LOG block and diagnostic, before transport concatenation.
        self.node._merge_transport = lambda image, selected, segs, enabled, log, diagnostic, **_kw: (
            image, selected, segs, enabled, log, None, diagnostic)
        original_backend = mm.resolve_restore_backend
        self.addCleanup(setattr, mm, "resolve_restore_backend", original_backend)
        mm.resolve_restore_backend = lambda name: (StubEnhancer(), {"backend": "STUB", "file": "/m/stub.pth"})
        original_execute = mm.RestoreFaceAdvanced.execute
        self.addCleanup(setattr, mm.RestoreFaceAdvanced, "execute", original_execute)

        def execute(engine_self, **kwargs):
            self.calls["restore"].append(kwargs)
            return original_execute(engine_self, **kwargs)

        mm.RestoreFaceAdvanced.execute = execute
        self.detected = [seg(box) for box in self.BOXES]

    def detect(self, image, detect_model, opt_sam_model=None, **kwargs):
        self.calls["detect"] += 1
        self.assertIsNone(opt_sam_model, "RESTORE never uses SAM")
        return ((240, 160), list(self.detected))

    def restore_run(self, select_face="All Faces", restore_model="stub.pth", restore_visibility=1.0, **params):
        values = {"detect_model": "bbox/face/face_yolov8m.pt", "refine_mode": "Off", "detect_crop_factor": 1.2}
        values.update(params)
        # The restore configuration arrives with FACE from the Prepare.
        face_pipe = {"face_mode": "restore", "face_image": IMAGE, "face_global_enable": True,
                     "face_restore_model": restore_model, "face_restore_visibility": restore_visibility}
        return self.node._run_restore(face_pipe, select_face, values, opt_log=None, VISUAL=None, opt_diagnostic=None,
                                      unique_id="e", cache_key=None, cache_detail="")

    def test_only_assigned_faces_are_restored_and_claimed(self):
        image, selected, segs, enabled, log, _visual, diagnostic = self.restore_run()
        assigned = self.calls["restore"][0]["assigned_faces"]
        self.assertEqual(len(assigned[0]), 1, "face 2 is ambiguous, face 3 has no InsightFace face")
        self.assertIs(assigned[0][0], self.faces[0], "the complete InsightFace object is handed over")
        self.assertEqual([tuple(s.crop_region) for s in selected[1]], [self.BOXES[0]])
        statuses = [row["status"] for row in diagnostic["metadata"]["faces"]]
        self.assertEqual(statuses, ["OK", "AMBIGUOUS", "UNMATCHED"])
        lines = log["blocks"][-1]["lines"] if isinstance(log, dict) else str(log)
        text = "\n".join(lines) if isinstance(lines, list) else lines
        self.assertIn("SKIPPED · AMBIGUOUS ASSIGNMENT", text)
        self.assertIn("SKIPPED · NO INSIGHTFACE ASSIGNMENT", text)
        self.assertIn("FACES RESTORED  : 1", text)
        self.assertTrue(enabled)
        changed = (image - IMAGE).abs().amax(-1)[0]
        x1, y1, x2, y2 = self.BOXES[0]
        self.assertGreater(float(changed[y1:y2, x1:x2].max()), 0.1)
        self.assertEqual(float(changed[:, 95:].max()), 0.0, "skipped faces stay untouched")

    def test_selection_is_yolo(self):
        _image, selected, *_rest, diagnostic = self.restore_run(select_face="Leftmost")
        self.assertEqual(len(diagnostic["metadata"]["faces"]), 1)
        self.assertEqual([tuple(s.crop_region) for s in selected[1]], [self.BOXES[0]])
        self.faces = [face((102, 32, 158, 98))]  # unambiguous now
        _image, selected, *_rest = self.restore_run(select_face="Center")
        self.assertEqual([tuple(s.crop_region) for s in selected[1]], [self.BOXES[1]])

    def test_restore_order_is_ascending_face_area(self):
        self.faces = [face((26, 36, 74, 94)), face((101, 31, 159, 99)), face((172, 32, 228, 98))]
        self.restore_run()
        order = self.calls["restore"][0]["assigned_faces"][0]
        areas = [float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) for f in order]
        self.assertEqual(len(order), 3)
        self.assertEqual(areas, sorted(areas), "2.7 order: faces are pasted by ascending area")

    def test_selected_face_without_assignment_is_not_replaced(self):
        _image, selected, _segs, _enabled, _log, _visual, diagnostic = self.restore_run(select_face="Rightmost")
        self.assertEqual(diagnostic["metadata"]["faces"][0]["status"], "UNMATCHED")
        self.assertEqual(selected[1], [], "no substitute face")
        self.assertEqual(self.calls["restore"], [], "nothing to restore")

    def test_no_face_detected(self):
        self.detected = []
        image, selected, *_rest, log, _visual, diagnostic = self.restore_run()
        self.assertIs(image, IMAGE)
        self.assertIn("No face detected", diagnostic["summary"])
        self.assertIn("no face detected by YOLO", diagnostic["warnings"])

    def test_configuration_errors(self):
        with self.assertRaisesRegex(ValueError, "requires a YOLO face detector"):
            self.restore_run(detect_model="none")
        mm.resolve_restore_backend = restore.resolve_restore_backend
        with self.assertRaisesRegex(ValueError, "requires a restore model"):
            self.restore_run(restore_model="none")

    def test_dropped_widgets(self):
        optional = EXECUTE.INPUT_TYPES()["optional"]
        self.assertNotIn("restore_facedetection", optional)
        self.assertNotIn("restore_codeformer_weight", optional)
        self.assertNotIn("restore_model", optional, "restore configuration belongs to the Prepare")
        self.assertNotIn("restore_visibility", optional)
        prepare = mm.CMKFaceProcessPrepareMultiModelPipe.INPUT_TYPES()["required"]
        self.assertIn("restore_model", prepare)
        self.assertEqual(prepare["restore_visibility"][1]["default"], 0.5)


class Engine(unittest.TestCase):
    def test_assigned_faces_equal_the_engine_selection(self):
        rgb = (np.random.default_rng(5).random((160, 240, 3)) * 255).astype(np.uint8)
        image = torch.from_numpy(rgb.astype(np.float32) / 255)[None]
        target = face((40, 30, 120, 130))
        other = face((150, 30, 220, 120))

        class Analyser:
            def get(self, _rgb):
                return [target, other]

        original = engine.get_face_analyser
        self.addCleanup(setattr, engine, "get_face_analyser", original)
        engine.get_face_analyser = lambda *a: Analyser()
        own = engine.RestoreFaceAdvanced().execute(image=image, model="x", face_selection="filter", sort_by="x_position",
                                                   enhancer=StubEnhancer())[0]
        engine.get_face_analyser = lambda *a: (_ for _ in ()).throw(AssertionError("no detection with assigned faces"))
        assigned = engine.RestoreFaceAdvanced().execute(image=image, model="x", assigned_faces=[[target]],
                                                        enhancer=StubEnhancer())[0]
        self.assertTrue(torch.equal(own, assigned))

    def test_one_face_list_per_batch_item(self):
        with self.assertRaisesRegex(ValueError, "one face list per batch item"):
            engine.RestoreFaceAdvanced().execute(image=torch.zeros(2, 8, 8, 3), model="x", assigned_faces=[[]],
                                                 enhancer=StubEnhancer())


if __name__ == "__main__":
    unittest.main()
