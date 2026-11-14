"""FaceProcess 2.8 Advanced (F5): Advanced Prepare, branch identity per FACE
slot and the layer merge.

Branch results come from the real FaceProcess paths with the F4 stubs
(inverting restore enhancer, detailer stub); the merge must compose exactly
their processing layers in slot order.  Executor behaviour (lazy order,
materialisation, HIT/MISS) is covered by the PromptExecutor harness.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_faceprocess_layers_tests"  # shared with test_faceprocess_layers (same classes)

sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

import torch  # noqa: E402

advanced = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_faceprocess_advanced_multimodel")
mm = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_faceprocess_multimodel")
layers_mod = importlib.import_module(f"{TEST_PACKAGE}.utils.face_layers")
face_diag = importlib.import_module(f"{TEST_PACKAGE}.utils.face_diagnostic")
boundary = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_module_boundary_cache")
cache = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_persistent_cache")
native = importlib.import_module(f"{TEST_PACKAGE}.engine.native_detailer")
mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")

CMK_TEST_TIER = "T0"
PREPARE = advanced.CMKFaceProcessPrepareAdvancedMultiModelPipe
MERGE = advanced.CMKFaceProcessMergeMultiModel
EXECUTE = mm.CMKFaceProcessMultiModelPipe
STANDARD = mm.CMKFaceProcessPrepareMultiModelPipe
PROCESS = {"result_contract": "family_neutral", "seed": 7, "prompt_pos": "a face"}
SDXL_SPEC = {"model_family": "sdxl", "ckpt_name": "ck.safetensors", "materialized": False}
ZIT_SPEC = {"model_family": "z_image_turbo", "diffusion_model_name": "z.safetensors", "materialized": False}
SLOT = {"refine_mode": "Off", "restore_model": "GFPGANv1.4.pth", "restore_visibility": 0.5, "steps": 30, "cfg": 7.0,
        "sampling_entry": 95.0, "sampler": "euler", "scheduler": "karras", "zit_steps": 8, "zit_cfg": 1.0,
        "zit_sampling_entry": 85.0, "zit_sampler": "res_multistep", "zit_scheduler": "simple", "zit_model_shift": 3.0}
SHARED = {"sam_model_name": "sam.pth", "sam_device_mode": "CPU", "use_prompt_lora_from_sampler": False,
          "stop_at_clip_layer": -2, "pag_scale": 2.5, "sampling": "v_prediction", "zsnr": True, "freeu_enabled": True,
          "freeu_b1": 1.3, "freeu_b2": 1.4, "freeu_s1": 0.9, "freeu_s2": 0.2}


def inputs(modes=("Restore", "SDXL", "Z-Image Turbo"), enable="ENABLE", **overrides):
    values = dict(SHARED)
    for slot, mode in enumerate(modes, 1):
        values[f"process_mode_{slot}"] = mode
        values.update({f"{name}_{slot}": value for name, value in SLOT.items()})
    values["faceprocess_global_enable"] = enable
    values.update(overrides)
    return values


def prompt(modes=("Restore", "SDXL", "Z-Image Turbo"), enable="ENABLE", **overrides):
    """Image and model source, Advanced Prepare, three branches, merge, boundary."""
    p = {
        "src": {"class_type": "CMKImageLoadAndResizePipe", "inputs": {"image": "a.png"}},
        "models": {"class_type": "CMKPostProcessModelSourcePipe", "inputs": {"ckpt_name": "ck", "diffusion_model": "z"}},
        "p": {"class_type": "CMKFaceProcessPrepareAdvancedMultiModelPipe", "inputs": {
            **inputs(modes, enable, **overrides), "MODEL SDXL": ["models", 0], "MODEL ZIT": ["models", 1],
            "PROCESS": ["src", 1], "IMAGE": ["src", 2], "LOG": ["src", 3]}},
        "m": {"class_type": "CMKFaceProcessMergeMultiModel", "inputs": {"IMAGE": ["src", 2], "LOG": ["p", 4]}},
        "b": {"class_type": "CMKFaceBoundaryCache", "inputs": {"MODEL": ["models", 0], "IMAGE": ["m", 0], "LOG": ["m", 1],
                                                              "diagnostic": ["m", 3]}},
    }
    for slot in (1, 2, 3):
        p[f"e{slot}"] = {"class_type": "CMKFaceProcessMultiModelPipe", "inputs": {
            "FACE": ["p", slot - 1], "opt_log": ["p", 4], "opt_diagnostic": ["p", 5], "select_face": "Largest",
            "output_image_proceed": True, "detect_model": "bbox/face/face_yolov8m.pt", "detail_paste_feather": 20}}
        p["m"]["inputs"].update({f"SEGS {slot}": [f"e{slot}", 2], f"LOG {slot}": [f"e{slot}", 4],
                                 f"diagnostic {slot}": [f"e{slot}", 6]})
    return p


def key(p, slot):
    return EXECUTE.dependency_cache_key(p, f"e{slot}")[0]


class Contract(unittest.TestCase):
    def test_registration_and_ports(self):
        self.assertIs(mappings.NODE_CLASS_MAPPINGS["CMKFaceProcessPrepareAdvancedMultiModelPipe"], PREPARE)
        self.assertIs(mappings.NODE_CLASS_MAPPINGS["CMKFaceProcessMergeMultiModel"], MERGE)
        self.assertEqual(PREPARE.RETURN_NAMES, ("FACE 1", "FACE 2", "FACE 3", "ENABLE", "LOG", "diagnostic"))
        self.assertEqual(PREPARE.RETURN_TYPES[:3], (mm.FACE_MULTIMODEL_PIPE,) * 3, "the Standard FACE contract")
        self.assertEqual(MERGE.RETURN_NAMES, ("IMAGE", "LOG", "VISUAL", "diagnostic", "diagnostic 1", "diagnostic 2",
                                              "diagnostic 3"))

    def test_widget_order(self):
        names = list(PREPARE.INPUT_TYPES()["required"])
        self.assertEqual(names[:3], ["MODEL SDXL", "PROCESS", "IMAGE"])
        self.assertEqual(names[-1], "faceprocess_global_enable")
        for slot in (1, 2, 3):
            first = names.index(f"process_mode_{slot}")
            self.assertEqual(names[first:first + 4], [f"process_mode_{slot}", f"refine_mode_{slot}",
                                                      f"restore_model_{slot}", f"restore_visibility_{slot}"])
        self.assertEqual(PREPARE.INPUT_TYPES()["required"]["restore_visibility_1"][1]["default"], 0.5)
        self.assertNotIn("prompt_pos", names, "promptless like the Advanced Detailer")

    def test_faces_carry_the_standard_contract(self):
        image = torch.zeros(1, 8, 8, 3)
        faces = PREPARE().prepare(image, PROCESS=PROCESS, **{**inputs(), "MODEL SDXL": SDXL_SPEC, "MODEL ZIT": ZIT_SPEC})
        restore, sdxl, zit = faces[:3]
        self.assertEqual([face["face_mode"] for face in faces[:3]], ["restore", "sdxl", "z_image_turbo"])
        self.assertEqual([face["face_contract"] for face in faces[:3]], [mm.FACE_CONTRACT] * 3)
        self.assertEqual((restore["face_restore_model"], restore["face_restore_visibility"]), ("GFPGANv1.4.pth", 0.5))
        self.assertNotIn("face_model_spec", restore, "RESTORE carries no model")
        self.assertIs(sdxl["face_model_spec"], SDXL_SPEC)
        self.assertIs(zit["face_model_spec"], ZIT_SPEC)
        self.assertEqual(sdxl["face_prepare_settings"]["prompt_pos"], "")
        self.assertEqual(sdxl["face_prepare_settings"]["lora_name"], "None")
        self.assertTrue(faces[3])
        # Same FACE as the Standard Prepare with the same settings.
        standard = STANDARD().prepare(image, process_mode="SDXL", PROCESS=PROCESS, refine_mode="Off",
                                      restore_model="GFPGANv1.4.pth", restore_visibility=0.5,
                                      **PREPARE._slot_settings({**inputs(), "MODEL SDXL": SDXL_SPEC, "MODEL ZIT": ZIT_SPEC}, 2))[0]
        self.assertEqual({k: v for k, v in sdxl.items() if k != "face_slot"}, standard)

    def test_none_and_global_disable(self):
        image = torch.zeros(1, 8, 8, 3)
        faces = PREPARE().prepare(image, **inputs(("None", "Restore", "None")))
        self.assertEqual([face["face_mode"] for face in faces[:3]], ["none", "restore", "none"])
        self.assertEqual(faces[1]["face_refine_mode"], "Off")
        off = PREPARE().prepare(image, **inputs(enable="DISABLE", refine_mode_2="Sharpen"))
        self.assertEqual([face["face_mode"] for face in off[:3]], ["none"] * 3)
        self.assertFalse(off[3])
        self.assertIn("GLOBAL          : DISABLE", str(off[4]))

    def test_refine_per_branch(self):
        faces = PREPARE().prepare(torch.zeros(1, 8, 8, 3), **inputs(("Restore", "Restore", "None"),
                                                                   refine_mode_1="Detail", refine_mode_2="Smooth",
                                                                   refine_mode_3="Sharpen"))
        self.assertEqual([face["face_refine_mode"] for face in faces[:3]], ["Detail", "Smooth", "Off"])

    def test_lazy_requests_follow_the_modes(self):
        node = PREPARE()

        def needed(modes, enable="ENABLE"):
            values = {**inputs(modes, enable), "IMAGE": torch.zeros(1), "PROCESS": None, "LOG": None,
                      "MODEL SDXL": None, "MODEL ZIT": None}
            return sorted(node.check_lazy_status(**values))

        self.assertEqual(needed(("Restore", "None", "Restore")), ["LOG"], "RESTORE needs no model and no PROCESS")
        self.assertEqual(needed(("Restore", "SDXL", "None")), ["LOG", "MODEL SDXL", "PROCESS"])
        self.assertEqual(needed(("Z-Image Turbo", "None", "None")), ["LOG", "MODEL ZIT", "PROCESS"])
        self.assertEqual(needed(("Z-Image Turbo", "SDXL", "Restore"), "DISABLE"), ["LOG"])


class BranchIdentity(unittest.TestCase):
    def test_mode_per_slot(self):
        p = prompt(("Restore", "SDXL", "None"))
        self.assertEqual([mm.branch_mode(p, f"e{slot}") for slot in (1, 2, 3)], ["restore", "sdxl", "none"])
        self.assertEqual(EXECUTE.dependency_revision(p, "e3", key(p, 3)), cache.DISABLED_REVISION)
        off = prompt(enable="DISABLE")
        self.assertEqual([mm.branch_mode(off, f"e{slot}") for slot in (1, 2, 3)], ["none"] * 3)

    def test_other_slots_do_not_invalidate(self):
        base = prompt()
        changed = prompt(steps_2=40, zit_steps_3=12, refine_mode_2="Smooth", restore_visibility_3=0.9)
        self.assertEqual(key(changed, 1), key(base, 1))

    def test_own_relevant_settings_invalidate(self):
        base = prompt()
        for slot, change in ((1, {"refine_mode_1": "Detail"}), (1, {"restore_visibility_1": 0.8}),
                             (1, {"restore_model_1": "GPEN-BFR-512.onnx"}), (2, {"steps_2": 40}), (2, {"cfg_2": 5.0}),
                             (2, {"pag_scale": 1.0}), (3, {"zit_sampling_entry_3": 60.0}), (3, {"sam_model_name": "other.pth"})):
            with self.subTest(slot=slot, change=change):
                self.assertNotEqual(key(prompt(**change), slot), key(base, slot))

    def test_settings_outside_the_mode_do_not_invalidate(self):
        base = prompt()
        same = {1: {"steps_1": 50, "zit_steps_1": 3, "pag_scale": 1.0, "sam_model_name": "other.pth"},
                2: {"zit_steps_2": 3, "restore_visibility_2": 0.9, "restore_model_2": "GPEN-BFR-512.onnx"},
                3: {"steps_3": 50, "pag_scale": 1.0, "restore_visibility_3": 0.9},
                0: {"faceprocess_global_enable": "ENABLE"}}
        for slot, change in same.items():
            for checked in ((slot,) if slot else (1, 2, 3)):
                with self.subTest(slot=checked, change=change):
                    self.assertEqual(key(prompt(**change), checked), key(base, checked))

    def test_slot_position_and_mode_are_identity(self):
        base = prompt(("Restore", "Restore", "None"))
        self.assertNotEqual(key(base, 1), key(base, 2), "own node identity per branch")
        self.assertNotEqual(key(prompt(("SDXL", "Restore", "None")), 1), key(base, 1))

    def test_standard_branch_keys_unchanged(self):
        standard = {
            "p": {"class_type": "CMKFaceProcessPrepareMultiModelPipe", "inputs": {"process_mode": "Restore",
                                                                                 "restore_model": "GFPGANv1.4.pth"}},
            "e": {"class_type": "CMKFaceProcessMultiModelPipe", "inputs": {"FACE": ["p", 0]}},
        }
        self.assertEqual(mm._prepare_link(standard, "e"), ("p", standard["p"], None))
        self.assertEqual(mm.branch_mode(standard, "e"), "restore")

    def test_boundary_module_state(self):
        def disabled(p):
            return boundary._faceprocess_disabled_state(p, "b")[0]

        self.assertFalse(disabled(prompt(("None", "Restore", "None"))))
        self.assertTrue(disabled(prompt(("None", "None", "None"))))
        self.assertTrue(disabled(prompt(enable="DISABLE")))


# -- the merge on real branch results ------------------------------------------
from test_faceprocess_layers import BOXES, H, W, Branches, face, image  # noqa: E402


def merge_branches(branches, source, modes=None):
    modes = modes or tuple("Restore" if result is not None else "None" for result in branches)
    kwargs = {}
    for slot, result in enumerate(branches, 1):
        if result is not None:
            kwargs[f"SEGS {slot}"] = result[2]
            kwargs[f"diagnostic {slot}"] = result[6]
    return MERGE().merge(source, prompt=prompt(modes), unique_id="m", **kwargs)


class Merge(Branches):
    def merge(self, branches, source, modes=None):
        return merge_branches(branches, source, modes)

    def test_single_branch_is_its_layers(self):
        source = image()
        branch = self.branch("restore", source, refine_mode="Sharpen", restore_visibility=0.6)
        merged = self.merge([branch, None, None], source)[0]
        self.assertTrue(torch.equal(merged, layers_mod.compose_layers(source, branch[2].cmk_layers)))
        self.assertEqual(float((merged - branch[0]).abs().max()), 0.0, "on the 8-bit grid: exactly the branch result")

    def test_each_branch_keeps_its_own_result(self):
        source = image()
        d1 = self.branch("restore", source, select_face="Leftmost", refine_mode="Detail", restore_visibility=0.7)
        d2 = self.branch("z_image_turbo", source, select_face="Center", refine_mode="Smooth")
        d3 = self.branch("sdxl", source, select_face="Rightmost")
        merged, _log, _visual, diagnostic, *slot_diagnostics = self.merge([d1, d2, d3], source,
                                                                          ("Restore", "Z-Image Turbo", "SDXL"))

        def region(branch):
            area = torch.zeros(H, W, dtype=torch.bool)
            x1, y1, x2, y2 = layers_mod.layers_roi(branch[2].cmk_layers)
            area[y1:y2, x1:x2] = True
            return area

        regions = [region(branch) for branch in (d1, d2, d3)]
        self.assertTrue((regions[0] & regions[1]).any(), "the refine ellipses of D1 and D2 overlap")
        for index, branch in enumerate((d1, d2, d3)):
            own = regions[index] & ~torch.stack([r for i, r in enumerate(regions) if i != index]).any(0)
            deviation = (merged - branch[0]).abs().amax(-1)[0]
            self.assertEqual(float(deviation[own].max()), 0.0, f"D{index + 1}: exactly its result in its own region")
        overlap = regions[0] & regions[1]
        self.assertEqual(float((merged - d2[0]).abs().amax(-1)[0][overlap & (d1[0] == source).all(-1)[0]].max()), 0.0,
                         "where D1 changed nothing, the overlap is D2's result")
        untouched = ~torch.stack(regions).any(0)
        self.assertEqual(float((merged - source).abs().amax(-1)[0][untouched].max()), 0.0, "outside: the input")
        self.assertEqual([d["metadata"] for d in slot_diagnostics], [d1[6]["metadata"], d2[6]["metadata"], d3[6]["metadata"]],
                         "single diagnostics pass unchanged")
        self.assertEqual(diagnostic["metadata"]["slot_order"], [1, 2, 3])
        self.assertEqual(diagnostic["metadata"]["modes"], {"D1": "restore", "D2": "z_image_turbo", "D3": "sdxl"})

    def test_overlap_composes_in_slot_order_with_the_masks(self):
        source = image(on_grid=False)
        d1 = self.branch("restore", source, refine_mode="Sharpen", restore_visibility=0.6)
        d2 = self.branch("z_image_turbo", source, refine_mode="Detail")
        merged = self.merge([d1, d2, None], source, ("Restore", "Z-Image Turbo", "None"))[0]
        expected = layers_mod.compose_layers(layers_mod.compose_layers(source, d1[2].cmk_layers), d2[2].cmk_layers)
        self.assertTrue(torch.equal(merged, expected))
        swapped = self.merge([d2, d1, None], source, ("Z-Image Turbo", "Restore", "None"))[0]
        self.assertGreater(float((merged - swapped).abs().max()), 0.0, "D1 -> D2 -> D3 decides the overlap")
        # Not a rectangle paste: D2's soft detailer mask keeps D1 where its alpha is below 1.
        detailer = next(layer for layer in d2[2].cmk_layers if layer["kind"] == "detailer")
        self.assertTrue(((detailer["alpha"] > 0) & (detailer["alpha"] < 1)).any())

    def test_none_slot_and_unconnected_slot(self):
        source = image()
        d2 = self.branch("restore", source)
        merged, log, _visual, diagnostic, diag1, diag2, diag3 = self.merge([None, d2, None], source)
        self.assertEqual(float((merged - d2[0]).abs().max()), 0.0)
        self.assertEqual(type(diag1).__name__, "ExecutionBlocker")
        self.assertIs(diag2, d2[6])
        self.assertIn("NONE · BYPASS", str(log))

    def test_foreign_branch_is_rejected(self):
        source = image()
        foreign = self.branch("restore", image(seed=9))
        with self.assertRaises(RuntimeError):
            self.merge([foreign, None, None], source)
        with self.assertRaises(ValueError):
            MERGE().merge(source, prompt=prompt(("Restore", "None", "None")), unique_id="m",
                          **{"SEGS 1": ((W, H), [])})

    def test_lazy_requests_in_resource_order(self):
        node = MERGE()
        p = prompt(("SDXL", "Z-Image Turbo", "Restore"))
        pending = {f"{name} {slot}": None for slot in (1, 2, 3) for name in ("SEGS", "LOG", "diagnostic")}
        order = []
        while True:
            needed = node.check_lazy_status(prompt=p, unique_id="m", **pending)
            if not needed:
                break
            order.append(sorted({int(name[-1]) for name in needed}))
            pending.update({name: object() for name in needed})
        self.assertEqual(order, [[3], [2], [1]], "RESTORE, then ZIT, then SDXL")
        none = prompt(("None", "SDXL", "None"))
        pending = {f"{name} {slot}": None for slot in (1, 2, 3) for name in ("SEGS", "LOG", "diagnostic")}
        self.assertEqual(node.check_lazy_status(prompt=none, unique_id="m", **pending), ["SEGS 2", "LOG 2", "diagnostic 2"])
        pending.update({"SEGS 2": 1, "LOG 2": 1, "diagnostic 2": 1})
        self.assertEqual(node.check_lazy_status(prompt=none, unique_id="m", **pending), ["diagnostic 1", "diagnostic 3"],
                         "of a NONE slot only its diagnostic, last")


class _Runner(Branches):
    """The F4 branch stubs; its transport returns the branch's own diagnostic."""

    def runTest(self):
        pass


class Diagnostics(unittest.TestCase):
    """Spatial diagnostics: observation only, from the branch's own data."""

    def setUp(self):
        self.runner = _Runner()
        self.runner.setUp()
        self.addCleanup(self.runner.doCleanups)
        self.node = self.runner.node

    def branch(self, *args, **kwargs):
        return self.runner.branch(*args, **kwargs)

    def merge(self, branches, source, modes=None):
        return merge_branches(branches, source, modes)

    def own(self, result):
        return result[6]

    def titles(self, result):
        return [stage["title"] for stage in self.own(result)["stages"]]

    def test_stage_sequences(self):
        source = image()
        self.assertEqual(self.titles(self.branch("z_image_turbo", source)),
                         ["01 Source", "02 Detection", "03 Sampling ROI", "04 Mask", "05 Pasteback", "06 Primary Detailer",
                          "07 Difference Source→Primary", "08 Final"], "the eight Detailer stages")
        self.assertEqual(self.titles(self.branch("restore", source, restore_visibility=0.6)),
                         ["01 Source", "02 Detection · Assignment", "03 Restore Region", "04 Restore Alpha", "05 Visibility",
                          "06 Primary Restore", "07 Difference Source→Primary", "08 Final"], "no sampling stages for RESTORE")
        refined = self.titles(self.branch("sdxl", source, refine_mode="Detail", select_face="Center"))
        self.assertEqual(refined[7:], ["08 Refine Mask", "09 Refine Detail", "10 Difference Primary→Refine",
                                       "11 Difference Source→Final", "12 Final"])

    def test_detection_separates_detected_selected_assigned(self):
        self.runner.faces = [face(BOXES[0])]
        restore = self.own(self.branch("restore", image(), select_face="All Faces"))
        self.assertEqual(restore["stages"][1]["subtitle"], "detected 3 · selected 3 · assigned 1")
        detailer = self.own(self.branch("z_image_turbo", image(), select_face="Leftmost"))
        self.assertEqual(detailer["stages"][1]["subtitle"], "detected 3 · selected 1")

    def test_effective_weights_are_the_layer_blends(self):
        source = image()
        for mode, refine in (("restore", "Off"), ("restore", "Smooth"), ("z_image_turbo", "Off"), ("sdxl", "Sharpen")):
            with self.subTest(mode=mode, refine=refine):
                result = self.branch(mode, source, refine_mode=refine, restore_visibility=0.6)
                weight = face_diag.layer_weight(result[2].cmk_layers, H, W)
                changed = (result[0][0] != source[0]).any(-1).numpy()
                self.assertFalse((changed & (weight == 0)).any(), "every changed pixel carries a layer weight")
        restore = next(layer for layer in self.branch("restore", source, restore_visibility=0.6)[2].cmk_layers
                       if layer["kind"] == "restore")
        weight = face_diag.layer_weight([restore], H, W, ("restore",))
        self.assertAlmostEqual(float(weight.max()), float(restore["alpha"].max()) * 0.6, places=5, msg="alpha x visibility")

    def test_diagnostics_do_not_change_the_image(self):
        source = image()
        with_stages = self.branch("restore", source, refine_mode="Sharpen", restore_visibility=0.6)
        original = mm.with_face_stages
        mm.with_face_stages = lambda payload, *args, **kwargs: payload
        try:
            without = self.branch("restore", source, refine_mode="Sharpen", restore_visibility=0.6)
        finally:
            mm.with_face_stages = original
        self.assertTrue(torch.equal(with_stages[0], without[0]))
        self.assertTrue(torch.equal(with_stages[2].cmk_branch_image, without[2].cmk_branch_image))

    def test_cache_entries_without_stages_are_completed_on_hit(self):
        source = image()
        result = self.branch("restore", source, restore_visibility=0.6)
        own = self.own(result)
        legacy = dict(own, metadata={key: value for key, value in own["metadata"].items() if key != face_diag.FACE_STAGES_FLAG},
                      stages=[stage for stage in own["stages"] if stage["title"].split(" ", 1)[1] in
                              ("Source", "Detection · Assignment", "Primary Restore", "Difference Source→Primary", "Final")])
        legacy["stages"][1] = dict(legacy["stages"][1], title="02 Detected Faces")
        completed = EXECUTE._merge_transport(result[0], result[1], result[2], True, "", legacy, before=source)[6]
        self.assertEqual([stage["title"] for stage in completed["stages"]],
                         ["01 Source", "02 Detected Faces", "03 Restore Region", "04 Restore Alpha", "05 Visibility",
                          "06 Primary Restore", "07 Difference Source→Primary", "08 Final"])
        again = EXECUTE._merge_transport(result[0], result[1], result[2], True, "", completed, before=source)[6]
        self.assertEqual(len(again["stages"]), len(completed["stages"]), "idempotent")

    def test_none_gets_no_spatial_diagnostic(self):
        face_spec = mm.CMKFaceProcessPrepareMultiModelPipe().prepare(image(), process_mode="None")[0]
        diagnostic = self.node._bypass(face_spec, None, "e", None, None)[6]
        self.assertNotIn(face_diag.FACE_STAGES_FLAG, diagnostic.get("metadata", {}))

    def test_merge_stages_show_masks_order_and_shares(self):
        source = image()
        d1 = self.branch("restore", source, refine_mode="Sharpen", restore_visibility=0.7)
        d2 = self.branch("z_image_turbo", source, refine_mode="Detail")
        d3 = self.branch("sdxl", source, select_face="Rightmost")
        merged, _log, _visual, diagnostic, *_ = self.merge([d1, d2, d3], source, ("Restore", "Z-Image Turbo", "SDXL"))
        self.assertEqual([stage["title"] for stage in diagnostic["stages"]],
                         ["01 · INPUT LAYERS", "02 · OVERLAP", "03 · COMPOSITE", "04 · FINAL"])
        metrics = diagnostic["metrics"]
        self.assertAlmostEqual(sum(metrics["share_px"].values()) + metrics["source_px"], H * W, delta=1.0,
                               msg="shares and source sum to 1 in every pixel")
        self.assertIn("D1xD2", metrics["alpha_overlap_px"])
        self.assertGreater(metrics["reduced_by_later_px"]["D1"], 0, "D2 on top in the overlap")
        weights = {slot: face_diag.layer_weight(branch[2].cmk_layers, H, W) for slot, branch in ((1, d1), (2, d2))}
        alpha_overlap = int(((weights[1] > 0) & (weights[2] > 0)).sum())
        self.assertEqual(metrics["alpha_overlap_px"]["D1xD2"], alpha_overlap, "the overlap of the actual alphas")
        self.assertNotEqual(alpha_overlap, metrics["region_overlap_px"], "not the overlap of rectangles")
        final = diagnostic["stages"][3]["image"]
        final = torch.as_tensor(final[0] if hasattr(final, "ndim") and final.ndim == 4 else final)
        expected = (merged[0].clamp(0, 1) * 255).round().to(torch.uint8)
        self.assertLessEqual(int((final.to(torch.int16) - expected.to(torch.int16)).abs().max()), 1, "FINAL is the merge result")
        self.assertIn("Effective alphas overlap, later branch on top: D1xD2", " ".join(diagnostic["warnings"]))


del Branches


if __name__ == "__main__":
    unittest.main()
