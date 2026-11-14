"""FaceProcess 2.8 F2 code contracts: deferred Prepare, Multi-Model execution,
branch identity per mode and the boundary's disabled state.

Executor behaviour (requests, loads, HIT/MISS) is proven with the
PromptExecutor harness in test_s1_faceprocess28_f2.py.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
TEST_PACKAGE = "_cmk_faceprocess_multimodel_tests"

sys.path.insert(0, str(COMFY_ROOT))
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(TEST_PACKAGE, package)

import torch  # noqa: E402

mm = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_faceprocess_multimodel")
legacy_prepare = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_faceprocess_prepare")
legacy_pipe = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_faceprocess")
detailer_mm = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_detailer_prepare_multimodel")
boundary = importlib.import_module(f"{TEST_PACKAGE}.pipe.cmk_module_boundary_cache")
zit_loader = importlib.import_module(f"{TEST_PACKAGE}.pipe.loaders.z_image_turbo_loader")
native = importlib.import_module(f"{TEST_PACKAGE}.engine.native_detailer")
mappings = importlib.import_module(f"{TEST_PACKAGE}.cmk_mappings")

CMK_TEST_TIER = "T0"
PREPARE = mm.CMKFaceProcessPrepareMultiModelPipe
EXECUTE = mm.CMKFaceProcessMultiModelPipe
IMAGE = torch.zeros(1, 8, 8, 3)
PROCESS = {"result_contract": "family_neutral", "seed": 7, "prompt_pos": "a face"}
SDXL_SPEC = {"model_family": "sdxl", "ckpt_name": "ck.safetensors", "materialized": False}
ZIT_SPEC = {"model_family": "z_image_turbo", "diffusion_model_name": "z.safetensors", "materialized": False}


def settings(**overrides):
    values = {
        "refine_mode": "Off", "restore_model": "GFPGANv1.4.pth", "restore_visibility": 0.5, "sam_model_name": "sam.pth", "sam_device_mode": "CPU", "use_prompt_lora_from_sampler": True,
        "use_lora_from_1st_pass": False, "use_1st_pass_sampling": True, "lora_name": "None", "strength_model": 1.0,
        "strength_clip": 1.0, "prompt_pos": "", "prompt_neg": "", "steps": 30, "cfg": 3.0, "sampling_entry": 95.0,
        "sampler": "euler", "scheduler": "simple", "zit_steps": 8, "zit_cfg": 1.0, "zit_sampling_entry": 85.0,
        "zit_sampler": "res_multistep",
        "zit_scheduler": "simple", "zit_model_shift": 3.0, "stop_at_clip_layer": -2, "pag_scale": 2.5,
        "sampling": "v_prediction", "zsnr": True, "freeu_enabled": True, "freeu_b1": 1.3, "freeu_b2": 1.4,
        "freeu_s1": 0.9, "freeu_s2": 0.2,
    }
    values.update(overrides)
    return values


def prompt(mode="Restore", prepare=None, execute=None):
    """API prompt: image source, model source, Prepare, execution, boundary."""
    p = {
        "src": {"class_type": "CMKImageLoadAndResizePipe", "inputs": {"image": "a.png"}},
        "models": {"class_type": "CMKPostProcessModelSourcePipe", "inputs": {"ckpt_name": "ck", "diffusion_model": "z"}},
        "p": {"class_type": "CMKFaceProcessPrepareMultiModelPipe", "inputs": {
            **settings(**(prepare or {})), "process_mode": mode, "MODEL SDXL": ["models", 0], "MODEL ZIT": ["models", 1],
            "PROCESS": ["src", 1], "IMAGE": ["src", 2], "LOG": ["src", 3]}},
        "e": {"class_type": "CMKFaceProcessMultiModelPipe", "inputs": {
            "FACE": ["p", 0], "opt_log": ["p", 1], "opt_diagnostic": ["p", 2], "select_face": "Largest",
            "output_image_proceed": True, "detect_model": "bbox/face/face_yolov8m.pt",
            "detail_paste_feather": 20, **(execute or {})}},
        "b": {"class_type": "CMKFaceBoundaryCache", "inputs": {"MODEL": ["models", 0], "IMAGE": ["e", 0], "LOG": ["e", 4]}},
    }
    return p


def key(p):
    return EXECUTE.dependency_cache_key(p, "e")[0]


class Registration(unittest.TestCase):
    def test_nodes(self):
        self.assertIs(mappings.NODE_CLASS_MAPPINGS["CMKFaceProcessPrepareMultiModelPipe"], PREPARE)
        self.assertIs(mappings.NODE_CLASS_MAPPINGS["CMKFaceProcessMultiModelPipe"], EXECUTE)
        self.assertEqual(PREPARE.RETURN_NAMES, ("FACE", "LOG", "diagnostic"))
        self.assertEqual(PREPARE.RETURN_TYPES[0], "CMK_FACE_MULTIMODEL_PIPE")
        self.assertEqual(EXECUTE.RETURN_NAMES, legacy_pipe.CMKFaceProcessPipe.RETURN_NAMES, "public outputs unchanged")

    def test_prepare_inputs(self):
        spec = PREPARE.INPUT_TYPES()
        self.assertEqual(spec["required"]["process_mode"], (["None", "Restore", "SDXL", "Z-Image Turbo"], {"default": "None"}))
        self.assertNotIn("face_global_enable", spec["required"], "the mode replaces the global switch")
        for name in ("MODEL SDXL", "PROCESS", "IMAGE"):
            self.assertTrue(spec["required"][name][1].get("lazy"), name)
        self.assertEqual(set(spec["optional"]), {"LOG", "MODEL ZIT", "LoRA ZIT"})
        self.assertTrue(all(value[1].get("lazy") for value in spec["optional"].values()))

    def test_parameters_are_the_multimodel_detailer_parameters(self):
        ours, detailer = PREPARE.INPUT_TYPES(), detailer_mm.CMKDetailerPrepareMultiModelPipe.INPUT_TYPES()
        mine = {**ours["required"], **ours["optional"]}
        theirs = {**detailer["required"], **detailer["optional"]}
        for name in ("process_mode", "refine_mode", "restore_model", "restore_visibility"):
            mine.pop(name)
        theirs.pop("model_family")
        self.assertEqual(mine, theirs, "same names, types, defaults and ranges")
        self.assertEqual(ours["required"]["sampling_entry"][1]["default"], 95.0)
        self.assertEqual(ours["required"]["zit_sampling_entry"][1]["default"], 85.0)
        self.assertNotIn("detail_denoise", EXECUTE.INPUT_TYPES()["optional"], "replaced by the sampling entries")

    def test_execute_has_no_own_switch(self):
        spec = EXECUTE.INPUT_TYPES()
        self.assertNotIn("enable", spec["required"])
        self.assertNotIn("process_mode", spec["required"])
        self.assertEqual(spec["required"]["FACE"], ("CMK_FACE_MULTIMODEL_PIPE", {"lazy": True}))
        self.assertEqual(list(spec["required"]), ["FACE", "output_image_proceed", "select_face"])
        self.assertEqual(spec["required"]["output_image_proceed"], ("BOOLEAN", {"default": True}))
        self.assertNotIn("refine_mode", spec["optional"], "Refine is configured in the Prepare")
        self.assertFalse({"restore_model", "restore_visibility"} & set(spec["optional"]), "restore is configured in the Prepare")
        order = list(PREPARE.INPUT_TYPES()["required"])
        self.assertEqual(order[order.index("process_mode"):order.index("process_mode") + 4],
                         ["process_mode", "refine_mode", "restore_model", "restore_visibility"])
        self.assertEqual(PREPARE.INPUT_TYPES()["required"]["restore_visibility"],
                         ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}))
        self.assertEqual(PREPARE.INPUT_TYPES()["required"]["refine_mode"], (["Off", "Detail", "Sharpen", "Smooth"], {"default": "Off"}))

    def test_legacy_nodes_unchanged(self):
        self.assertIn("face_global_enable", legacy_prepare.CMKFaceProcessPreparePipe.INPUT_TYPES()["required"])
        self.assertIn("process_mode", legacy_pipe.CMKFaceProcessPipe.INPUT_TYPES()["required"])
        self.assertEqual(legacy_pipe.CMKFaceProcessPipe._CACHE_SCHEMA, "cmk_faceprocess_branch_v8")


class PrepareLazy(unittest.TestCase):
    def requests(self, mode, **present):
        kwargs = {"IMAGE": None, "LOG": None, "PROCESS": None, "MODEL SDXL": None, "MODEL ZIT": None}
        kwargs.update(present)
        return PREPARE().check_lazy_status(process_mode=mode, **kwargs)

    def test_requests_per_mode(self):
        self.assertEqual(self.requests("None"), ["IMAGE", "LOG"])
        self.assertEqual(self.requests("Restore"), ["IMAGE", "LOG"])
        self.assertEqual(self.requests("SDXL"), ["IMAGE", "LOG", "PROCESS", "MODEL SDXL"])
        self.assertEqual(self.requests("Z-Image Turbo"), ["IMAGE", "LOG", "PROCESS", "MODEL ZIT"])
        self.assertEqual(self.requests("Z-Image Turbo", **{"LoRA ZIT": None}), ["IMAGE", "LOG", "PROCESS", "MODEL ZIT", "LoRA ZIT"])
        for mode in ("None", "Restore", "SDXL"):
            self.assertNotIn("LoRA ZIT", self.requests(mode, **{"LoRA ZIT": None}), mode)

    def test_unconnected_optional_inputs_are_never_requested(self):
        self.assertEqual(PREPARE().check_lazy_status(process_mode="Z-Image Turbo", IMAGE=IMAGE, PROCESS=PROCESS), [])
        self.assertEqual(self.requests("SDXL", IMAGE=IMAGE, LOG={}, PROCESS=PROCESS, **{"MODEL SDXL": SDXL_SPEC}), [])


class PrepareSpec(unittest.TestCase):
    def setUp(self):
        # Nothing may load, encode or materialise in Prepare.
        def forbidden(*_args, **_kwargs):
            raise AssertionError("Prepare must not load")

        for owner, name in ((native.CMKSAMLoader, "load"), (legacy_prepare.CMKFaceProcessPreparePipe, "prepare"),
                            (detailer_mm.CMKDetailerPrepareMultiModelPipe, "prepare")):
            original = getattr(owner, name)
            self.addCleanup(setattr, owner, name, original)
            setattr(owner, name, forbidden)

    def face(self, mode, **extra):
        values = settings()
        values.update(extra)
        return PREPARE().prepare(IMAGE=IMAGE, process_mode=mode, PROCESS=PROCESS, LOG=None, **values)

    def test_restore_and_none_carry_no_model_and_no_process(self):
        for mode, enabled in (("None", False), ("Restore", True)):
            face, log, diagnostic = self.face(mode, **{"MODEL SDXL": SDXL_SPEC, "MODEL ZIT": ZIT_SPEC})
            with self.subTest(mode=mode):
                self.assertIs(face["face_image"], IMAGE)
                self.assertEqual(face["face_global_enable"], enabled)
                self.assertTrue(face["face_deferred"])
                self.assertFalse({"face_model_spec", "source_pipe", "face_prepare_settings"} & set(face))
                self.assertIn("MODEL SOURCE    : NOT REQUESTED", log["blocks"][-1]["lines"])
                self.assertFalse(diagnostic["metadata"]["models_loaded"])

    def test_diffusion_modes_carry_only_their_specification(self):
        face, log, _diag = self.face("SDXL", **{"MODEL SDXL": SDXL_SPEC, "MODEL ZIT": None})
        self.assertIs(face["face_model_spec"], SDXL_SPEC)
        self.assertEqual(face["face_seed"], 7)
        self.assertNotIn("MODEL SDXL", face["face_prepare_settings"])
        self.assertFalse({"face_model", "face_clip", "face_vae", "face_sam_model", "face_conditioning_pos"} & set(face))
        face, _log, _diag = self.face("Z-Image Turbo", **{"MODEL SDXL": None, "MODEL ZIT": ZIT_SPEC})
        self.assertIs(face["face_model_spec"], ZIT_SPEC)

    def test_missing_model_source_is_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "MODEL ZIT"):
            self.face("Z-Image Turbo", **{"MODEL SDXL": SDXL_SPEC, "MODEL ZIT": None})


class BranchIdentity(unittest.TestCase):
    def test_mode_resolution_and_disabled_revision(self):
        self.assertEqual(mm.branch_mode(prompt("Restore"), "e"), "restore")
        self.assertEqual(mm.branch_mode(prompt("Z-Image Turbo"), "e"), "z_image_turbo")
        linked = prompt("Restore")
        linked["p"]["inputs"]["process_mode"] = ["ctrl", 1]
        self.assertIsNone(mm.branch_mode(linked, "e"), "a link is unresolved")
        self.assertEqual(EXECUTE.dependency_revision(prompt("None"), "e", key(prompt("None"))), "disabled")

    def test_none_requests_only_the_spec(self):
        self.assertEqual(EXECUTE().check_lazy_status(FACE=None, prompt=prompt("None"), unique_id="e"), ["FACE"])

    def test_restore_ignores_diffusion_and_detail_settings(self):
        base = key(prompt("Restore"))
        for variant in (prompt("Restore", prepare={"steps": 40, "zit_steps": 12, "prompt_pos": "x", "sam_model_name": "o"}),
                        prompt("Restore", prepare={"sampling_entry": 70.0, "zit_sampling_entry": 60.0}),
                        prompt("Restore", execute={"detail_paste_feather": 5})):
            self.assertEqual(key(variant), base)
        changed = prompt("Restore")
        changed["models"]["inputs"]["ckpt_name"] = "other"
        self.assertEqual(key(changed), base, "the model source is no RESTORE identity")
        self.assertEqual(key(prompt("Restore", execute={"output_image_proceed": False})), base, "an output switch only")
        for variant in (prompt("Restore", prepare={"refine_mode": "Smooth"}),
                        prompt("Restore", execute={"select_face": "Leftmost"}),
                        prompt("Restore", prepare={"restore_visibility": 0.8}),
                        prompt("Restore", prepare={"restore_model": "codeformer-v0.1.0.pth"}),
                        prompt("SDXL")):
            self.assertNotEqual(key(variant), base)

    def test_family_settings_belong_to_their_family(self):
        sdxl = key(prompt("SDXL"))
        self.assertNotEqual(key(prompt("SDXL", prepare={"sampling_entry": 90.0})), sdxl)
        self.assertEqual(key(prompt("SDXL", prepare={"zit_sampling_entry": 70.0})), sdxl)
        self.assertNotEqual(key(prompt("Z-Image Turbo", prepare={"zit_sampling_entry": 70.0})), key(prompt("Z-Image Turbo")))
        self.assertEqual(key(prompt("Z-Image Turbo", prepare={"sampling_entry": 90.0})), key(prompt("Z-Image Turbo")))
        lora = prompt("Z-Image Turbo")
        lora["lz"] = {"class_type": "CMKZImageTurboLoRALoaderPipe", "inputs": {"lora_1": "a.safetensors"}}
        lora["p"]["inputs"]["LoRA ZIT"] = ["lz", 0]
        self.assertNotEqual(key(lora), key(prompt("Z-Image Turbo")), "LoRA ZIT is ZIT identity")
        lora_sdxl = prompt("SDXL")
        lora_sdxl["lz"] = lora["lz"]; lora_sdxl["p"]["inputs"]["LoRA ZIT"] = ["lz", 0]
        self.assertEqual(key(lora_sdxl), sdxl, "LoRA ZIT is no SDXL identity")
        self.assertEqual(key(prompt("SDXL", prepare={"zit_steps": 12})), sdxl)
        self.assertEqual(key(prompt("SDXL", prepare={"restore_visibility": 0.8})), sdxl)
        self.assertEqual(key(prompt("SDXL", prepare={"restore_model": "codeformer-v0.1.0.pth"})), sdxl)
        self.assertEqual(key(prompt("Z-Image Turbo", prepare={"restore_visibility": 0.8})), key(prompt("Z-Image Turbo")))
        self.assertNotEqual(key(prompt("SDXL", prepare={"steps": 40})), sdxl)
        changed = prompt("SDXL")
        changed["models"]["inputs"]["ckpt_name"] = "other"
        self.assertNotEqual(key(changed), sdxl, "the SDXL source is SDXL identity")
        zit = key(prompt("Z-Image Turbo"))
        self.assertEqual(key(prompt("Z-Image Turbo", prepare={"steps": 40, "cfg": 9.0})), zit)
        self.assertNotEqual(key(prompt("Z-Image Turbo", prepare={"zit_steps": 12})), zit)
        self.assertNotEqual(zit, sdxl)

    def test_transport_is_no_identity(self):
        base = key(prompt("Restore"))
        moved = prompt("Restore")
        moved["e"]["inputs"]["opt_log"] = ["src", 3]
        moved["p"]["inputs"]["LOG"] = ["other", 0]
        self.assertEqual(key(moved), base)


class Materialisation(unittest.TestCase):
    def test_restore_builds_a_model_free_pipe(self):
        face = PREPARE().prepare(IMAGE=IMAGE, process_mode="Restore", **settings())[0]
        pipe = EXECUTE()._materialize_face(face)
        self.assertEqual(pipe["face_mode"], "restore")
        self.assertIsNone(pipe["face_sam_model"])
        self.assertFalse({"face_model", "face_clip", "face_vae", "face_conditioning_pos"} & set(pipe))

    def materialise(self, mode, **extra):
        calls = []
        original = detailer_mm.CMKDetailerPrepareMultiModelPipe.prepare
        self.addCleanup(setattr, detailer_mm.CMKDetailerPrepareMultiModelPipe, "prepare", original)

        def fake_prepare(self, **kwargs):
            calls.append(kwargs)
            return ({"detailer_model": "M", "detailer_clip": "C" if kwargs["model_family"] == "SDXL" else None,
                     "detailer_vae": "V", "detailer_conditioning_pos": "P", "detailer_conditioning_neg": "N",
                     "detailer_seed": 7, "detailer_steps": 9, "detailer_cfg": 1.0, "detailer_sampler": "s",
                     "detailer_scheduler": "k", "detailer_sam_model": "S",
                     "detailer_sampling_entry": kwargs["sampling_entry" if kwargs["model_family"] == "SDXL" else "zit_sampling_entry"],
                     "source_pipe": PROCESS}, None, None)

        detailer_mm.CMKDetailerPrepareMultiModelPipe.prepare = fake_prepare
        face = PREPARE().prepare(IMAGE=IMAGE, process_mode=mode, PROCESS=PROCESS, **settings(**extra))[0]
        return calls, EXECUTE()._materialize_face(face)

    def test_sdxl_runs_the_multimodel_detailer_preparation(self):
        calls, pipe = self.materialise("SDXL", steps=33, sampling_entry=92.0, **{"MODEL SDXL": SDXL_SPEC, "MODEL ZIT": ZIT_SPEC,
                                                                                 "LoRA ZIT": {"family": "z_image_turbo"}})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model_family"], "SDXL")
        self.assertIs(calls[0]["MODEL SDXL"], SDXL_SPEC)
        self.assertEqual(calls[0]["PROCESS"], PROCESS)
        self.assertEqual((calls[0]["steps"], calls[0]["sampling_entry"]), (33, 92.0))
        self.assertNotIn("MODEL ZIT", calls[0])
        self.assertNotIn("LoRA ZIT", calls[0], "no ZIT LoRA on the SDXL path")
        self.assertEqual((pipe["face_model"], pipe["face_clip"], pipe["face_sampling_entry"], pipe["face_mode"]),
                         ("M", "C", 92.0, "sdxl"))

    def test_zit_runs_the_multimodel_detailer_preparation(self):
        bundle = {"family": "z_image_turbo", "active_loras": "<lora:z.safetensors:1>"}
        calls, pipe = self.materialise("Z-Image Turbo", zit_steps=9, zit_sampling_entry=87.0,
                                       **{"MODEL ZIT": ZIT_SPEC, "MODEL SDXL": None, "LoRA ZIT": bundle})
        self.assertEqual(calls[0]["model_family"], "Z-Image Turbo")
        self.assertIs(calls[0]["MODEL ZIT"], ZIT_SPEC)
        self.assertIs(calls[0]["LoRA ZIT"], bundle)
        self.assertEqual((calls[0]["zit_steps"], calls[0]["zit_sampling_entry"]), (9, 87.0))
        self.assertEqual((pipe["face_model"], pipe["face_vae"], pipe["face_steps"], pipe["face_sam_model"]), ("M", "V", 9, "S"))
        self.assertEqual((pipe["face_conditioning_pos"], pipe["face_conditioning_neg"], pipe["face_sampling_entry"]), ("P", "N", 87.0))
        self.assertTrue(pipe["face_global_enable"])
        self.assertIsInstance(pipe["face_clip"], str, "the text encoder is released; the pipe check needs a value")

    def test_foreign_face_pipe_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Prepare Multi-Model"):
            EXECUTE()._materialize_face({"face_image": IMAGE, "face_global_enable": True})


class Bypass(unittest.TestCase):
    def test_none_passes_the_image_through_without_processing(self):
        node = EXECUTE()
        node.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no processing"))
        face = PREPARE().prepare(IMAGE=IMAGE, process_mode="None", **settings())[0]
        image, selected, segs, enabled, log, _visual, diagnostic = node.run_pipe(
            FACE=face, prompt=prompt("None"), unique_id="e", refine_mode="Sharpen")
        self.assertIs(image, IMAGE)
        self.assertFalse(enabled)
        self.assertEqual(segs[1], [])
        self.assertIn("REFINE          : SKIPPED", log["blocks"][-1]["lines"] if isinstance(log, dict) else str(log))


class RefineAndOutput(unittest.TestCase):
    def test_refine_is_carried_by_the_prepare(self):
        self.assertEqual(PREPARE().prepare(IMAGE=IMAGE, process_mode="None", **settings(refine_mode="Sharpen"))[0]["face_refine_mode"],
                         "Off", "NONE: no Refine")
        for mode, extra in (("Restore", {}), ("SDXL", {"MODEL SDXL": SDXL_SPEC}), ("Z-Image Turbo", {"MODEL ZIT": ZIT_SPEC})):
            face = PREPARE().prepare(IMAGE=IMAGE, process_mode=mode, PROCESS=PROCESS, **settings(refine_mode="Smooth"), **extra)[0]
            with self.subTest(mode=mode):
                self.assertEqual(face["face_refine_mode"], "Smooth")
                self.assertNotIn("refine_mode", face.get("face_prepare_settings") or {}, "not a Detailer Prepare argument")

    def test_restore_configuration_is_carried_by_the_prepare(self):
        face = PREPARE().prepare(IMAGE=IMAGE, process_mode="Restore",
                                 **settings(restore_model="GFPGANv1.4.pth", restore_visibility=0.7))[0]
        self.assertEqual((face["face_restore_model"], face["face_restore_visibility"]), ("GFPGANv1.4.pth", 0.7))
        pipe = EXECUTE()._materialize_face(face)
        self.assertEqual((pipe["face_restore_model"], pipe["face_restore_visibility"]), ("GFPGANv1.4.pth", 0.7))
        self.assertEqual(PREPARE().prepare(IMAGE=IMAGE, process_mode="Restore")[0]["face_restore_visibility"], 0.5)
        for mode, extra in (("None", {}), ("SDXL", {"MODEL SDXL": SDXL_SPEC}), ("Z-Image Turbo", {"MODEL ZIT": ZIT_SPEC})):
            face = PREPARE().prepare(IMAGE=IMAGE, process_mode=mode, PROCESS=PROCESS, **settings(), **extra)[0]
            with self.subTest(mode=mode):
                self.assertFalse({"face_restore_model", "face_restore_visibility"} & set(face))
                self.assertFalse({"restore_model", "restore_visibility"} & set(face.get("face_prepare_settings") or {}))

    def test_execution_refines_with_the_prepare_value(self):
        captured = []
        node = EXECUTE()
        node._run_restore = lambda face_pipe, select_face, params, **_kw: (captured.append(params), (IMAGE, None, None, True, None, None, None))[1]
        face = PREPARE().prepare(IMAGE=IMAGE, process_mode="Restore", **settings(refine_mode="Detail"))[0]
        node.run_pipe(FACE=face, prompt=None, unique_id="e", select_face="Largest")
        self.assertEqual(captured[0]["refine_mode"], "Detail")

    def test_output_image_proceed_like_the_smart_detailer(self):
        node = EXECUTE()
        node._run_restore = lambda face_pipe, select_face, params, **_kw: (IMAGE, "faces", "segs", True, "log", "visual", "diag")
        face = PREPARE().prepare(IMAGE=IMAGE, process_mode="Restore", **settings())[0]
        on = node.run_pipe(FACE=face, output_image_proceed=True, prompt=None, unique_id="e")
        off = node.run_pipe(FACE=face, output_image_proceed=False, prompt=None, unique_id="e")
        self.assertIs(on[0], IMAGE)
        self.assertIsNone(off[0], "IMAGE PROCEED stays empty")
        self.assertEqual(off[1:], on[1:], "SEGS, ENABLED, LOG, VISUAL and diagnostic unchanged")
        none = PREPARE().prepare(IMAGE=IMAGE, process_mode="None", **settings())[0]
        self.assertIsNone(node.run_pipe(FACE=none, output_image_proceed=False, prompt=None, unique_id="e")[0])


class BoundaryState(unittest.TestCase):
    def test_disabled_follows_the_mode(self):
        self.assertTrue(boundary._faceprocess_disabled_state(prompt("None"), "b")[0])
        for mode in ("Restore", "SDXL", "Z-Image Turbo"):
            self.assertFalse(boundary._faceprocess_disabled_state(prompt(mode), "b")[0], mode)
        linked = prompt("None")
        linked["p"]["inputs"]["process_mode"] = ["ctrl", 1]
        self.assertFalse(boundary._faceprocess_disabled_state(linked, "b")[0], "an unresolved selection is active")

    def test_manifest_contains_the_branch(self):
        manifest, _detail = boundary.CMKFaceBoundaryCache()._dependencies(prompt("Restore"), "b")
        self.assertEqual([entry["class_type"] for entry in manifest["branches"]], ["CMKFaceProcessMultiModelPipe"])
        self.assertEqual(manifest["branches"][0]["cache_key"], key(prompt("Restore")))


if __name__ == "__main__":
    unittest.main()
