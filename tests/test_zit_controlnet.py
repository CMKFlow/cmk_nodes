import ast
import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ZITControlNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepare_source = (
            ROOT / "pipe" / "controlnet" / "cmk_zit_controlnet_prepare.py"
        ).read_text(encoding="utf-8")
        cls.sampler_source = (ROOT / "pipe" / "cmk_z_image_turbo.py").read_text(
            encoding="utf-8"
        )
        cls.sampler_runner_source = (
            ROOT / "pipe" / "cmk_pipe_sampler.py"
        ).read_text(encoding="utf-8")
        cls.loader_source = (
            ROOT / "pipe" / "loaders" / "z_image_turbo_loader.py"
        ).read_text(encoding="utf-8")
        cls.prepare_tree = ast.parse(cls.prepare_source)

    def test_public_node_is_zit_typed_and_optional_by_default(self):
        self.assertIn('"PROCESS": ("CMK_PROCESS_Z_IMAGE",)', self.prepare_source)
        self.assertIn(
            '"PROCESS", "IMAGE", "LOG", "VISUAL", "diagnostic", "CONTROLNET IMAGE",',
            self.prepare_source,
        )
        self.assertIn('"ENABLE": ("BOOLEAN", {"default": False})', self.prepare_source)
        self.assertNotIn("CMK_PROCESS_SDXL", self.prepare_source)

    def test_authoritative_image_is_forwarded_for_zit_inpaint(self):
        self.assertIn(
            'bool(PROCESS.get("boolean_inpaint_mode", False))',
            self.prepare_source,
        )
        self.assertIn(
            'kwargs.get("IMAGE"),\n                log,\n                visual,',
            self.prepare_source,
        )
        self.assertIn(
            '"Unchanged authoritative IMAGE for the following ZIT module."',
            self.prepare_source,
        )

    def test_ui_follows_sdxl_controlnet_source_pattern(self):
        for field in (
            '"IMAGE SOURCE"',
            '"REFERENCE IMAGE"',
            '"APPLY MASK"',
            '"STRENGTH"',
        ):
            self.assertIn(field, self.prepare_source)
        mapping = (ROOT / "cmk_mappings.py").read_text(encoding="utf-8")
        self.assertIn(
            '"CMKZITControlNetPreparePipe": '
            '"CMK Flow · 05 ControlNet ZIT"',
            mapping,
        )

    def test_prepare_uses_registered_native_reference_nodes(self):
        self.assertIn('("ImageScaleToMaxDimension",)', self.prepare_source)
        self.assertIn('("Canny",)', self.prepare_source)
        self.assertNotIn("cv2", self.prepare_source)

    def test_patch_load_and_apply_stay_inside_gated_zit_sampler(self):
        self.assertNotIn('("ModelPatchLoader",)', self.prepare_source)
        self.assertIn('_call_node_kwargs(("ModelPatchLoader",)', self.sampler_source)
        self.assertIn(
            '("QwenImageDiffsynthControlnet", "ZImageFunControlnet")',
            self.sampler_source,
        )
        control_guard = self.sampler_source.index("if controlnet_enabled:")
        patch_load = self.sampler_source.index('("ModelPatchLoader",)')
        model_sampling = self.sampler_source.index('("ModelSamplingAuraFlow",)')
        self.assertLess(control_guard, patch_load)
        self.assertLess(patch_load, model_sampling)

    def test_loaded_patch_is_exposed_for_post_sampling_unload(self):
        self.assertIn('"zit_controlnet_model_patch": controlnet_model_patch', self.sampler_source)
        self.assertIn('"zit_inpaint_model_patch": inpaint_model_patch_resource', self.sampler_source)
        self.assertIn('"zit_controlnet_model_patch"', self.sampler_runner_source)
        self.assertIn('"zit_inpaint_model_patch"', self.sampler_runner_source)
        self.assertIn("has_zit_patch", self.sampler_runner_source)
        self.assertIn("_unload_completed_controlnet(pipe)", self.sampler_runner_source)
        self.assertIn("unload_model_and_clones(", self.sampler_runner_source)
        self.assertIn("pipe.pop(key, None)", self.sampler_runner_source)
        self.assertIn('patches.pop(name, None)', self.sampler_runner_source)
        self.assertIn('[CMK ControlNet]', self.sampler_runner_source)
        self.assertIn('CONTROLNET MODEL: {controlnet_model_status}', self.sampler_source)

    def test_text2image_path_remains_the_default(self):
        self.assertIn(
            'controlnet_enabled = bool(PROCESS.get("boolean_controlnet_enable", False))',
            self.sampler_source,
        )
        self.assertIn('else ("ControlNet" if controlnet_enabled else "Text2Image")', self.sampler_source)

    def test_inpaint_path_uses_unfilled_source_and_native_zit_patch(self):
        self.assertNotIn('("InpaintModelConditioning",)', self.sampler_source)
        self.assertIn('("ZImageFunControlnet",)', self.sampler_source)
        self.assertIn(
            'inpaint_source_image = PROCESS.get("inpaint_source_image")',
            self.sampler_source,
        )
        self.assertIn("inpaint_image=inpaint_source_image", self.sampler_source)
        self.assertIn(
            '_call_node(("VAEEncode",), vae, inpaint_source_image)',
            self.sampler_source,
        )
        self.assertIn("mask=mask", self.sampler_source)
        self.assertIn('return ["IMAGE"]', self.sampler_source)

    def test_direct_inpaint_samples_full_frame_and_defers_visibility_to_finish(self):
        start = self.sampler_source.index("        if inpaint_enabled:")
        end = self.sampler_source.index("        elif hybrid_mode:", start)
        inpaint_source = self.sampler_source[start:end]

        self.assertEqual(inpaint_source.count("mask=mask"), 1)
        self.assertNotIn("noise_mask", inpaint_source)
        self.assertNotIn("grow_mask_for_sampling", inpaint_source)
        self.assertNotIn("PROCESS[\"mask\"] =", inpaint_source)
        self.assertIn('"zit_inpaint_source_image": inpaint_source_image', self.sampler_source)
        self.assertIn('"zit_inpaint_finish_mask": mask', self.sampler_source)
        self.assertIn('"zit_inpaint_masked_finish": True', self.sampler_source)
        self.assertIn(
            "_hybrid_masked_composite(source_image, image, finish_mask)",
            self.sampler_source,
        )

    def test_interrupted_zit_sampling_cleans_patch_and_invalidates_resource_cache(self):
        self.assertIn("def _cleanup_interrupted_sampling(pipe):", self.sampler_runner_source)
        self.assertIn("finally:", self.sampler_runner_source)
        self.assertIn(
            "_cleanup_interrupted_sampling(SAMPLER)",
            self.sampler_runner_source,
        )
        self.assertIn(
            'invalidate_zit_resource_cache("10 ZIT interrupted")',
            self.sampler_runner_source,
        )
        self.assertIn(
            "def invalidate_zit_resource_cache(",
            self.loader_source,
        )

    def test_direct_inpaint_patch_cleanup_runs_without_controlnet_boolean(self):
        tree = ast.parse(self.sampler_runner_source)
        helper = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_unload_completed_controlnet"
        )
        namespace = {}
        exec(
            compile(
                ast.Module(body=[helper], type_ignores=[]),
                "cmk_pipe_sampler.py",
                "exec",
            ),
            namespace,
        )

        unloaded = []
        management = types.ModuleType("comfy.model_management")
        management.unload_model_and_clones = (
            lambda model, unload_additional_models=False: unloaded.append(model)
        )
        management.soft_empty_cache = lambda force=False: None
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy.model_management = management
        previous_comfy = sys.modules.get("comfy")
        previous_management = sys.modules.get("comfy.model_management")
        sys.modules["comfy"] = comfy
        sys.modules["comfy.model_management"] = management

        class PatchedModel:
            def __init__(self):
                self.model_options = {
                    "transformer_options": {
                        "patches": {
                            "double_block": [object()],
                            "noise_refiner": [object()],
                        }
                    }
                }

        control_patch = object()
        inpaint_patch = object()
        model = PatchedModel()
        sampled_model = PatchedModel()
        pipe = {
            "boolean_controlnet_enable": False,
            "unload_models_after_use": True,
            "zit_controlnet_model_patch": control_patch,
            "zit_inpaint_model_patch": inpaint_patch,
            "model": model,
            "model_patched": sampled_model,
        }
        try:
            status = namespace["_unload_completed_controlnet"](pipe)
        finally:
            if previous_comfy is None:
                sys.modules.pop("comfy", None)
            else:
                sys.modules["comfy"] = previous_comfy
            if previous_management is None:
                sys.modules.pop("comfy.model_management", None)
            else:
                sys.modules["comfy.model_management"] = previous_management

        self.assertEqual([control_patch, inpaint_patch], unloaded)
        self.assertNotIn("zit_controlnet_model_patch", pipe)
        self.assertNotIn("zit_inpaint_model_patch", pipe)
        self.assertNotIn("patches", model.model_options["transformer_options"])
        self.assertNotIn("patches", sampled_model.model_options["transformer_options"])
        self.assertIn("UNLOADED (2 model", status)

    def test_sampler_wrapper_preserves_primary_error_after_interrupt_cleanup(self):
        tree = ast.parse(self.sampler_runner_source)
        sampler_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CMKKSamplerPipe"
        )
        cleanup_calls = []
        namespace = {
            "_cleanup_interrupted_sampling": cleanup_calls.append,
        }
        exec(
            compile(
                ast.Module(body=[sampler_class], type_ignores=[]),
                "cmk_pipe_sampler.py",
                "exec",
            ),
            namespace,
        )
        sampler = namespace["CMKKSamplerPipe"]()

        def fail(*_args, **_kwargs):
            raise RuntimeError("primary sampler failure")

        sampler._run = fail
        pipe = {"model_family": "z_image_turbo"}
        with self.assertRaisesRegex(RuntimeError, "primary sampler failure"):
            sampler.run(pipe)
        self.assertEqual([pipe], cleanup_calls)

    def test_zit_ui_resets_hidden_outpaint_before_serialization(self):
        source = (
            ROOT / "web" / "js" / "cmk_flow_start_guidance_v51.js"
        ).read_text(encoding="utf-8")
        self.assertIn("function enforceDirectZitInpaintContract(state)", source)
        self.assertIn('if (outpaint) outpaint.value = false;', source)
        serializer = source[source.index("function installStableWidgetSerialization"):]
        self.assertIn("enforceDirectZitInpaintContract(state);", serializer)
        handler_start = serializer.index("node.onSerialize = function")
        handler = serializer[
            handler_start:
            serializer.index(
                "node._cmkStableWidgetSerializationInstalled",
                handler_start,
            )
        ]
        self.assertLess(
            handler.index("enforceDirectZitInpaintContract(state);"),
            handler.index("originalOnSerialize?.apply"),
        )

    def test_user_reference_zit_inpaint_workflows_disable_outpaint(self):
        workflow_root = (
            ROOT.parents[1]
            / "user"
            / "default"
            / "workflows"
            / "SHOWCASE"
            / "REFERENCES"
            / "TASK WORKFLOWS"
        )
        expected = {
            "CMK 2.4 - CMK InOutpaint Z-Image Turbo.json": 8334,
            "CMK - TASK WORKFLOWS I.json": 8301,
        }
        checked = 0
        for filename, node_id in expected.items():
            workflow_path = workflow_root / filename
            if not workflow_path.is_file():
                continue
            document = json.loads(workflow_path.read_text(encoding="utf-8"))
            node = next(item for item in document["nodes"] if item.get("id") == node_id)
            self.assertFalse(node["widgets_values"][7], filename)
            self.assertFalse(node["widgets_values_named"]["outpaint_on"], filename)
            checked += 1
        if not checked:
            self.skipTest("user-managed reference workflows are being rebuilt")

    def test_flow_browser_metadata_is_registered(self):
        metadata = json.loads(
            (ROOT / "web" / "flow_node_metadata.json").read_text(encoding="utf-8")
        )["nodes"]["CMKZITControlNetPreparePipe"]
        self.assertEqual(
            metadata["recommendedAfter"],
            [{
                "label": "10 KSampler Z-Image Turbo",
                "targetId": "92cbb212-4e34-4b9a-8aa7-903d96ce39f2",
            }],
        )

    def test_zit_reference_picker_has_its_own_widget_normalization(self):
        source = (
            ROOT / "web" / "js" / "cmk_controlnet_prepare_preview.js"
        ).read_text(encoding="utf-8")
        self.assertIn("function normalizeZITPickerValue", source)
        self.assertIn(
            'if (nodeData.name === "CMKZITControlNetPreparePipe")',
            source,
        )
        self.assertIn("normalized.splice(pickerIndex, 0, null)", source)
        self.assertIn('typeof normalized[8] === "number"', source)
        self.assertIn(
            'normalized[8] = "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
