import json
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FamilyProcessContractTests(unittest.TestCase):
    SDXL_SUBGRAPHS = (
        "CMK Flow · 10 KSampler SDXL 1st Pass.json",
        "CMK Flow · 20 Refiner SDXL.json",
        "CMK Flow · 23 Detailer SDXL.json",
        "CMK Flow · 23 Detailer SDXL · Advanced.json",
        "CMK Flow · 30 FaceProcess SDXL.json",
        "CMK Flow · 30 FaceProcess SDXL · Advanced.json",
    )

    FAMILY_GATED_SUBGRAPHS = {
        "CMK Flow · 10 KSampler SDXL 1st Pass.json": "CMKFamilyBranchGateSDXLSampled",
        "CMK Flow · 20 Refiner SDXL.json": "CMKFamilyBranchGateSDXL",
        "CMK Flow · 10 KSampler Z-Image Turbo.json": "CMKFamilyBranchGateZImage",
    }

    def test_sdxl_subgraphs_expose_only_sdxl_process_contracts(self):
        for filename in self.SDXL_SUBGRAPHS:
            with self.subTest(filename=filename):
                document = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )
                definition = document["definitions"]["subgraphs"][0]
                process_inputs = [
                    item for item in definition["inputs"] if item["name"] == "PROCESS"
                ]
                process_outputs = [
                    item for item in definition["outputs"] if item["name"] == "PROCESS"
                ]
                self.assertTrue(process_inputs)
                self.assertTrue(process_outputs)
                self.assertTrue(
                    all(item["type"] == "CMK_PROCESS_SDXL" for item in process_inputs)
                )
                self.assertTrue(
                    all(item["type"] == "CMK_PROCESS_SDXL" for item in process_outputs)
                )
                self.assertNotIn("CMK_PIPE", document["nodes"][0]["outputs"][1]["type"])

    def test_family_contract_names_are_distinct(self):
        self.assertNotEqual("CMK_PROCESS_SDXL", "CMK_PROCESS_Z_IMAGE")

    def test_module_bypass_gate_resolves_only_the_selected_path(self):
        path = ROOT / "pipe" / "cmk_family_result.py"
        spec = importlib.util.spec_from_file_location("cmk_bypass_gate_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        gate = module.CMKModuleBypassGate()
        prompt = {
            "gate": {
                "inputs": {
                    "MODEL BYPASS": ["source", 0],
                    "IMAGE BYPASS": ["source", 1],
                    "LOG BYPASS": ["source", 2],
                    "MODEL ACTIVE": ["cache", 0],
                    "IMAGE ACTIVE": ["cache", 1],
                    "LOG ACTIVE": ["cache", 2],
                    "DIAGNOSTIC ACTIVE": ["execute", 0],
                },
            },
        }
        self.assertEqual(
            gate.check_lazy_status(ENABLE=False, prompt=prompt, unique_id="gate"),
            ["MODEL BYPASS"],
        )
        bypass = {"MODEL BYPASS": {}, "IMAGE BYPASS": object(), "LOG BYPASS": {}}
        self.assertEqual(
            gate.check_lazy_status(ENABLE=False, prompt=prompt, unique_id="gate", **bypass),
            [],
        )
        self.assertEqual(
            gate.check_lazy_status(ENABLE=True, prompt=prompt, unique_id="gate"),
            ["MODEL ACTIVE"],
        )

        active_image = object()
        active = gate.gate(
            ENABLE=True,
            **{
                "MODEL ACTIVE": None,
                "IMAGE ACTIVE": active_image,
                "LOG ACTIVE": {},
                "DIAGNOSTIC ACTIVE": None,
            },
        )
        self.assertIsNone(active[0])
        self.assertIs(active[1], active_image)
        self.assertEqual(active[2], {})
        self.assertEqual(active[3]["type"], "CMK_DIAGNOSTIC")
        self.assertTrue(active[3]["metadata"]["diagnostic_fallback"])

    def test_controlnet_gates_accept_text2image_without_a_base_image(self):
        path = ROOT / "pipe" / "cmk_family_result.py"
        spec = importlib.util.spec_from_file_location("cmk_controlnet_gate_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        zit_inputs = {
            "PROCESS ACTIVE": {},
            "IMAGE ACTIVE": None,
            "LOG ACTIVE": {},
            "DIAGNOSTIC ACTIVE": {},
        }
        zit_gate = module.CMKZITControlNetBypassGate()
        self.assertEqual([], zit_gate.check_lazy_status(ENABLE=True, **zit_inputs))
        self.assertIsNone(zit_gate.gate(ENABLE=True, **zit_inputs)[1])

        combined_inputs = {
            "PROCESS SDXL ACTIVE": {},
            "PROCESS ZIT ACTIVE": {},
            "IMAGE ACTIVE": None,
            "LOG ACTIVE": {},
            "DIAGNOSTIC ACTIVE": {},
        }
        combined_gate = module.CMKCombinedControlNetBypassGate()
        self.assertEqual([], combined_gate.check_lazy_status(ENABLE=True, **combined_inputs))
        self.assertIsNone(combined_gate.gate(ENABLE=True, **combined_inputs)[2])

    def test_bypass_subgraphs_route_public_results_through_lazy_gate(self):
        expected = {
            "CMK Flow · 15 InstantID-Sampler SDXL.json": "CMKSamplerBypassGate",
            "CMK Flow · 23 Detailer SDXL.json": "CMKModuleBypassGate",
            "CMK Flow · 23 Detailer SDXL · Advanced.json": "CMKModuleBypassGate",
            "CMK Flow · 25 FaceRebuild SDXL.json": "CMKModuleBypassGate",
            "CMK Flow · 25 FaceRebuild SDXL · Advanced.json": "CMKModuleBypassGate",
            "CMK Flow · 30 FaceProcess SDXL.json": "CMKModuleBypassGate",
            "CMK Flow · 30 FaceProcess SDXL · Advanced.json": "CMKModuleBypassGate",
            "CMK Flow · 40 FaceSwap.json": "CMKModuleBypassGate",
            "CMK Flow · 40 FaceSwap · Advanced.json": "CMKModuleBypassGate",
        }
        for filename, gate_type in expected.items():
            with self.subTest(filename=filename):
                definition = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )["definitions"]["subgraphs"][0]
                gate = next(node for node in definition["nodes"] if node["type"] == gate_type)
                self.assertIsNotNone(gate["inputs"][0].get("link"))
                providers = [node for node in definition["nodes"] if node["type"] == "CMKVisualProvider"]
                for provider in providers:
                    enable = next(item for item in provider["inputs"] if item["name"] == "enable")
                    self.assertIsNotNone(enable.get("link"))

        controlnet = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · 05 ControlNet SDXL.json").read_text(
                encoding="utf-8"
            )
        )["definitions"]["subgraphs"][0]
        self.assertTrue(any(
            node["type"] == "CMKControlNetBypassGate"
            for node in controlnet["nodes"]
        ))

    def test_faceprocess_public_process_bypasses_inactive_boundary(self):
        for filename in (
            "CMK Flow · 30 FaceProcess SDXL.json",
            "CMK Flow · 30 FaceProcess SDXL · Advanced.json",
        ):
            with self.subTest(filename=filename):
                definition = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )["definitions"]["subgraphs"][0]
                process_slot = next(
                    index for index, item in enumerate(definition["outputs"])
                    if item["name"] == "PROCESS"
                )
                output_link = next(
                    link for link in definition["links"]
                    if link["target_id"] == -20
                    and link["target_slot"] == process_slot
                )
                origin = next(
                    node for node in definition["nodes"]
                    if node["id"] == output_link["origin_id"]
                )
                self.assertEqual(origin["type"], "CMKProcessForwardPipe")

    def test_standalone_image_input_has_complete_neutral_result_contract(self):
        loader_source = (
            ROOT / "pipe" / "loaders" / "cmk_image_load_resize.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG", "diagnostic")',
            loader_source,
        )
        self.assertNotIn('"CMK_PIXEL_MODEL"', loader_source)
        self.assertIn('"MODEL SDXL (opt)": ("CMK_MODEL_PIPE",)', loader_source)
        self.assertIn('"CMK_MODEL_PIPE"', loader_source)
        self.assertIn('"CMK_PROCESS_SDXL"', loader_source)
        self.assertIn('inputs.get("MODEL SDXL (opt)")', loader_source)
        self.assertIn('"result_contract": "family_neutral"', loader_source)
        self.assertIn('"source_model_family": "image"', loader_source)
        self.assertNotIn('"pixel_only": True', loader_source)

        result_source = (ROOT / "pipe" / "cmk_family_result.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("neutral_image_path = (", result_source)
        self.assertIn('family == "image"', result_source)
        self.assertNotIn('MODEL.get("pixel_only") is True', result_source)
        self.assertIn("complete CMK image-input path", result_source)

        for filename in (
            "CMK Flow · 23 Detailer SDXL.json",
            "CMK Flow · 30 FaceProcess SDXL.json",
        ):
            with self.subTest(filename=filename):
                definition = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )["definitions"]["subgraphs"][0]
                model_input = next(
                    item for item in definition["inputs"] if item["name"] == "MODEL"
                )
                self.assertEqual(model_input["type"], "CMK_MODEL_PIPE")

    def test_standalone_image_input_process_connects_to_sdxl_processors(self):
        loader_source = (
            ROOT / "pipe" / "loaders" / "cmk_image_load_resize.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"CMK_MODEL_PIPE",\n        "CMK_PROCESS_SDXL",',
            loader_source,
        )

        for filename in (
            "CMK Flow · 23 Detailer SDXL.json",
            "CMK Flow · 30 FaceProcess SDXL.json",
        ):
            with self.subTest(filename=filename):
                definition = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )["definitions"]["subgraphs"][0]
                process_input = next(
                    item for item in definition["inputs"] if item["name"] == "PROCESS"
                )
                self.assertEqual(process_input["type"], "CMK_PROCESS_SDXL")

    def test_pixel_only_finish_path_does_not_require_a_model_placeholder(self):
        checkpoint_source = (
            ROOT / "pipe" / "loaders" / "checkpoint_vae_loader.py"
        ).read_text(encoding="utf-8")
        self.assertIn('RETURN_NAMES = ("MODEL SDXL",)', checkpoint_source)

        result_source = (ROOT / "pipe" / "cmk_family_result.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"optional": {\n                "MODEL (opt)": (CMK_FINISH_INPUT,)', result_source)
        self.assertIn('MODEL = kwargs.get("MODEL (opt)", MODEL)', result_source)

        boundary_source = (
            ROOT / "pipe" / "cmk_module_boundary_cache.py"
        ).read_text(encoding="utf-8")
        face_start = boundary_source.index("class CMKFaceSwapBoundaryCache:")
        face_boundary = boundary_source[face_start:]
        self.assertIn('"optional": {\n                "MODEL (opt)": ("CMK_MODEL_PIPE", {"lazy": True})', face_boundary)
        self.assertNotIn('(\"MODEL\", MODEL),\n            (\"PROCESS\", PROCESS)', face_boundary)

        save_source = (ROOT / "nodes" / "io" / "save_project_image.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"optional": {\n                "MODEL (opt)": ("CMK_MODEL_PIPE",)', save_source)

    def test_faceswap_boundary_resolves_a_connected_model_on_cache_hit(self):
        source = (ROOT / "pipe" / "cmk_module_boundary_cache.py").read_text(
            encoding="utf-8"
        )
        section = source[source.index("class CMKFaceSwapBoundaryCache:"):]
        self.assertIn(
            'model_connected = "MODEL (opt)" in ((node or {}).get("inputs") or {})',
            section,
        )
        self.assertGreaterEqual(
            section.count('return ["MODEL (opt)"] if model_missing else []'),
            2,
        )

    def test_optional_model_nodes_use_runtime_input_order_in_saved_subgraphs(self):
        expected = {
            "CMKResultUnpackPipe": ["MODEL (opt)", "PROCESS", "IMAGE", "LOG"],
            "CMKResultPackPipe": ["MODEL (opt)", "PROCESS", "IMAGE", "LOG"],
            "CMKFaceSwapBoundaryCache": ["MODEL (opt)", "IMAGE", "LOG", "diagnostic"],
            "CMK_SaveProjectImage": [
                "MODEL (opt)",
                "PROCESS",
                "IMAGE",
                "LOG",
                "SAVE ENABLED",
                "FILENAME PREFIX",
                "OUTPUT FOLDER",
                "USE DATE FOLDER",
                "PROJECT FOLDER",
            ],
        }
        for filename in (
            "CMK Flow · 40 FaceSwap.json",
            "CMK Flow · 40 FaceSwap · Advanced.json",
            "CMK Flow · 90 Upscale & Save.json",
        ):
            with self.subTest(filename=filename):
                definition = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )["definitions"]["subgraphs"][0]
                for node in definition["nodes"]:
                    if node["type"] not in expected:
                        continue
                    self.assertEqual(
                        [item["name"] for item in node["inputs"]],
                        expected[node["type"]],
                    )
                    for slot, item in enumerate(node["inputs"]):
                        if item.get("link") is None:
                            continue
                        link = next(
                            link for link in definition["links"]
                            if link["id"] == item["link"]
                        )
                        self.assertEqual(link["target_slot"], slot)

    def test_shared_module_public_inputs_keep_optional_model_first(self):
        expected = {
            "CMK Flow · 40 FaceSwap.json": [
                "MODEL (opt)", "PROCESS", "IMAGE_TARGET", "LOG", "VISUAL",
                "FACESWAP ENABLE", "image",
            ],
            "CMK Flow · 40 FaceSwap · Advanced.json": [
                "MODEL (opt)", "PROCESS", "IMAGE_TARGET", "LOG", "VISUAL",
                "FACESWAP ENABLE",
            ],
            "CMK Flow · 90 Upscale & Save.json": [
                "MODEL (opt)", "PROCESS", "IMAGE", "LOG", "SAVE ENABLED",
                "FILENAME PREFIX", "OUTPUT FOLDER", "USE DATE FOLDER", "enable",
            ],
        }
        for filename, input_names in expected.items():
            with self.subTest(filename=filename):
                document = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )
                definition = document["definitions"]["subgraphs"][0]
                self.assertEqual(
                    [item["name"] for item in definition["inputs"]], input_names
                )
                self.assertEqual(
                    [item["name"] for item in document["nodes"][0]["inputs"]],
                    input_names,
                )

                public_slots = {
                    link["id"]: link["origin_slot"]
                    for link in definition["links"]
                    if link["origin_id"] == -10
                }
                for slot, public_input in enumerate(definition["inputs"]):
                    for link_id in public_input.get("linkIds", []):
                        self.assertEqual(public_slots[link_id], slot)

    def test_curated_processing_order_and_family_boundary_are_explicit(self):
        expected = {
            "CMK Flow · 20 Refiner SDXL.json": 20,
            "CMK Flow · 23 Detailer SDXL.json": 23,
            "CMK Flow · 30 FaceProcess SDXL.json": 30,
            "CMK Flow · 40 FaceSwap.json": 40,
            "CMK Flow · 90 Upscale & Save.json": 90,
        }
        metadata = {}
        for filename, order in expected.items():
            document = json.loads(
                (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
            )
            flow = document["extra"]["CMKFlow"]
            self.assertEqual(flow["order"], order)
            metadata[filename] = flow

        detailer_after = metadata["CMK Flow · 23 Detailer SDXL.json"]["recommendedAfter"]
        face_after = metadata["CMK Flow · 30 FaceProcess SDXL.json"]["recommendedAfter"]
        self.assertIn("30 FaceProcess SDXL", detailer_after)
        self.assertNotIn("40 FaceSwap", metadata["CMK Flow · 30 FaceProcess SDXL.json"]["recommendedBefore"])
        self.assertIn("40 FaceSwap", face_after)

    def test_boundary_caches_do_not_carry_process(self):
        boundary_classes = (
            "CMKRefinerBoundaryCache",
            "CMKDetailerBoundaryCache",
            "CMKFaceRebuildBoundaryCache",
            "CMKZImageBoundaryCache",
            "CMKFaceBoundaryCache",
            "CMKFaceSwapBoundaryCache",
        )
        for filename in (
            ROOT / "pipe" / "cmk_refiner_boundary_cache.py",
            ROOT / "pipe" / "cmk_module_boundary_cache.py",
        ):
            source = filename.read_text(encoding="utf-8")
            starts = [
                (source.index(f"class {name}:"), name)
                for name in boundary_classes
                if f"class {name}:" in source
            ]
            starts.sort()
            for index, (start, name) in enumerate(starts):
                end = starts[index + 1][0] if index + 1 < len(starts) else len(source)
                class_source = source[start:end]
                with self.subTest(name=name):
                    self.assertNotIn('"PROCESS": (', class_source)
                    self.assertNotIn('"PROCESS",', class_source)

        for path in (ROOT / "subgraphs").glob("*.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            for definition in document.get("definitions", {}).get("subgraphs", []):
                for node in definition.get("nodes", []):
                    if not str(node.get("type", "")).endswith("BoundaryCache"):
                        continue
                    with self.subTest(path=path.name, node=node["type"]):
                        self.assertNotIn("PROCESS", [item["name"] for item in node.get("inputs", [])])
                        self.assertNotIn("PROCESS", [item["name"] for item in node.get("outputs", [])])

    def test_faceprocess_prepare_remains_an_sdxl_only_lazy_family_gate(self):
        source = (ROOT / "pipe" / "cmk_module_boundary_cache.py").read_text(
            encoding="utf-8"
        )
        start = source.index("class CMKFaceBoundaryCache:")
        face_boundary = source[start:]
        self.assertNotIn('"PROCESS": (', face_boundary)

        prepare_source = (ROOT / "pipe" / "cmk_faceprocess_prepare.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"PROCESS": ("CMK_PROCESS_SDXL", {"lazy": True})', prepare_source)
        self.assertIn('accepts only PROCESS SDXL', prepare_source)

    def test_disabled_detailer_uses_static_passthrough_without_cache_analysis(self):
        source = (ROOT / "pipe" / "cmk_module_boundary_cache.py").read_text(
            encoding="utf-8"
        )
        start = source.index("class CMKDetailerBoundaryCache:")
        end = source.index("class CMKZImageBoundaryCache:")
        detailer_boundary = source[start:end]
        self.assertNotIn('"PROCESS": (', detailer_boundary)
        self.assertIn("_detailer_disabled_state", detailer_boundary)
        self.assertIn("DISABLED PASSTHROUGH -> CACHE SKIPPED", detailer_boundary)
        self.assertIn('return MODEL, IMAGE, LOG', detailer_boundary)
        lazy_body = detailer_boundary[detailer_boundary.index("def check_lazy_status("):]
        disabled_pos = lazy_body.index("if module_disabled:")
        cache_pos = lazy_body.index("cache_key, detail = self._cache_key")
        self.assertLess(disabled_pos, cache_pos)

    def test_detailer_cache_miss_materializes_image_before_public_model(self):
        source = (ROOT / "pipe" / "cmk_module_boundary_cache.py").read_text(
            encoding="utf-8"
        )
        start = source.index("class CMKDetailerBoundaryCache:")
        end = source.index("class CMKZImageBoundaryCache:")
        detailer_boundary = source[start:end]
        lazy_body = detailer_boundary[detailer_boundary.index("def check_lazy_status("):]
        self.assertIn('if branch_needed:\n            return branch_needed', lazy_body)
        self.assertIn('if MODEL is None:\n            return ["MODEL"]', lazy_body)
        self.assertLess(
            lazy_body.index("if branch_needed:"),
            lazy_body.index("if MODEL is None:\n            return [\"MODEL\"]", lazy_body.index("if branch_needed:")),
        )

    def test_detailer_prepare_materializes_upstream_image_before_model(self):
        source = (ROOT / "pipe" / "cmk_detailer_prepare.py").read_text(
            encoding="utf-8"
        )
        start = source.index("def check_lazy_status(")
        end = source.index("@staticmethod", start)
        lazy_body = source[start:end]
        self.assertIn("if upstream_needed:\n            return upstream_needed", lazy_body)
        self.assertIn("if not bool(detailer_global_enable):\n            return []", lazy_body)
        self.assertIn('model_needed.append("MODEL")', lazy_body)
        self.assertLess(
            lazy_body.index("if upstream_needed:"),
            lazy_body.index('model_needed.append("MODEL")'),
        )

    def test_detailer_inherits_prompt_lora_and_sampling_independently(self):
        source = (ROOT / "pipe" / "cmk_detailer_prepare.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"use_prompt_lora_from_sampler": ("BOOLEAN", {"default": True})', source)
        self.assertIn('"use_lora_from_1st_pass": ("BOOLEAN", {"default": False})', source)
        self.assertIn('"use_1st_pass_sampling": ("BOOLEAN", {"default": True})', source)
        self.assertIn('source_pipe.get("sampler", sampler)', source)
        self.assertIn('source_pipe.get("scheduler", scheduler)', source)
        self.assertIn('source_pipe.get("sampling", sampling)', source)
        self.assertIn('source_pipe.get("zsnr", zsnr)', source)

        frontend = (
            ROOT / "web" / "js" / "cmk_detailer_prepare_labels.js"
        ).read_text(encoding="utf-8")
        self.assertIn('USE PROMPT FROM 1ST PASS', frontend)
        self.assertIn('USE LORA FROM 1ST PASS', frontend)
        self.assertIn('USE SAMPLING FROM 1ST PASS', frontend)
        self.assertIn('migrated.splice(4, 0, false, true)', frontend)

    def test_native_image_compare_is_proxied_and_press_hold(self):
        backend = (
            ROOT / "nodes" / "utils" / "native_flow_helpers.py"
        ).read_text(encoding="utf-8")
        frontend = (
            ROOT / "web" / "js" / "cmk_image_compare_hold.js"
        ).read_text(encoding="utf-8")
        self.assertIn('return {"optional": {"image_a": ("IMAGE",), "image_b": ("IMAGE",)}}', backend)
        self.assertNotIn('CMK_IMAGE_COMPARE', backend)
        self.assertNotIn("ResizeObserver", frontend)
        self.assertIn('"cmk_compare_images": [source_info, result_info]', backend)
        self.assertIn('"images": [source_info, result_info]', backend)
        self.assertIn('VIEWPORT_SELECTOR = \'[data-testid="image-compare-viewport"]\'', frontend)
        self.assertIn('viewport.classList.add("cmk-hold-compare")', frontend)
        self.assertIn('viewport.classList.add("cmk-show-source")', frontend)
        self.assertIn('viewport.classList.remove("cmk-show-source")', frontend)

    def test_faceprocess_inherits_prompt_lora_and_sampling_independently(self):
        source = (ROOT / "pipe" / "cmk_faceprocess_prepare.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"use_prompt_lora_from_sampler": ("BOOLEAN", {"default": True})', source)
        self.assertIn('"use_lora_from_1st_pass": ("BOOLEAN", {"default": False})', source)
        self.assertIn('"use_1st_pass_sampling": ("BOOLEAN", {"default": True})', source)
        self.assertIn('source_pipe.get("sampler", sampler)', source)
        self.assertIn('source_pipe.get("scheduler", scheduler)', source)
        self.assertIn('source_pipe.get("sampling", sampling)', source)
        self.assertIn('source_pipe.get("zsnr", zsnr)', source)

        frontend = (
            ROOT / "web" / "js" / "cmk_faceprocess_prepare_labels.js"
        ).read_text(encoding="utf-8")
        self.assertIn('USE PROMPT FROM 1ST PASS', frontend)
        self.assertIn('USE LORA FROM 1ST PASS', frontend)
        self.assertIn('USE SAMPLING FROM 1ST PASS', frontend)
        self.assertIn('migrated.splice(4, 0, false, true)', frontend)

    def test_advanced_faceprocess_keeps_compare_internal_without_routing_through_it(self):
        for filename in ("CMK Flow · 30 FaceProcess SDXL · Advanced.json",):
            graph = json.loads((ROOT / "subgraphs" / filename).read_text(encoding="utf-8"))
            definition = graph["definitions"]["subgraphs"][0]
            compare = next(node for node in definition["nodes"] if node["type"] == "ImageCompare")
            image_link = next(
                link for link in definition["links"]
                if link["target_id"] == -20 and link["target_slot"] == 2
            )
            compare_input = next(
                link for link in definition["links"]
                if link["target_id"] == compare["id"] and link["target_slot"] == 0
            )
            compare_gate = next(
                node for node in definition["nodes"]
                if node["id"] == compare_input["origin_id"]
                and node["type"] == "CMKImageCompareEnableGate"
            )
            result_link_id = next(
                item["link"] for item in compare_gate["inputs"]
                if item["name"] == "IMAGE A"
            )
            result_input = next(
                link for link in definition["links"]
                if link["id"] == result_link_id
            )
            bypass_gate = next(
                node for node in definition["nodes"]
                if node["type"] == "CMKModuleBypassGate"
            )
            active_image_link_id = next(
                item["link"] for item in bypass_gate["inputs"]
                if item["name"] == "IMAGE ACTIVE"
            )
            active_image = next(
                link for link in definition["links"]
                if link["id"] == active_image_link_id
            )
            self.assertEqual(active_image["origin_id"], result_input["origin_id"])
            self.assertEqual(active_image["origin_slot"], result_input["origin_slot"])
            self.assertEqual(compare["outputs"], [])
            self.assertNotIn("proxyWidgets", graph["nodes"][0]["properties"])

    def test_refiner_materializes_first_pass_before_loading_refiner_model(self):
        prepare_source = (ROOT / "pipe" / "cmk_refiner_prepare.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"MODEL": ("CMK_MODEL_PIPE", {"lazy": True})', prepare_source)
        self.assertIn('"SAMPLED": ("CMK_SAMPLED_PIPE", {"lazy": True})', prepare_source)
        start = prepare_source.index("def check_lazy_status(")
        end = prepare_source.index("def prepare(", start)
        lazy_body = prepare_source[start:end]
        sampled_pos = lazy_body.index('if SAMPLED is None:')
        model_pos = lazy_body.index('if MODEL is None:')
        self.assertLess(sampled_pos, model_pos)

        boundary_source = (
            ROOT / "pipe" / "cmk_refiner_boundary_cache.py"
        ).read_text(encoding="utf-8")
        start = boundary_source.index("def check_lazy_status(")
        end = boundary_source.index("def boundary(", start)
        boundary_lazy = boundary_source[start:end]
        first_pos = boundary_lazy.index("if IMAGE_1ST_PASS is None:")
        refined_pos = boundary_lazy.index("if IMAGE_REFINED is None:")
        public_model_pos = boundary_lazy.index("if MODEL is None:")
        log_pos = boundary_lazy.index("if LOG is None:")
        self.assertLess(public_model_pos, first_pos)
        self.assertLess(log_pos, first_pos)
        self.assertLess(first_pos, refined_pos)

    def test_refiner_boundary_has_no_process_port(self):
        source = (ROOT / "pipe" / "cmk_refiner_boundary_cache.py").read_text(
            encoding="utf-8"
        )
        class_source = source[source.index("class CMKRefinerBoundaryCache:"):]
        self.assertNotIn('"PROCESS": (', class_source)
        self.assertNotIn('        "PROCESS",', class_source)

        definition = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · 20 Refiner SDXL.json").read_text(
                encoding="utf-8"
            )
        )["definitions"]["subgraphs"][0]
        node = next(
            node for node in definition["nodes"]
            if node["type"] == "CMKRefinerBoundaryCache"
        )
        self.assertEqual(
            [item["name"] for item in node["inputs"]],
            ["MODEL", "IMAGE_1ST_PASS", "IMAGE_REFINED", "LOG"],
        )
        self.assertEqual(
            [item["name"] for item in node["outputs"]],
            ["MODEL", "IMAGE 1ST PASS", "IMAGE REFINED", "LOG"],
        )
        for slot, item in enumerate(node["inputs"]):
            if item.get("link") is None:
                continue
            link = next(link for link in definition["links"] if link["id"] == item["link"])
            self.assertEqual(link["target_slot"], slot)
        for slot, item in enumerate(node["outputs"]):
            for link_id in item.get("links") or []:
                link = next(link for link in definition["links"] if link["id"] == link_id)
                self.assertEqual(link["origin_slot"], slot)

    def test_pipe_upscaler_releases_diffusion_memory_before_model_load(self):
        source = (ROOT / "nodes" / "image" / "smart_upscale.py").read_text(
            encoding="utf-8"
        )
        pipe_start = source.index("class CMK_SmartUpscalerPipe")
        pipe_source = source[pipe_start:]
        self.assertIn("model_management.unload_all_models()", pipe_source)
        self.assertIn("model_management.soft_empty_cache(True)", pipe_source)
        release_call = pipe_source.index("self._release_diffusion_memory()")
        model_load = pipe_source.index("model = self.load_upscale_model", release_call)
        self.assertLess(release_call, model_load)

    def test_expensive_finish_and_loader_phases_have_timing_markers(self):
        timing_source = (ROOT / "utils" / "cmk_timing.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("[CMK Timing] START", timing_source)
        self.assertIn("[CMK Timing] END", timing_source)
        self.assertIn("isoformat(timespec=\"milliseconds\")", timing_source)
        expected = {
            ROOT / "loader" / "checkpoint_vae_loader.py": "CHECKPOINT LOAD",
            ROOT / "pipe" / "cmk_pipe_sampler.py": "10 KSAMPLER SAMPLE",
            ROOT / "pipe" / "cmk_refiner.py": "20 REFINER SAMPLE",
            ROOT / "nodes" / "image" / "smart_upscale.py": "90 UPSCALE MODEL LOAD",
            ROOT / "nodes" / "io" / "save_project_image.py": "90 PNG SAVE",
        }
        for path, marker in expected.items():
            with self.subTest(path=path.name):
                self.assertIn(marker, path.read_text(encoding="utf-8"))
        self.assertIn(
            '@cmk_timed_call("10 SAMPLER PREPARE")',
            (ROOT / "pipe" / "cmk_sampler_prepare.py").read_text(encoding="utf-8"),
        )
        self.assertIn(
            '@cmk_timed_call("20 REFINER PREPARE")',
            (ROOT / "pipe" / "cmk_refiner_prepare.py").read_text(encoding="utf-8"),
        )

    def test_family_subgraphs_gate_computed_outputs_with_materialized_process_signal(self):
        for filename, gate_type in self.FAMILY_GATED_SUBGRAPHS.items():
            with self.subTest(filename=filename):
                document = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )
                definition = document["definitions"]["subgraphs"][0]
                gate = next(
                    node for node in definition["nodes"] if node["type"] == gate_type
                )
                process_slot = next(
                    index for index, item in enumerate(definition["inputs"])
                    if item["name"] == "PROCESS"
                )
                process_gate_slot = next(
                    index for index, item in enumerate(gate["inputs"])
                    if item["name"] == "PROCESS"
                )
                process_link = next(
                    link for link in definition["links"]
                    if link["target_id"] == gate["id"]
                    and link["target_slot"] == process_gate_slot
                )
                if filename == "CMK Flow · 10 KSampler Z-Image Turbo.json":
                    boundary = next(
                        node for node in definition["nodes"]
                        if node["type"] == "CMKZImageBoundaryCache"
                    )
                    self.assertEqual(process_link["origin_id"], -10)
                    self.assertEqual(process_link["origin_slot"], process_slot)
                    process_selector = next(
                        node for node in definition["nodes"]
                        if node["type"] == "CMKZImageProcessForwardPipe"
                    )
                    self.assertEqual(
                        [item["name"] for item in process_selector["inputs"]],
                        ["PROCESS"],
                    )
                else:
                    allowed_process_sources = {-10} | {
                        node["id"] for node in definition["nodes"]
                        if node["type"] in {
                            "CMKProcessForwardPipe",
                            "CMKRefinerBoundaryCache",
                            "CMKDetailerBoundaryCache",
                        }
                    }
                    self.assertIn(process_link["origin_id"], allowed_process_sources)
                    if process_link["origin_id"] == -10:
                        self.assertEqual(process_link["origin_slot"], process_slot)
                for output_slot in range(len(definition["outputs"])):
                    output_name = definition["outputs"][output_slot]["name"]
                    output_link = next(
                        link for link in definition["links"]
                        if link["target_id"] == -20
                        and link["target_slot"] == output_slot
                    )
                    if output_name == "diagnostic":
                        self.assertNotEqual(output_link["origin_id"], gate["id"])
                    elif output_name == "VISUAL":
                        visual_nodes = {
                            node["id"]
                            for node in definition["nodes"]
                            if node["type"] in {"CMKVisualPass", "CMKVisualProvider"}
                        }
                        self.assertIn(output_link["origin_id"], visual_nodes)
                    elif output_slot == 1:
                        if filename == "CMK Flow · 10 KSampler Z-Image Turbo.json":
                            self.assertEqual(output_link["origin_id"], process_selector["id"])
                            self.assertEqual(output_link["origin_slot"], 0)
                        else:
                            process_output_sources = {gate["id"]} | {
                                node["id"] for node in definition["nodes"]
                                if node["type"] in {
                                    "CMKProcessForwardPipe",
                                    "CMKRefinerBoundaryCache",
                                    "CMKDetailerBoundaryCache",
                                }
                            }
                            self.assertIn(output_link["origin_id"], process_output_sources)
                    elif output_slot == 2 and any(
                        node["type"] == "ImageCompare"
                        for node in definition["nodes"]
                    ):
                        compare = next(
                            node for node in definition["nodes"]
                            if node["type"] == "ImageCompare"
                        )
                        compare_input = next(
                            link for link in definition["links"]
                            if link["target_id"] == compare["id"]
                            and link["target_slot"] == 0
                        )
                        compare_gate = next(
                            node for node in definition["nodes"]
                            if node["id"] == compare_input["origin_id"]
                            and node["type"] == "CMKImageCompareEnableGate"
                        )
                        result_link_id = next(
                            item["link"] for item in compare_gate["inputs"]
                            if item["name"] == "IMAGE A"
                        )
                        result_input = next(
                            link for link in definition["links"]
                            if link["id"] == result_link_id
                        )
                        if result_input["origin_id"] == output_link["origin_id"]:
                            authoritative = output_link
                        elif output_link["origin_id"] == gate["id"]:
                            gate_image_slot = next(
                                index for index, item in enumerate(gate["inputs"])
                                if item["name"] == "IMAGE"
                            )
                            authoritative = next(
                                link for link in definition["links"]
                                if link["target_id"] == gate["id"]
                                and link["target_slot"] == gate_image_slot
                            )
                        else:
                            authoritative = output_link
                        self.assertEqual(authoritative["origin_id"], result_input["origin_id"])
                        self.assertEqual(authoritative["origin_slot"], result_input["origin_slot"])
                    elif output_slot == 2 and any(
                        node["type"] == "CMKImagePreviewForward"
                        for node in definition["nodes"]
                    ):
                        preview = next(
                            node for node in definition["nodes"]
                            if node["type"] == "CMKImagePreviewForward"
                        )
                        self.assertEqual(output_link["origin_id"], preview["id"])
                        preview_input = next(
                            link for link in definition["links"]
                            if link["target_id"] == preview["id"]
                        )
                        self.assertEqual(preview_input["origin_id"], gate["id"])
                    else:
                        self.assertEqual(output_link["origin_id"], gate["id"])

    def test_sdxl_sampler_loader_is_gated_before_loading_the_checkpoint(self):
        document = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · 10 KSampler SDXL 1st Pass.json")
            .read_text(encoding="utf-8")
        )
        definition = document["definitions"]["subgraphs"][0]
        loader = next(
            node for node in definition["nodes"]
            if node["type"] == "CMKCheckpointVAELoaderPipe"
        )
        loader_process_slot = next(
            index for index, item in enumerate(loader["inputs"])
            if item["name"] == "PROCESS"
        )
        process_link = next(
            link for link in definition["links"]
            if link["target_id"] == loader["id"]
            and link["target_slot"] == loader_process_slot
        )
        public_process_slot = next(
            index for index, item in enumerate(definition["inputs"])
            if item["name"] == "PROCESS"
        )
        self.assertEqual(process_link["origin_id"], -10)
        self.assertEqual(process_link["origin_slot"], public_process_slot)

    def test_inactive_family_gate_does_not_request_computed_inputs(self):
        path = ROOT / "pipe" / "cmk_family_result.py"
        spec = importlib.util.spec_from_file_location("cmk_family_gate_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for gate_class in (
            module.CMKFamilyBranchGateSDXL,
            module.CMKFamilyBranchGateSDXLSampled,
            module.CMKFamilyBranchGateZImage,
        ):
            with self.subTest(gate=gate_class.__name__):
                gate = gate_class()
                self.assertEqual(gate.check_lazy_status(PROCESS=None), ["PROCESS"])
                result = gate.gate(PROCESS=None)
                self.assertTrue(all(
                    isinstance(value, module.ExecutionBlocker)
                    for value in result
                ))

    def test_only_sampled_sdxl_gate_exposes_materialized_process_output(self):
        path = ROOT / "pipe" / "cmk_family_result.py"
        spec = importlib.util.spec_from_file_location("cmk_family_ports_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for gate_class in (
            module.CMKFamilyBranchGateSDXL,
            module.CMKFamilyBranchGateSDXLSampled,
            module.CMKFamilyBranchGateZImage,
        ):
            with self.subTest(gate=gate_class.__name__):
                if gate_class is module.CMKFamilyBranchGateSDXLSampled:
                    self.assertIn("PROCESS", gate_class.RETURN_NAMES)
                else:
                    self.assertNotIn("PROCESS", gate_class.RETURN_NAMES)
                self.assertNotIn(
                    "RESULT PROCESS",
                    gate_class.INPUT_TYPES().get("optional", {}),
                )

    def test_zit_process_forward_does_not_wait_for_sampled_result(self):
        path = ROOT / "pipe" / "cmk_family_result.py"
        spec = importlib.util.spec_from_file_location("cmk_zit_forward_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        forward = module.CMKZImageProcessForwardPipe()
        process = {
            "model_family": "z_image_turbo",
            "family_active": True,
            "width": 768,
            "height": 512,
        }
        self.assertEqual(forward.check_lazy_status(PROCESS=process), [])
        result, = forward.forward(PROCESS=process)
        self.assertIs(result, process)

    def test_family_gate_saved_input_order_matches_runtime_contract(self):
        for filename, gate_type in self.FAMILY_GATED_SUBGRAPHS.items():
            with self.subTest(filename=filename):
                definition = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )["definitions"]["subgraphs"][0]
                gate = next(
                    node for node in definition["nodes"] if node["type"] == gate_type
                )
                expected = (
                    ["MODEL", "PROCESS", "SAMPLED", "LOG"]
                    if gate_type == "CMKFamilyBranchGateSDXLSampled"
                    else ["MODEL", "PROCESS", "IMAGE", "LOG"]
                )
                self.assertEqual([item["name"] for item in gate["inputs"]], expected)
                for slot, item in enumerate(gate["inputs"]):
                    if item.get("link") is None:
                        continue
                    link = next(
                        link for link in definition["links"] if link["id"] == item["link"]
                    )
                    self.assertEqual(link["target_slot"], slot)

    def test_merge_requests_only_the_selected_lazy_branch(self):
        path = ROOT / "pipe" / "cmk_family_result.py"
        spec = importlib.util.spec_from_file_location("cmk_family_result_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        merge = module.CMKFamilyResultMergePipe()
        self.assertEqual(
            merge.check_lazy_status(),
            ["PROCESS ZIT"],
        )
        self.assertEqual(
            merge.check_lazy_status(**{
                "PROCESS SDXL": {"model_family": "sdxl", "family_active": False},
            }),
            ["PROCESS ZIT"],
        )
        self.assertEqual(
            merge.check_lazy_status(**{
                "PROCESS SDXL": {"model_family": "sdxl", "family_active": True},
                "PROCESS ZIT": {"model_family": "z_image_turbo", "family_active": False},
            }),
            ["MODEL SDXL"],
        )
        self.assertEqual(
            merge.check_lazy_status(**{
                "PROCESS SDXL": {"model_family": "sdxl", "family_active": False},
                "PROCESS ZIT": {"model_family": "z_image_turbo", "family_active": True},
            }),
            ["MODEL ZIT"],
        )

        zit_inputs = {
            "PROCESS SDXL": {"model_family": "sdxl", "family_active": False},
            "PROCESS ZIT": {"model_family": "z_image_turbo", "family_active": True},
            "MODEL ZIT": {},
        }
        self.assertEqual(merge.check_lazy_status(**zit_inputs), ["IMAGE ZIT"])
        zit_inputs["IMAGE ZIT"] = object()
        self.assertEqual(merge.check_lazy_status(**zit_inputs), ["LOG ZIT"])

    def test_image_family_gate_resolves_converging_inputs_sequentially(self):
        path = ROOT / "pipe" / "cmk_family_result.py"
        spec = importlib.util.spec_from_file_location("cmk_image_gate_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        gate = module.CMKFamilyBranchGateZImage()
        process = {"model_family": "z_image_turbo", "family_active": True}
        self.assertEqual(gate.check_lazy_status(PROCESS=process), ["MODEL"])
        self.assertEqual(
            gate.check_lazy_status(PROCESS=process, MODEL={}), ["IMAGE"]
        )
        self.assertEqual(
            gate.check_lazy_status(PROCESS=process, MODEL={}, IMAGE=object()),
            ["LOG"],
        )

    def test_sampled_family_gate_resolves_converging_inputs_sequentially(self):
        path = ROOT / "pipe" / "cmk_family_result.py"
        spec = importlib.util.spec_from_file_location("cmk_sampled_gate_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        gate = module.CMKFamilyBranchGateSDXLSampled()
        process = {"model_family": "sdxl", "family_active": True}

        self.assertEqual(gate.check_lazy_status(PROCESS=process), ["MODEL"])
        self.assertEqual(
            gate.check_lazy_status(PROCESS=process, MODEL={}),
            ["SAMPLED"],
        )
        self.assertEqual(
            gate.check_lazy_status(PROCESS=process, MODEL={}, SAMPLED={}),
            ["LOG"],
        )

    def test_first_pass_visual_prefers_clean_instantid_x0(self):
        source = (ROOT / "pipe" / "cmk_family_result.py").read_text(encoding="utf-8")
        self.assertIn('sampled.get("instantid_keypoints_latent")', source)
        self.assertIn('sampled.get("latent_1st_pass", sampled.get("latent_image"))', source)

    def test_public_diagnostics_bypass_family_gates(self):
        for filename in (
            "CMK Flow · 10 KSampler SDXL 1st Pass.json",
            "CMK Flow · 10 KSampler Z-Image Turbo.json",
            "CMK Flow · 20 Refiner SDXL.json",
            "CMK Flow · 23 Detailer SDXL.json",
            "CMK Flow · 23 Detailer SDXL · Advanced.json",
        ):
            with self.subTest(filename=filename):
                definition = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )["definitions"]["subgraphs"][0]
                gate_ids = {
                    node["id"]
                    for node in definition["nodes"]
                    if node["type"].startswith("CMKFamilyBranchGate")
                }
                diagnostic_output = next(
                    item for item in definition["outputs"]
                    if item["name"] == "diagnostic"
                )
                public_link = next(
                    link for link in definition["links"]
                    if link["id"] in diagnostic_output["linkIds"]
                )
                self.assertNotIn(public_link["origin_id"], gate_ids)

                for gate in (
                    node for node in definition["nodes"] if node["id"] in gate_ids
                ):
                    self.assertNotIn("diagnostic", {item["name"] for item in gate["inputs"]})
                    self.assertNotIn("diagnostic", {item["name"] for item in gate["outputs"]})

    def test_faceprocess_branch_loads_lazy_cache_before_requiring_face(self):
        source = (ROOT / "pipe" / "cmk_faceprocess.py").read_text(encoding="utf-8")
        run_source = source[source.index("    def run_pipe("):]
        self.assertIn("FACE=None", run_source[:250])
        cache_load = run_source.index("load_pickle(self._CACHE_SCOPE, cache_key)")
        face_validation = run_source.index(
            'raise ValueError("CMK FaceProcess -Pipe-: FACE is missing")'
        )
        self.assertLess(cache_load, face_validation)

    def test_faceprocess_detailer_center_uses_its_own_detected_segs(self):
        pipe_source = (ROOT / "pipe" / "cmk_faceprocess.py").read_text(encoding="utf-8")
        engine_source = (ROOT / "nodes" / "swap" / "face_process.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('if str(select_face) == "Center":', pipe_source)
        self.assertIn('"select_face_selection": "center"', pipe_source)
        self.assertIn('if normalized_selection == "center":', engine_source)
        self.assertIn("image=image,\n                selection=select_face_selection", engine_source)

    def test_faceprocess_restore_center_is_resolved_by_restore_detector(self):
        pipe_source = (ROOT / "pipe" / "cmk_faceprocess.py").read_text(encoding="utf-8")
        restore_source = (ROOT / "engine" / "native_face_restore.py").read_text(encoding="utf-8")

        self.assertIn('if str(select_face) == "Center":', pipe_source)
        self.assertIn('if str(selection) == "center":', restore_source)
        self.assertIn("image_shape=original.shape", restore_source)
        self.assertIn("distance_from_image_center", restore_source)
        self.assertIn("cmk_faceprocess_branch_v7", pipe_source)

    def test_externally_driven_faceprocess_mode_serializes_both_parameter_sets(self):
        frontend = (
            ROOT / "web" / "js" / "cmk_faceprocess_pipe_ui_v6.js"
        ).read_text(encoding="utf-8")
        self.assertIn("function processModeIsExternallyDriven(node)", frontend)
        self.assertIn('item?.name === "process_mode"', frontend)
        self.assertIn('return processModeIsExternallyDriven(node) ? "external"', frontend)
        self.assertIn('mode === "external" || widgetBelongsToMode(name, mode)', frontend)

    def test_finish_accepts_direct_family_or_neutral_result_contract(self):
        document = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · 90 Upscale & Save.json").read_text(
                encoding="utf-8"
            )
        )
        definition = document["definitions"]["subgraphs"][0]
        process_input = next(
            item for item in definition["inputs"] if item["name"] == "PROCESS"
        )
        self.assertEqual(process_input["type"], "*")

        bridge = next(
            node
            for node in definition["nodes"]
            if node["type"] == "CMKResultUnpackPipe"
        )
        self.assertEqual([item["type"] for item in bridge["inputs"]], ["*", "*", "*", "*"])
        self.assertIn("CMKResultPackPipe", {node["type"] for node in definition["nodes"]})
        self.assertEqual(definition["outputs"][2]["type"], "IMAGE")
        self.assertEqual(definition["outputs"][3]["type"], "CMK_LOG_PIPE")

        path = ROOT / "pipe" / "cmk_family_result.py"
        spec = importlib.util.spec_from_file_location("cmk_finish_input_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        unpack = module.CMKResultUnpackPipe.unpack
        for family in ("sdxl", "z_image_turbo"):
            with self.subTest(family=family):
                model, process, image, log = unpack(
                    {"model_family": family},
                    {"model_family": family, "family_active": True},
                    object(),
                    {"blocks": []},
                )
                self.assertEqual(process["result_contract"], "family_neutral")
                self.assertEqual(process["source_model_family"], family)

        model, process, image, log = unpack(
            None,
            {
                "result_contract": "family_neutral",
                "source_model_family": "image",
                "pipe_origin": "CMK Image Load and Resize -Pipe-",
            },
            object(),
            {"blocks": []},
        )
        self.assertIsNone(model)
        self.assertEqual(process["source_model_family"], "image")

        model, process, image, log = unpack(
            {"model_family": "sdxl"},
            {
                "result_contract": "family_neutral",
                "source_model_family": "image",
                "pipe_origin": "CMK Image Load and Resize -Pipe-",
            },
            object(),
            {"blocks": []},
        )
        self.assertEqual(model["model_family"], "sdxl")
        self.assertEqual(process["source_model_family"], "image")

        with self.assertRaisesRegex(ValueError, "different families"):
            unpack(
                {"model_family": "z_image_turbo"},
                {
                    "result_contract": "family_neutral",
                    "source_model_family": "image",
                    "pipe_origin": "CMK Image Load and Resize -Pipe-",
                },
                object(),
                {"blocks": []},
            )

    def test_faceswap_accepts_direct_family_and_outputs_neutral_contract(self):
        for filename in (
            "CMK Flow · 40 FaceSwap.json",
            "CMK Flow · 40 FaceSwap · Advanced.json",
        ):
            with self.subTest(filename=filename):
                document = json.loads(
                    (ROOT / "subgraphs" / filename).read_text(encoding="utf-8")
                )
                definition = document["definitions"]["subgraphs"][0]
                self.assertEqual(
                    next(item for item in definition["inputs"] if item["name"] == "PROCESS")["type"],
                    "*",
                )
                unpack = next(
                    node for node in definition["nodes"]
                    if node["type"] == "CMKResultUnpackPipe"
                )
                self.assertEqual(
                    [item["type"] for item in unpack["inputs"]],
                    ["*", "*", "*", "*"],
                )
                self.assertEqual(
                    next(item for item in definition["outputs"] if item["name"] == "PROCESS")["type"],
                    "CMK_RESULT_PROCESS",
                )


if __name__ == "__main__":
    unittest.main()
