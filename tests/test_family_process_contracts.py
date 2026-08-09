import json
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FamilyProcessContractTests(unittest.TestCase):
    SDXL_SUBGRAPHS = (
        "CMK Flow · 10 KSampler SDXL 1st Pass.json",
        "CMK Flow · 20 Refiner SDXL.json",
        "CMK Flow · 25 Detailer SDXL.json",
        "CMK Flow · 25 Detailer SDXL · Advanced.json",
        "CMK Flow · 30 FaceProcess SDXL.json",
        "CMK Flow · 30 FaceProcess SDXL · Advanced.json",
    )

    FAMILY_GATED_SUBGRAPHS = {
        "CMK Flow · 10 KSampler SDXL 1st Pass.json": "CMKFamilyBranchGateSDXLSampled",
        "CMK Flow · 20 Refiner SDXL.json": "CMKFamilyBranchGateSDXL",
        "CMK Flow · 25 Detailer SDXL.json": "CMKFamilyBranchGateSDXL",
        "CMK Flow · 25 Detailer SDXL · Advanced.json": "CMKFamilyBranchGateSDXL",
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
            "CMK Flow · 25 Detailer SDXL.json",
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
            "CMK Flow · 25 Detailer SDXL.json",
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
        self.assertIn('"optional": {\n                "MODEL": (CMK_FINISH_INPUT,)', result_source)
        self.assertIn('def unpack(MODEL=None, PROCESS=None, IMAGE=None, LOG=None):', result_source)
        self.assertIn('def pack(MODEL=None, PROCESS=None, IMAGE=None, LOG=None):', result_source)

        boundary_source = (
            ROOT / "pipe" / "cmk_module_boundary_cache.py"
        ).read_text(encoding="utf-8")
        face_start = boundary_source.index("class CMKFaceSwapBoundaryCache:")
        face_boundary = boundary_source[face_start:]
        self.assertIn('"optional": {\n                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True})', face_boundary)
        self.assertNotIn('(\"MODEL\", MODEL),\n            (\"PROCESS\", PROCESS)', face_boundary)

        save_source = (ROOT / "nodes" / "io" / "save_project_image.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"optional": {\n                "MODEL": ("CMK_MODEL_PIPE",)', save_source)

    def test_optional_model_nodes_use_runtime_input_order_in_saved_subgraphs(self):
        expected = {
            "CMKResultUnpackPipe": ["PROCESS", "IMAGE", "LOG", "MODEL"],
            "CMKResultPackPipe": ["PROCESS", "IMAGE", "LOG", "MODEL"],
            "CMKFaceSwapBoundaryCache": ["PROCESS", "IMAGE", "LOG", "MODEL"],
            "CMK_SaveProjectImage": [
                "PROCESS",
                "IMAGE",
                "LOG",
                "SAVE ENABLED",
                "FILENAME PREFIX",
                "OUTPUT FOLDER",
                "USE DATE FOLDER",
                "PROJECT FOLDER",
                "MODEL",
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

    def test_shared_module_public_inputs_place_optional_model_after_result_path(self):
        expected = {
            "CMK Flow · 40 FaceSwap.json": [
                "PROCESS", "IMAGE_TARGET", "LOG", "MODEL", "ENABLE"
            ],
            "CMK Flow · 40 FaceSwap · Advanced.json": [
                "PROCESS", "IMAGE_TARGET", "LOG", "MODEL", "ENABLE"
            ],
            "CMK Flow · 90 Upscale & Save.json": [
                "PROCESS", "IMAGE", "LOG", "MODEL", "SAVE ENABLED",
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
            "CMK Flow · 25 Detailer SDXL.json": 25,
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

        detailer_after = metadata["CMK Flow · 25 Detailer SDXL.json"]["recommendedAfter"]
        face_after = metadata["CMK Flow · 30 FaceProcess SDXL.json"]["recommendedAfter"]
        self.assertIn("30 FaceProcess SDXL", detailer_after)
        self.assertNotIn("40 FaceSwap", metadata["CMK Flow · 30 FaceProcess SDXL.json"]["recommendedBefore"])
        self.assertIn("40 FaceSwap", face_after)

    def test_faceprocess_boundary_is_an_sdxl_only_lazy_family_gate(self):
        source = (ROOT / "pipe" / "cmk_module_boundary_cache.py").read_text(
            encoding="utf-8"
        )
        start = source.index("class CMKFaceBoundaryCache:")
        face_boundary = source[start:]
        self.assertIn('"PROCESS": ("CMK_PROCESS_SDXL", {"lazy": True})', face_boundary)
        return_types = face_boundary[face_boundary.index("RETURN_TYPES = ("):]
        return_types = return_types[:return_types.index(")")]
        self.assertIn('"CMK_MODEL_PIPE"', return_types)
        self.assertIn('"CMK_PROCESS_SDXL"', return_types)
        self.assertIn('if PROCESS is None:\n            return ["PROCESS"]', face_boundary)
        self.assertIn('accepts only PROCESS SDXL', face_boundary)

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
        self.assertIn('if PROCESS is None:\n            return ["PROCESS"]', detailer_boundary)
        self.assertIn("_detailer_disabled_state", detailer_boundary)
        self.assertIn("DISABLED PASSTHROUGH -> CACHE SKIPPED", detailer_boundary)
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

    def test_family_subgraphs_gate_computed_outputs_with_direct_process_signal(self):
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
                    boundary_process_slot = next(
                        index for index, item in enumerate(boundary["inputs"])
                        if item["name"] == "PROCESS"
                    )
                    boundary_link = next(
                        link for link in definition["links"]
                        if link["target_id"] == boundary["id"]
                        and link["target_slot"] == boundary_process_slot
                    )
                    self.assertNotEqual(boundary_link["origin_id"], -10)
                else:
                    self.assertEqual(process_link["origin_id"], -10)
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
                    elif output_slot == 1:
                        if filename == "CMK Flow · 10 KSampler Z-Image Turbo.json":
                            self.assertEqual(output_link["origin_id"], process_selector["id"])
                            self.assertEqual(output_link["origin_slot"], 0)
                        else:
                            forward_types = {
                                "CMKProcessForwardPipe", "CMKZImageProcessForwardPipe"
                            }
                            forward_ids = {
                                node["id"] for node in definition["nodes"]
                                if node["type"] in forward_types
                            }
                            self.assertIn(output_link["origin_id"], forward_ids)
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
                self.assertEqual(gate.check_lazy_status(PROCESS=None), [])
                result = gate.gate(PROCESS=None)
                self.assertTrue(all(
                    isinstance(value, module.ExecutionBlocker)
                    for value in result
                ))

    def test_family_gates_do_not_expose_a_redundant_process_output(self):
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
                self.assertNotIn("PROCESS", gate_class.RETURN_NAMES)
                self.assertNotIn(
                    "RESULT PROCESS",
                    gate_class.INPUT_TYPES().get("optional", {}),
                )

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
                    ["PROCESS", "MODEL", "SAMPLED", "LOG"]
                    if gate_type == "CMKFamilyBranchGateSDXLSampled"
                    else ["PROCESS", "MODEL", "IMAGE", "LOG"]
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
            ["PROCESS SDXL", "PROCESS ZIT"],
        )
        self.assertEqual(
            merge.check_lazy_status(**{
                "PROCESS SDXL": {"model_family": "sdxl", "family_active": True},
                "PROCESS ZIT": {"model_family": "z_image_turbo", "family_active": False},
            }),
            ["MODEL SDXL", "IMAGE SDXL", "LOG SDXL"],
        )
        self.assertEqual(
            merge.check_lazy_status(**{
                "PROCESS SDXL": {"model_family": "sdxl", "family_active": False},
                "PROCESS ZIT": {"model_family": "z_image_turbo", "family_active": True},
            }),
            ["MODEL ZIT", "IMAGE ZIT", "LOG ZIT"],
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

    def test_public_diagnostics_bypass_family_gates(self):
        for filename in (
            "CMK Flow · 10 KSampler SDXL 1st Pass.json",
            "CMK Flow · 10 KSampler Z-Image Turbo.json",
            "CMK Flow · 20 Refiner SDXL.json",
            "CMK Flow · 25 Detailer SDXL.json",
            "CMK Flow · 25 Detailer SDXL · Advanced.json",
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
