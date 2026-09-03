import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InstantIDModuleContractTests(unittest.TestCase):
    def test_sampler_node_has_sdxl_flow_contract(self):
        source = (ROOT / "pipe/instantid/cmk_instantid_sampler.py").read_text(encoding="utf-8")
        self.assertIn('"MODEL": ("CMK_MODEL_PIPE",)', source)
        self.assertIn('"PROCESS": ("CMK_PROCESS_SDXL",)', source)
        self.assertIn('"SAMPLED": ("CMK_SAMPLED_PIPE",)', source)
        self.assertIn('"LOG": ("CMK_LOG_PIPE",)', source)
        self.assertIn('"source_face": ("IMAGE",)', source)
        self.assertIn('"CMK_SAMPLED_PIPE",', source)
        self.assertNotIn("OUTPUT_NODE = True", source)
        self.assertIn("class CMKInstantIDBoundary:", source)
        self.assertIn('"prompt": "PROMPT"', source)
        self.assertIn('"unique_id": "UNIQUE_ID"', source)
        self.assertIn('"dynprompt": "DYNPROMPT"', source)
        self.assertIn("_INSTANTID_BOUNDARY_CACHE", source)
        self.assertIn("CMK InstantID Boundary] HIT", source)
        self.assertIn("_reset_instantid_boundary_cache(dynprompt)", source)

    def test_dependency_is_resolved_only_when_enabled(self):
        source = (ROOT / "pipe/instantid/cmk_instantid_sampler.py").read_text(encoding="utf-8")
        self.assertIn('("CMKInternalInstantIDModelLoader",)', source)
        self.assertIn('("CMKInternalInstantIDFaceAnalysis",)', source)
        self.assertIn('("CMKInternalApplyInstantIDAdvanced",)', source)
        self.assertIn("_sample_with_midpoint_guard(", source)
        self.assertIn('("VAEDecode",)', source)
        self.assertIn('image_kps=layout_image', source)
        self.assertIn("get_content_guard", source)
        self.assertIn("_guard_source_images(faceanalysis, source_face)", source)
        self.assertNotIn("EmptyLatentImage", source)
        self.assertIn('sampler_name=str(sampler_name)', source)
        self.assertIn('scheduler=str(scheduler)', source)
        self.assertIn('["CMK prepared", "InstantID reference"]', source)
        self.assertIn('("CLIPTextEncode",), clip=clean_clip, text=translation_pos.text', source)
        self.assertIn("15 INSTANTID REFERENCE CONDITIONING", source)
        self.assertIn('sampled_in.get("model_identity")', source)
        self.assertIn("source LoRAs + FreeU; no local Hyper-LoRA / PAG / layout sampling patch", source)
        self.assertIn('"instantid_start": ("FLOAT",', source)
        self.assertIn('"instantid_end": ("FLOAT",', source)
        self.assertIn('"tooltip": (', source)
        for stage in (
            "15 INSTANTID KEYPOINTS X0 VAE DECODE",
            "15 INSTANTID FACEANALYSIS LOAD",
            "15 INSTANTID MODEL LOAD",
            "15 INSTANTID CONTROLNET LOAD",
            "15 INSTANTID APPLY",
            "15 INSTANTID KSAMPLER",
        ):
            self.assertIn(stage, source)
        self.assertIn("15 INSTANTID CONTENTGUARD TARGET", source)
        self.assertIn('process.get("instantid_enabled", False)', source)
        self.assertIn('sampled_in.get("steps_1st_pass"', source)
        self.assertIn('sampled_in.get("instantid_end_at_step", 2)', source)
        self.assertIn("callback=guarded_preview_callback", source)
        self.assertIn('add_noise = "enable" if', source)

    def test_start_switch_and_shared_schedule_handoff(self):
        start = (ROOT / "pipe/cmk_pipe_image.py").read_text(encoding="utf-8")
        prepare = (ROOT / "pipe/cmk_sampler_prepare.py").read_text(encoding="utf-8")
        self.assertIn('"instantid_enabled": (', start)
        self.assertIn('"instantid_enabled": bool(', start)
        self.assertIn('"instantid_end_at_step": (', prepare)
        self.assertIn('"min": 1,', prepare)
        self.assertIn('"max": 10,', prepare)
        self.assertIn('effective_sampler = sampler', prepare)
        self.assertIn('effective_scheduler = scheduler', prepare)
        sampler = (ROOT / "pipe/cmk_pipe_sampler.py").read_text(encoding="utf-8")
        self.assertIn('last_step=end_at_step', sampler)
        self.assertIn('force_full_denoise=False', sampler)
        self.assertIn('latent_preview.prepare_callback(model, steps, x0_output)', sampler)
        self.assertIn('callback=preview_callback', sampler)
        self.assertIn('new_pipe["instantid_keypoints_latent"]', sampler)

        instantid = (ROOT / "pipe/instantid/cmk_instantid_sampler.py").read_text(encoding="utf-8")
        self.assertIn('sampled_in.get("steps_1st_pass", sampled_in.get("steps", 20))', instantid)
        self.assertIn('sampled_in.get("instantid_end_at_step", 2)', instantid)
        self.assertIn('sampled_in.get("seed", process.get("seed", 0))', instantid)
        self.assertIn('sampled_in.get("sampler", "euler_ancestral")', instantid)
        self.assertIn('sampled_in.get("scheduler", "karras")', instantid)
        self.assertIn('sampled_in.get("denoise", 1.0)', instantid)
        self.assertIn('cfg=float(cfg)', instantid)

    def test_sampler_prepare_exports_identity_model_without_lcm_sampling(self):
        source = (ROOT / "pipe/cmk_sampler_prepare.py").read_text(encoding="utf-8")
        source_lora_at = source.index("identity_source_model = prepared_model")
        local_lora_at = source.index("self._apply_single_lora", source_lora_at)
        identity_at = source.index("identity_model = identity_source_model", local_lora_at)
        freeu_at = source.index("self._apply_freeu", identity_at)
        sampling_at = source.index("self._apply_sampling", identity_at)
        self.assertLess(source_lora_at, local_lora_at)
        self.assertLess(local_lora_at, identity_at)
        self.assertLess(identity_at, freeu_at)
        self.assertLess(freeu_at, sampling_at)
        self.assertLess(identity_at, sampling_at)
        self.assertIn('new_pipe["model_identity"]', source)

    def test_upstream_runtime_is_vendored_with_license(self):
        upstream = ROOT / "pipe/instantid/upstream"
        for name in ("InstantID.py", "CrossAttentionPatch.py", "resampler.py", "utils.py", "LICENSE", "NOTICE"):
            self.assertTrue((upstream / name).is_file(), name)

    def test_vendored_runtime_uses_current_image_transform_apis(self):
        source = (ROOT / "pipe/instantid/upstream/InstantID.py").read_text(encoding="utf-8")
        self.assertIn("transform_type.from_estimate(lmk, dst)", source)
        self.assertIn("torchvision_functional.to_tensor(image)", source)
        self.assertIn('if hasattr(transform_type, "from_estimate"):', source)
        self.assertNotIn("T.ToTensor()", source)

    def test_brownian_rounding_filter_is_exact_and_sampling_scoped(self):
        sources = "\n".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in (
                "pipe/cmk_pipe_sampler.py",
                "pipe/instantid/cmk_instantid_sampler.py",
                "pipe/cmk_refiner.py",
            )
        )
        self.assertEqual(sources.count("with ignore_torchsde_boundary_rounding():"), 3)
        helper = (ROOT / "utils/cmk_sampling_warnings.py").read_text(encoding="utf-8")
        self.assertIn('module=r"torchsde\\._brownian\\.brownian_interval"', helper)
        self.assertIn("ta>=t0|tb<=t1", helper)
        self.assertNotIn('warnings.simplefilter("ignore")', helper)

    def test_mapping_is_registered(self):
        source = (ROOT / "cmk_mappings.py").read_text(encoding="utf-8")
        self.assertIn('"CMKInstantIDSamplerSDXLPipe": CMKInstantIDSamplerSDXLPipe', source)
        self.assertIn('"CMK InstantID Sampler SDXL -Pipe-"', source)

    def test_sources_are_valid_python(self):
        ast.parse((ROOT / "pipe/instantid/cmk_instantid_sampler.py").read_text(encoding="utf-8"))

    def test_content_guard_decode_is_internal_only(self):
        source = (ROOT / "pipe/instantid/cmk_instantid_sampler.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_guard_generated_latent"
        )
        self.assertTrue(any(
            isinstance(node, ast.Constant) and node.value == "VAEDecode"
            for node in ast.walk(helper)
        ))
        self.assertFalse(any(
            isinstance(node, ast.Return) and node.value is not None
            for node in ast.walk(helper)
        ))
        self.assertIn("del decoded", ast.get_source_segment(source, helper))
        guard_at = source.index("_guard_generated_latent(result, vae, faceanalysis, target_face)")
        pipe_at = source.index("sampled_out = dict(sampled_in)")
        self.assertLess(guard_at, pipe_at)
        sampled_block = source[pipe_at:source.index("lines = [", pipe_at)]
        self.assertNotIn("decoded", sampled_block)

    def test_midpoint_guard_preserves_preview_and_is_early_rejection_only(self):
        source = (ROOT / "pipe/instantid/cmk_instantid_sampler.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        midpoint_guard = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_guard_midpoint_x0"
        )
        guard_source = ast.get_source_segment(source, midpoint_guard)
        self.assertIn("inspect_content", guard_source)
        self.assertNotIn("inspect_image", guard_source)
        self.assertNotIn("faceanalysis", guard_source)
        self.assertFalse(any(
            isinstance(node, ast.Return) and node.value is not None
            for node in ast.walk(midpoint_guard)
        ))
        self.assertIn("del decoded", guard_source)
        snapshot_at = guard_source.index(
            "loaded_models(only_currently_used=True)"
        )
        decode_at = guard_source.index('_call_node(("VAEDecode",)')
        restore_at = guard_source.index("load_models_gpu(loaded_models)")
        self.assertLess(snapshot_at, decode_at)
        self.assertLess(decode_at, restore_at)
        restore_call = next(
            node for node in ast.walk(midpoint_guard)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load_models_gpu"
        )
        self.assertTrue(any(
            restore_call in (
                descendant
                for statement in node.finalbody
                for descendant in ast.walk(statement)
            )
            for node in ast.walk(midpoint_guard)
            if isinstance(node, ast.Try)
        ))

        sampler = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_sample_with_midpoint_guard"
        )
        sampler_source = ast.get_source_segment(source, sampler)
        callback_at = sampler_source.index("preview_callback(step, x0, x, total_steps)")
        guard_at = sampler_source.index("_guard_midpoint_x0(x0_latent, vae)")
        self.assertLess(callback_at, guard_at)
        self.assertIn("model.model.process_latent_out(x0.detach().cpu())", sampler_source)
        self.assertIn("callback=guarded_preview_callback", sampler_source)
        self.assertEqual(sampler_source.count("_guard_midpoint_x0("), 1)

        final_guard_at = source.index(
            "_guard_generated_latent(result, vae, faceanalysis, target_face)"
        )
        sampled_pipe_at = source.index("sampled_out = dict(sampled_in)")
        self.assertLess(final_guard_at, sampled_pipe_at)

    def test_guard_approved_visual_image_never_becomes_a_public_image_port(self):
        import json

        document = json.loads(
            (ROOT / "subgraphs/CMK Flow · 15 InstantID-Sampler SDXL.json").read_text(
                encoding="utf-8"
            )
        )
        definition = document["definitions"]["subgraphs"][0]
        self.assertNotIn("IMAGE", [item["type"] for item in definition["outputs"]])
        sampler = next(
            node for node in definition["nodes"]
            if node["type"] == "CMKInstantIDSamplerSDXLPipe"
        )
        image_output = next(item for item in sampler["outputs"] if item["type"] == "IMAGE")
        provider = next(node for node in definition["nodes"] if node["type"] == "CMKVisualProvider")
        visual_link = next(link for link in definition["links"] if link["id"] == image_output["links"][0])
        self.assertEqual(visual_link["target_id"], provider["id"])
        self.assertEqual(visual_link["target_slot"], 0)

    def test_flow_metadata_declares_dependency_and_placement(self):
        import json

        metadata = json.loads((ROOT / "web/flow_node_metadata.json").read_text(encoding="utf-8"))
        item = metadata["nodes"]["CMKInstantIDSamplerSDXLPipe"]
        self.assertEqual(item["status"], "STABLE")
        self.assertEqual(item["version"], "1.0.0")
        self.assertIn("von CMK Nodes mitinstalliert", item["dependencyNote"])
        self.assertNotIn("separate InstantID-Custom-Node", item["dependencyNote"])

        english = json.loads(
            (ROOT / "web/browser_content_en.json").read_text(encoding="utf-8")
        )["flows"]
        for key in ("CMKInstantIDSamplerSDXLPipe", "CMK Flow · 15 InstantID-Sampler SDXL"):
            self.assertIn(key, english)
            self.assertTrue(english[key]["description"])
            self.assertTrue(english[key]["features"])
            self.assertTrue(english[key]["placementNote"])
            self.assertTrue(english[key]["dependencyNote"])

    def test_subgraph_has_only_canonical_ports_and_internal_source_face(self):
        import json

        path = ROOT / "subgraphs/CMK Flow · 15 InstantID-Sampler SDXL.json"
        workflow = json.loads(path.read_text(encoding="utf-8"))
        outer = workflow["nodes"][0]
        self.assertEqual(outer["size"], [600, 1225])
        self.assertEqual([item["name"] for item in outer["inputs"]], ["MODEL", "PROCESS", "SAMPLED", "LOG", "VISUAL"])
        self.assertEqual([item["name"] for item in outer["outputs"]], ["MODEL", "PROCESS", "SAMPLED", "LOG", "VISUAL", "diagnostic"])
        self.assertEqual(outer["properties"]["proxyWidgets"], [["1501", "image"]])
        self.assertEqual(outer["properties"]["cmkVisualProviders"][0]["label"], "Identity")
        definition = workflow["definitions"]["subgraphs"][0]
        flow_metadata = workflow["extra"]["CMKFlow"]
        self.assertEqual(flow_metadata["status"], "STABLE")
        self.assertEqual(flow_metadata["version"], "1.0.0")
        types = {node["type"] for node in definition["nodes"]}
        self.assertIn("CMKLoadImage", types)
        self.assertIn("CMKInstantIDSamplerSDXLPipe", types)
        self.assertIn("CMKInstantIDBoundary", types)
        self.assertIn("CMKVisualProvider", types)
        self.assertNotIn("PreviewImage", types)
        sampler = next(node for node in definition["nodes"] if node["type"] == "CMKInstantIDSamplerSDXLPipe")
        input_names = [item["name"] for item in sampler["inputs"]]
        self.assertIn("cfg", input_names)
        self.assertIn("instantid_start", input_names)
        self.assertIn("instantid_end", input_names)
        self.assertNotIn("start", input_names)
        self.assertEqual(
            [item["name"] for item in sampler["outputs"][-2:]],
            ["IDENTITY IMAGE", "diagnostic"],
        )
        provider = next(node for node in definition["nodes"] if node["type"] == "CMKVisualProvider")
        self.assertEqual(provider["widgets_values"], ["Identity", 15, "CMKInstantIDSamplerSDXLPipe", "", ""])
        self.assertNotIn("end", input_names)
        self.assertNotIn("denoise", input_names)
        self.assertNotIn("sampler_name", input_names)
        self.assertNotIn("scheduler", input_names)
        self.assertEqual(sampler["widgets_values"][:4], [1, 0.7, "Largest", 0])
        self.assertEqual(sampler["widgets_values"][-3:], ["InstantID reference", 0.75, 8])

    def test_sampler_ui_separates_effect_controls(self):
        source = (ROOT / "web/js/cmk_instantid_sampler_ui.js").read_text(encoding="utf-8")
        self.assertIn('data-testid="widget-layout-field-label"', source)
        self.assertIn('name === "conditioning_mode"', source)
        self.assertIn('label.parentElement?.classList.add(ROW_CLASS)', source)
        self.assertIn("cmk-instantid-sampling-section", source)
        self.assertIn("MutationObserver", source)
        self.assertNotIn("conditioningMode.label", source)
        self.assertNotIn("addDOMWidget", source)
        self.assertNotIn("node.widgets.splice", source)
        self.assertNotIn("widgets_values", source)


if __name__ == "__main__":
    unittest.main()
