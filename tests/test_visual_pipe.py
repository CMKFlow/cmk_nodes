import importlib.util
import json
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "pipe" / "cmk_visual.py"
SPEC = importlib.util.spec_from_file_location("cmk_visual_contract", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
VISUAL_TYPE = MODULE.VISUAL_TYPE
CMKVisualPass = MODULE.CMKVisualPass
CMKVisualProvider = MODULE.CMKVisualProvider
normalize_visual = MODULE.normalize_visual
register_provider = MODULE.register_provider
ROOT = PATH.parents[1]


class VisualPipeTests(unittest.TestCase):
    def test_curated_subgraphs_do_not_share_provider_ids_between_stages(self):
        import json

        owners = {}
        for path in (ROOT / "subgraphs").glob("CMK Flow · *.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            for provider in document.get("nodes", [{}])[0].get("properties", {}).get("cmkVisualProviders", []):
                provider_id = provider["provider_id"]
                owner = (
                    provider.get("branch", ""),
                    provider.get("stage_key", provider.get("key", "")),
                )
                previous = owners.get(provider_id)
                if previous is not None:
                    self.assertEqual(
                        previous,
                        owner,
                        f"{provider_id} is shared by unrelated providers",
                    )
                owners[provider_id] = owner

    def test_accumulates_without_copying_images(self):
        first_image = object()
        second_image = object()
        visual = register_provider(None, module_instance_id="10", module_type="Sampler", module_label="1st Pass", sequence=10, channels={"result": first_image})
        visual = register_provider(visual, module_instance_id="20", module_type="Refiner", module_label="Refiner", sequence=20, channels={"result": second_image})
        self.assertEqual(VISUAL_TYPE, visual["type"])
        self.assertEqual(["1st Pass", "Refiner"], [item["module_label"] for item in visual["providers"]])
        self.assertIs(first_image, visual["providers"][0]["channels"]["result"])
        self.assertIs(second_image, visual["providers"][1]["channels"]["result"])

    def test_pass_is_identity(self):
        visual = normalize_visual(None)
        self.assertIs(visual, CMKVisualPass.forward(visual)[0])

    def test_provider_enable_is_the_authoritative_registration_gate(self):
        visual = register_provider(
            None,
            module_instance_id="20",
            module_type="CMKVisualProvider",
            module_label="Refiner",
            sequence=20,
            channels={"result": object()},
        )
        result = CMKVisualProvider.publish(
            object(), "Detailer", 23, VISUAL=visual, enable=False
        )[0]
        self.assertEqual(["Refiner"], [item["module_label"] for item in result["providers"]])

    def test_reregister_replaces_provider(self):
        visual = register_provider(None, module_instance_id="10", module_type="Sampler", module_label="Waiting", sequence=10, channels={}, status="waiting")
        visual = register_provider(visual, module_instance_id="10", module_type="Sampler", module_label="Done", sequence=10, channels={"result": object()})
        self.assertEqual(1, len(visual["providers"]))
        self.assertEqual("completed", visual["providers"][0]["status"])

    def test_late_convenience_provider_cannot_overwrite_an_intermediate_stage(self):
        identity = object()
        visual = register_provider(None, module_instance_id="15", module_type="Identity", module_label="Identity", sequence=15, channels={"result": identity})
        visual = register_provider(visual, module_instance_id="10", module_type="Sampler", module_label="1st Pass", sequence=10, channels={"result": object()})
        self.assertEqual(["Identity"], [item["module_label"] for item in visual["providers"]])
        self.assertIs(identity, visual["providers"][0]["channels"]["result"])

    def test_multichannel_declares_compare(self):
        visual = register_provider(None, module_instance_id="40", module_type="FaceSwap", module_label="FaceSwap", sequence=40, channels={"before": object(), "after": object()})
        provider = visual["providers"][0]
        self.assertTrue(provider["capabilities"]["multi_source"])
        self.assertTrue(provider["capabilities"]["compare"])

    def test_execution_identity_survives_normalization(self):
        visual = MODULE.empty_visual("prompt-2")
        self.assertEqual("prompt-2", normalize_visual(visual)["execution_id"])

    def test_provider_can_carry_branch_stable_stage_identity(self):
        visual = register_provider(
            None,
            module_instance_id="5",
            module_type="ControlNet",
            module_label="ControlNet",
            sequence=5,
            channels={"result": object()},
            branch="sdxl",
            stage_key="sdxl.controlnet",
        )
        provider = visual["providers"][0]
        self.assertEqual("sdxl", provider["branch"])
        self.assertEqual("sdxl.controlnet", provider["stage_key"])

    def test_sampler_and_refiner_form_a_visual_chain(self):
        sampler = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · 10 KSampler SDXL 1st Pass.json").read_text(encoding="utf-8")
        )["definitions"]["subgraphs"][0]
        refiner = json.loads(
            (ROOT / "subgraphs" / "CMK Flow · 20 Refiner SDXL.json").read_text(encoding="utf-8")
        )["definitions"]["subgraphs"][0]
        self.assertEqual(["LOG", "VISUAL", "diagnostic"], [item["name"] for item in sampler["outputs"][-3:]])
        self.assertEqual([12701], sampler["outputs"][-2]["linkIds"])
        sampler_provider = next(node for node in sampler["nodes"] if node["id"] == 7000)
        self.assertEqual("CMKVisualProvider", sampler_provider["type"])
        self.assertEqual("1st Pass", sampler_provider["widgets_values"][0])
        self.assertEqual("sdxl.first_pass", sampler_provider["widgets_values"][4])
        providers = [node for node in refiner["nodes"] if node["type"] == "CMKVisualProvider"]
        self.assertEqual(["Refiner"], [node["widgets_values"][0] for node in providers])
        self.assertEqual(["CMKRefinerPipe"], [node["widgets_values"][2] for node in providers])
        self.assertEqual(["LOG", "VISUAL", "diagnostic"], [item["name"] for item in refiner["outputs"][-3:]])
        self.assertEqual([12713], refiner["inputs"][-1]["linkIds"])
        self.assertEqual([12716], refiner["outputs"][-2]["linkIds"])

    def test_visualizer_accepts_an_unconnected_visual_pipe(self):
        self.assertNotIn("required", MODULE.CMKVisualizer.INPUT_TYPES())
        self.assertIn("VISUAL", MODULE.CMKVisualizer.INPUT_TYPES()["optional"])

    def test_visualizer_owns_frontend_history_follow_and_compare(self):
        frontend = (ROOT / "web" / "js" / "cmk_visualizer.js").read_text(encoding="utf-8")
        for contract in (
            "cmkVisualProviders",
            "loadWaitingProviders",
            "scheduleProviderRefresh",
            '"onAdded", "onRemoved", "onConnectionsChange"',
            'status: "waiting"',
            "state.autoFollow",
            "state.runActive",
            "onpointerdown",
            "onlostpointercapture",
            "cmk_visual_images",
            "cmk-visual-click-compare",
            'api.addEventListener("b_preview"',
            'api.addEventListener("b_preview_with_metadata"',
            "realNodeId",
            "real_node_id",
            "displayNodeId",
            "display_node_id",
            "preview_metadata",
            "moduleMatches.length === 1",
            "function exposeProvider(state, provider)",
            "state.catalog = graphProviders()",
            "state.providers = []",
            "function remappedDeclaredProvider(outerNode, item)",
            "outerNode?.subgraph?.nodes",
            "live_node_type",
            "cmk-CMKVisualProvider-${visualProvider.id}",
            "fallbackLiveTypes",
            '"first-pass": "CMKKSamplerPipe"',
            'controlnet: "CMKControlNetPreparePipe"',
            "live_node_resolved: Boolean(liveNode)",
            "live_node_resolved: Boolean(item.live_node_resolved)",
            "!current.live_node_resolved && provider.live_node_resolved",
            "lastMetadataPreviewBlob = accepted ? blob : null",
            'api.addEventListener("executed"',
            "detail.output?.images?.[0]",
            "function acceptExecutedImage(state, provider, descriptor)",
            'api.addEventListener("execution_success"',
            "function activateProvider(state, provider)",
            "provider.liveUrl",
            "function providerSemanticKey(provider)",
            "const declarationBySemanticKey = new Map",
            "provider_id: providerId",
            "const completed = new Map()",
            "state.providers = [...completed.values()]",
            "image_index: Number(channel.image_index)",
            "state.images = incomingImages",
            "state.completedProviderIds = new Set(completed.keys())",
            "state.completedProviderIds.has(provider.provider_id)",
            "declaredItem.enable_widget",
            "`${declaredItem.key}_global_enable`",
            "URL.revokeObjectURL",
        ):
            self.assertIn(contract, frontend)
        self.assertNotIn('slider.type = "range"', frontend)
        self.assertNotIn("compare.onclick", frontend)
        self.assertNotIn("function nextLiveProvider", frontend)
        self.assertNotIn("state.providers = graphProviders()", frontend)
        self.assertNotIn('api.addEventListener("executing"', frontend)
        self.assertNotIn('api.addEventListener("progress"', frontend)
        self.assertNotIn('api.addEventListener("progress_state"', frontend)
        self.assertNotIn('api.addEventListener("execution_cached"', frontend)
        self.assertNotIn("state.providers.sort", frontend)
        self.assertNotIn("[...merged.values()].sort", frontend)
        self.assertNotIn("for (const previous of state.providers) merged.set", frontend)
        self.assertIn("if (!sourceNodeIds.length) return", frontend)
        self.assertNotIn("cmk-visual-status", frontend)
        self.assertNotIn("LIVE ·", frontend)
        backend = (ROOT / "pipe" / "cmk_visual.py").read_text(encoding="utf-8")
        self.assertIn('"cmk_visual_images": ui_images', backend)
        self.assertNotIn('"images": ui_images', backend)


if __name__ == "__main__":
    unittest.main()
