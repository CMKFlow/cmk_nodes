import unittest
import json
from pathlib import Path

from pipe.cmk_module_cache_contract import (
    ARTIFACTS_FIELD,
    CURRENT_ARTIFACT_FIELD,
    artifact_for,
    build_artifact_key,
    current_artifact,
    stamp_artifact,
)


class ModuleCacheContractTests(unittest.TestCase):
    def test_same_effective_contract_has_same_identity(self):
        first = build_artifact_key(
            "sdxl.refiner",
            "sampled-result-123",
            {"steps": 25, "cfg": 4.8, "nested": {"b": 2, "a": 1}},
            schema="refiner-v1",
        )
        second = build_artifact_key(
            "sdxl.refiner",
            "sampled-result-123",
            {"nested": {"a": 1, "b": 2}, "cfg": 4.8, "steps": 25},
            schema="refiner-v1",
        )
        self.assertEqual(first, second)

    def test_upstream_result_change_invalidates_identity(self):
        settings = {"steps": 25, "cfg": 4.8}
        first = build_artifact_key(
            "sdxl.detailer", "refiner-a", settings, schema="detailer-v1"
        )
        second = build_artifact_key(
            "sdxl.detailer", "refiner-b", settings, schema="detailer-v1"
        )
        self.assertNotEqual(first, second)

    def test_effective_setting_change_invalidates_identity(self):
        first = build_artifact_key(
            "sdxl.detailer",
            "refiner-a",
            {"denoise": 0.55},
            schema="detailer-v1",
        )
        second = build_artifact_key(
            "sdxl.detailer",
            "refiner-a",
            {"denoise": 0.60},
            schema="detailer-v1",
        )
        self.assertNotEqual(first, second)

    def test_artifact_stamp_is_copy_on_write_and_preserves_lineage(self):
        source = {
            "model_family": "sdxl",
            ARTIFACTS_FIELD: {"sdxl.first_pass": "first-pass-a"},
        }
        result = stamp_artifact(source, "sdxl.refiner", "refiner-a")

        self.assertIsNone(artifact_for(source, "sdxl.refiner"))
        self.assertEqual("first-pass-a", artifact_for(result, "sdxl.first_pass"))
        self.assertEqual("refiner-a", artifact_for(result, "sdxl.refiner"))
        self.assertEqual("refiner-a", current_artifact(result))
        self.assertNotIn(CURRENT_ARTIFACT_FIELD, source)
        self.assertIsNot(source[ARTIFACTS_FIELD], result[ARTIFACTS_FIELD])

    def test_runtime_objects_are_rejected_from_identity(self):
        with self.assertRaises(TypeError):
            build_artifact_key(
                "sdxl.detailer",
                "refiner-a",
                {"model": object()},
                schema="detailer-v1",
            )

    def test_refiner_result_is_the_only_upstream_identity_for_detailer(self):
        refiner_key = build_artifact_key(
            "sdxl.refiner",
            "sampled-a",
            {"start": 0.8, "steps": 25},
            schema="refiner-v1",
        )
        process = stamp_artifact({}, "sdxl.refiner", refiner_key)
        detailer_key = build_artifact_key(
            "sdxl.detailer",
            artifact_for(process, "sdxl.refiner"),
            {"denoise": 0.45, "model": "face.pt"},
            schema="detailer-v1",
        )

        self.assertEqual(64, len(detailer_key))
        self.assertEqual(refiner_key, artifact_for(process, "sdxl.refiner"))

    def test_refiner_process_output_bypasses_its_boundary(self):
        root = Path(__file__).resolve().parents[1]
        document = json.loads(
            (root / "subgraphs" / "CMK Flow · 20 Refiner SDXL.json").read_text(
                encoding="utf-8"
            )
        )
        definition = document["definitions"]["subgraphs"][0]
        boundary = next(
            node for node in definition["nodes"]
            if node["type"] == "CMKRefinerBoundaryCache"
        )
        self.assertNotIn("PROCESS", [item["name"] for item in boundary["inputs"]])
        self.assertNotIn("PROCESS", [item["name"] for item in boundary["outputs"]])
        forward = next(
            node for node in definition["nodes"]
            if node["type"] == "CMKProcessForwardPipe"
        )
        process_slot = next(
            index for index, item in enumerate(definition["outputs"])
            if item["name"] == "PROCESS"
        )
        process_link = next(
            link for link in definition["links"]
            if link["target_id"] == -20 and link["target_slot"] == process_slot
        )
        self.assertEqual(forward["id"], process_link["origin_id"])

    def test_refiner_visuals_are_routed_from_cache_boundary_images(self):
        root = Path(__file__).resolve().parents[1]
        document = json.loads(
            (root / "subgraphs" / "CMK Flow · 20 Refiner SDXL.json").read_text(
                encoding="utf-8"
            )
        )
        definition = document["definitions"]["subgraphs"][0]
        boundary = next(
            node for node in definition["nodes"]
            if node["type"] == "CMKRefinerBoundaryCache"
        )
        providers = [
            node for node in definition["nodes"]
            if node["type"] == "CMKVisualProvider"
        ]
        links = {link["id"]: link for link in definition["links"]}

        for provider in providers:
            for item in provider["inputs"]:
                if item["name"] in {"IMAGE", "BEFORE", "AFTER"} and item.get("link") is not None:
                    self.assertEqual(boundary["id"], links[item["link"]]["origin_id"])

    def test_identity_refiner_detailer_lineage_is_preserved(self):
        process = stamp_artifact({}, "sdxl.identity", "identity-a")
        process = stamp_artifact(process, "sdxl.refiner", "refiner-a")
        process = stamp_artifact(process, "sdxl.detailer", "detailer-a")

        self.assertEqual("identity-a", artifact_for(process, "sdxl.identity"))
        self.assertEqual("refiner-a", artifact_for(process, "sdxl.refiner"))
        self.assertEqual("detailer-a", artifact_for(process, "sdxl.detailer"))
        self.assertEqual("detailer-a", current_artifact(process))

    def test_current_artifact_tracks_the_immediate_predecessor(self):
        process = stamp_artifact({}, "sdxl.first_pass", "first-a")
        self.assertEqual("first-a", current_artifact(process))
        process = stamp_artifact(process, "sdxl.refiner", "refiner-a")
        self.assertEqual("refiner-a", current_artifact(process))
        process = stamp_artifact(process, "sdxl.detailer", "detailer-a")
        self.assertEqual("detailer-a", current_artifact(process))
        self.assertEqual("refiner-a", artifact_for(process, "sdxl.refiner"))

    def test_detailer_uses_current_artifact_before_legacy_refiner_fallback(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "nodes" / "image" / "smart_detailer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "current_artifact(source_pipe) or artifact_for(\n"
            "                source_pipe, self._UPSTREAM_STAGE_KEY,",
            source,
        )
        self.assertIn('_CACHE_SCHEMA = "cmk_detailer_branch_v6"', source)
        self.assertEqual(
            2,
            source.count('exclude_inputs=("output_image_proceed", "opt_log", "opt_diagnostic")'),
        )

        boundary = (root / "pipe" / "cmk_module_boundary_cache.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"cmk_detailer_branch_v6"', boundary)
        self.assertIn('_SCHEMA = "cmk_detailer_boundary_v3"', boundary)

    def test_first_pass_process_output_is_routed_from_sampled_boundary(self):
        root = Path(__file__).resolve().parents[1]
        document = json.loads(
            (root / "subgraphs" / "CMK Flow · 10 KSampler SDXL 1st Pass.json").read_text(
                encoding="utf-8"
            )
        )
        definition = document["definitions"]["subgraphs"][0]
        boundary = next(
            node for node in definition["nodes"]
            if node["type"] == "CMKFamilyBranchGateSDXLSampled"
        )
        process_output = next(
            output for output in boundary["outputs"] if output["name"] == "PROCESS"
        )
        links = {link["id"]: link for link in definition["links"]}

        self.assertEqual([12229], process_output["links"])
        self.assertEqual(boundary["id"], links[12229]["origin_id"])


if __name__ == "__main__":
    unittest.main()
