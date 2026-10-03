import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_single_boundary():
    path = ROOT / "pipe" / "cmk_family_result.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "_CMKPostProcessBoundaryBase",
        "_CMKSinglePostProcessBoundary",
        "CMKPostProcessBoundarySDXLPipe",
    }
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    namespace = {"ExecutionBlocker": lambda value: value}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["CMKPostProcessBoundarySDXLPipe"]


class PostProcessBoundary35Tests(unittest.TestCase):
    def test_three_public_boundary_nodes_are_registered(self):
        source = (ROOT / "cmk_mappings.py").read_text(encoding="utf-8")
        expected = {
            "CMKPostProcessBoundarySDXLPipe": "CMK Flow · PostProcess Boundary SDXL",
            "CMKPostProcessBoundaryZITPipe": "CMK Flow · PostProcess Boundary ZIT",
            "CMKFamilyResultMergePipe": "CMK Flow · PostProcess Boundary Combined",
        }
        for node_type, title in expected.items():
            self.assertIn(f'"{node_type}"', source)
            self.assertIn(f'"{node_type}": "{title}"', source)

    def test_boundary_contract_is_family_neutral_and_model_is_deferred(self):
        family = (ROOT / "pipe" / "cmk_family_result.py").read_text(encoding="utf-8")
        loader = (ROOT / "pipe" / "loaders" / "checkpoint_vae_loader.py").read_text(encoding="utf-8")
        self.assertIn('"CMK_RESULT_PROCESS"', family)
        self.assertIn('"result_contract": "family_neutral"', family)
        self.assertIn('"model_role": "postprocess"', loader)
        self.assertIn('"materialized": False', loader)
        self.assertIn("def resolve_postprocess_model", loader)

    def test_boundary_offers_a_safe_none_checkpoint_contract(self):
        family = (ROOT / "pipe" / "cmk_family_result.py").read_text(encoding="utf-8")
        loader = (ROOT / "pipe" / "loaders" / "checkpoint_vae_loader.py").read_text(encoding="utf-8")
        self.assertIn("POSTPROCESS_MODEL_NONE", family)
        self.assertIn('"postprocess_model_disabled": disabled', loader)
        self.assertIn("MODEL LOAD         : DISABLED", family)

    def test_inactive_single_family_does_not_request_model(self):
        source = (ROOT / "pipe" / "cmk_family_result.py").read_text(encoding="utf-8")
        marker = "class _CMKSinglePostProcessBoundary"
        section = source[source.index(marker):source.index("class CMKPostProcessBoundarySDXLPipe")]
        self.assertLess(section.index('not process.get("family_active", True)'), section.index('for name in ("MODEL", "IMAGE", "LOG")'))

    def test_single_family_boundary_resolves_and_forwards_connected_visual(self):
        boundary_type = _load_single_boundary()
        boundary = boundary_type()
        process = {"model_family": "sdxl", "family_active": True}
        visual = {
            "type": "CMK_VISUAL_PIPE",
            "version": 1,
            "providers": [
                {"provider_id": "sampling"},
                {"provider_id": "identity"},
            ],
        }
        resolved = {
            "PROCESS": process,
            "MODEL": {},
            "IMAGE": object(),
            "LOG": {},
            "VISUAL": None,
        }

        self.assertEqual(["VISUAL"], boundary.check_lazy_status(**resolved))
        resolved["VISUAL"] = visual
        self.assertEqual([], boundary.check_lazy_status(**resolved))

        boundary_type._finish = staticmethod(
            lambda family, model, process, image, log, visual, checkpoint, vae, checkpoint_vae:
            (model, process, image, log, visual)
        )
        result = boundary.boundary(
            postprocess_checkpoint="checkpoint",
            postprocess_vae="vae",
            use_checkpoint_vae=True,
            **resolved,
        )
        self.assertIs(visual, result[4])

    def test_postprocess_consumers_resolve_model_lazily(self):
        for relative in (
            "pipe/cmk_detailer_prepare.py",
            "pipe/cmk_faceprocess_prepare.py",
            "nodes/instantid_face_detailer.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("resolve_postprocess_model", source, relative)
            self.assertIn("family_neutral", source, relative)

    def test_direct_image_input_exposes_the_neutral_process_contract(self):
        source = (ROOT / "pipe" / "loaders" / "cmk_image_load_resize.py").read_text(encoding="utf-8")
        self.assertIn('"CMK_RESULT_PROCESS",', source)
        self.assertIn('"result_contract": "family_neutral"', source)
        self.assertIn('"source_model_family": "image"', source)

    def test_direct_image_input_does_not_hide_local_postprocess_prompts(self):
        for relative in ("pipe/cmk_detailer_prepare.py", "pipe/cmk_faceprocess_prepare.py"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn('key in source_pipe for key in ("prompt_pos", "prompt_neg")', source)

    def test_packaged_postprocess_subgraphs_use_result_process(self):
        paths = sorted((ROOT / "subgraphs").glob("CMK Flow · Detailer*.json"))
        paths += sorted((ROOT / "subgraphs").glob("CMK Flow · FaceRebuild*.json"))
        paths += sorted((ROOT / "subgraphs").glob("CMK Flow · FaceProcess*.json"))
        self.assertEqual(6, len(paths))
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            text = json.dumps(data, ensure_ascii=False)
            self.assertIn("CMK_RESULT_PROCESS", text, path.name)
            forward_types = [
                node.get("type")
                for definition in data.get("definitions", {}).get("subgraphs", [])
                for node in definition.get("nodes", [])
            ]
            self.assertNotIn("CMKProcessForwardPipe", forward_types, path.name)

    def test_faceprocess_mode_is_owned_only_by_execute_nodes(self):
        paths = sorted((ROOT / "subgraphs").glob("CMK Flow · FaceProcess*.json"))
        self.assertEqual(2, len(paths))
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            prepares = [
                node
                for definition in data.get("definitions", {}).get("subgraphs", [])
                for node in definition.get("nodes", [])
                if node.get("type") == "CMKFaceProcessPreparePipe"
            ]
            self.assertEqual(1, len(prepares), path.name)
            self.assertNotIn(
                "process_mode",
                [item.get("name") for item in prepares[0].get("inputs", [])],
                path.name,
            )
            executes = [
                node
                for definition in data.get("definitions", {}).get("subgraphs", [])
                for node in definition.get("nodes", [])
                if node.get("type") == "CMKFaceProcessPipe"
            ]
            self.assertGreaterEqual(len(executes), 1, path.name)
            for execute in executes:
                self.assertIn(
                    "process_mode",
                    [item.get("name") for item in execute.get("inputs", [])],
                    path.name,
                )

    def test_visualizer_owns_terminal_unload(self):
        source = (ROOT / "pipe" / "cmk_visual.py").read_text(encoding="utf-8")
        self.assertIn('model_role") == "postprocess"', source)
        self.assertIn("unload_model_pipe", source)

    def test_full_flow_places_boundary_before_postprocess_and_save_after_it(self):
        flow = json.loads((ROOT / "workflows" / "showcase" / "CMK 2.5 · Full Flow .json").read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in flow["nodes"]}
        merge = next(node for node in nodes.values() if node["type"] == "CMKFamilyResultMergePipe")
        self.assertEqual("CMK_RESULT_PROCESS", next(output for output in merge["outputs"] if output["name"] == "PROCESS")["type"])
        visualizer = next(node for node in nodes.values() if node["type"] == "CMKVisualizer")
        links = flow["links"]
        incoming = [link for link in links if link[3] == merge["id"]]
        self.assertTrue(any(link[5] == "CMK_PROCESS_Z_IMAGE" for link in incoming))
        self.assertTrue(any(link[5] == "CMK_PROCESS_SDXL" for link in incoming))
        first_postprocess = next(link for link in links if link[1] == merge["id"] and link[5] == "CMK_RESULT_PROCESS")
        self.assertNotEqual(first_postprocess[3], visualizer["id"])
        for input_spec in visualizer["inputs"][:5]:
            self.assertIsNotNone(input_spec.get("link"))


if __name__ == "__main__":
    unittest.main()
