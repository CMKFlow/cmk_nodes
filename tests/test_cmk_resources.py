from contextlib import redirect_stdout
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_cmk_resources.py"


def load_module():
    spec = importlib.util.spec_from_file_location("cmk_resource_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CmkResourceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit_module = load_module()

    def test_manifest_has_separate_instantid_resources(self):
        ids = {resource.resource_id for resource in self.audit_module.RESOURCES}
        self.assertIn("instantid-adapter", ids)
        self.assertIn("instantid-controlnet", ids)
        self.assertNotEqual(
            next(r for r in self.audit_module.RESOURCES if r.resource_id == "instantid-adapter").target_path,
            next(r for r in self.audit_module.RESOURCES if r.resource_id == "instantid-controlnet").target_path,
        )

    def test_audit_finds_resource_in_an_additional_shared_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / "shared"
            target = shared / "controlnet" / "instantid" / "diffusion_pytorch_model.safetensors"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"test")
            comfy = root / "ComfyUI"
            (comfy / "models").mkdir(parents=True)
            (comfy / "main.py").touch()
            (comfy / "custom_nodes").mkdir()
            results = self.audit_module.audit(comfy, [shared])
            found = dict((resource.resource_id, path) for resource, path in results)
            self.assertEqual(target.resolve(), found["instantid-controlnet"])

    def test_shared_comfy_root_resolves_to_nested_models_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            shared = Path(directory) / "ComfyUI-Shared"
            models = shared / "models"
            models.mkdir(parents=True)

            self.assertEqual(
                models.resolve(),
                self.audit_module._models_root(shared).resolve(),
            )
            self.assertEqual(
                models.resolve(),
                self.audit_module._models_root(models).resolve(),
            )

    def test_downloadable_resources_have_destination_and_url(self):
        for resource in self.audit_module.RESOURCES:
            if resource.download_url:
                self.assertTrue(resource.target_path, resource.resource_id)

    def test_manifest_uses_exact_files_instead_of_nonempty_model_directories(self):
        for resource in self.audit_module.RESOURCES:
            for relative in resource.relative_paths:
                if resource.resource_id == "insightface-buffalo-l":
                    continue
                self.assertTrue(Path(relative).suffix, resource.resource_id)

    def test_all_model_files_selected_by_showcase_and_subgraphs_are_audited(self):
        extensions = {".safetensors", ".pth", ".pt", ".onnx", ".bin"}
        selected = set()

        def visit(value):
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
            elif isinstance(value, str) and Path(value).suffix.lower() in extensions:
                selected.add(Path(value).name)

        for folder in (ROOT / "workflows" / "showcase", ROOT / "subgraphs"):
            for path in folder.glob("*.json"):
                visit(json.loads(path.read_text(encoding="utf-8")))

        audited = {
            Path(relative).name
            for resource in self.audit_module.RESOURCES
            for relative in resource.relative_paths
        }
        self.assertEqual(set(), selected - audited)

    def test_zit_controlnet_weights_are_installed_as_model_patches(self):
        for resource_id in ("zit-controlnet", "zit-inpaint-model-patch"):
            resource = next(
                item for item in self.audit_module.RESOURCES
                if item.resource_id == resource_id
            )
            self.assertTrue(resource.target_path.startswith("model_patches/"))

    def test_download_copy_reports_percentage_and_size(self):
        payload = b"x" * 4096
        response = io.BytesIO(payload)
        response.headers = {"Content-Length": str(len(payload))}
        output = io.BytesIO()
        terminal = io.StringIO()

        with redirect_stdout(terminal):
            transferred = self.audit_module._copy_with_progress(
                response,
                output,
                chunk_size=1024,
            )

        self.assertEqual(len(payload), transferred)
        self.assertEqual(payload, output.getvalue())
        self.assertIn("100.00%", terminal.getvalue())
        self.assertIn("4.0 KiB", terminal.getvalue())


if __name__ == "__main__":
    unittest.main()
