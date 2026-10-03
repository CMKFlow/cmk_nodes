import importlib.util
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

    def test_downloadable_resources_have_destination_and_url(self):
        for resource in self.audit_module.RESOURCES:
            if resource.download_url:
                self.assertTrue(resource.target_path, resource.resource_id)


if __name__ == "__main__":
    unittest.main()
