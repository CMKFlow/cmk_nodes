from contextlib import redirect_stdout
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


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

    def test_reference_downloads_are_pinned_to_expected_hashes(self):
        pinned = {
            "sdxl-checkpoint-juggernaut",
            "sdxl-checkpoint-pony",
            "sdxl-vae-clear",
            "sdxl-refiner",
            "sdxl-refiner-vae",
            "sdxl-controlnet",
            "insightface-buffalo-l",
            "ultralytics-face-yolov8m",
            "ultralytics-face-yolov8s",
            "ultralytics-hand-yolov8n",
            "ultralytics-hand-yolov8s",
            "faceswap-inswapper-128",
            "faceswap-hyperswap-1b",
            "faceswap-hyperswap-1c",
            "gpen-bfr-512",
            "realesrgan-x2-legacy",
            "refiner-hyper-sdxl-lora",
            "sdxl-dmd2-lora",
        }
        by_id = {resource.resource_id: resource for resource in self.audit_module.RESOURCES}
        for resource_id in pinned:
            self.assertTrue(by_id[resource_id].expected_sha256, resource_id)

    def test_checksum_verification_rejects_changed_file(self):
        resource = self.audit_module.Resource(
            "test",
            "Test",
            "Test",
            ("test.bin",),
            expected_sha256="0" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.bin"
            path.write_bytes(b"different")
            with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
                self.audit_module._verify_download(resource, path)

    def test_archive_extraction_uses_declared_members_only(self):
        resource = self.audit_module.Resource(
            "test-archive",
            "Test archive",
            "Test",
            ("models/test",),
            archive_members=("one.onnx", "two.onnx"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "models.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("one.onnx", b"one")
                bundle.writestr("two.onnx", b"two")
                bundle.writestr("ignored.txt", b"ignored")
            target = root / "models"
            installed = self.audit_module._extract_archive(resource, archive, target)
            self.assertEqual(target, installed)
            self.assertEqual({"one.onnx", "two.onnx"}, {path.name for path in target.iterdir()})

    def test_buffalo_pack_is_only_found_when_every_required_file_exists(self):
        resource = next(
            item for item in self.audit_module.RESOURCES
            if item.resource_id == "insightface-buffalo-l"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / resource.relative_paths[0]
            first.parent.mkdir(parents=True)
            first.touch()
            self.assertIsNone(self.audit_module.locate(resource, [root]))

            for relative in resource.relative_paths[1:]:
                (root / relative).touch()
            self.assertEqual(first, self.audit_module.locate(resource, [root]))

    def test_every_audited_resource_has_an_approved_download(self):
        without_download = [
            resource.resource_id
            for resource in self.audit_module.RESOURCES
            if not resource.download_url
        ]
        self.assertEqual([], without_download)

    def test_manifest_uses_exact_files_instead_of_nonempty_model_directories(self):
        for resource in self.audit_module.RESOURCES:
            for relative in resource.relative_paths:
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

    def test_detector_selections_use_the_audited_bbox_or_segmentation_path(self):
        selected = set()

        def visit(value):
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
            elif isinstance(value, str) and value.startswith(("bbox/", "segm/")):
                selected.add("ultralytics/" + value)

        for folder in (ROOT / "workflows" / "showcase", ROOT / "subgraphs"):
            for path in folder.glob("*.json"):
                visit(json.loads(path.read_text(encoding="utf-8")))

        audited = {
            relative
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
