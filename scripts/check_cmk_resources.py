#!/usr/bin/env python3
"""Audit and install CMK model resources without touching workflows.

The audit deliberately separates Python dependencies from model files.  It
searches the ComfyUI model directory and an optional shared model directory,
reports every known resource independently, and only downloads resources for
which CMK has a verified public source URL.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Resource:
    resource_id: str
    label: str
    feature: str
    relative_paths: tuple[str, ...]
    download_url: str | None = None
    target_path: str | None = None
    license_note: str = ""
    require_all: bool = False


RESOURCES = (
    Resource(
        "sdxl-checkpoint",
        "SDXL checkpoint",
        "SDXL / HYBRID generation",
        ("checkpoints",),
        license_note="Choose and install a compatible SDXL checkpoint from its official source.",
    ),
    Resource(
        "vae",
        "VAE",
        "SDXL / HYBRID generation",
        ("vae",),
        license_note="Choose and install the VAE required by the selected checkpoint.",
    ),
    Resource(
        "controlnet-model",
        "ControlNet model",
        "ControlNet",
        ("controlnet",),
        license_note="Choose and install the ControlNet model required by the workflow.",
    ),
    Resource(
        "instantid-adapter",
        "InstantID adapter",
        "InstantID / FaceRebuild",
        ("instantid/ip-adapter.bin", "ip-adapter.bin"),
        "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin",
        "instantid/ip-adapter.bin",
        "InstantX/InstantID (Apache-2.0)",
    ),
    Resource(
        "instantid-controlnet",
        "InstantID ControlNet",
        "InstantID / FaceRebuild",
        (
            "controlnet/instantid/diffusion_pytorch_model.safetensors",
            "controlnet/diffusion_pytorch_model.safetensors",
            "instantid/diffusion_pytorch_model.safetensors",
        ),
        "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors",
        "controlnet/instantid/diffusion_pytorch_model.safetensors",
        "InstantX/InstantID (Apache-2.0; approximately 2.5 GB)",
    ),
    Resource(
        "insightface-antelopev2",
        "InsightFace antelopev2",
        "InstantID / FaceRebuild",
        (
            "insightface/models/antelopev2",
            "insightface/antelopev2",
        ),
        license_note="InsightFace model licence must be accepted and the official archive installed manually.",
    ),
    Resource(
        "sam-vit-b",
        "SAM ViT-B detector",
        "Detailer / FaceProcess",
        ("sams/sam_vit_b_01ec64.pth",),
        license_note="Install from the official Segment Anything release; no silent third-party download.",
    ),
    Resource(
        "ultralytics-detector",
        "Ultralytics detector",
        "Detailer / segmentation",
        ("ultralytics",),
        license_note="Install the detector weights required by the selected Detailer workflow.",
    ),
    Resource(
        "face-restore",
        "Face restore model (GFPGAN or CodeFormer)",
        "FaceProcess Restore",
        (
            "facerestore_models/GFPGANv1.4.pth",
            "facerestore_models/codeformer.pth",
            "facerestore_models/codeformer-v0.1.0.pth",
        ),
        license_note="Optional; install one supported face-restore model from its official source.",
    ),
    Resource(
        "fooocus-inpaint",
        "Fooocus inpaint head and patch",
        "Inpaint / InOutpaint",
        (
            "inpaint/fooocus_inpaint_head.pth",
            "inpaint/inpaint_v25.fooocus.patch",
        ),
        license_note="Both files are required for the Fooocus inpaint path; install from the approved upstream source.",
        require_all=True,
    ),
)


def find_comfy_root(repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    direct = repo_root.parent.parent
    if repo_root.parent.name == "custom_nodes" and (direct / "main.py").is_file():
        return direct
    for parent in repo_root.parents:
        if (parent / "main.py").is_file() and (parent / "custom_nodes").is_dir():
            return parent
    raise RuntimeError("CMK must be installed below ComfyUI/custom_nodes/cmk_nodes.")


def _candidate_roots(comfy_root: Path, explicit: Iterable[Path] = ()) -> list[Path]:
    roots = [Path(value).expanduser() for value in explicit if value]
    roots.append(comfy_root / "models")
    shared = os.environ.get("CMK_SHARED_MODELS")
    if shared:
        roots.append(Path(shared).expanduser())
    config = comfy_root / "extra_model_paths.yaml"
    if config.is_file():
        for line in config.read_text(encoding="utf-8", errors="ignore").splitlines():
            value = line.split("#", 1)[0].strip()
            if value.startswith("base_path:"):
                base = value.split(":", 1)[1].strip().strip("'\"")
                if base:
                    base_path = Path(base).expanduser()
                    roots.extend((base_path, base_path / "models"))
    return list(dict.fromkeys(path.resolve() for path in roots))


def locate(resource: Resource, roots: Iterable[Path]) -> Path | None:
    for root in roots:
        if resource.require_all:
            candidates = [root / relative for relative in resource.relative_paths]
            if all(candidate.is_file() for candidate in candidates):
                return candidates[0]
            continue
        for relative in resource.relative_paths:
            candidate = root / relative
            if candidate.is_file():
                return candidate
            if candidate.is_dir() and any(path.is_file() for path in candidate.rglob("*")):
                return candidate
    return None


def audit(comfy_root: Path, extra_roots: Iterable[Path] = ()) -> list[tuple[Resource, Path | None]]:
    roots = _candidate_roots(comfy_root, extra_roots)
    return [(resource, locate(resource, roots)) for resource in RESOURCES]


def _download(resource: Resource, destination: Path) -> Path:
    if not resource.download_url or not resource.target_path:
        raise RuntimeError(f"{resource.resource_id} has no approved automatic download source.")
    target = destination / resource.target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Target already exists: {target}")
    temporary = target.with_name(target.name + ".part")
    request = Request(resource.download_url, headers={"User-Agent": "CMK-resource-installer/1"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", action="append", type=Path, default=[], help="additional model root; repeatable")
    parser.add_argument("--install", metavar="RESOURCE_ID", help="download exactly one resource into the selected model root")
    parser.add_argument("--target-root", type=Path, help="destination model root for --install (defaults to ComfyUI/models)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    comfy_root = find_comfy_root(repo_root)
    roots = _candidate_roots(comfy_root, args.models_root)
    print(f"ComfyUI root : {comfy_root}")
    print("Model roots  :")
    for root in roots:
        print(f"  - {root}")

    if args.install:
        resource = next((item for item in RESOURCES if item.resource_id == args.install), None)
        if resource is None:
            raise RuntimeError(f"Unknown resource: {args.install}")
        target_root = (args.target_root or (comfy_root / "models")).expanduser().resolve()
        print(f"Installing   : {resource.label}")
        if resource.license_note:
            print(f"Notice       : {resource.license_note}")
        print(f"Source       : {resource.download_url}")
        print(f"Destination  : {target_root / resource.target_path}")
        print(f"Installed    : {_download(resource, target_root)}")
        return 0

    print("\nCMK resource audit")
    for resource, path in audit(comfy_root, args.models_root):
        status = "FOUND" if path else "MISSING"
        detail = f" -> {path}" if path else (f" | {resource.license_note}" if resource.license_note else "")
        print(f"[{status:7}] {resource.resource_id:24} {resource.label}{detail}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"CMK resource setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
