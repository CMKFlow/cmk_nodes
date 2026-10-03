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
import os
from pathlib import Path
import sys
import time
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
        "sdxl-checkpoint-juggernaut",
        "SDXL checkpoint: juggernautXL_ragnarok.safetensors",
        "SDXL / HYBRID generation",
        ("checkpoints/juggernautXL_ragnarok.safetensors",),
        license_note="Showcase selection; install manually from its authorised model page.",
    ),
    Resource(
        "sdxl-checkpoint-pony",
        "PostProcess checkpoint: Realism By Stable Yogi (Pony)XL_V3VAE.safetensors",
        "PostProcess",
        ("checkpoints/Realism By Stable Yogi (Pony)XL_V3VAE.safetensors",),
        license_note="Showcase selection; install manually from its authorised model page.",
    ),
    Resource(
        "sdxl-vae-clear",
        "SDXL VAE: ClearVAE_V2.2.safetensors",
        "SDXL / HYBRID / PostProcess",
        ("vae/ClearVAE_V2.2.safetensors",),
        license_note="Showcase selection; install manually from its authorised model page.",
    ),
    Resource(
        "sdxl-refiner",
        "SDXL Refiner 1.0",
        "20 Refiner",
        ("checkpoints/refiner/sd_xl_refiner_1.0.safetensors",),
        license_note="Install the official SDXL Refiner model manually after accepting its licence.",
    ),
    Resource(
        "sdxl-refiner-vae",
        "SDXL Refiner VAE",
        "20 Refiner",
        ("vae/sdxl_vae.safetensors",),
        license_note="Install the VAE used by the SDXL Refiner workflow.",
    ),
    Resource(
        "sdxl-controlnet",
        "SDXL ControlNet: controlnetxlCNXL_2vxpswa7AnytestV4.safetensors",
        "05 ControlNet SDXL / Combined",
        ("controlnet/controlnetxlCNXL_2vxpswa7AnytestV4.safetensors",),
        license_note="Showcase selection; install manually from its authorised model page.",
    ),
    Resource(
        "zit-diffusion-model",
        "Z-Image Turbo diffusion model",
        "Z-Image Turbo / HYBRID generation",
        ("diffusion_models/z_image_turbo_bf16.safetensors",),
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        "diffusion_models/z_image_turbo_bf16.safetensors",
        "Comfy-Org repack of Tongyi-MAI/Z-Image-Turbo (Apache-2.0; large download)",
    ),
    Resource(
        "zit-text-encoder",
        "Z-Image Turbo text encoder: qwen_3_4b.safetensors",
        "Z-Image Turbo / HYBRID generation",
        ("text_encoders/qwen_3_4b.safetensors",),
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
        "text_encoders/qwen_3_4b.safetensors",
        "Comfy-Org Z-Image Turbo package (Apache-2.0; large download)",
    ),
    Resource(
        "zit-vae",
        "Z-Image Turbo VAE: ae.safetensors",
        "Z-Image Turbo / HYBRID generation",
        ("vae/ae.safetensors",),
        "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
        "vae/ae.safetensors",
        "Comfy-Org Z-Image Turbo package (Apache-2.0)",
    ),
    Resource(
        "zit-controlnet",
        "Z-Image Turbo ControlNet Union",
        "05 ControlNet ZIT",
        ("model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors",),
        "https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors",
        "model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors",
        "Alibaba-PAI/VideoX-Fun (Apache-2.0; approximately 3.1 GB)",
    ),
    Resource(
        "zit-inpaint-model-patch",
        "Z-Image Turbo Inpaint model patch 2.1",
        "Z-Image Turbo Inpaint",
        ("model_patches/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors",),
        "https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors",
        "model_patches/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors",
        "Alibaba-PAI/VideoX-Fun (Apache-2.0; approximately 6.7 GB)",
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
        "insightface-buffalo-l",
        "InsightFace buffalo_l",
        "InstantID / FaceRebuild / FaceSwap",
        (
            "insightface/models/buffalo_l",
            "insightface/buffalo_l",
        ),
        license_note="InsightFace model licence must be accepted; install the official buffalo_l pack manually.",
    ),
    Resource(
        "sam-vit-b",
        "SAM ViT-B detector",
        "Detailer / FaceProcess",
        ("sams/sam_vit_b_01ec64.pth",),
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sams/sam_vit_b_01ec64.pth",
        "Meta Segment Anything model checkpoint (Apache-2.0; approximately 375 MB)",
    ),
    Resource(
        "ultralytics-face-yolov8m",
        "Ultralytics face detector: face_yolov8m.pt",
        "FaceProcess / Detailer",
        ("ultralytics/bbox/face/face_yolov8m.pt",),
        license_note="Install this third-party detector weight manually from its authorised model page.",
    ),
    Resource(
        "ultralytics-face-yolov8s",
        "Ultralytics face detector: face_yolov8s.pt",
        "Detailer",
        ("ultralytics/bbox/face/face_yolov8s.pt",),
        license_note="Install this third-party detector weight manually from its authorised model page.",
    ),
    Resource(
        "ultralytics-hand-yolov8s",
        "Ultralytics hand detector: hand_yolov8s.pt",
        "Detailer",
        ("ultralytics/bbox/div/hand_yolov8s.pt",),
        license_note="Install this third-party detector weight manually from its authorised model page.",
    ),
    Resource(
        "ultralytics-hand-segm-yolov8n",
        "Ultralytics hand segmentation: hand_yolov8n.pt",
        "Detailer",
        ("ultralytics/segm/div/hand_yolov8n.pt",),
        license_note="Install this third-party detector weight manually from its authorised model page.",
    ),
    Resource(
        "ultralytics-hand-segm-yolov8s",
        "Ultralytics hand segmentation: hand_yolov8s.pt",
        "Detailer",
        ("ultralytics/segm/div/hand_yolov8s.pt",),
        license_note="Install this third-party detector weight manually from its authorised model page.",
    ),
    Resource(
        "gfpgan-v1.4",
        "GFPGAN v1.4 face restore model",
        "FaceProcess Restore / FaceSwap enhancement",
        ("facerestore_models/GFPGANv1.4.pth",),
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "facerestore_models/GFPGANv1.4.pth",
        "TencentARC/GFPGAN (Apache-2.0; approximately 332 MB)",
    ),
    Resource(
        "gfpgan-v1.3",
        "GFPGAN v1.3 face restore model",
        "Legacy FaceRestore examples",
        ("facerestore_models/GFPGANv1.3.pth",),
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        "facerestore_models/GFPGANv1.3.pth",
        "TencentARC/GFPGAN (Apache-2.0; approximately 332 MB)",
    ),
    Resource(
        "gpen-bfr-512",
        "GPEN-BFR-512 face restore model",
        "Legacy video FaceSwap enhancement",
        ("facerestore_models/GPEN-BFR-512.onnx",),
        license_note="Install manually from the official GPEN project after reviewing its model terms.",
    ),
    Resource(
        "faceswap-inswapper-128",
        "FaceSwap model: inswapper_128.onnx",
        "FaceSwap / Legacy video FaceSwap",
        ("insightface/inswapper_128.onnx", "insightface/models/inswapper_128.onnx"),
        license_note="Install manually only from a source whose model licence you have accepted.",
    ),
    Resource(
        "faceswap-hyperswap-1b",
        "FaceSwap model: hyperswap_1b_256.onnx",
        "FaceSwap",
        ("insightface/hyperswap/hyperswap_1b_256.onnx", "insightface/hyperswap_1b_256.onnx"),
        license_note="Install manually only from a source whose model licence you have accepted.",
    ),
    Resource(
        "faceswap-hyperswap-1c",
        "FaceSwap model: hyperswap_1c_256.onnx",
        "FaceSwap",
        ("insightface/hyperswap/hyperswap_1c_256.onnx", "insightface/hyperswap_1c_256.onnx"),
        license_note="Install manually only from a source whose model licence you have accepted.",
    ),
    Resource(
        "fooocus-inpaint-head",
        "Fooocus inpaint head",
        "SDXL Inpaint / InOutpaint",
        ("inpaint/fooocus_inpaint_head.pth",),
        "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth",
        "inpaint/fooocus_inpaint_head.pth",
        "Official Fooocus inpaint resource",
    ),
    Resource(
        "fooocus-inpaint-v25-patch",
        "Fooocus inpaint v2.5 patch",
        "SDXL Inpaint / InOutpaint",
        ("inpaint/inpaint_v25.fooocus.patch",),
        "https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v25.fooocus.patch",
        "inpaint/inpaint_v25.fooocus.patch",
        "Official Fooocus v2.5 inpaint resource (large download)",
    ),
    Resource(
        "realesrgan-x4plus",
        "RealESRGAN x4plus upscaler",
        "Visualizer / Upscale & Save",
        ("upscale_models/RealESRGAN_x4plus.pth",),
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "upscale_models/RealESRGAN_x4plus.pth",
        "Xintao/Real-ESRGAN (BSD-3-Clause)",
    ),
    Resource(
        "realesrgan-x2plus",
        "RealESRGAN x2plus upscaler",
        "Visualizer / Upscale & Save",
        ("upscale_models/RealESRGAN-x2plus.pth", "upscale_models/RealESRGAN_x2plus.pth"),
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "upscale_models/RealESRGAN-x2plus.pth",
        "Xintao/Real-ESRGAN (BSD-3-Clause)",
    ),
    Resource(
        "realesrgan-x2-legacy",
        "Legacy RealESRGAN x2 upscaler: RealESRGAN_x2.pth",
        "Legacy examples",
        ("upscale_models/RealESRGAN_x2.pth",),
        license_note="Legacy workflow selection; install manually from its authorised source if needed.",
    ),
    Resource(
        "refiner-hyper-sdxl-lora",
        "Hyper-SDXL 1-step Refiner LoRA",
        "20 Refiner",
        ("loras/refiner/Hyper-SDXL-1step-lora.safetensors",),
        license_note="Optional Refiner LoRA; install manually from the official model repository.",
    ),
    Resource(
        "sdxl-dmd2-lora",
        "DMD2 SDXL 4-step LoRA",
        "SDXL sampling",
        ("loras/misc/dmd2_sdxl_4step_lora_fp16.safetensors",),
        license_note="Optional workflow LoRA; install manually from the official model repository.",
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


def _models_root(path: Path) -> Path:
    """Accept either a shared ComfyUI root or its actual models directory."""

    path = Path(path).expanduser()
    nested = path / "models"
    return nested if path.name != "models" and nested.is_dir() else path


def _candidate_roots(comfy_root: Path, explicit: Iterable[Path] = ()) -> list[Path]:
    roots = [_models_root(Path(value)) for value in explicit if value]
    roots.append(comfy_root / "models")
    shared = os.environ.get("CMK_SHARED_MODELS")
    if shared:
        roots.append(_models_root(Path(shared)))
    config = comfy_root / "extra_model_paths.yaml"
    if config.is_file():
        for line in config.read_text(encoding="utf-8", errors="ignore").splitlines():
            value = line.split("#", 1)[0].strip()
            if value.startswith("base_path:"):
                base = value.split(":", 1)[1].strip().strip("'\"")
                if base:
                    roots.append(_models_root(Path(base)))
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


def _human_size(value: int) -> str:
    size = float(max(0, value))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def _copy_with_progress(response, output, chunk_size: int = 1024 * 1024) -> int:
    header = response.headers.get("Content-Length") if response.headers else None
    total = int(header) if header and str(header).isdigit() else 0
    transferred = 0
    started = time.monotonic()
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        output.write(chunk)
        transferred += len(chunk)
        elapsed = max(0.001, time.monotonic() - started)
        speed = _human_size(int(transferred / elapsed)) + "/s"
        if total:
            percent = min(100.0, transferred * 100.0 / total)
            progress = (
                f"{percent:6.2f}% · {_human_size(transferred)} / "
                f"{_human_size(total)} · {speed}"
            )
        else:
            progress = f"{_human_size(transferred)} · {speed}"
        print(f"\rDownloading : {progress}", end="", flush=True)
    print()
    return transferred


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
            _copy_with_progress(response, output)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root",
        action="append",
        type=Path,
        default=[],
        help="shared ComfyUI root or models directory; repeatable",
    )
    parser.add_argument("--install", metavar="RESOURCE_ID", help="download exactly one resource into the selected model root")
    parser.add_argument("--target-root", type=Path, help="destination model root for --install (defaults to ComfyUI/models)")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="report missing resources without asking to install downloadable ones",
    )
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
        target_root = _models_root(args.target_root or (comfy_root / "models")).resolve()
        print(f"Installing   : {resource.label}")
        if resource.license_note:
            print(f"Notice       : {resource.license_note}")
        print(f"Source       : {resource.download_url}")
        print(f"Destination  : {target_root / resource.target_path}")
        print(f"Installed    : {_download(resource, target_root)}")
        return 0

    print("\nCMK resource audit")
    target_root = (
        args.target_root
        or (args.models_root[0] if args.models_root else None)
        or (comfy_root / "models")
    )
    target_root = _models_root(target_root).resolve()
    for resource, path in audit(comfy_root, args.models_root):
        status = "FOUND" if path else "MISSING"
        detail = f" -> {path}" if path else (f" | {resource.license_note}" if resource.license_note else "")
        print(
            f"[{status:7}] {resource.resource_id:30} {resource.label}"
            f" | {resource.feature}{detail}"
        )
        if path or not resource.download_url or args.non_interactive:
            continue
        answer = input(f"Install {resource.label} now into {target_root}? [y/N] ").strip().lower()
        if answer not in {"y", "yes", "j", "ja"}:
            print(f"Skipped     : {resource.resource_id}")
            continue
        print(f"Installing  : {resource.label}")
        print(f"Destination : {target_root / resource.target_path}")
        print(f"Source      : {resource.download_url}")
        installed = _download(resource, target_root)
        print(f"Installed   : {installed}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"CMK resource setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
