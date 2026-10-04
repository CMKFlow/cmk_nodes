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
import getpass
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Iterable
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
import zipfile


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
    expected_sha256: str | None = None
    archive_members: tuple[str, ...] = ()


RESOURCES = (
    Resource(
        "sdxl-checkpoint-juggernaut",
        "SDXL checkpoint: juggernautXL_ragnarok.safetensors",
        "SDXL / HYBRID generation",
        ("checkpoints/juggernautXL_ragnarok.safetensors",),
        "https://civitai.com/api/download/models/1759168?fileId=1659952",
        "checkpoints/juggernautXL_ragnarok.safetensors",
        "Civitai model version 1759168; creator terms apply (approximately 6.6 GiB)",
        expected_sha256="dd08fa32f98d05a2443ca1419e46df1575a0811f6e3b246d9dd47ff20f5eb66a",
    ),
    Resource(
        "sdxl-checkpoint-pony",
        "PostProcess checkpoint: Realism By Stable Yogi (Pony)XL_V3VAE.safetensors",
        "PostProcess",
        ("checkpoints/Realism By Stable Yogi (Pony)XL_V3VAE.safetensors",),
        "https://civitai.com/api/download/models/992946",
        "checkpoints/Realism By Stable Yogi (Pony)XL_V3VAE.safetensors",
        "Civitai model version 992946; creator terms apply (approximately 6.5 GiB)",
        expected_sha256="4796b66fd03fd9c8330df6cd44f6bf0cfbeccef6a010a80e61cb8a70d8edf56f",
    ),
    Resource(
        "sdxl-vae-clear",
        "SDXL VAE: ClearVAE_V2.2.safetensors",
        "SDXL / HYBRID / PostProcess",
        ("vae/ClearVAE_V2.2.safetensors",),
        "https://huggingface.co/theboylzh/ClearVAE/resolve/main/ClearVAE_V2.2.safetensors",
        "vae/ClearVAE_V2.2.safetensors",
        "theboylzh/ClearVAE (OpenRAIL; approximately 319 MiB)",
        expected_sha256="54b156d6ce34d0627ca0b63a824f58f5bf9c4e879549eb84ec499662726c4013",
    ),
    Resource(
        "sdxl-refiner",
        "SDXL Refiner 1.0",
        "20 Refiner",
        ("checkpoints/refiner/sd_xl_refiner_1.0.safetensors",),
        "https://civitai.com/api/download/models/126613",
        "checkpoints/refiner/sd_xl_refiner_1.0.safetensors",
        "SDXL Refiner 1.0, Civitai model version 126613 (approximately 5.7 GiB)",
        expected_sha256="7440042bbdc8a24813002c09b6b69b64dc90fded4472613437b7f55f9b7d9c5f",
    ),
    Resource(
        "sdxl-refiner-vae",
        "SDXL Refiner VAE",
        "20 Refiner",
        ("vae/sdxl_vae.safetensors",),
        "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
        "vae/sdxl_vae.safetensors",
        "Stability AI SDXL VAE (approximately 319 MiB)",
        expected_sha256="63aeecb90ff7bc1c115395962d3e803571385b61938377bc7089b36e81e92e2e",
    ),
    Resource(
        "sdxl-controlnet",
        "SDXL ControlNet: controlnetxlCNXL_2vxpswa7AnytestV4.safetensors",
        "05 ControlNet SDXL / Combined",
        ("controlnet/controlnetxlCNXL_2vxpswa7AnytestV4.safetensors",),
        "https://civitai.com/api/download/models/1296881?fileId=1201251",
        "controlnet/controlnetxlCNXL_2vxpswa7AnytestV4.safetensors",
        "ControlNetXL (CNXL) 2vXpSwA7 Anytest v4, Civitai model version 1296881 (approximately 2.3 GiB)",
        expected_sha256="807aa29189c10660dff77a5bbfcf5cf39d60f7780199db36db36a9096e11ace7",
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
            "insightface/models/buffalo_l/genderage.onnx",
            "insightface/models/buffalo_l/2d106det.onnx",
            "insightface/models/buffalo_l/det_10g.onnx",
            "insightface/models/buffalo_l/1k3d68.onnx",
            "insightface/models/buffalo_l/w600k_r50.onnx",
        ),
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "insightface/models/buffalo_l",
        "Official InsightFace buffalo_l pack; InsightFace pretrained-model terms apply (approximately 275 MiB)",
        require_all=True,
        expected_sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
        archive_members=(
            "genderage.onnx",
            "2d106det.onnx",
            "det_10g.onnx",
            "1k3d68.onnx",
            "w600k_r50.onnx",
        ),
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
        "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt",
        "ultralytics/bbox/face/face_yolov8m.pt",
        "Bingsu/adetailer face detector",
        expected_sha256="717923c19b3f4bbf5250b728f1fa6b2cb72a33aed1d236ea9caf0e21ad943e5f",
    ),
    Resource(
        "ultralytics-face-yolov8s",
        "Ultralytics face detector: face_yolov8s.pt",
        "Detailer",
        ("ultralytics/bbox/face/face_yolov8s.pt",),
        "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt",
        "ultralytics/bbox/face/face_yolov8s.pt",
        "Bingsu/adetailer face detector",
        expected_sha256="c7237eff25787377de196961140ceaed324d859ee8de5a775d93d33a0e3fab78",
    ),
    Resource(
        "ultralytics-hand-yolov8n",
        "Ultralytics hand detector: hand_yolov8n.pt",
        "Detailer",
        ("ultralytics/bbox/div/hand_yolov8n.pt",),
        "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt",
        "ultralytics/bbox/div/hand_yolov8n.pt",
        "Bingsu/adetailer hand detector",
        expected_sha256="f3f23b865741cc8373a76dfac31a71ffd71356a480ca43266f294815b608e174",
    ),
    Resource(
        "ultralytics-hand-yolov8s",
        "Ultralytics hand detector: hand_yolov8s.pt",
        "Detailer",
        ("ultralytics/bbox/div/hand_yolov8s.pt",),
        "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt",
        "ultralytics/bbox/div/hand_yolov8s.pt",
        "Bingsu/adetailer hand detector",
        expected_sha256="70b540063fbc385736d8258970744a4afbc4cbf7932134bae3b24cdadeadec06",
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
        "https://github.com/visomaster/visomaster-assets/releases/download/v0.1.0/GPEN-BFR-512.onnx",
        "facerestore_models/GPEN-BFR-512.onnx",
        "GPEN ONNX conversion distributed by VisoMaster; GPEN model terms apply (approximately 271 MiB)",
        expected_sha256="0960f836488735444d508b588e44fb5dfd19c68fde9163ad7878aa24d1d5115e",
    ),
    Resource(
        "faceswap-inswapper-128",
        "FaceSwap model: inswapper_128.onnx",
        "FaceSwap / Legacy video FaceSwap",
        ("insightface/inswapper_128.onnx", "insightface/models/inswapper_128.onnx"),
        "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx",
        "insightface/inswapper_128.onnx",
        "Official InsightFace model-zoo asset; non-commercial research terms apply (approximately 529 MiB)",
        expected_sha256="e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
    ),
    Resource(
        "faceswap-hyperswap-1b",
        "FaceSwap model: hyperswap_1b_256.onnx",
        "FaceSwap",
        ("insightface/hyperswap/hyperswap_1b_256.onnx", "insightface/hyperswap_1b_256.onnx"),
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1b_256.onnx",
        "insightface/hyperswap/hyperswap_1b_256.onnx",
        "Official FaceFusion asset (ResearchRAIL; approximately 384 MiB)",
        expected_sha256="5124031789c42f71b9558fb71954ef7aedb6da7ed9fac79293e23c61a792a73e",
    ),
    Resource(
        "faceswap-hyperswap-1c",
        "FaceSwap model: hyperswap_1c_256.onnx",
        "FaceSwap",
        ("insightface/hyperswap/hyperswap_1c_256.onnx", "insightface/hyperswap_1c_256.onnx"),
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1c_256.onnx",
        "insightface/hyperswap/hyperswap_1c_256.onnx",
        "Official FaceFusion asset (ResearchRAIL; approximately 384 MiB)",
        expected_sha256="5528c2d76fe9986c99d829278987ef9f3a630cb606db7628d02b57b330f406a5",
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
        "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth",
        "upscale_models/RealESRGAN_x2.pth",
        "ai-forever/Real-ESRGAN legacy x2 weight",
        expected_sha256="c830d067d54fc767b9543a8432f36d91bc2de313584e8bbfe4ac26a47339e899",
    ),
    Resource(
        "refiner-hyper-sdxl-lora",
        "Hyper-SDXL 1-step Refiner LoRA",
        "20 Refiner",
        (
            "loras/refiner/Hyper-SDXL-1step-lora.safetensors",
            "loras/SDXL/refiner/Hyper-SDXL-1step-lora.safetensors",
        ),
        "https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-1step-lora.safetensors",
        "loras/refiner/Hyper-SDXL-1step-lora.safetensors",
        "Official ByteDance Hyper-SD repository",
        expected_sha256="c912df184c5116792d2c604d26c6bc2aa916685f4a793755255cda1c43a3c78a",
    ),
    Resource(
        "sdxl-dmd2-lora",
        "DMD2 SDXL 4-step LoRA",
        "SDXL sampling",
        (
            "loras/misc/dmd2_sdxl_4step_lora_fp16.safetensors",
            "loras/SDXL/misc/dmd2_sdxl_4step_lora_fp16.safetensors",
        ),
        "https://huggingface.co/tianweiy/DMD2/resolve/main/dmd2_sdxl_4step_lora_fp16.safetensors",
        "loras/misc/dmd2_sdxl_4step_lora_fp16.safetensors",
        "Official tianweiy/DMD2 repository",
        expected_sha256="b3d9173815a4b595991c3a7a0e0e63ad821080f314a0b2a3cc31ecd7fcf2cbb8",
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


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_download(resource: Resource, path: Path) -> None:
    if not resource.expected_sha256:
        return
    actual = _sha256(path)
    if actual.lower() != resource.expected_sha256.lower():
        raise RuntimeError(
            f"Checksum mismatch for {resource.resource_id}: "
            f"expected {resource.expected_sha256}, got {actual}"
        )
    print(f"Verified    : SHA-256 {actual}")


def _extract_archive(resource: Resource, archive: Path, target: Path) -> Path:
    if not resource.archive_members:
        raise RuntimeError(f"{resource.resource_id} has no archive member manifest.")
    staging = target.with_name(target.name + ".extracting")
    if staging.exists():
        raise FileExistsError(f"Archive staging directory already exists: {staging}")
    try:
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
            missing = [name for name in resource.archive_members if name not in names]
            if missing:
                raise RuntimeError(
                    f"Archive for {resource.resource_id} is missing: {', '.join(missing)}"
                )
            staging.mkdir(parents=True, exist_ok=False)
            for name in resource.archive_members:
                member_target = staging / Path(name).name
                with bundle.open(name) as source, member_target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def _is_civitai_download(resource: Resource) -> bool:
    return bool(resource.download_url and resource.download_url.startswith("https://civitai.com/"))


def _download_request(resource: Resource, civitai_token: str | None = None) -> Request:
    headers = {"User-Agent": "CMK-resource-installer/1"}
    url = str(resource.download_url)
    if civitai_token and _is_civitai_download(resource):
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["token"] = civitai_token
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    return Request(url, headers=headers)


def _download(
    resource: Resource,
    destination: Path,
    civitai_token: str | None = None,
) -> Path:
    if not resource.download_url or not resource.target_path:
        raise RuntimeError(f"{resource.resource_id} has no approved automatic download source.")
    target = destination / resource.target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Target already exists: {target}")
    suffix = ".zip.part" if resource.archive_members else ".part"
    temporary = target.with_name(target.name + suffix)
    request = _download_request(resource, civitai_token)
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            _copy_with_progress(response, output)
        _verify_download(resource, temporary)
        if resource.archive_members:
            installed = _extract_archive(resource, temporary, target)
            temporary.unlink(missing_ok=True)
            return installed
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _collect_masked_input(read_character, output) -> str:
    characters = []
    while True:
        character = read_character()
        if character in ("\r", "\n", ""):
            break
        if character == "\x03":
            raise KeyboardInterrupt
        if character in ("\x08", "\x7f"):
            if characters:
                characters.pop()
                output.write("\b \b")
                output.flush()
            continue
        if character.isprintable():
            characters.append(character)
            output.write("*")
            output.flush()
    return "".join(characters)


def _masked_input(prompt: str) -> str:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return getpass.getpass(prompt)

    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        if os.name == "nt":
            import msvcrt

            value = _collect_masked_input(msvcrt.getwch, sys.stdout)
        else:
            import termios
            import tty

            descriptor = sys.stdin.fileno()
            previous = termios.tcgetattr(descriptor)
            try:
                tty.setraw(descriptor)
                value = _collect_masked_input(lambda: sys.stdin.read(1), sys.stdout)
            finally:
                termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return value


def _download_with_civitai_retry(
    resource: Resource,
    destination: Path,
    civitai_token: str | None,
    allow_prompt: bool,
) -> tuple[Path | None, str | None]:
    try:
        return _download(resource, destination, civitai_token), civitai_token
    except HTTPError as exc:
        if exc.code not in (401, 403) or not _is_civitai_download(resource):
            raise
        print("Civitai requires authentication for this resource.")
        print("Create an API key in your Civitai account settings:")
        print("https://civitai.com/user/account")
        if not allow_prompt:
            raise RuntimeError(
                "Civitai authentication required. Set CIVITAI_API_TOKEN and run again."
            ) from exc
        token = _masked_input(
            "Civitai API key (shown as *; press Enter to skip): "
        ).strip()
        if not token:
            return None, civitai_token
        print(f"API key received: {len(token)} characters.")
        try:
            return _download(resource, destination, token), token
        except HTTPError as retry_exc:
            if retry_exc.code in (401, 403):
                raise RuntimeError(
                    "Civitai rejected the API key. Create a Personal API Key in "
                    "Civitai account settings, copy the complete value, and try again."
                ) from retry_exc
            raise


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
    civitai_token = os.environ.get("CIVITAI_API_TOKEN", "").strip() or None

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
        try:
            installed, civitai_token = _download_with_civitai_retry(
                resource,
                target_root,
                civitai_token,
                allow_prompt=not args.non_interactive,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"Failed       : {resource.resource_id} | {exc}", file=sys.stderr)
            return 1
        if installed is None:
            print(f"Skipped      : {resource.resource_id}")
            return 1
        print(f"Installed    : {installed}")
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
        print(f"Source      : {resource.download_url}")
        print(f"Destination : {target_root / resource.target_path}")
        if resource.expected_sha256:
            print(f"SHA-256     : {resource.expected_sha256}")
        answer = input(f"Install {resource.label} now into {target_root}? [y/N] ").strip().lower()
        if answer not in {"y", "yes", "j", "ja"}:
            print(f"Skipped     : {resource.resource_id}")
            continue
        print(f"Installing  : {resource.label}")
        try:
            installed, civitai_token = _download_with_civitai_retry(
                resource,
                target_root,
                civitai_token,
                allow_prompt=True,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"Failed       : {resource.resource_id} | {exc}", file=sys.stderr)
            continue
        if installed is None:
            print(f"Skipped     : {resource.resource_id}")
            continue
        print(f"Installed   : {installed}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"CMK resource setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
