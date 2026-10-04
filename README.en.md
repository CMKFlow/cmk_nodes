<p align="center">
  <img src="web/assets/brand/cmk-logo.png" width="140" alt="CMK Flow logo">
</p>

# CMK Flow

Modular custom-node package for ComfyUI.

**Current release: CMK 2.5.4**

[Deutsch](README.md) · **English**

[Source](https://github.com/CMKFlow/cmk_nodes) · [Documentation](#documents) · [Support the project](https://paypal.me/CMKFlow)

Copyright (C) 2026 Carsten Kirschner

CMK Flow deliberately separates two concepts:

```text
CMK **** -Pipe- → closed, guided ecosystem
all other nodes → open experimental toolbox
```

[`ARCHITECTURE.md`](ARCHITECTURE.md) is the authoritative architecture and interface contract.

## Inspiration and acknowledgement

The [ComfyUI LoRA Manager by willmiao](https://github.com/willmiao/ComfyUI-Lora-Manager)
strongly influenced the expectations for the integrated CMK Flow Browser.
CMK Flow is an independent project with no official affiliation. This
acknowledgement credits the design inspiration and the work behind that
open-source project; it does not claim code reuse.

## Getting started

The `CMK Flow Browser` is the central interface between users and CMK. Inside
ComfyUI, it provides a concise overview, quick access to nodes and subgraphs,
real previews, and curated reference workflows. It is a working tool rather
than a manual or technical reference.

CMK 2.5 separates the model-specific generation stage from the
family-neutral post-processing stage:

```text
Preparation: Loader / LoRA → 01 START HERE
Generation:  02 Regional Conditioning (optional, SDXL)
             → 05 ControlNet (optional)
             → 10 KSampler
             → 15 InstantID (optional, SDXL)
             → 20 Refiner (SDXL)
Handoff:     PostProcess Boundary (SDXL / Z-Image Turbo / Combined)
PostProcess: FaceRebuild · FaceSwap · FaceProcess · Detailer · MaskDetailer
Output:      Visualizer and/or Upscale & Save
```

SDXL, Z-Image Turbo, and HYBRID remain technically separate during generation.
HYBRID first builds the image with SDXL, then hands it to Z-Image Turbo for the
final finish. The appropriate `PostProcess Boundary` ends generation and
provides an independent, family-neutral context for subsequent processing. In
workflows with parallel model families, the `Combined` variant accepts only the
active SDXL, ZIT, or HYBRID branch.

The public transport roles are:

```text
MODEL | PROCESS | IMAGE | LOG | VISUAL
```

Between the SDXL sampler, InstantID, and refiner, the proprietary latent
handoff type `SAMPLED` replaces `IMAGE`. Visible titles are presentational
only; technical identity and navigation use metadata, node classes, provider
keys, and UUIDs.

## Key features

- explicit separation of model resources, process state, image data, logs, and visualization;
- proprietary Prepare/Execute contracts that prevent ambiguous wiring;
- separate SDXL, Z-Image Turbo, and HYBRID generation paths followed by a
  shared family-neutral post-processing stage;
- parallel Smart Detailer and FaceProcess instances;
- dynamic `SEGS`, `LOG BLOCK`, and `DIAGNOSTIC` inputs;
- persistent branch caches for unchanged parallel instances;
- mandatory module boundaries before comparers, downstream modules, and public outputs;
- a central `VISUAL` chain for processing stages registered with the Visualizer;
- standalone post-processing modules as either discrete workflows or packaged subgraphs;
- aspect-ratio-safe Fit/Crop preparation, positioned cropping, mask alignment,
  Replace, Remove, Extend, and controlled outpainting overlap;
- local FaceSwap image/video paths with mandatory ContentGuard;
- a Flow Browser reference catalog organized into Task Workflows, Module
  Workflows, Comparisons, Real-World Workflows, System Workflow, and Legacy,
  including 20 task workflows ordered by increasing functional complexity;
- dedicated Z-Image Turbo loader, sampler, and optional ControlNet modules.

ZIT-Inpaint is explicitly marked `EXPERIMENTAL` and temporarily frozen because
of its very high memory and runtime requirements.

## Installation

### Install with ComfyUI Manager

Once CMK has been published to the Comfy Registry, search for `CMK Flow` in
ComfyUI Manager, install the entry published by `CMKFlow`, and restart ComfyUI.
The Manager downloads CMK and installs the Python dependencies from
`requirements.txt` automatically. Model weights are intentionally kept outside
the Python package and are checked afterwards with CMK's resource audit.

The Registry package id is `cmk-flow`. CMK itself does not depend on a fixed
installation-directory name, so Manager-normalized and manually selected folder
names are both supported.

The Manager installation does not download model resources automatically and
does not adopt or migrate files from historical CMK installations. The separate
resource audit remains the explicitly intended next step.

### Alternative: manual installation from GitHub

Fully quit ComfyUI before starting the installation.

#### 1. Open the correct ComfyUI folder

Open the root folder of your ComfyUI installation. This is the folder that
contains `main.py` and the `custom_nodes` subfolder. Open a terminal **in that
exact folder**.

If `custom_nodes/cmk_nodes` already exists there, do not install over it. Back
up or completely remove the existing folder first.

#### 2. Download CMK and install its dependencies

Copy these two commands into the terminal one after the other:

```bash
git clone https://github.com/CMKFlow/cmk_nodes.git custom_nodes/cmk_nodes
python3 custom_nodes/cmk_nodes/scripts/install_cmk_requirements.py
```

The installer derives the Python environment actually used by ComfyUI from the
repository location, installs `requirements.txt` into that environment, and
then verifies every mandatory CMK import. This matters especially with ComfyUI
Desktop, where the system Python, `standalone-env`, and `ComfyUI/.venv` may all
exist side by side.

The installation succeeded when the final line reads:

```text
CMK dependency check: OK
```

### Audit model resources

Python dependencies do not include model weights. Run an explicit resource
audit against the installation and, when used, the shared model directory:

```bash
python3 custom_nodes/cmk_nodes/scripts/check_cmk_resources.py \\
  --models-root "/path/to/ComfyUI-Shared"
```

After a Manager installation, the same script normally resides at
`custom_nodes/cmk-flow/scripts/check_cmk_resources.py`. The audit locates
ComfyUI from its own file location and works independently of the CMK
installation-directory name.

When a shared model directory is used, it is the preferred destination for
missing resources and is supplied through `--models-root`. CMK does not move or
copy models from older installations; existing resources are searched only in
the explicitly supplied model roots registered with ComfyUI.

On the first run with `--models-root`, the helper also registers that shared
model directory in `ComfyUI/extra_model_paths.yaml`, so ComfyUI can use the
discovered models at runtime. Existing configuration is preserved; the CMK
block is added at most once. Restart ComfyUI after the audit so the paths are
loaded.

The helper checks the exact filenames selected by the bundled CMK 2.5
workflows; a merely non-empty model directory does not count as a match. Every
resource is reported independently as `FOUND` or `MISSING`, together with the
CMK feature that uses it. Immediately after a missing resource with an
unambiguous public upstream download, the helper asks whether to install it.
`n` continues the audit; `y` installs that file first, with download progress,
into the correct ComfyUI model directory.

When several installable resources are missing, two modes are available:
`INSTALL ALL MISSING` confirms and installs all of them in one run;
`INSTALL ONE BY ONE` asks for each resource separately. Scripts can select a
mode directly with `--install-mode all` or `--install-mode one-by-one`.

The fixed CMK resources for InstantID, Z-Image Turbo, SAM, GFPGAN, Fooocus
Inpaint, and RealESRGAN can be installed directly. Checkpoints, detector
weights, and FaceSwap weights that involve a model choice or separate licence
terms are all reported but are deliberately not downloaded automatically. In
those cases the report identifies the affected feature and why manual
installation is required:

```bash
python3 custom_nodes/cmk_nodes/scripts/check_cmk_resources.py \\
  --models-root "/path/to/ComfyUI-Shared" \\
  --install instantid-adapter \\
  --target-root "/path/to/ComfyUI-Shared"

python3 custom_nodes/cmk_nodes/scripts/check_cmk_resources.py \\
  --models-root "/path/to/ComfyUI-Shared" \\
  --install instantid-controlnet \\
  --target-root "/path/to/ComfyUI-Shared"
```

Several Z-Image and ControlNet files are multiple gigabytes in size, so each
one requires separate confirmation. The resource audit never changes workflows
or technical identities.

Use `--non-interactive` for automated audits without prompts.

### Restart ComfyUI

Start ComfyUI again only after the success message appears. The Flow Browser is
registered automatically when the CMK package loads.

### Updating

For Manager installations, use **Update** in ComfyUI Manager; the Manager also
maintains the dependencies declared in `requirements.txt`.

For a manual Git update, open a terminal in the ComfyUI root folder again and
run:

```bash
git -C custom_nodes/cmk_nodes pull --ff-only
python3 custom_nodes/cmk_nodes/scripts/install_cmk_requirements.py
```

A manual Git clone never runs `pip` on its own. The second command is therefore
a required part of every installation and update.

Do not leave historical individual files beside the current package. The JSON
files in `subgraphs/` remain part of the node pack; additional copies under
`user/default/subgraphs/` create duplicate blueprint entries.

## Required ComfyUI frontend

> **Validated release target**
>
> CMK 2.5.4 was fully validated with **ComfyUI 0.38.2** and
> **comfyui-frontend-package 1.53.6**. This includes Flow Browser navigation,
> packaged subgraphs and reference workflows, embedded previews, cache paths,
> and the serialized link, socket, UUID, and topology contracts.

CMK Flow requires **Vue Nodes / Nodes 2.0** in the active ComfyUI user profile.
Without it, dynamic CMK nodes fall back to legacy LiteGraph rendering and their
Advanced switching, dropdowns, shapes, and automatic sizing do not work as
designed. CMK shows a startup notice when this setting is disabled.

For sampler and refiner previews, use **Comfy → Execution → Live preview
method → auto**.

## Runtime dependencies

The CMK core nodes for detection, `SEGS`, Detailer, pasteback, FaceProcess
restore, SAM loading, and the included ControlNet preprocessors do not require
third-party custom-node packs. The optional `LoRA Stack · SDXL`, `LoRA Stack ·
ZIT`, and `LoRA Stack · Combined` subgraphs, as well as references that use
them, require the
[ComfyUI LoRA Manager](https://github.com/willmiao/ComfyUI-Lora-Manager).
All other CMK modules remain usable without it.

Python dependencies are listed in `requirements.txt`. Model-backed features
still require the selected Ultralytics, SAM, InsightFace, optional face-restore,
ControlNet, checkpoint, VAE, and upscale models. Missing models fail only the
selected feature path with a clear runtime message.

## FaceSwap ContentGuard

All public CMK FaceSwap paths use a mandatory local ContentGuard. It checks the
source and target before a swap and every target frame in video processing.
Explicit content, a target age estimate below 18, a source estimate below the
conservative threshold of 25, or missing/invalid protection models cause a hard
failure. No public bypass is provided. Disabled FaceSwap nodes remain true
pass-through paths and do not load the guard.

The guard runs locally with NudeNet and InsightFace. It is a technical risk
control, not a consent check and not a guarantee against misclassification.
See [`CONTENT_GUARD.md`](CONTENT_GUARD.md) for the authoritative policy.

## Cache behavior

Internal CMK caches are stored under:

```text
ComfyUI/temp/cmk/
```

The first execution after a restart, code change, or cache cleanup normally
produces `MISS → STORED`. Unchanged branches and module boundaries may then
resolve as `HIT` during the same ComfyUI session. Restarting ComfyUI clears
these image-boundary caches; only the explicitly persistent video workflow has
a cross-session continuation contract.

Cache entries carry revision and dependency markers. A branch is reused only
when its manifest matches the currently materialized revisions. Caches are
temporary accelerators, not a portable project format.

## Video processing

`CMK Split Video into Segments` reads from `input/video/`, writes source-scoped
segments to `output/video/segments/<video_name>/`, and returns the persistent
`CMK_VIDEO_SEGMENTS` work context.

`CMK Merge and Save Video` merges overlap-safe segments into
`output/video/merged/<video_name>/`, previews the result, and can publish it
without another encode. `CMK FaceSwap Video Loader` adds video and source-image
selection to the persistent Split path.

`CMK FaceSwap Video` is the historical legacy reference that started the CMK
project. It dates back to CMK 1.0, was intentionally not modernized to the CMK
2.5 architecture, and continues to run unchanged in the current environment.

## Documents

| Document | Purpose |
|---|---|
| `ARCHITECTURE.md` | authoritative interfaces and architecture |
| `CMK_Design_Guidelines.md` | UI, naming, and presentation rules |
| `README.md` | German installation and project overview |
| `README.en.md` | English installation and project overview |
| `CHANGELOG.md` | historical changes, not the current API contract |
| `WORKFLOWS.md` | subgraph, workflow, and installation overview |
| `SUBGRAPH_AUDIT.md` | subgraph inventory and safe cleanup plan |
| `TOOLBOX.md` | open-toolbox product boundary and maintenance plan |
| `CMK_FLOW_COMPATIBILITY.md` | draft integration contract for third parties |
| `CONTENT_GUARD.md` | authoritative local FaceSwap protection policy |
| `RELEASE_AUDIT_CMK_2_5.md` | final audit of CMK 2.5 contracts, tests, and release deviations |

## License

CMK Flow is free software under the **GNU General Public License version 3 or,
at your option, any later version** (`GPL-3.0-or-later`). It may be used
privately and commercially, studied, modified, and redistributed. Distributed
versions and modifications must remain under the same license, preserve the
notices, and make their source available. The software is provided without
warranty. See [`LICENSE`](LICENSE) for the complete terms. Independent licenses
for models and other dependencies continue to apply.

## Development rule

A working technical solution is not yet a CMK solution when it creates
unnecessary user decisions, makes miswiring easy, moves internal complexity
into the main workflow, or reruns unchanged expensive modules. In those cases,
the architecture wins.
