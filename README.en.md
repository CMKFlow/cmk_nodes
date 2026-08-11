<p align="center">
  <img src="web/assets/brand/cmk-logo.png" width="140" alt="CMK Flow logo">
</p>

# CMK Flow

Modular custom-node package for ComfyUI.

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

The `CMK Flow Browser` provides the current Flow modules, the open toolbox, and
curated reference workflows directly inside ComfyUI.

The model families use separate, type-safe process paths:

```text
SDXL: 01 → optional 05 → 10 → 20 → optional 25 → optional 30 → optional 40 → 90
ZIT:  01 → optional 05 → 10                                → optional 40 → 90
```

`25 Detailer SDXL` and `30 FaceProcess SDXL` are intentionally SDXL-only.
`40 FaceSwap` and `90 Upscale & Save` are shared. A single family can connect
to them directly; only a combined SDXL/ZIT workflow needs `35 Active Family
Result` to select the active path first.

The public transport roles are:

```text
MODEL | PROCESS | IMAGE | LOG
```

Between the SDXL sampler and refiner, the proprietary latent handoff type
`SAMPLED` replaces `IMAGE`.

## Key features

- explicit separation of model resources, process state, image data, and logs;
- proprietary Prepare/Execute contracts that prevent ambiguous wiring;
- lazy SDXL/ZIT family routing and cache-safe optional modules;
- parallel Smart Detailer and FaceProcess instances;
- dynamic `SEGS`, `LOG BLOCK`, and `DIAGNOSTIC` inputs;
- standalone Detailer and FaceProcess operation through CMK loaders;
- aspect-ratio-safe Fit/Crop preparation, positioned cropping, mask alignment,
  Replace, Remove, Extend, and controlled outpainting overlap;
- local FaceSwap image/video paths with mandatory ContentGuard;
- curated SDXL, ZIT, ControlNet, FaceSwap, Inpaint, and combined Full Flow
  references in the Flow Browser;
- dedicated Z-Image Turbo loader, sampler, and optional ControlNet modules.

ZIT-Inpaint is explicitly marked `EXPERIMENTAL` and temporarily frozen because
of its very high memory and runtime requirements.

## Installation

Run inside the target ComfyUI installation:

```bash
cd /path/to/ComfyUI
git clone https://github.com/CMKFlow/cmk_nodes.git custom_nodes/cmk_nodes
python -m pip install -r custom_nodes/cmk_nodes/requirements.txt
```

Fully stop and restart ComfyUI. To update:

```bash
git -C custom_nodes/cmk_nodes pull --ff-only
python -m pip install -r custom_nodes/cmk_nodes/requirements.txt
```

For an existing manual installation, back up and completely replace the old
folder. Do not leave historical individual files beside the current package.
The JSON files in `subgraphs/` remain part of the node pack; additional copies
under `user/default/subgraphs/` create duplicate blueprint entries.

## Required ComfyUI frontend

> **Validated release target / temporary standby**
>
> This release is comprehensively validated with **ComfyUI 0.28.2** and
> **comfyui-frontend-package 1.45.21**. With **ComfyUI 0.31.1** and frontend
> **1.48.7**, live and final previews on outer subgraph nodes may remain blank
> even though execution, image transport, saving, and results remain correct.
> Adaptation to newer ComfyUI versions is temporarily on **standby** while the
> upstream subgraph-preview behavior is being revised. CMK will re-test after a
> relevant frontend update before adding package-specific workarounds.
>
> Upstream context: [missing subgraph live previews](https://github.com/Comfy-Org/ComfyUI_frontend/issues/9859) and [custom-node preview detection](https://github.com/Comfy-Org/ComfyUI_frontend/issues/10531).

CMK Flow requires **Vue Nodes / Nodes 2.0** in the active ComfyUI user profile.
Without it, dynamic CMK nodes fall back to legacy LiteGraph rendering and their
Advanced switching, dropdowns, shapes, and automatic sizing do not work as
designed. CMK shows a startup notice when this setting is disabled.

For sampler and refiner previews, use **Comfy → Execution → Live preview
method → auto**.

## Runtime dependencies

The CMK core nodes for detection, `SEGS`, Detailer, pasteback, FaceProcess
restore, SAM loading, and the included ControlNet preprocessors do not require
third-party custom-node packs. The optional `02 SDXL LoRA Stack` subgraph and
references that use it require the
[ComfyUI LoRA Manager](https://github.com/willmiao/ComfyUI-Lora-Manager).
All other CMK modules remain usable without it.

Python dependencies are listed in `requirements.txt`. Model-backed features
still require the selected Ultralytics, SAM, InsightFace, optional face-restore,
ControlNet, checkpoint, VAE, and upscale models. Missing models fail only the
selected feature path with a clear runtime message.

`Remove Object` uses the locally executed
[LaMa model](https://github.com/advimman/lama). On first use, CMK downloads
`big-lama.pt` to `ComfyUI/models/inpaint/` and verifies its known MD5 checksum.
Afterwards the feature runs locally. LaMa and the used
[IOPaint model distribution](https://github.com/Sanster/IOPaint) retain their
independent Apache License 2.0 terms.

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

The technical `CMK FaceSwap Video` reference remains separate because its
project, segment, and continuation contract needs dedicated migration and
compatibility testing.

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
