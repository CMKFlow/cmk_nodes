"""Scenario functions executed inside a (fresh) harness process.

Each scenario boots the runtime for a given work directory, runs prompts and
returns a JSON-serialisable record.  Flat API prompts here contain no
subgraph instances; prompts with subgraphs come exclusively from fixtures
exported by the real ComfyUI frontend (see FIXTURES_EXPORT.md).
"""

from __future__ import annotations

import json
from pathlib import Path

SELFCHECK_INPUT = "cmk_s1_input.png"


FIXTURE_IMAGES = Path(__file__).resolve().parents[1] / "fixtures" / "images"


def write_selfcheck_input(workdir: Path, source: str = SELFCHECK_INPUT) -> None:
    """Place a fixture image under the harness input name (fixtures/make_images.py)."""
    import shutil

    target = Path(workdir) / "input" / SELFCHECK_INPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURE_IMAGES / source, target)


def selfcheck_prompt() -> dict:
    """Image input → Upscale & Save boundary → Visualizer, without any model."""
    return {
        "1": {"class_type": "CMKImageLoadAndResizePipe", "inputs": {
            "image": SELFCHECK_INPUT, "RESOLUTION": "1024x1024",
            "SWAP DIMENSIONS": False, "RESIZE METHOD": "lanczos",
            "CROP": True, "CROP POSITION": "center",
        }},
        "2": {"class_type": "CMKUpscaleSaveBoundaryCache", "inputs": {
            "IMAGE": ["1", 2], "LOG": ["1", 3], "diagnostic": ["1", 4],
        }},
        "3": {"class_type": "CMKVisualizer", "inputs": {
            "PROCESS": ["1", 1], "IMAGE": ["2", 1], "LOG": ["2", 2],
            "SAVE ENABLED": False, "filename prefix": "s1",
        }},
    }


def _scope_status(workdir: Path, scope: str):
    path = Path(workdir) / "temp" / "cmk" / scope / "last_status.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("status")


def selfcheck_chain(workdir: str, probes: bool = False, runs: int = 2, mode: str = "classic", seams: bool = False,
                    extra_model_paths: str | None = None) -> dict:
    from .comfy_runtime import boot, tensor_hash
    from .probes import install_probes
    from .seams import Seams, install_standard

    workdir = Path(workdir)
    runtime = boot(workdir, extra_model_paths=extra_model_paths)
    write_selfcheck_input(workdir)
    runtime.new_executor(mode)
    installed = install_probes(runtime) if probes else None
    seam_set = Seams()
    if seams:
        install_standard(seam_set, runtime)
    records = []
    try:
        for _ in range(runs):
            record = runtime.run(selfcheck_prompt())
            records.append({
                "succeeded": record.succeeded,
                "executed": record.executed,
                "entered": record.entered_nodes,
                "cached": record.cached,
                "boundary_status": _scope_status(workdir, "upscale_save_boundary"),
                "image_hash": tensor_hash(runtime.output("2")[1]),
                "visual": json.loads(record.ui.get("3", {}).get("cmk_visual", ["{}"])[0]),
            })
    finally:
        snapshot = installed.snapshot() if installed else None
        if installed:
            installed.uninstall()
        seam_set.uninstall()
    return {"runs": records, "probes": snapshot}


def scope_inventory(workdir: str) -> dict:
    """Collect every declared persistent cache scope from the loaded CMK modules.

    Scopes are read from module constants and class attributes whose names end
    in ``SCOPE`` (the convention used by every CMK cache), not from source text.
    """
    import inspect

    from .comfy_runtime import boot

    runtime = boot(Path(workdir))
    declared = {}
    for module_name, module in runtime.cmk_modules.items():
        short = module_name.split(".")[-1]
        for name, value in vars(module).items():
            if name.upper().endswith("SCOPE") and isinstance(value, str):
                declared.setdefault(value, set()).add(f"{short}.{name}")
            if inspect.isclass(value) and value.__module__ == module.__name__:
                for attribute, attribute_value in vars(value).items():
                    if attribute.upper().endswith("SCOPE") and isinstance(attribute_value, str):
                        declared.setdefault(attribute_value, set()).add(f"{short}.{value.__name__}.{attribute}")
    # Caches with their own directory helper instead of a SCOPE constant
    # (e.g. the Refiner boundary) are found by calling that helper inside the
    # isolated work directory.
    temp_cmk = (Path(workdir) / "temp" / "cmk").resolve()
    for module_name, module in runtime.cmk_modules.items():
        helper = vars(module).get("_cache_directory")
        if callable(helper) and not inspect.signature(helper).parameters:
            directory = Path(helper()).resolve()
            if directory.parent == temp_cmk:
                declared.setdefault(directory.name, set()).add(f"{module_name.split('.')[-1]}._cache_directory")
    persistent = runtime.cmk_module("pipe.cmk_persistent_cache")
    completion = runtime.cmk_module("pipe.cmk_generation_completion_cache")
    return {
        "declared": {scope: sorted(owners) for scope, owners in declared.items()},
        "registered": list(persistent.PERSISTENT_DISK_SCOPES),
        "completion_scope": completion.CACHE_SCOPE,
    }


POSTPROCESS_MODULES = (
    "CMK Flow · Detailer SDXL", "CMK Flow · Detailer SDXL · Advanced",
    "CMK Flow · FaceProcess SDXL", "CMK Flow · FaceProcess SDXL · Advanced",
    "CMK Flow · FaceRebuild SDXL", "CMK Flow · FaceRebuild SDXL · Advanced",
    "CMK Flow · FaceSwap", "CMK Flow · FaceSwap · Advanced",
    "CMK Flow · MaskDetailer SDXL", "CMK Flow · Upscale & Save",
)
TRANSPORT_ROLES = ("MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL")


def _type_string(spec):
    value = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
    return value if isinstance(value, str) else None  # combo lists are not link types


def graph_type_check(workdir: str) -> dict:
    """V-C4 with the real node classes and ComfyUI's validate_node_input."""
    import sys

    from .comfy_runtime import TESTS_ROOT, boot

    boot(Path(workdir))
    import nodes
    from comfy_execution.validation import validate_node_input

    sys.path.insert(0, str(TESTS_ROOT))
    from validator import checks
    from validator.graph_model import load_workflow
    from validator.report import PACKAGE_ROOT, canonical_definitions, shipped_workflows

    catalog = {}
    for class_type, cls in nodes.NODE_CLASS_MAPPINGS.items():
        try:
            declared = cls.INPUT_TYPES()
        except Exception:
            continue
        inputs = {}
        for group in ("required", "optional"):
            for name, spec in (declared.get(group) or {}).items():
                text = _type_string(spec)
                if text is not None:
                    inputs[name] = text
        catalog[class_type] = {"inputs": inputs, "outputs": list(getattr(cls, "RETURN_TYPES", ()))}

    findings = []
    files = sorted((PACKAGE_ROOT / "subgraphs").glob("*.json")) + shipped_workflows()
    for path in files:
        top, definitions = load_workflow(path)
        local = dict(catalog)
        for definition in definitions:
            interface = definition.public_interface()
            local[definition.id] = {
                "inputs": dict(interface["inputs"]),
                "outputs": [type_ for _name, type_ in interface["outputs"]],
            }
        for graph in [top, *definitions]:
            for finding in checks.link_type_compatibility(graph, local, validate_node_input):
                findings.append({"file": path.name, "graph": graph.name, **finding})

    # Composability of regular PostProcess modules: A's public outputs into B's
    # same-role public inputs, judged by ComfyUI's own type rule.
    canon = canonical_definitions()
    incompatible = []
    for source in POSTPROCESS_MODULES:
        outputs = dict(canon[source].public_interface()["outputs"])
        for target in POSTPROCESS_MODULES:
            inputs = canon[target].public_interface()["inputs"]
            for role in TRANSPORT_ROLES:
                out_type = next((t for n, t in outputs.items() if n.split(" ")[0] == role), None)
                in_type = next((t for n, t in inputs if n.split(" ")[0] == role or (role == "IMAGE" and n == "IMAGE_TARGET")), None)
                if out_type and in_type and not validate_node_input(out_type, in_type):
                    incompatible.append([source, target, role, out_type, in_type])
    return {"link_findings": findings, "incompatible_compositions": incompatible}


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load_api_prompt(name: str) -> dict:
    return json.loads((FIXTURES / "api_prompts" / f"{name}.json").read_text(encoding="utf-8"))


def validate_fixture(workdir: str, fixture: str, extra_model_paths: str | None = None) -> dict:
    """Run ComfyUI's validate_prompt on an exported fixture; nothing is executed."""
    import asyncio
    import uuid

    from .comfy_runtime import boot

    runtime = boot(Path(workdir), extra_model_paths=extra_model_paths)
    write_selfcheck_input(Path(workdir))
    import execution

    valid, error, outputs, node_errors = asyncio.run(
        execution.validate_prompt(str(uuid.uuid4()), load_api_prompt(fixture), None)
    )
    return {"valid": bool(valid), "error": error, "outputs": list(outputs), "node_errors": node_errors}


# -- fixture scenarios (S1-T6) -------------------------------------------------

def load_fixture_workflow(name: str) -> dict:
    return json.loads((FIXTURES / "workflows" / f"{name}.json").read_text(encoding="utf-8"))


def _definitions(workflow: dict) -> dict:
    return {item["id"]: item for item in (workflow.get("definitions") or {}).get("subgraphs", [])}


def widget_targets(fixture: str, instance_id: int, input_name: str) -> list[tuple[str, str]]:
    """Inner (node, input) pairs that a subgraph instance input feeds.

    Read from the definition's explicit links (origin -10), i.e. the same
    wiring the frontend expands.  test_s1_fixture_scenarios verifies this
    mapping against the exported API prompts.
    """
    workflow = load_fixture_workflow(fixture)
    instance = next(node for node in workflow["nodes"] if node["id"] == instance_id)
    definition = _definitions(workflow)[instance["type"]]
    slot = next(index for index, item in enumerate(definition["inputs"]) if item["name"] == input_name)
    nodes = {node["id"]: node for node in definition["nodes"]}
    targets = []
    for link in definition["links"]:
        origin, origin_slot, target, target_slot = (
            (link["origin_id"], link["origin_slot"], link["target_id"], link["target_slot"])
            if isinstance(link, dict) else link[1:5]
        )
        if origin == -10 and origin_slot == slot and target in nodes:
            targets.append((f"{instance_id}:{target}", nodes[target]["inputs"][target_slot]["name"]))
    return targets


def instance_widget_values(fixture: str, instance_id: int) -> dict:
    workflow = load_fixture_workflow(fixture)
    instance = next(node for node in workflow["nodes"] if node["id"] == instance_id)
    names = [item["name"] for item in instance.get("inputs", []) if item.get("widget")]
    return dict(zip(names, instance.get("widgets_values", [])))


def inner_ids(fixture: str, class_type: str) -> list[str]:
    workflow = load_fixture_workflow(fixture)
    definitions = _definitions(workflow)
    result = []
    for node in workflow["nodes"]:
        definition = definitions.get(node["type"])
        if definition is None:
            if node["type"] == class_type:
                result.append(str(node["id"]))
            continue
        for inner in definition["nodes"]:
            if inner["type"] == class_type:
                result.append(f"{node['id']}:{inner['id']}")
    return result


UPSTREAM_PROVIDER = "cmk_s1_upstream"


def _inject_upstream_visual(prompt: dict, fixture: str) -> dict:
    """Feed the module's VISUAL input from an upstream provider so that VISUAL
    accumulation through the module becomes observable."""
    prompt[UPSTREAM_PROVIDER] = {"class_type": "CMKVisualProvider", "inputs": {
        "label": "Upstream", "sequence": 5, "IMAGE": ["1", 2], "enable": True}}
    for node_id, name in widget_targets(fixture, 8417, "VISUAL"):
        prompt[node_id]["inputs"][name] = [UPSTREAM_PROVIDER, 0]
    return prompt


def _apply_overrides(prompt: dict, fixture: str, overrides: dict, widget_overrides: dict) -> dict:
    prompt = json.loads(json.dumps(prompt))
    for key, value in (widget_overrides or {}).items():
        instance_id, input_name = key.split(".", 1)
        targets = widget_targets(fixture, int(instance_id), input_name)
        if not targets:
            raise KeyError(f"{fixture}: no inner input fed by {key}")
        for node_id, name in targets:
            prompt[node_id]["inputs"][name] = value
    for key, value in (overrides or {}).items():
        node_id, name = key.rsplit(".", 1)
        prompt[node_id]["inputs"][name] = value
    return prompt


def _output_files(workdir: Path) -> dict:
    root = Path(workdir) / "output"
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in root.rglob("*") if path.is_file()
    }


def _png_hash(path: Path) -> str:
    """Pixel hash of a PNG written by the run (Visualizer previews, saved files)."""
    import hashlib

    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()


def _image_shape(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return [image.width, image.height]


def _errors(record) -> list[dict]:
    return [
        {key: error.get(key) for key in ("node_id", "node_type", "exception_type", "exception_message")}
        for error in record.errors
    ]


def _collect(runtime, record, workdir, probes, seams, inspect, files_before) -> dict:
    from .comfy_runtime import tensor_hash

    outputs = {}
    for label, ref in (inspect or {}).items():
        node_id, output_index = ref.rsplit(":", 1)
        value = runtime.output(node_id)
        item = None if value is None else value[int(output_index)]
        if item is None:
            outputs[label] = None
            continue
        tensor = item[0] if isinstance(item, (list, tuple)) else item
        if isinstance(tensor, dict) and isinstance(tensor.get("blocks"), list):
            outputs[label] = {"type": "log", "blocks": [[b.get("title"), b.get("lines")] for b in tensor["blocks"]]}
            continue
        if isinstance(tensor, dict) and tensor.get("type") == "CMK_DIAGNOSTIC":
            outputs[label] = {"type": "diagnostic", "title": tensor.get("title"), "mode": tensor.get("mode")}
            continue
        outputs[label] = (
            {"hash": tensor_hash(tensor), "shape": list(tensor.shape)}
            if hasattr(tensor, "shape") else {"type": type(tensor).__name__}
        )
    visual = {}
    visual_meta = {}
    for node_id, ui in record.ui.items():
        if "cmk_visual" in ui:
            payload = json.loads(ui["cmk_visual"][0])
            images = ui.get("cmk_visual_images", [])
            # Complete provider records apart from the channel images.
            visual_meta[node_id] = [{key: value for key, value in p.items() if key != "channels"}
                                    for p in payload.get("providers", [])]
            visual[node_id] = [
                {"provider_id": p.get("provider_id"), "status": p.get("status"),
                 "module_type": p.get("module_type"),
                 "channels": [c["name"] for c in p.get("channels", [])],
                 "channel_hashes": {c["name"]: _png_hash(Path(workdir) / "temp" / images[c["image_index"]]["subfolder"]
                                                     / images[c["image_index"]]["filename"])
                                    for c in p.get("channels", [])}}
                for p in payload.get("providers", [])
            ]
    files_after = _output_files(workdir)
    new_files = sorted(set(files_after) - set(files_before))
    return {
        "succeeded": record.succeeded,
        "errors": _errors(record),
        "entered": record.entered_nodes,
        "cached": record.cached,
        "function_calls": {key: len(value) for key, value in probes.function_calls.items()},
        "lazy_requests": {key: [list(item) for item in value] for key, value in probes.lazy_requests.items()},
        "statuses": [list(item) for item in probes.statuses],
        "seams": seams.snapshot(),
        "outputs": outputs,
        "visual": visual,
        "visual_meta": visual_meta,
        "new_files": {name: _image_shape(Path(workdir) / "output" / name) for name in new_files if name.endswith(".png")},
        "new_file_hashes": {name: _png_hash(Path(workdir) / "output" / name)
                            for name in new_files if name.endswith(".png")},
        "mask_ui": {node_id: ui.get("cmk_mask_status") for node_id, ui in record.ui.items() if "cmk_mask_status" in ui},
        "ready": None,
    }


def _session(workdir, extra_model_paths, mode, install_seams, stop_prepare, input_source, fake_postprocess=False):
    from .comfy_runtime import boot
    from .probes import install_probes
    from .seams import Seams, install_standard

    workdir = Path(workdir)
    runtime = boot(workdir, extra_model_paths=extra_model_paths)
    write_selfcheck_input(workdir, input_source)
    runtime.new_executor(mode)
    probes = install_probes(runtime)
    seams = Seams()
    if install_seams:
        install_standard(seams, runtime, stop_prepare=stop_prepare, fake_postprocess=fake_postprocess)
    return runtime, probes, seams


def _submit(runtime, prompt, gate_pruning):
    """The prompt as /prompt hands it to the queue: through CMK's on_prompt handler."""
    if not gate_pruning:
        return prompt, []
    pruning = runtime.cmk_module("cmk_prompt_gate_pruning")
    submitted = pruning.on_prompt({"prompt": prompt})["prompt"]
    pruned = sorted(node_id for node_id, node in submitted.items()
                    if pruning.PRUNED_META_KEY in (node.get("_meta") or {}))
    return submitted, pruned


def _run_once(runtime, probes, seams, workdir, prompt, inspect):
    probes.function_calls.clear()
    probes.lazy_requests.clear()
    probes.statuses.clear()
    seams.reset_counters()
    files_before = _output_files(workdir)
    record = runtime.run(prompt)
    return _collect(runtime, record, workdir, probes, seams, inspect, files_before)


def run_fixture(
    workdir: str,
    fixture: str,
    runs: int = 1,
    mode: str = "classic",
    overrides: dict | None = None,
    widget_overrides: dict | None = None,
    stop_prepare: bool = True,
    install_seams: bool = True,
    extra_model_paths: str | None = None,
    input_source: str = SELFCHECK_INPUT,
    inspect: dict | None = None,
    evict_between_runs: bool = False,
    fake_faces: int = 1,
    upstream_visual: bool = False,
    drop_model: bool = False,
    fake_postprocess: bool = False,
    gate_pruning: bool = False,
    run_overrides: list | None = None,
    remove_inputs: list | None = None,
) -> dict:
    """Execute an exported fixture with the real PromptExecutor.

    ``inspect`` maps a label to "node_id:output_index" whose tensor is hashed
    after each run.  ``upstream_visual`` adds a provider in front of the
    module; ``drop_model`` disconnects the checkpoint so MODEL is None.
    """
    from . import seams as seam_module

    seam_module.FAKE_FACES = int(fake_faces)
    runtime, probes, seams = _session(workdir, extra_model_paths, mode, install_seams, stop_prepare, input_source,
                                      fake_postprocess)
    prompt = _apply_overrides(load_api_prompt(fixture), fixture, overrides or {}, widget_overrides or {})
    if upstream_visual:
        prompt = _inject_upstream_visual(prompt, fixture)
    if drop_model:
        prompt["1"]["inputs"].pop("MODEL SDXL (opt)", None)
    for key in remove_inputs or []:
        node_id, name = key.rsplit(".", 1)
        prompt[node_id]["inputs"].pop(name, None)
    records = []
    try:
        for index in range(runs):
            run_prompt = prompt
            if run_overrides is not None:
                run_prompt = _apply_overrides(prompt, fixture, run_overrides[index], {})
            run_prompt, pruned = _submit(runtime, run_prompt, gate_pruning)
            record = _run_once(runtime, probes, seams, Path(workdir), run_prompt, inspect)
            record["pruned_gates"] = pruned
            records.append(record)
            if evict_between_runs and index + 1 < runs:
                runtime.force_ram_release()
    finally:
        seams.uninstall()
        probes.uninstall()
    return {"runs": records}


HYBRID_START = "8332"
HYBRID_SDXL_GATE = "8430:7417"
HYBRID_SCOPES = ("sdxl_hybrid_boundary", "sdxl_handoff_boundary")


def _scope_listing(workdir: Path) -> dict:
    root = Path(workdir) / "temp" / "cmk"
    return {scope: sorted(p.name for p in (root / scope).glob("*.pkl")) if (root / scope).is_dir() else []
            for scope in HYBRID_SCOPES}


def _payload_view(value):
    """Comparable view of a LOG / diagnostic / VISUAL payload (tensors hashed)."""
    from .comfy_runtime import tensor_hash

    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"tensor": tensor_hash(value) if hasattr(value, "detach") else "non-tensor", "shape": list(value.shape)}
    if isinstance(value, dict):
        return {str(k): _payload_view(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_payload_view(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def _tensor_png_hash(runtime, image) -> str:
    """The pixels the Visualizer writes for this IMAGE; comparable with _png_hash."""
    import hashlib

    rgb = runtime.cmk_module("utils.preview_payload")._as_rgb_uint8(image)
    return hashlib.sha256(rgb.tobytes()).hexdigest()


REGIONAL_NODE = "9001"


def tensor_hash_of(value) -> str:
    from .comfy_runtime import tensor_hash

    return tensor_hash(value) if hasattr(value, "detach") else type(value).__name__


def _add_postprocess_tail(prompt: dict) -> None:
    visualizer = prompt["8334"]["inputs"]
    family = {name: visualizer.pop(name) for name in list(visualizer)
              if name.endswith((" ZIT", " SDXL"))}
    prompt["9100"] = {"class_type": "CMKPostProcessSetupPipe", "inputs": {
        **family, "postprocess_checkpoint": "juggernautXL_ragnarok.safetensors",
        "postprocess_vae": "ClearVAE_V2.2.safetensors", "use_checkpoint_vae": True,
    }}
    prompt["9101"] = {"class_type": "CMKModelForwardPipe", "inputs": {"MODEL": ["9100", 0]}}
    visualizer.update({"MODEL (opt)": ["9101", 0], "PROCESS": ["9100", 1], "IMAGE": ["9100", 2],
                       "LOG": ["9100", 3], "VISUAL": ["9100", 4]})


def _add_mask_detailer_tail(prompt: dict) -> None:
    """Module nodes of the maskdetailer fixture behind PostProcess Setup."""
    for node_id, node in load_api_prompt("maskdetailer").items():
        if not node_id.startswith(("8417:", "8419")):
            continue
        node = json.loads(json.dumps(node))
        for name, value in node["inputs"].items():
            if isinstance(value, list) and value[0] == "1":
                node["inputs"][name] = ["9100", value[1]]
        prompt[node_id] = node
    prompt["8334"]["inputs"].update({name: ["8417:8447", slot] for slot, name in enumerate(
        ("MODEL (opt)", "PROCESS", "IMAGE", "LOG", "VISUAL"))})


def mask_detailer_active_process(workdir: str, extra_model_paths: str | None = None) -> dict:
    """The active route: Prepare keeps the source operation for the Visualizer."""
    import torch

    runtime = boot_runtime(workdir, extra_model_paths)
    module = runtime.cmk_module("nodes.image.mask_detailer")
    paths = runtime.cmk_module("utils.cmk_save_path")
    model = {"model": object(), "clip": object(), "vae": object()}
    module.resolve_postprocess_model = lambda value: model
    module._node = lambda *args, **kwargs: ([],)
    image = torch.zeros((1, 64, 64, 3))
    results = {}
    for label, process in (
        ("inpaint", {"boolean_inpaint_mode": True, "source_model_family": "sdxl"}),
        ("hybrid_inpaint", {"hybrid_inpaint_mode": True, "boolean_inpaint_mode": False,
                            "source_model_family": "z_image_turbo"}),
        ("text2image", {"boolean_inpaint_mode": False, "source_model_family": "sdxl"}),
        ("inpaint_instantid", {"boolean_inpaint_mode": True, "instantid_enabled": True, "source_model_family": "sdxl"}),
        ("no_instantid", {"boolean_inpaint_mode": True, "instantid_enabled": False, "source_model_family": "sdxl"}),
    ):
        source = module._terminal_source(image, torch.ones((1, 64, 64)), module._source_metadata(process), {}, "s")
        output = module.CMKMaskDetailerPrepare.prepare(
            {"model_role": "postprocess"}, "None", 1.0, "", LOG={"blocks": []},
            **{"MASK DETAILER SOURCE": source},
        )
        results[label] = paths.save_automatic_folders(output[1])
    return results


def mask_detailer_snapshot_folders(workdir: str, extra_model_paths: str | None = None) -> dict:
    """Save folders after Intake and Prepare, read from a locked snapshot.

    "stored" keeps the sidecar written at capture time, "older" removes the
    operation flags as a snapshot locked before they existed has none.
    """
    from PIL import Image

    import torch

    runtime = boot_runtime(workdir, extra_model_paths)
    module = runtime.cmk_module("nodes.image.mask_detailer")
    paths = runtime.cmk_module("utils.cmk_save_path")
    model = {"model": object(), "clip": object(), "vae": object()}
    module.resolve_postprocess_model = lambda value: model
    module._node = lambda *args, **kwargs: ([],)
    Path(workdir, "input").mkdir(parents=True, exist_ok=True)
    process = {"boolean_inpaint_mode": True, "instantid_enabled": True, "source_model_family": "sdxl",
               "model_family": "sdxl"}
    image = torch.zeros((1, 64, 64, 3))
    results = {}

    def folders(label, **inputs):
        output = module.CMKMaskDetailerIntake.intake(IMAGE=image, LOG={"blocks": []}, unique_id=label, **inputs)
        prepared = module.CMKMaskDetailerPrepare.prepare(
            {"model_role": "postprocess"}, "None", 1.0, "", LOG={"blocks": []},
            **{"MASK DETAILER SOURCE": output["result"][3]},
        )
        return paths.save_automatic_folders(prepared[1])

    for label in ("stored", "older"):
        module.CMKMaskDetailerIntake.intake(PROCESS=process, IMAGE=image, LOG={"blocks": []}, unique_id=label)
        captured = module.CMKMaskDetailerIntake.intake(capture_current=True, unique_id=label)
        snapshot = captured["result"][3]["snapshot"]
        painted = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
        painted.paste((0, 0, 0, 0), (8, 8, 32, 32))
        painted.save(Path(workdir, "input", snapshot))
        sidecar = module._snapshot_files(label)[2]
        if label == "older":
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            for key in module._SOURCE_OPERATION_KEYS:
                metadata["process"].pop(key, None)
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
        results[label] = {"with_process": folders(label, PROCESS=process, image=snapshot)}
        results[label]["from_sidecar"] = folders(label, image=snapshot)
    return results


def _add_inpaint_source(prompt: dict, image: str) -> None:
    prompt["8437"] = {"class_type": "CMKLoadImage", "inputs": {"image": image, "opt_in": False}}
    prompt[HYBRID_START]["inputs"].update({
        "PROCESS": ["8437", 0], "IMAGE": ["8437", 1], "MASK": ["8437", 2], "FILENAME": ["8437", 3],
        "LOG": ["8437", 4], "INPAINT_MODE": "Inpaint",
    })


def _insert_regional_node(runtime, prompt: dict, spec: dict) -> None:
    """A regional node between START HERE and the SDXL branch, as in InOutpaint."""
    start_outputs = {0: 0, 2: 1, 3: 2, 4: 3}  # PROCESS SDXL, IMAGE, LOG, VISUAL
    for node_id, node in prompt.items():
        if node_id.split(":")[0] not in {"8430", "8438", "8433"}:
            continue
        for name, value in node["inputs"].items():
            if isinstance(value, list) and value[0] == HYBRID_START and value[1] in start_outputs:
                node["inputs"][name] = [REGIONAL_NODE, start_outputs[value[1]]]
    links = {"PROCESS": [HYBRID_START, 0], "IMAGE": [HYBRID_START, 2], "LOG": [HYBRID_START, 3],
             "VISUAL": [HYBRID_START, 4]}
    if "layout" in spec:
        regions = [{"id": f"region-id-{index}", "name": f"REGION NAME {index}", "prompt": text, "strength": 1.5,
                    "end": 0.65, "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}}
                   for index, text in enumerate(spec["layout"])]
        layout = json.dumps({"version": 1, "width": 1024, "height": 1024, "regions": regions})
        prompt[REGIONAL_NODE] = {"class_type": "CMKRegionalLayoutSDXL",
                                 "inputs": {**links, "width": 1024, "height": 1024, "regions_json": layout}}
    else:
        node = runtime.cmk_module("pipe.cmk_regional_conditioning").CMKRegionalConditioningSDXL
        widgets = {name: (spec_[1].get("default") if isinstance(spec_[1:], tuple) and spec_[1:] else "")
                   for name, spec_ in node.INPUT_TYPES()["required"].items() if name not in links}
        texts = {f"region_{index}_area_prompt": text for index, text in enumerate(spec["conditioning"], 1)}
        prompt[REGIONAL_NODE] = {"class_type": "CMKRegionalConditioningSDXL", "inputs": {**widgets, **links, **texts}}


def boot_runtime(workdir, extra_model_paths):
    from .comfy_runtime import boot

    return boot(Path(workdir), extra_model_paths=extra_model_paths)


def _write_inpaint_input(workdir: Path, spec: dict) -> None:
    """RGBA source: uniform shade, alpha 0 (= masked) inside mask_box."""
    from PIL import Image, ImageDraw

    shade = int(spec.get("shade", 128))
    image = Image.new("RGBA", (256, 256), (shade, shade, shade, 255))
    ImageDraw.Draw(image).rectangle(tuple(spec.get("mask_box", (64, 64, 192, 192))), fill=(shade, shade, shade, 0))
    image.save(Path(workdir) / "input" / spec["name"])


def hybrid_stop_sequence(workdir: str, fixture: str = "hybrid_stop", stops=(False, True, False, True),
                         mode: str = "classic", extra_model_paths: str | None = None, guard: str | None = None,
                         evict_between_runs: bool = False, seed: int | None = None,
                         preseed: dict | None = None, inspect: dict | None = None,
                         overrides: dict | None = None, seeds: list | None = None,
                         start_inputs: list | None = None, remove_inputs: list | None = None,
                         input_images: list | None = None, regional: list | None = None,
                         inpaint_source: str | None = None, postprocess_tail: bool = False,
                         visualizer_inputs: dict | None = None, mask_detailer_tail: bool = False,
                         gate_pruning: bool = False) -> dict:
    """Run a Hybrid fixture several times in one executor with STOP per run.

    guard: None keeps the fixture's ContentGuard OFF; "pass" / "reject" turns
    it ON and replaces CMKContentGuard.inspect_content (counted per role);
    "reject" refuses only the SDXL/ZIT handoff role ("target_intermediate").  ``preseed`` writes files
    into the cache root before the first run (e.g. a v1 entry).  ``input_images`` holds one entry per run
    ({"name", "shade", "mask_box"}): that RGBA file is written into the input folder before the run, and
    node 8437 reads it, so a file can change while its name stays the same.  ``regional`` holds one entry
    per run: None, or {"layout": [texts]} / {"conditioning": [texts]} for a regional node on the SDXL
    PROCESS branch.  ``inpaint_source`` adds the Load Image node of InOutpaint to a fixture without one.
    ``postprocess_tail`` feeds the Visualizer through PostProcess Setup and a module's MODEL forward, as a
    workflow with PostProcess modules does.
    """
    from . import seams as seam_module
    from .seams import install_generation_seams, reset_generation_calls

    workdir = Path(workdir)
    runtime, probes, seams = _session(workdir, extra_model_paths, mode, True, False, SELFCHECK_INPUT,
                                      fake_postprocess=True)
    install_generation_seams(seams, runtime)
    guard_calls = []
    if guard is not None:
        content_guard = runtime.cmk_module("engine.content_guard")

        def fake_inspect(self, image_rgb, role):
            guard_calls.append(role)
            if guard == "reject" and role == "target_intermediate":
                raise content_guard.ContentGuardBlocked("CG_EXPLICIT_CONTENT", role)
        seams.substitute(content_guard.CMKContentGuard, "inspect_content", "guard.inspect_content", fake_inspect)
    for relative, data in (preseed or {}).items():
        target = workdir / "temp" / "cmk" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes.fromhex(data))
    import comfy.utils
    import hashlib

    import numpy as np
    from comfy_execution.utils import get_executing_context

    previews = []

    def preview_hook(value, total, preview_image, prompt_id=None, node_id=None):
        if preview_image is None:
            return
        image = preview_image[1]
        node = node_id
        if node is None:
            context = get_executing_context()
            node = None if context is None else context.node_id
        previews.append([str(node), hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()])

    previous_hook = comfy.utils.PROGRESS_BAR_HOOK
    comfy.utils.set_progress_bar_global_hook(preview_hook)
    base = _apply_overrides(load_api_prompt(fixture), fixture, overrides or {}, {})
    for key in remove_inputs or []:  # e.g. an older embedded definition without these links
        node_id, name = key.rsplit(".", 1)
        base[node_id]["inputs"].pop(name)
    records = []
    try:
        for index, stop in enumerate(stops):
            prompt = json.loads(json.dumps(base))
            start = prompt[HYBRID_START]["inputs"]
            start["stop_after_handoff"] = bool(stop)
            start.update((start_inputs or [{}] * len(stops))[index])
            if guard is not None:
                start["content_guard_enabled"] = True
            run_seed = seeds[index] if seeds is not None else seed
            if run_seed is not None:
                for node in prompt.values():
                    if node["class_type"] in {"CMKSamplerPrepareSDXLPipe", "CMKSamplerPrepareZImageTurboPipe"}:
                        node["inputs"]["seed"] = int(run_seed)
            if inpaint_source is not None:
                _add_inpaint_source(prompt, inpaint_source)
            if postprocess_tail:
                _add_postprocess_tail(prompt)
            if mask_detailer_tail:
                _add_mask_detailer_tail(prompt)
            prompt["8334"]["inputs"].update(visualizer_inputs or {})
            regional_spec = (regional or [None] * len(stops))[index]
            if regional_spec is not None:
                _insert_regional_node(runtime, prompt, regional_spec)
            image_spec = (input_images or [None] * len(stops))[index]
            if image_spec is not None:
                _write_inpaint_input(workdir, image_spec)
                prompt["8437"]["inputs"]["image"] = image_spec["name"]
            reset_generation_calls()
            guard_calls.clear()
            previews.clear()
            prompt, pruned = _submit(runtime, prompt, gate_pruning)
            record = _run_once(runtime, probes, seams, workdir, prompt, inspect)
            record["pruned_gates"] = pruned
            gate = runtime.output(HYBRID_SDXL_GATE)
            refiner_gate = runtime.output("8433:7812")
            record.update({
                "stop": bool(stop),
                "public_image_png": _tensor_png_hash(runtime, refiner_gate[1][0]) if refiner_gate and refiner_gate[1] else None,
                "visualizer_model": (lambda output: None if not output else _payload_view(output[0][0]))(runtime.output("9101")),
                "setup_image": (lambda output: None if not output else tensor_hash_of(output[2][0]))(runtime.output("9100")),
                "setup_guard": (lambda output: None if not output or not isinstance(output[1][0], dict)
                                else output[1][0].get("content_guard_enabled", "missing"))(runtime.output("9100")),
                "sampling": dict(seam_module.GENERATION_CALLS),
                "inpaint_prepares": [dict(item) for item in seam_module.INPAINT_PREPARES],
                "sampler_noise_masks": list(seam_module.SAMPLER_NOISE_MASKS),
                "sampler_last_steps": list(seam_module.SAMPLER_LAST_STEPS),
                "zit_prompts": list(seam_module.ZIT_PROMPTS),
                "regional_areas": [dict(item) for item in seam_module.REGIONAL_AREAS],
                "instantid_latents": [dict(item) for item in seam_module.INSTANTID_LATENTS],
                "guard_calls": list(guard_calls),
                "final_previews": [list(item) for item in previews],
                "scopes": _scope_listing(workdir),
                "sdxl_gate": None if gate is None else {
                    "model": _payload_view(gate[0][0]) if gate[0] else None,
                    "sampled_blocked": type(gate[1][0]).__name__ if gate[1] else None,
                    "log": _payload_view(gate[2][0]) if gate[2] else None,
                    "image": _payload_view(gate[4][0]) if gate[4] else None,
                    "visual": _payload_view(gate[5][0]) if gate[5] else None,
                    "diagnostic": _payload_view(gate[6][0]) if gate[6] else None,
                },
            })
            records.append(record)
            if evict_between_runs and index + 1 < len(stops):
                runtime.force_ram_release()
    finally:
        comfy.utils.set_progress_bar_global_hook(previous_hook)
        seams.uninstall()
        probes.uninstall()
    return {"runs": records}


def stop_contract_matrix(workdir: str, extra_model_paths: str | None = None) -> dict:
    """STOP decision of the real CMKPipeCreateImage PROCESS next to the
    prompt-side decision, for every START HERE combination."""
    import itertools

    import torch

    runtime = boot_runtime(workdir, extra_model_paths)
    distribution = runtime.cmk_module("cmk_distribution")
    node = runtime.cmk_module("pipe.cmk_pipe_image").CMKPipeCreateImage()
    image = torch.zeros((1, 64, 64, 3))
    mask = torch.zeros((1, 64, 64))
    rows = []
    for family, mode, enabled, requested, stop in itertools.product(
        ("SDXL", "Hybrid", "Z-Image Turbo"), ("Text2Image", "Inpaint", True, False), (True, False),
        ("HYBRID", "ZERO-PASS", "unexpected", None), (True, False),
    ):
        widgets = {"model_family": family, "INPAINT_MODE": mode, "instantid_enabled": enabled,
                   "instantid_mode": requested, "stop_after_handoff": stop}
        result = node.create_image(**widgets, IMAGE=image, MASK=mask, FILENAME="x.png", seed=1)
        process = result["result"][0] if isinstance(result, dict) else result[0]
        prompt = {"1": {"class_type": "CMKPipeCreateImage", "inputs": widgets}}
        rows.append({"widgets": widgets, "process_mode": process.get("instantid_mode"),
                     "process": distribution.stop_after_handoff_active(process),
                     "prompt": distribution.stop_after_handoff_prompt_state(prompt)})
    return {"rows": rows}


def input_content_keys(workdir: str, extra_model_paths: str | None = None) -> dict:
    """build_node_fingerprint around a same-name file replacement, with the
    same prompt object: the memo must not hide the new content."""
    runtime = boot_runtime(workdir, extra_model_paths)
    cache = runtime.cmk_module("pipe.cmk_persistent_cache")
    refiner = runtime.cmk_module("pipe.cmk_refiner_boundary_cache")
    workdir = Path(workdir)
    (workdir / "input").mkdir(parents=True, exist_ok=True)
    prompt = load_api_prompt("instantid_inpaint")
    prompt["8437"]["inputs"]["image"] = "same_name.png"
    keys = {}

    def key(mode):
        prompt["8332"]["inputs"]["INPAINT_MODE"] = mode
        fingerprint, _detail = cache.build_node_fingerprint(prompt, "8430:7415", ("CMKKSamplerPipe",), "s1")
        refined, _detail = refiner.build_refiner_fingerprint(prompt, "8433:7810")
        return [fingerprint, refined]

    for label, spec in (("a", {"shade": 40}), ("pixels", {"shade": 200}), ("mask", {"shade": 200, "mask_box": (8, 8, 32, 32)})):
        _write_inpaint_input(workdir, {"name": "same_name.png", **spec})
        keys[label] = {"inpaint": key("Inpaint"), "text2image": key("Text2Image")}
    return keys


def handoff_keys(workdir: str, extra_model_paths: str | None = None) -> dict:
    """KSampler fingerprint per technical handoff, computed twice per mode."""
    runtime = boot_runtime(workdir, extra_model_paths)
    cache = runtime.cmk_module("pipe.cmk_persistent_cache")
    result = {}

    for mode in ("Text2Image", "Inpaint"):
        keys = {}
        for label in ("first", "again"):
            keys[label] = []
            for handoff in (20, 30, 40, 50, 60, 70, 80):
                prompt = load_api_prompt("instantid_inpaint")
                prompt["8332"]["inputs"].update({"INPAINT_MODE": mode, "instantid_handoff_percent": handoff})
                keys[label].append(cache.build_node_fingerprint(prompt, "8430:7415", ("CMKKSamplerPipe",), "s1")[0])
        result[mode] = keys
    return result


def standalone_faceswap_guard(workdir: str, extra_model_paths: str | None = None) -> dict:
    """The standalone FaceSwap image flow as shipped: loader -> FaceSwap with
    the PROCESS input left open.  Returns the guard state handed to the engine."""
    import shutil

    from . import seams as seam_module

    workdir = Path(workdir)
    runtime, probes, seams = _session(workdir, extra_model_paths, "classic", True, True, SELFCHECK_INPUT)
    for name in ("cmk_s1_target.png", "cmk_s1_source.png"):
        shutil.copy(workdir / "input" / SELFCHECK_INPUT, workdir / "input" / name)
    pair = json.dumps({"target": "cmk_s1_target.png", "source": "cmk_s1_source.png"})
    results = {}
    try:
        for label, guard, process_link, with_widget in (
            ("on", True, False, True), ("off", False, False, True),
            ("on_unpack", True, True, True), ("off_unpack", False, True, True),
        ):
            loader = {"RESOLUTION": "SDXL 832x1152", "SWAP DIMENSIONS": False, "RESIZE METHOD": "lanczos",
                      "CROP": False, "CROP POSITION": "center", "IMAGE PAIR": pair}
            if with_widget:
                loader["content_guard_enabled"] = guard
            swap = {"GLOBAL ENABLE": True, "ENABLE": True, "SWAP MODEL": "hyperswap_1b_256.onnx",
                    "DETECT MODEL": "buffalo_l", "TARGET FACE": "Largest", "SOURCE FACE": "Largest",
                    "PATCH ENHANCER": False, "PATCH ENHANCER MODEL": "GFPGAN", "PATCH ENHANCER STRENGTH": 0.75,
                    "bbox_dilation": 0, "drop_size": 10, "QUALITY PROFILE": "Automatic", "POST RESTORE": False,
                    "POST RESTORE MODEL": "GFPGANv1.4.pth", "RESTORE VISIBILITY": 1,
                    "IMAGE_TARGET": ["1", 1], "IMAGE_SOURCE": ["1", 2], "opt_log": ["1", 3]}
            prompt = {"1": {"class_type": "CMKSwapImageLoaderPipe", "inputs": loader},
                      "2": {"class_type": "CMKFaceSwapImagePipe", "inputs": swap},
                      "3": {"class_type": "CMKVisualizer", "inputs": {"PROCESS": ["1", 0], "IMAGE": ["2", 0],
                                                                       "LOG": ["2", 2], "SAVE ENABLED": False}}}
            if process_link:
                prompt["4"] = {"class_type": "CMKResultUnpackPipe", "inputs": {
                    "PROCESS": ["1", 0], "IMAGE": ["1", 1], "LOG": ["1", 3]}}
                swap["PROCESS"] = ["4", 1]
            seam_module.SWAP_GUARD_STATES.clear()
            record = _run_once(runtime, probes, seams, workdir, prompt, {})
            results[label] = {"succeeded": record["succeeded"], "errors": record["errors"],
                              "engine_guard": list(seam_module.SWAP_GUARD_STATES)}
    finally:
        seams.uninstall()
        probes.uninstall()
    return results


def completion_guard_keys(workdir: str, extra_model_paths: str | None = None) -> dict:
    """Generation-completion key for the same generation with the guard ON and OFF."""
    runtime = boot_runtime(workdir, extra_model_paths)
    completion = runtime.cmk_module("pipe.cmk_generation_completion_cache")
    keys = {}
    for label, guard in (("on", True), ("off", False), ("on_again", True)):
        prompt = load_api_prompt("hybrid_stop")
        prompt[HYBRID_START]["inputs"]["content_guard_enabled"] = guard
        roots = {"IMAGE": ["8431:7655", 1], "PROCESS": ["8431:7654", 0]}
        keys[label] = completion.build_completion_key(prompt, roots, model_family="hybrid",
                                                      generation_mode="sdxl_zit_hybrid")[0]
    return keys


def multi_image_loader_keys(workdir: str, extra_model_paths: str | None = None) -> dict:
    """ComfyUI cache keys and CMK fingerprints for prompts that the real
    frontend produced with a CMK Multi Image Loader (fixtures/multi_image_loader)."""
    import asyncio
    import hashlib
    import shutil

    runtime = boot_runtime(workdir, extra_model_paths)
    import execution
    from comfy_execution.caching import CacheKeySetInputSignature
    from comfy_execution.graph import DynamicPrompt

    sampler = runtime.cmk_module("pipe.cmk_pipe_sampler")
    refiner = runtime.cmk_module("pipe.cmk_refiner_boundary_cache")
    family = runtime.cmk_module("pipe.cmk_family_result")
    loader = runtime.cmk_module("pipe.loaders.cmk_load_image").CMKLoadImage
    folder = Path(__file__).resolve().parents[1] / "fixtures" / "multi_image_loader"
    result = {}
    for path in sorted(folder.glob("*.json")):
        prompt = json.loads(path.read_text(encoding="utf-8"))
        dynamic = DynamicPrompt(prompt)
        is_changed = execution.IsChangedCache("cmk-multi", dynamic, None)
        keyset = CacheKeySetInputSignature(dynamic, list(prompt), is_changed)
        asyncio.run(keyset.add_keys(list(prompt)))
        comfy = {node_id: hashlib.sha256(repr(keyset.get_data_key(node_id)).encode()).hexdigest()
                 for node_id in prompt}
        result[path.stem] = {
            "comfy": comfy,
            "references": {node_id: node["inputs"]["opt_image_file"] for node_id, node in prompt.items()
                           if isinstance(node["inputs"].get("opt_image_file"), str)},
            "handoff": sampler._sdxl_handoff_cache_key(prompt, "8430:7415"),
            "instantid_boundary": family._sampled_boundary_key(prompt, "8438:7915", include_content_guard=True),
            "refiner": refiner.build_refiner_fingerprint(prompt, "8433:7810")[0],
        }
    # Content changes behind an unchanged reference keep reaching the consumer.
    target = Path(workdir) / "input" / "cmk_multi_content.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    digests = []
    for source in (SELFCHECK_INPUT, "cmk_s1_input_replacement.png"):
        write_selfcheck_input(Path(workdir), source)
        shutil.copy(Path(workdir) / "input" / SELFCHECK_INPUT, target)
        digests.append(loader.IS_CHANGED("ignored.png", opt_image_file="cmk_multi_content.png", opt_in=True))
    result["content_digests"] = digests
    return result


def mask_detailer_sequence(workdir: str, extra_model_paths: str | None = None) -> dict:
    """Drive the MaskDetailer READY state machine with the real Intake node.

    S0 live → S1 capture (capture_current = True) → S2 locked, empty mask →
    S3 locked, painted mask → input image changed → back to S0.  The mask
    editor is simulated by writing an alpha channel into the locked snapshot,
    which is exactly what the Intake reads (mask = 1 - alpha).
    """
    from PIL import Image

    fixture = "maskdetailer"
    workdir = Path(workdir)
    runtime, probes, seams = _session(workdir, extra_model_paths, "classic", True, True, SELFCHECK_INPUT)
    intake = inner_ids(fixture, "CMKMaskDetailerIntake")[0]
    process = inner_ids(fixture, "CMKMaskDetailerProcess")[0]
    gate = inner_ids(fixture, "CMKMaskDetailerFlowGate")[0]
    safe = intake.replace(":", "-")
    snapshot_rel = f"cmk_mask_detailer/snapshot-{safe}.png"
    snapshot = workdir / "input" / snapshot_rel
    base = load_api_prompt(fixture)
    inspect = {"gate_image": f"{gate}:2", "loader_image": "1:2", "process_image": f"{process}:2"}
    steps = []

    def step(label, **intake_inputs):
        prompt = json.loads(json.dumps(base))
        prompt[intake]["inputs"].update(intake_inputs)
        result = _run_once(runtime, probes, seams, workdir, prompt, inspect)
        value = runtime.output(intake)
        result["ready"] = None if value is None else bool(_first(value[4]))
        gate_value = runtime.output(gate)
        process_value = None if gate_value is None else _first(gate_value[1])
        result["public_process_keys"] = sorted(process_value) if isinstance(process_value, dict) else None
        result["public_process_has_tensor"] = _contains_tensor(process_value)
        result["label"] = label
        steps.append(result)

    try:
        step("S0", capture_current=False)
        step("S1", capture_current=True)
        step("S2", capture_current=False, image=snapshot_rel)
        with Image.open(snapshot) as opened:
            rgba = opened.convert("RGBA")
        width, height = rgba.size
        alpha = Image.new("L", (width, height), 255)
        alpha.paste(0, (width // 4, height // 4, width // 2, height // 2))
        rgba.putalpha(alpha)
        rgba.save(snapshot)
        step("S3", capture_current=False, image=snapshot_rel)
        write_selfcheck_input(workdir, "cmk_s1_input_replacement.png")
        step("S0_after_input_change", capture_current=False, image=snapshot_rel)
    finally:
        seams.uninstall()
        probes.uninstall()
    return {"steps": steps, "intake": intake, "process": process, "gate": gate}


def _first(value):
    """ComfyUI stores every node output as a list of per-batch results."""
    return value[0] if isinstance(value, list) and value else value


def _contains_tensor(value, depth=0):
    import torch

    if depth > 6:
        return False
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item, depth + 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item, depth + 1) for item in value)
    return False


def _class_surface() -> dict:
    """What ComfyUI reads from node classes for caching, lazy evaluation and order."""
    import hashlib
    import inspect

    import nodes

    surface = {}
    for class_type, cls in sorted(nodes.NODE_CLASS_MAPPINGS.items()):
        if not class_type.startswith("CMK"):
            continue
        try:
            declared = json.dumps(cls.INPUT_TYPES(), sort_keys=True, default=str)
        except Exception as exc:  # pragma: no cover - recorded, not hidden
            declared = f"ERROR {type(exc).__name__}"
        surface[class_type] = {
            "input_types": hashlib.sha256(declared.encode()).hexdigest(),
            "return_types": [str(t) for t in getattr(cls, "RETURN_TYPES", ())],
            "function": getattr(cls, "FUNCTION", None),
            "check_lazy_status": id(inspect.getattr_static(cls, "check_lazy_status", None)),
            "is_changed": id(inspect.getattr_static(cls, "IS_CHANGED", None)),
            "not_idempotent": getattr(cls, "NOT_IDEMPOTENT", None),
            "output_node": getattr(cls, "OUTPUT_NODE", None),
        }
    return surface


def _cache_signatures(fixture: str) -> dict:
    import asyncio

    import execution
    from comfy_execution.caching import CacheKeySetInputSignature
    from comfy_execution.graph import DynamicPrompt

    prompt = load_api_prompt(fixture)
    dynprompt = DynamicPrompt(prompt)
    is_changed = execution.IsChangedCache("cmk-s1-signatures", dynprompt, None)
    keyset = CacheKeySetInputSignature(dynprompt, list(prompt), is_changed)
    asyncio.run(keyset.add_keys(list(prompt)))
    keys = {node_id: repr(keyset.get_data_key(node_id)) for node_id in sorted(prompt)}
    volatile = sorted(
        node_id for node_id in prompt
        if _is_nan_marker(asyncio.run(is_changed.get(node_id)))
    )
    return {"keys": keys, "volatile": volatile}


def _is_nan_marker(value) -> bool:
    import math

    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, (list, tuple)):
        return any(_is_nan_marker(item) for item in value)
    return False


def seam_neutrality(workdir: str, fixture: str, extra_model_paths: str | None = None) -> dict:
    """Class surface and ComfyUI cache signatures before and after installing
    the standard seams, computed without executing any node."""
    from .comfy_runtime import boot
    from .seams import Seams, install_standard

    runtime = boot(Path(workdir), extra_model_paths=extra_model_paths)
    write_selfcheck_input(Path(workdir))
    surface_before = _class_surface()
    keys_before = _cache_signatures(fixture)
    seams = Seams()
    install_standard(seams, runtime)
    try:
        surface_after = _class_surface()
        keys_after = _cache_signatures(fixture)
    finally:
        seams.uninstall()
    surface_restored = _class_surface()
    volatile = sorted(set(keys_before["volatile"]) | set(keys_after["volatile"]))
    stable_before = {k: v for k, v in keys_before["keys"].items() if k not in volatile}
    stable_after = {k: v for k, v in keys_after["keys"].items() if k not in volatile}
    return {
        "surface_equal": surface_before == surface_after,
        "surface_restored": surface_before == surface_restored,
        "surface_diff": sorted(k for k in surface_before if surface_before[k] != surface_after.get(k)),
        "keys_equal": stable_before == stable_after,
        "keys_diff": sorted(k for k in stable_before if stable_before[k] != stable_after.get(k)),
        "volatile_nodes": volatile,
        "compared_nodes": len(stable_before),
    }


def boundary_key(workdir: str, fixture: str, boundary_class: str, input_source: str = SELFCHECK_INPUT,
                 extra_model_paths: str | None = None, model_source_class: str | None = None,
                 schema: str | None = None) -> dict:
    """Boundary cache key exactly as the boundary computes it, for the given
    content of the (same-named) input file.  ``model_source_class`` replaces
    the class of the node feeding CMK Load Image's MODEL input (H-2)."""
    from .comfy_runtime import boot

    runtime = boot(Path(workdir), extra_model_paths=extra_model_paths)
    write_selfcheck_input(Path(workdir), input_source)
    import nodes

    node_id = inner_ids(fixture, boundary_class)[0]
    instance = nodes.NODE_CLASS_MAPPINGS[boundary_class]()
    if schema:
        instance._SCHEMA = schema  # compare across a deliberate schema bump (K-5)
    prompt = load_api_prompt(fixture)
    if model_source_class:
        source = prompt["1"]["inputs"]["MODEL SDXL (opt)"][0]
        prompt[source]["class_type"] = model_source_class
    key, detail = instance._cache_key(prompt, node_id)
    dependencies = instance._dependencies(prompt, node_id)[0] if hasattr(instance, "_dependencies") else None
    return {"key": key, "detail": detail, "dependencies": dependencies}


def forward_only(workdir: str, fixture: str, source: str, extra_model_paths: str | None = None,
                 drop_model: bool = False) -> dict:
    """Request only one output (PROCESS or MODEL) through an added PreviewAny
    output node and record what executed.  ``drop_model`` disconnects the
    checkpoint so MODEL is None."""
    runtime, probes, seams = _session(workdir, extra_model_paths, "classic", True, True, SELFCHECK_INPUT)
    prompt = load_api_prompt(fixture)
    if drop_model:
        prompt["1"]["inputs"].pop("MODEL SDXL (opt)", None)
    node_id, index = source.rsplit(":", 1)
    prompt["cmk_s1_probe"] = {"class_type": "PreviewAny", "inputs": {"source": [node_id, int(index)]}}
    try:
        probes.function_calls.clear()
        seams.reset_counters()
        record = runtime.run(prompt, outputs=["cmk_s1_probe"])
        return {
            "succeeded": record.succeeded,
            "errors": _errors(record),
            "function_calls": {key: len(value) for key, value in probes.function_calls.items()},
            "seams": seams.snapshot(),
        }
    finally:
        seams.uninstall()
        probes.uninstall()


def manifest_keys(workdir: str, fixture: str, extra_model_paths: str | None = None,
                  overrides: dict | None = None) -> dict:
    """K5-B / K5-C: per manifest branch, the key and revision the boundary's
    dependency manifest records and the key under which the branch node
    publishes its revision."""
    from .comfy_runtime import boot

    runtime = boot(Path(workdir), extra_model_paths=extra_model_paths)
    write_selfcheck_input(Path(workdir), SELFCHECK_INPUT)
    import nodes

    pc = runtime.cmk_module("pipe.cmk_persistent_cache")
    prompt = _apply_overrides(load_api_prompt(fixture), fixture, overrides or {}, {})
    rows = []
    for boundary_class in ("CMKDetailerBoundaryCache", "CMKFaceBoundaryCache"):
        for boundary_id in inner_ids(fixture, boundary_class):
            boundary = nodes.NODE_CLASS_MAPPINGS[boundary_class]()
            manifest = boundary._dependencies(prompt, boundary_id)[0]
            for entry in manifest["branches"]:
                branch = nodes.NODE_CLASS_MAPPINGS[entry["class_type"]]
                if hasattr(branch, "dependency_cache_key"):
                    published = branch.dependency_cache_key(prompt, entry["node_id"])[0]
                elif entry["class_type"] == "CMK_SmartDetailerPipe":  # pre-K5-B formula of _publish_dependency_revision
                    published = pc.build_node_fingerprint(
                        prompt, entry["node_id"], ("CMK_SmartDetailerPipe",), branch._CACHE_SCHEMA,
                        exclude_inputs=("output_image_proceed", "opt_log", "VISUAL", "opt_diagnostic"),
                        include_node_identity=True)[0]
                else:
                    published = branch()._cache_key(prompt, entry["node_id"])[0]
                rows.append({"boundary": boundary_id, "node_id": entry["node_id"], "class_type": entry["class_type"],
                             "scope": entry["scope"], "manifest_key": entry["cache_key"], "published_key": published,
                             "branch_scope": branch._CACHE_SCOPE, "revision": entry["revision"],
                             "enable": prompt[entry["node_id"]]["inputs"].get("enable"),
                             "complete": manifest["complete"],
                             "boundary_key": boundary._cache_key(prompt, boundary_id)[0]})
    return {"rows": rows}


FULL_FLOW_REFERENCES = {  # consumer of the identity reference per module
    "instantid": "8465:8400", "facerebuild": "8435:7822", "facerebuild_adv": "8440:8380",
    "faceswap": "8436:7850", "faceswap_adv": "8441:8462",
}
FULL_FLOW_MODULE_SWITCHES = {  # inner nodes fed by the module's ENABLE widget
    "facerebuild": {"8435:7827.ENABLE", "8435:7823.FACEREBUILD ENABLE", "8435:7826.enable"},
    "faceswap": {"8436:7856.ENABLE", "8436:7849.GLOBAL ENABLE", "8436:7855.enable"},
}
FULL_FLOW_START = "8332"


def full_flow_prompt(instantid=False, stop=False, facerebuild=False, faceswap=True, references=None, overrides=None):
    """The shipped Full Flow with every identity consumer fed a literal reference,
    as the Multi Image Loader delivers it."""
    prompt = load_api_prompt("full_flow")
    start = prompt[FULL_FLOW_START]["inputs"]
    start["instantid_enabled"] = bool(instantid)
    start["stop_after_handoff"] = bool(stop)
    for module, enabled in (("facerebuild", facerebuild), ("faceswap", faceswap)):
        for key in FULL_FLOW_MODULE_SWITCHES[module]:
            node_id, name = key.rsplit(".", 1)
            prompt[node_id]["inputs"][name] = bool(enabled)
    refs = {module: "CMK Package · faceswap_reference.png" for module in FULL_FLOW_REFERENCES}
    refs.update(references or {})
    for module, node_id in FULL_FLOW_REFERENCES.items():
        prompt[node_id]["inputs"]["opt_image_file"] = refs[module]
    for key, value in (overrides or {}).items():
        node_id, name = key.rsplit(".", 1)
        prompt[node_id]["inputs"][name] = value
    return prompt


def gate_pruning_keys(workdir: str, cases: dict, extra_model_paths: str | None = None) -> dict:
    """ComfyUI cache signatures and CMK fingerprints of Full Flow prompts, raw and
    after CMK's on_prompt handler.  ``cases`` maps a label to full_flow_prompt kwargs."""
    import asyncio
    import copy
    import hashlib

    runtime = boot_runtime(workdir, extra_model_paths)
    import execution
    from comfy_execution.caching import CacheKeySetInputSignature
    from comfy_execution.graph import DynamicPrompt

    import nodes

    # Third-party nodes of the shipped flow (LoRA Manager) are not loaded by the
    # harness; a signature needs only their class name and inputs.
    for node in load_api_prompt("full_flow").values():
        if node["class_type"] not in nodes.NODE_CLASS_MAPPINGS:
            nodes.NODE_CLASS_MAPPINGS[node["class_type"]] = type(
                "CMKSignatureStub", (), {"INPUT_TYPES": classmethod(lambda cls: {"required": {}})})
    pruning = runtime.cmk_module("cmk_prompt_gate_pruning")
    persistent = runtime.cmk_module("pipe.cmk_persistent_cache")
    refiner = runtime.cmk_module("cmk_refiner_boundary_cache")
    family = runtime.cmk_module("pipe.cmk_family_result")

    def keys(prompt):
        work = copy.deepcopy(prompt)
        dynamic = DynamicPrompt(work)
        is_changed = execution.IsChangedCache("cmk-gate-pruning", dynamic, None)
        keyset = CacheKeySetInputSignature(dynamic, list(work), is_changed)
        asyncio.run(keyset.add_keys(list(work)))
        comfy = {node_id: hashlib.sha256(repr(keyset.get_data_key(node_id)).encode()).hexdigest()[:16]
                 for node_id in work}
        cmk = {"refiner": (refiner.build_refiner_fingerprint(copy.deepcopy(prompt), "8433:7810")[0] or "")[:16]}
        for node_id, node in prompt.items():
            if "Boundary" in node["class_type"]:
                cmk[node_id] = (persistent.build_node_fingerprint(
                    copy.deepcopy(prompt), node_id, [node["class_type"]], "probe", include_node_identity=True)[0]
                    or "")[:16]
        completion = family.CMKPostProcessSetupPipe._completion_key(copy.deepcopy(prompt), "8434")
        cmk["completion"] = (completion[0] or "")[:16] or f"none: {completion[1]}"
        return {"comfy": comfy, "cmk": cmk}

    result = {}
    for label, kwargs in cases.items():
        raw = full_flow_prompt(**kwargs)
        pruned = pruning.on_prompt({"prompt": raw})["prompt"]
        again = pruning.on_prompt({"prompt": pruned})["prompt"]
        raw_keys = keys(raw)
        repeat = keys(raw)["comfy"]
        result[label] = {
            # Output nodes whose IS_CHANGED differs per evaluation (Visualizer,
            # Visual Compare) and their descendants: never comparable.
            "unstable": sorted(n for n in repeat if repeat[n] != raw_keys["comfy"][n]
                               or raw[n]["class_type"] in {"CMKVisualizer", "CMKVisualCompare"}),
            "raw": raw_keys, "pruned": keys(pruned),
            "pruned_gates": sorted(n for n, node in pruned.items()
                                   if pruning.PRUNED_META_KEY in (node.get("_meta") or {})),
            "idempotent": again == pruned,
            "active_inputs_kept": all(pruned[n]["inputs"] == raw[n]["inputs"] for n in raw
                                      if pruning.PRUNED_META_KEY not in (pruned[n].get("_meta") or {})),
        }
    return result


def swap_loader_packaged_references(workdir: str, extra_model_paths: str | None = None) -> dict:
    """CMKSwapImageLoaderPipe with an empty input folder: shipped pairs name package files plainly."""
    from pathlib import Path as _Path

    runtime = boot_runtime(workdir, extra_model_paths)
    module = runtime.cmk_module("pipe.loaders.cmk_swap_image_loader")
    loader = module.CMKSwapImageLoaderPipe
    pair = json.dumps({"target": "mask_detailer_reference.png", "source": "portrait_reference_00001.png"})
    missing = json.dumps({"target": "not_in_the_package.png", "source": "portrait_reference_00001.png"})
    input_dir = _Path(workdir) / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    fresh = {"validate": loader.VALIDATE_INPUTS(**{"IMAGE PAIR": pair}),
             "changed": loader.IS_CHANGED(**{"IMAGE PAIR": pair}),
             "resolved": _Path(module._resolve_image_path("portrait_reference_00001.png")).name,
             "missing": loader.VALIDATE_INPUTS(**{"IMAGE PAIR": missing})}
    (input_dir / "portrait_reference_00001.png").write_bytes((_Path(__file__).resolve().parents[2] / "assets" / "references" / "black.png").read_bytes())
    own = _Path(module._resolve_image_path("portrait_reference_00001.png")).resolve()
    fresh["user_file_wins"] = own == (input_dir / "portrait_reference_00001.png").resolve()
    return fresh
