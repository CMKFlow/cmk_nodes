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


def _session(workdir, extra_model_paths, mode, install_seams, stop_prepare, input_source):
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
        install_standard(seams, runtime, stop_prepare=stop_prepare)
    return runtime, probes, seams


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
) -> dict:
    """Execute an exported fixture with the real PromptExecutor.

    ``inspect`` maps a label to "node_id:output_index" whose tensor is hashed
    after each run.  ``upstream_visual`` adds a provider in front of the
    module; ``drop_model`` disconnects the checkpoint so MODEL is None.
    """
    from . import seams as seam_module

    seam_module.FAKE_FACES = int(fake_faces)
    runtime, probes, seams = _session(workdir, extra_model_paths, mode, install_seams, stop_prepare, input_source)
    prompt = _apply_overrides(load_api_prompt(fixture), fixture, overrides or {}, widget_overrides or {})
    if upstream_visual:
        prompt = _inject_upstream_visual(prompt, fixture)
    if drop_model:
        prompt["1"]["inputs"].pop("MODEL SDXL (opt)", None)
    records = []
    try:
        for index in range(runs):
            records.append(_run_once(runtime, probes, seams, Path(workdir), prompt, inspect))
            if evict_between_runs and index + 1 < runs:
                runtime.force_ram_release()
    finally:
        seams.uninstall()
        probes.uninstall()
    return {"runs": records}


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
                 extra_model_paths: str | None = None, model_source_class: str | None = None) -> dict:
    """Boundary cache key exactly as the boundary computes it, for the given
    content of the (same-named) input file.  ``model_source_class`` replaces
    the class of the node feeding CMK Load Image's MODEL input (H-2)."""
    from .comfy_runtime import boot

    runtime = boot(Path(workdir), extra_model_paths=extra_model_paths)
    write_selfcheck_input(Path(workdir), input_source)
    import nodes

    node_id = inner_ids(fixture, boundary_class)[0]
    instance = nodes.NODE_CLASS_MAPPINGS[boundary_class]()
    prompt = load_api_prompt(fixture)
    if model_source_class:
        source = prompt["1"]["inputs"]["MODEL SDXL (opt)"][0]
        prompt[source]["class_type"] = model_source_class
    key, detail = instance._cache_key(prompt, node_id)
    dependencies = instance._dependencies(prompt, node_id)[0] if hasattr(instance, "_dependencies") else None
    return {"key": key, "detail": detail, "dependencies": dependencies}


def forward_only(workdir: str, fixture: str, source: str, extra_model_paths: str | None = None) -> dict:
    """Request only one output (PROCESS or MODEL) through an added PreviewAny
    output node and record what executed."""
    runtime, probes, seams = _session(workdir, extra_model_paths, "classic", True, True, SELFCHECK_INPUT)
    prompt = load_api_prompt(fixture)
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
