import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "nodes" / "image" / "mask_detailer.py"
IMAGE_INPUT_SOURCE = ROOT / "pipe" / "loaders" / "cmk_image_load_resize.py"


def _source():
    return SOURCE.read_text(encoding="utf-8")


def _load_flow_gate():
    tree = ast.parse(_source())
    gate = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CMKMaskDetailerFlowGate"
    )

    def empty_visual():
        return {"type": "CMK_VISUAL_PIPE", "version": 1, "providers": []}

    def register_provider(visual, **provider):
        result = dict(visual)
        result["providers"] = [*visual.get("providers", []), provider]
        return result

    namespace = {
        "CMK_TERMINAL_INPUT": "*",
        "MASK_DETAILER_PROCESS": "CMK_MASK_DETAILER_PROCESS",
        "_source_metadata": lambda process: dict(process),
        "empty_visual": empty_visual,
        "register_provider": register_provider,
    }
    exec(compile(ast.Module(body=[gate], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["CMKMaskDetailerFlowGate"]


def test_mask_detailer_has_own_process_family_and_two_stage_contract():
    text = _source()
    assert 'MASK_DETAILER_PROCESS = "CMK_MASK_DETAILER_PROCESS"' in text
    assert "class CMKMaskDetailerPrepare" in text
    assert "class CMKMaskDetailerProcess" in text
    assert '("CMK_RESULT_PROCESS",)' in text
    assert '"MASK DETAILER PROCESS": (MASK_DETAILER_PROCESS,)' in text
    assert 'RETURN_NAMES = ("MODEL", "MASK DETAILER PROCESS", "IMAGE", "LOG", "VISUAL")' in text


def test_terminal_intake_locks_a_snapshot_and_severs_the_flow_contract():
    text = _source()
    assert 'MASK_DETAILER_SOURCE = "CMK_MASK_DETAILER_SOURCE"' in text
    assert "class CMKMaskDetailerIntake" in text
    assert '"PROCESS": (CMK_TERMINAL_INPUT, {"lazy": True})' in text
    assert '"IMAGE": ("IMAGE", {"lazy": True})' in text
    assert '"LOG": ("CMK_LOG_PIPE", {"lazy": True})' in text
    assert '"optional": {\n                # ComfyUI\'s native mask editor' in text
    assert '"image": (cls._available_snapshots(), {"image_upload": True' in text
    assert 'f"{name} [input]"' in text
    assert 'if bool(capture_current):\n            return []' in text
    assert '"cmk_mask_status": ["LIVE · CAPTURE CURRENT"]' in text
    assert 'current_fingerprint != locked_fingerprint' in text
    assert 'Path(snapshot_value).name.startswith("incoming-")' in text
    assert 'and not is_pending_snapshot' in text
    assert '"cmk_mask_status": ["LOCKED · PAINT MASK"]' in text
    assert 'if float(mask.max().detach().cpu()) <= 0.0:' in text
    assert '"FLOW CONTRACT    : TERMINATED"' in text
    assert '"NEXT             : MASK DETAILER / VISUALIZER ONLY"' in text
    assert 'RETURN_TYPES = (CMK_TERMINAL_INPUT, "IMAGE", "CMK_LOG_PIPE", MASK_DETAILER_SOURCE, "BOOLEAN")' in text
    assert 'RETURN_NAMES = ("PROCESS", "IMAGE", "LOG", "MASK DETAILER SOURCE", "READY")' in text
    assert '"result": (process, current_image, LOG, source, False)' in text
    assert '"result": (process, image, base_log, source, False)' in text
    assert '"result": (process, image, log, source, ready)' in text
    assert 'OUTPUT_NODE = True' not in text
    ui = (ROOT / "web" / "js" / "cmk_mask_detailer_intake.js").read_text(encoding="utf-8")
    assert 'CAPTURE CURRENT IMAGE' in ui
    assert 'CAPTURE REQUESTED · START WORKFLOW' in ui
    assert 'AKTUELLES BILD ÜBERNEHMEN' not in ui
    assert 'WORKFLOW STARTEN' not in ui
    assert 'capture.value = true' in ui
    assert 'hideWidget(image)' in ui
    assert 'message?.cmk_mask_snapshot?.[0] || message?.cmk_mask_preview?.[0]' in ui
    assert 'isPendingSnapshot(image.value)' in ui
    assert 'ensureImageOption(this, filename)' in ui
    assert "function ensureImageOption(node, value)" in ui
    assert "values.push(selected)" in ui
    assert "configuration?.widgets_values_named?.image" in ui
    assert "nodeType.prototype.onConfigure" in ui
    assert "loadedGraphNode(node)" in ui
    assert 'setButtonLocked(button, lock === "locked")' in ui
    assert "class CMKMaskDetailerCompareRelay" not in text


def test_mask_detailer_flow_gate_is_lazy_and_has_real_passthrough_paths():
    text = _source()
    assert "class CMKMaskDetailerFlowGate" in text
    assert '"READY": ("BOOLEAN", {"lazy": True})' in text
    assert '"PROCESS BYPASS": (CMK_TERMINAL_INPUT, {"lazy": True})' in text
    assert '"PROCESS ACTIVE": (MASK_DETAILER_PROCESS, {"lazy": True})' in text
    assert 'if not bool(enabled):' in text
    assert 'if READY is None:' in text
    assert 'self._pending_connected_visual(' in text
    assert '"VISUAL BYPASS", prompt, unique_id, inputs' in text
    intake_segment = text[text.index("class CMKMaskDetailerIntake"):text.index("class CMKMaskDetailerFlowGate")]
    gate_segment = text[text.index("class CMKMaskDetailerFlowGate"):text.index("class CMKMaskDetailerPrepare")]
    assert '"prompt": "PROMPT"' not in intake_segment
    assert '"prompt": "PROMPT"' in gate_segment
    assert '"mask_detailer_bypassed": True' in text
    assert '"snapshot not ready"' in text
    assert 'inputs.get("VISUAL BYPASS") or empty_visual()' in text
    assert '"VISUAL BYPASS": ("CMK_VISUAL_PIPE", {"lazy": True})' in text
    assert '"LOG BYPASS",\n        )' in text
    assert 'channels={"before": image, "after": image}' in text
    assert 'branch="mask-detailer"' in text
    assert 'stage_key="mask-detailer.result"' in text


def test_mask_detailer_disabled_bypass_preserves_upstream_visual():
    gate = _load_flow_gate()()
    upstream = {
        "type": "CMK_VISUAL_PIPE",
        "version": 1,
        "providers": [{"provider_id": "sampling"}, {"provider_id": "identity"}],
    }
    inputs = {
        "MODEL BYPASS": {},
        "PROCESS BYPASS": {"source_model_family": "sdxl"},
        "IMAGE BYPASS": object(),
        "LOG BYPASS": {},
        "VISUAL BYPASS": None,
    }

    prompt = {"gate": {"inputs": {"VISUAL BYPASS": ["upstream", 0]}}}
    assert gate.check_lazy_status(
        enabled=False, prompt=prompt, unique_id="gate", **inputs
    ) == ["VISUAL BYPASS"]
    fallback = gate.gate(enabled=False, unique_id="mask-detailer", **inputs)
    assert len(fallback[4]["providers"]) == 1
    assert fallback[4]["providers"][0]["status"] == "disabled"
    inputs["VISUAL BYPASS"] = upstream
    assert gate.check_lazy_status(
        enabled=False, prompt=prompt, unique_id="gate", **inputs
    ) == []
    result = gate.gate(enabled=False, unique_id="mask-detailer", **inputs)
    assert [provider["provider_id"] for provider in result[4]["providers"][:2]] == [
        "sampling", "identity",
    ]
    assert result[4]["providers"][-1]["status"] == "disabled"


def test_mask_detailer_waiting_bypass_preserves_upstream_visual():
    gate = _load_flow_gate()()
    upstream = {
        "type": "CMK_VISUAL_PIPE",
        "version": 1,
        "providers": [{"provider_id": "sampling"}, {"provider_id": "refiner"}],
    }
    inputs = {
        "MODEL BYPASS": {},
        "PROCESS BYPASS": {"source_model_family": "sdxl"},
        "IMAGE BYPASS": object(),
        "LOG BYPASS": {},
        "VISUAL BYPASS": None,
    }

    prompt = {"gate": {"inputs": {"VISUAL BYPASS": ["upstream", 0]}}}
    assert gate.check_lazy_status(
        enabled=True, READY=False, prompt=prompt, unique_id="gate", **inputs
    ) == ["VISUAL BYPASS"]
    fallback = gate.gate(
        enabled=True, READY=False, unique_id="mask-detailer", **inputs
    )
    assert len(fallback[4]["providers"]) == 1
    assert fallback[4]["providers"][0]["status"] == "waiting"
    inputs["VISUAL BYPASS"] = upstream
    assert gate.check_lazy_status(
        enabled=True, READY=False, prompt=prompt, unique_id="gate", **inputs
    ) == []
    result = gate.gate(enabled=True, READY=False, unique_id="mask-detailer", **inputs)
    assert [provider["provider_id"] for provider in result[4]["providers"][:2]] == [
        "sampling", "refiner",
    ]
    assert result[4]["providers"][-1]["status"] == "waiting"


def test_mask_detailer_active_gate_keeps_active_visual_contract():
    gate = _load_flow_gate()()
    visual = {
        "type": "CMK_VISUAL_PIPE",
        "version": 1,
        "providers": [{"provider_id": "sampling"}, {"provider_id": "mask-detailer"}],
    }
    inputs = {
        "MODEL ACTIVE": {},
        "PROCESS ACTIVE": {"type": "CMK_MASK_DETAILER_PROCESS"},
        "IMAGE ACTIVE": object(),
        "LOG ACTIVE": {},
        "VISUAL ACTIVE": visual,
    }

    assert gate.check_lazy_status(enabled=True, READY=True, **inputs) == []
    result = gate.gate(enabled=True, READY=True, **inputs)
    assert result[4] is visual


def test_mask_detailer_process_has_no_second_enable_switch():
    text = _source()
    section = text[text.index("class CMKMaskDetailerProcess:"):]
    assert '"enable": ("BOOLEAN"' not in section
    assert "def process(MODEL, seed, steps" in section
    assert '"STATUS           : DISABLED"' not in section


def test_mask_detailer_prepare_materializes_module_35_model_spec():
    text = _source()
    assert "from ...pipe.loaders.checkpoint_vae_loader import resolve_postprocess_model" in text
    assert "MODEL = resolve_postprocess_model(MODEL)" in text


def test_pure_visual_preview_compare_has_optional_socket_enable_and_no_finish_features():
    backend = (ROOT / "pipe" / "cmk_visual.py").read_text(encoding="utf-8")
    frontend = (ROOT / "web" / "js" / "cmk_visualizer.js").read_text(encoding="utf-8")
    mappings = (ROOT / "cmk_mappings.py").read_text(encoding="utf-8")
    section = backend[backend.index("class CMKVisualCompare:"):]
    assert '"required": {"VISUAL": (VISUAL_TYPE, {"lazy": True})}' in section
    assert '"optional": {"enable": ("BOOLEAN", {"forceInput": True})}' in section
    assert "def check_lazy_status(VISUAL=None, enable=True):" in section
    assert "def show(VISUAL=None, enable=True):" in section
    assert '"cmk_visual_enabled": [False]' in section
    assert 'RETURN_TYPES = ()' in section
    assert 'CMKVisualizer.show(VISUAL=compare_visual)' in section
    assert "CMK_SaveProjectImage" not in section
    assert "CMK_SmartUpscalerPipe" not in section
    assert 'const VISUAL_COMPARE = "CMKVisualCompare"' in frontend
    assert 'cmk-preview-compare' in frontend
    assert '"CMKVisualCompare": "CMK Preview & Compare"' in mappings


def test_mask_detailer_is_deliberately_not_flow_or_cache_coupled():
    text = _source()
    forbidden = (
        "CMKFamilyResult",
        "BoundaryCache",
        "boundary_cache",
        "persistent_cache",
        "artifact_for",
    )
    assert not any(token in text for token in forbidden)


def test_mask_detailer_nodes_are_published_as_toolbox_building_blocks():
    text = _source()
    assert text.count('CATEGORY = "CMK/Toolbox/Image"') == 3
    metadata = json.loads((ROOT / "web" / "toolbox_node_metadata.json").read_text(encoding="utf-8"))["nodes"]
    assert "CMKMaskDetailerIntake" in metadata
    assert "CMKMaskDetailerPrepare" in metadata
    assert "CMKMaskDetailerProcess" in metadata


def test_mask_detailer_reference_is_standalone_and_not_a_flow_subgraph():
    workflow = ROOT / "workflows" / "showcase" / "CMK MaskDetailer.json"
    metadata = json.loads(
        (ROOT / "workflows" / "showcase" / "metadata" / "CMK MaskDetailer.json").read_text(encoding="utf-8")
    )
    document = json.loads(workflow.read_text(encoding="utf-8"))
    assert workflow.is_file()
    assert metadata["published"] is True
    assert metadata["kind"] == "standalone"
    assert "Stand-alone" in metadata["cmkHighlight"]
    assert not (ROOT / "subgraphs" / "CMK MaskDetailer.json").exists()
    assert "CMKFlow" not in document.get("extra", {})
    outer = next(
        node for node in document["nodes"]
        if node["type"] == "f38abe24-7d28-4946-a39b-6da5bdb60633"
    )
    assert "seed" not in [item["name"] for item in outer["inputs"]]


def test_mask_detailer_95_has_a_lazy_bypass_and_forwards_without_snapshot():
    active_path = ROOT.parents[1] / "user" / "default" / "workflows" / "Ref #05 - CMK MaskDetailer.json"
    active = json.loads(
        active_path.read_text(encoding="utf-8")
    )
    packaged = json.loads(
        (ROOT / "workflows" / "showcase" / "CMK MaskDetailer.json").read_text(encoding="utf-8")
    )
    definition_id = "f38abe24-7d28-4946-a39b-6da5bdb60633"
    for document in (active, packaged):
        definition = next(item for item in document["definitions"]["subgraphs"] if item["id"] == definition_id)
        assert definition["name"] == "CMK Flow · MaskDetailer SDXL"
        assert [item["name"] for item in definition["inputs"]][4] == "enabled"
        assert [item["name"] for item in definition["inputs"]][-1] == "denoise"
        assert "fill_mask_holes" not in [item["name"] for item in definition["inputs"]]
        gate = next(node for node in definition["nodes"] if node["type"] == "CMKMaskDetailerFlowGate")
        intake = next(node for node in definition["nodes"] if node["type"] == "CMKMaskDetailerIntake")
        compare = next(node for node in definition["nodes"] if node["type"] == "CMKVisualCompare")
        process = next(node for node in definition["nodes"] if node["type"] == "CMKMaskDetailerProcess")
        assert intake["outputs"][4]["name"] == "READY"
        gate_inputs = {item["name"]: item for item in gate["inputs"]}
        assert gate_inputs["READY"]["link"] is not None
        assert gate_inputs["enabled"]["link"] is not None
        assert compare["inputs"][0]["link"] in gate["outputs"][4]["links"]
        assert compare["inputs"][0]["link"] not in next(
            node for node in definition["nodes"] if node["type"] == "CMKMaskDetailerProcess"
        )["outputs"][4]["links"]
        assert "enable" not in [item["name"] for item in process["inputs"]]
        outer = next(node for node in document["nodes"] if node["type"] == definition_id)
        assert [item["name"] for item in outer["inputs"]][4] == "enabled"
        assert [item["name"] for item in outer["inputs"]][-1] == "denoise"
        assert "fill_mask_holes" not in [item["name"] for item in outer["inputs"]]
        assert outer["size"] == [450, 230]
        assert outer["properties"]["cmkOuterSize"] == [450, 230]
        assert outer["properties"]["cmkManualSize"] == [450, 230]
        assert outer["widgets_values_named"]["enabled"] is True
        declaration = outer["properties"]["cmkVisualProviders"][0]
        assert declaration["key"] == "mask-detailer"
        assert declaration["enable_widget"] == "enabled"
        assert declaration["stage_key"] == "mask-detailer.result"
        assert declaration["live_node_id"] == "7568"
        assert declaration["capabilities"]["live"] is True

    packaged_definition = next(
        item for item in packaged["definitions"]["subgraphs"] if item["id"] == definition_id
    )
    packaged_intake = next(
        node for node in packaged_definition["nodes"] if node["type"] == "CMKMaskDetailerIntake"
    )
    assert packaged_intake["widgets_values"] == [False, "None"]


def test_mask_detailer_95_accepts_direct_postprocessors_and_faceswap_results():
    path = ROOT.parents[1] / "user" / "default" / "subgraphs" / "CMK Flow · MaskDetailer SDXL.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    definition = document["definitions"]["subgraphs"][0]
    inputs = {item["name"]: item for item in definition["inputs"]}
    expected_inputs = {
        "MODEL": "CMK_MODEL_PIPE,CMK_RESULT_MODEL",
        "PROCESS": "CMK_RESULT_PROCESS",
        "IMAGE": "IMAGE,CMK_RESULT_IMAGE",
        "LOG": "CMK_LOG_PIPE,CMK_RESULT_LOG",
    }
    assert {name: inputs[name]["type"] for name in expected_inputs} == expected_inputs

    outer = next(node for node in document["nodes"] if node["type"] == definition["id"])
    outer_inputs = {item["name"]: item["type"] for item in outer["inputs"]}
    assert {name: outer_inputs[name] for name in expected_inputs} == expected_inputs

    unpack = next(node for node in definition["nodes"] if node["type"] == "CMKResultUnpackPipe")
    assert [item["type"] for item in unpack["inputs"]] == ["*", "*", "*", "*"]
    links = {item["id"]: item for item in definition["links"]}
    for slot, name in enumerate(("MODEL", "PROCESS", "IMAGE", "LOG")):
        link = links[inputs[name]["linkIds"][0]]
        assert (link["origin_id"], link["origin_slot"]) == (-10, slot)
        assert (link["target_id"], link["target_slot"]) == (unpack["id"], slot)

    nodes = {node["type"]: node for node in definition["nodes"]}
    expected_targets = {
        0: [(nodes["CMKMaskDetailerPrepare"]["id"], 0), (nodes["CMKMaskDetailerFlowGate"]["id"], 0)],
        1: [(nodes["CMKMaskDetailerIntake"]["id"], 0), (nodes["CMKMaskDetailerFlowGate"]["id"], 1)],
        2: [(nodes["CMKMaskDetailerIntake"]["id"], 1), (nodes["CMKMaskDetailerFlowGate"]["id"], 2)],
        3: [(nodes["CMKMaskDetailerIntake"]["id"], 2), (nodes["CMKMaskDetailerFlowGate"]["id"], 3)],
    }
    for slot, targets in expected_targets.items():
        assert [
            (links[link_id]["target_id"], links[link_id]["target_slot"])
            for link_id in unpack["outputs"][slot]["links"]
        ] == targets

    def accepts(input_type, output_type):
        return bool(
            {item.strip() for item in input_type.split(",")}
            & {item.strip() for item in output_type.split(",")}
        )

    source_files = (
        "CMK Flow · Detailer SDXL.json",
        "CMK Flow · FaceRebuild SDXL.json",
        "CMK Flow · FaceProcess SDXL.json",
        "CMK Flow · FaceSwap.json",
        "CMK Flow · FaceSwap · Advanced.json",
    )
    for filename in source_files:
        source = json.loads((ROOT / "subgraphs" / filename).read_text(encoding="utf-8"))
        outputs = {
            item["name"]: item["type"]
            for item in source["definitions"]["subgraphs"][0]["outputs"]
        }
        for name in expected_inputs:
            assert accepts(inputs[name]["type"], outputs[name]), f"{filename}: {name}"
        if "FaceSwap" in filename:
            assert [outputs[name] for name in expected_inputs] == [
                "CMK_RESULT_MODEL",
                "CMK_RESULT_PROCESS",
                "CMK_RESULT_IMAGE",
                "CMK_RESULT_LOG",
            ]


def test_mask_detailer_reference_uses_the_packaged_masked_image():
    document = json.loads(
        (ROOT / "workflows" / "showcase" / "CMK MaskDetailer.json").read_text(encoding="utf-8")
    )
    image_node = next(node for node in document["nodes"] if node["type"] == "CMKImageLoadAndResizePipe")
    packaged = "CMK Package · mask_detailer_reference.png"
    assert image_node["widgets_values"][0] == packaged
    assert image_node["widgets_values_named"]["image"] == packaged
    assert image_node["properties"]["image"] == packaged
    assert (ROOT / "assets" / "references" / "mask_detailer_reference.png").is_file()
    for relative_path in (
        "__init__.py",
        "pipe/loaders/cmk_load_image.py",
        "pipe/loaders/cmk_image_load_resize.py",
    ):
        assert '"mask_detailer_reference.png"' in (ROOT / relative_path).read_text(encoding="utf-8")


def test_mask_detailer_compare_is_visualizer_only():
    tree = ast.parse(_source())
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "register_provider" in calls
    assert "CMKImageCompare" not in _source()
    assert 'channels={"before": before, "after": after}' in _source()


def test_mask_detailer_exposes_one_lora_and_one_small_positive_prompt():
    text = _source()
    assert '"lora_name": (loras' in text
    assert '"prompt_pos": ("STRING", {"default": "", "multiline": False})' in text
    assert "lora_name_2" not in text


def test_prepare_exposes_switchable_fill_mask_holes():
    text = _source()
    assert '"fill_mask_holes": ("BOOLEAN", {"default": False})' in text
    assert "mask = fill_mask_holes_fn(mask)" in text
    assert '"fill_mask_holes": bool(fill_mask_holes)' in text


def test_visualizer_declares_mask_detailer_as_live_preview_source():
    text = (ROOT / "web" / "js" / "cmk_visualizer.js").read_text(encoding="utf-8")
    assert 'nodeKind(node) === "CMKMaskDetailerProcess"' in text
    assert '"mask-detailer": "CMKMaskDetailerProcess"' in text
    assert 'live_node_id: String(node.id)' in text
    assert 'stage_key: "mask-detailer.result"' in text


def test_mask_detailer_uses_the_same_segs_core_as_detailer_23():
    text = _source()
    assert "CMKSEG, SEGSDetailer, SEGSPaste" in text
    assert "segs = _mask_to_segs(before, mask, crop_factor=2.0)" in text
    assert "detailed = SEGSDetailer().doit(" in text
    assert "after = SEGSPaste.doit(" in text
    assert "VAEEncodeForInpaint" not in text


def test_mask_detailer_has_explicit_log_chain():
    text = _source()
    assert '"LOG": ("CMK_LOG_PIPE",)' in text
    assert 'RETURN_NAMES = ("MODEL", "MASK DETAILER PROCESS", "LOG", "VISUAL")' in text
    assert 'RETURN_TYPES = ("CMK_MODEL_PIPE", MASK_DETAILER_PROCESS, "CMK_LOG_PIPE", "CMK_VISUAL_PIPE")' in text
    assert 'cmk_add_block(LOG, "Mask Detailer Prepare"' in text
    assert 'cmk_add_block(LOG, "Mask Detailer Process"' in text


def test_mask_detailer_threads_the_effective_model_and_records_it():
    text = _source()
    assert '"MODEL": ("CMK_MODEL_PIPE",)' in text
    assert 'return (MODEL, process, log,' in text
    assert 'return (MODEL, result_pipe, after, log, visual)' in text
    assert 'basic_pipe=(pipe["model"], pipe["clip"], pipe["vae"]' in text
    assert 'MODEL SOURCE     : MODEL' in text
    assert 'f"CHECKPOINT       : {MODEL.get(' in text
    assert 'f"MODEL            : {MODEL.get(' not in text


def test_mask_detailer_preserves_loaded_image_origin_for_output_folder():
    text = _source()
    assert '"source_model_family": source.get("source_model_family", "image")' in text
    result_text = (ROOT / "pipe" / "cmk_family_result.py").read_text(encoding="utf-8")
    assert 'normalized.setdefault("source_model_family", "image")' in result_text


def test_visualizer_result_unpack_accepts_mask_detailer_family():
    text = (ROOT / "pipe" / "cmk_family_result.py").read_text(encoding="utf-8")
    assert 'PROCESS.get("type") == "CMK_MASK_DETAILER_PROCESS"' in text
    assert 'normalized["result_contract"] = "mask_detailer"' in text
    assert 'return (MODEL, normalized, IMAGE, LOG)' in text


def test_result_unpack_errors_do_not_name_removed_module_90():
    text = (ROOT / "pipe" / "cmk_family_result.py").read_text(encoding="utf-8")
    assert "CMK 90" not in text
    assert "CMK Result Unpack requires a CMK LOG" in text


def test_image_input_preserves_native_editor_mask_for_mask_detailer():
    text = IMAGE_INPUT_SOURCE.read_text(encoding="utf-8")
    assert '"image": (' in text
    assert '{"image_upload": True, "label": "IMAGE"}' in text
    assert 'inputs.get("image", inputs.get("IMAGE", ""))' in text
    assert 'prefix = "clipspace-painted-masked-"' in text
    assert 'clean_name = "clipspace-mask-"' in text
    assert 'RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG", "diagnostic", "MASK")' in text
    assert '"mask": resized_mask' in text
    assert 'frame.getchannel("A")' in text
