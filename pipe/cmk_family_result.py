try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:  # Unit tests outside a running ComfyUI installation.
    class ExecutionBlocker:
        def __init__(self, message):
            self.message = message

try:
    from ..utils.cmk_timing import cmk_timed_call
except ImportError:  # Direct source loading in contract tests.
    def cmk_timed_call(_stage):
        return lambda function: function

try:
    from .cmk_module_cache_contract import stamp_artifact
except ImportError:  # Direct source loading in contract tests.
    from pipe.cmk_module_cache_contract import stamp_artifact

try:
    from .cmk_persistent_cache import build_node_fingerprint
except ImportError:  # Direct source loading in contract tests.
    from pipe.cmk_persistent_cache import build_node_fingerprint

try:
    from .cmk_generation_completion_cache import (
        build_completion_key,
        invalidate_completion,
        lookup_completion,
        store_completion,
    )
except ImportError:  # Direct source loading in contract tests.
    from pipe.cmk_generation_completion_cache import (
        build_completion_key,
        invalidate_completion,
        lookup_completion,
        store_completion,
    )

try:
    from ..cmk_distribution import content_guard_prompt_token
except ImportError:  # Direct source loading in contract tests.
    from cmk_distribution import content_guard_prompt_token

try:
    from .cmk_prompt_result_state import (
        IncompleteResultError,
        MISSING,
        PromptStateUnavailableError,
        UnsafeSnapshotTypeError,
        active_prompt_id,
        get_resource,
        get_result,
        make_key,
        publish_result,
        release_resources,
        store_resource,
    )
except ImportError:  # Direct source loading in contract tests.
    from pipe.cmk_prompt_result_state import (
        IncompleteResultError,
        MISSING,
        PromptStateUnavailableError,
        UnsafeSnapshotTypeError,
        active_prompt_id,
        get_resource,
        get_result,
        make_key,
        publish_result,
        release_resources,
        store_resource,
    )

class _CMKAnyType(str):
    def __ne__(self, other):
        return False


CMK_FINISH_INPUT = _CMKAnyType("*")


def _detailer_diag_context(unique_id=None):
    """Return prompt/node context for temporary Detailer diagnostics."""
    try:
        prompt_id = active_prompt_id()
    except Exception:
        prompt_id = None
    return (
        f"prompt_id={prompt_id if prompt_id is not None else 'UNAVAILABLE'} "
        f"node_id={unique_id if unique_id is not None else 'UNAVAILABLE'}"
    )


def _detailer_diag_sdxl_state(state_key):
    """Inspect SDXL prompt state without changing either state store."""
    if state_key is None:
        return {
            "result": False,
            "resource": False,
            "descriptor": False,
            "reason": "STATE_KEY_UNAVAILABLE",
        }
    try:
        result = get_result(state_key)
        resource = get_resource(state_key)
    except PromptStateUnavailableError:
        return {
            "result": False,
            "resource": False,
            "descriptor": False,
            "reason": "PROMPT_STATE_UNAVAILABLE",
        }
    has_result = result is not MISSING
    has_resource = resource is not MISSING
    has_descriptor = bool(
        isinstance(result, dict) and result.get("released_model_descriptor") is not None
    )
    if not has_result:
        reason = "RESULT_MISSING"
    elif has_resource:
        reason = "RESOURCE_AVAILABLE"
    elif has_descriptor:
        reason = "RELEASED_DESCRIPTOR_AVAILABLE"
    else:
        reason = "RESOURCE_AND_DESCRIPTOR_MISSING"
    return {
        "result": has_result,
        "resource": has_resource,
        "descriptor": has_descriptor,
        "reason": reason,
    }

_GENERATION_MODEL_DESCRIPTOR_TYPE = "cmk_generation_model_descriptor"
_GENERATION_MODEL_DESCRIPTOR_VERSION = 1
_GENERATION_MODEL_FAMILIES = {"sdxl", "z_image_turbo"}
_GENERATION_RESOURCE_RELEASE_STATUSES = {"released", "unloaded"}
_GENERATION_MODEL_DESCRIPTOR_FIELDS = {
    "descriptor_type",
    "descriptor_version",
    "model_role",
    "model_family",
    "generation_model_identity",
    "generation_vae_identity",
    "resource_status",
    "resource_release_status",
    "resources_released",
    "materialized",
}


def make_released_generation_model_descriptor(
    model_family, model_identity, vae_identity, resource_release_status
):
    """Describe generation resources only after their successful release."""
    if not all(
        isinstance(value, str)
        for value in (
            model_family,
            model_identity,
            vae_identity,
            resource_release_status,
        )
    ):
        raise TypeError("CMK generation MODEL descriptor identities and status must be strings")
    family = str(model_family or "").strip().lower()
    model_name = str(model_identity or "").strip()
    vae_name = str(vae_identity or "").strip()
    release_status = str(resource_release_status or "").strip().lower()
    if family not in _GENERATION_MODEL_FAMILIES:
        raise ValueError("CMK generation MODEL descriptor requires SDXL or Z-Image Turbo")
    if not model_name or not vae_name:
        raise ValueError("CMK generation MODEL descriptor requires model and VAE identities")
    if release_status not in _GENERATION_RESOURCE_RELEASE_STATUSES:
        raise ValueError("CMK generation MODEL descriptor requires a successful resource release status")
    return {
        "descriptor_type": _GENERATION_MODEL_DESCRIPTOR_TYPE,
        "descriptor_version": _GENERATION_MODEL_DESCRIPTOR_VERSION,
        "model_role": "generation_result",
        "model_family": family,
        "generation_model_identity": model_name,
        "generation_vae_identity": vae_name,
        "resource_status": "released",
        "resource_release_status": release_status.upper(),
        "resources_released": True,
        "materialized": False,
    }


def _released_generation_model_descriptor(value):
    if not isinstance(value, dict) or value.get("descriptor_type") != _GENERATION_MODEL_DESCRIPTOR_TYPE:
        return None
    if value.get("descriptor_version") != _GENERATION_MODEL_DESCRIPTOR_VERSION:
        raise ValueError("CMK generation MODEL descriptor has an unsupported version")
    unexpected = set(value) - _GENERATION_MODEL_DESCRIPTOR_FIELDS
    if unexpected:
        raise ValueError(
            "CMK generation MODEL descriptor contains unsupported fields: "
            + ", ".join(sorted(str(name) for name in unexpected))
        )
    if any(type(item) not in (str, int, bool) for item in value.values()):
        raise TypeError("CMK generation MODEL descriptor contains a non-primitive value")
    family = str(value.get("model_family", "")).strip().lower()
    if family not in _GENERATION_MODEL_FAMILIES:
        raise ValueError("CMK generation MODEL descriptor has an invalid model family")
    if not all(
        isinstance(value.get(name), str) and value[name].strip()
        for name in ("generation_model_identity", "generation_vae_identity")
    ):
        raise ValueError("CMK generation MODEL descriptor is missing model or VAE identity")
    release_status_value = value.get("resource_release_status")
    release_status = (
        release_status_value.strip().lower()
        if isinstance(release_status_value, str)
        else ""
    )
    if (
        value.get("model_role") != "generation_result"
        or value.get("resource_status") != "released"
        or value.get("resources_released") is not True
        or value.get("materialized") is not False
        or release_status not in _GENERATION_RESOURCE_RELEASE_STATUSES
    ):
        raise ValueError("CMK generation MODEL descriptor does not prove completed resource release")
    return value


class _CMKGenerationCompletionMode:
    __slots__ = (
        "name",
        "generation_mode",
        "cache_family",
        "runtime_family",
        "historical_root_roles",
        "root_ports",
        "required_root_roles",
        "graph_validator",
        "hit_validator",
        "finalization_validator",
        "finalization_uses_process",
        "descriptor_family",
        "log_label",
    )

    def __init__(
        self,
        *,
        name,
        generation_mode,
        cache_family,
        runtime_family,
        historical_root_roles,
        root_ports,
        required_root_roles,
        graph_validator,
        hit_validator,
        finalization_validator,
        finalization_uses_process,
        descriptor_family,
        log_label,
    ):
        self.name = name
        self.generation_mode = generation_mode
        self.cache_family = cache_family
        self.runtime_family = runtime_family
        self.historical_root_roles = tuple(historical_root_roles)
        self.root_ports = {
            role: tuple(names) for role, names in root_ports.items()
        }
        self.required_root_roles = frozenset(required_root_roles)
        self.graph_validator = graph_validator
        self.hit_validator = hit_validator
        self.finalization_validator = finalization_validator
        self.finalization_uses_process = bool(finalization_uses_process)
        self.descriptor_family = descriptor_family
        self.log_label = log_label


_GENERIC_ROOT_ROLES = ("MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL")
_SDXL_MERGE_ROOT_ROLES = (
    "MODEL SDXL",
    "PROCESS SDXL",
    "IMAGE SDXL",
    "LOG SDXL",
    "VISUAL SDXL",
)
_ZIT_MERGE_ROOT_ROLES = (
    "MODEL ZIT",
    "PROCESS ZIT",
    "IMAGE ZIT",
    "LOG ZIT",
    "VISUAL ZIT",
)


def _root_ports(historical_roles, *port_groups):
    return {
        role: tuple(group[index] for group in port_groups)
        for index, role in enumerate(historical_roles)
    }


_SDXL_COMPLETION = _CMKGenerationCompletionMode(
    name="SDXL",
    generation_mode="sdxl",
    cache_family="sdxl",
    runtime_family="sdxl",
    historical_root_roles=_GENERIC_ROOT_ROLES,
    root_ports=_root_ports(_GENERIC_ROOT_ROLES, _GENERIC_ROOT_ROLES, _SDXL_MERGE_ROOT_ROLES),
    required_root_roles=("MODEL", "PROCESS", "IMAGE", "LOG"),
    graph_validator="_sdxl_completion_graph",
    hit_validator="_sdxl_completion_hit",
    finalization_validator=None,
    finalization_uses_process=False,
    descriptor_family="sdxl",
    log_label="SDXL",
)
_ZIT_COMPLETION = _CMKGenerationCompletionMode(
    name="Z_IMAGE_TURBO",
    generation_mode="z_image_turbo",
    cache_family="z_image_turbo",
    runtime_family="z_image_turbo",
    historical_root_roles=_GENERIC_ROOT_ROLES,
    root_ports=_root_ports(_GENERIC_ROOT_ROLES, _GENERIC_ROOT_ROLES, _ZIT_MERGE_ROOT_ROLES),
    required_root_roles=("MODEL", "PROCESS", "IMAGE", "LOG"),
    graph_validator="_zit_completion_graph",
    hit_validator="_zit_completion_hit",
    finalization_validator="_zit_finalization_evidence",
    finalization_uses_process=False,
    descriptor_family="z_image_turbo",
    log_label="ZIT",
)
_SDXL_ZIT_HYBRID_COMPLETION = _CMKGenerationCompletionMode(
    name="SDXL_ZIT_HYBRID",
    generation_mode="sdxl_zit_hybrid",
    cache_family="hybrid",
    runtime_family="z_image_turbo",
    historical_root_roles=_ZIT_MERGE_ROOT_ROLES,
    root_ports=_root_ports(_ZIT_MERGE_ROOT_ROLES, _ZIT_MERGE_ROOT_ROLES),
    required_root_roles=_ZIT_MERGE_ROOT_ROLES,
    graph_validator="_hybrid_graph",
    hit_validator="_hybrid_completion_hit",
    finalization_validator="_hybrid_finalization_evidence",
    finalization_uses_process=True,
    descriptor_family="hybrid",
    log_label="HYBRID",
)
_INSTANTID_HYBRID_COMPLETION = _CMKGenerationCompletionMode(
    name="INSTANTID_HYBRID",
    generation_mode="instantid_hybrid",
    cache_family="hybrid",
    runtime_family="sdxl",
    historical_root_roles=_SDXL_MERGE_ROOT_ROLES,
    root_ports=_root_ports(_SDXL_MERGE_ROOT_ROLES, _SDXL_MERGE_ROOT_ROLES),
    required_root_roles=_SDXL_MERGE_ROOT_ROLES,
    graph_validator="_instantid_hybrid_graph",
    hit_validator="_instantid_hybrid_completion_hit",
    finalization_validator="_instantid_hybrid_finalization_evidence",
    finalization_uses_process=True,
    descriptor_family="sdxl",
    log_label="INSTANTID-HYBRID",
)
_INSTANTID_ZERO_PASS_COMPLETION = _CMKGenerationCompletionMode(
    name="INSTANTID_ZERO_PASS",
    generation_mode="instantid_zero_pass",
    cache_family="hybrid",
    runtime_family="sdxl",
    historical_root_roles=_SDXL_MERGE_ROOT_ROLES,
    root_ports=_root_ports(_SDXL_MERGE_ROOT_ROLES, _SDXL_MERGE_ROOT_ROLES),
    required_root_roles=_SDXL_MERGE_ROOT_ROLES,
    graph_validator="_instantid_zero_pass_graph",
    hit_validator="_instantid_zero_pass_completion_hit",
    finalization_validator="_instantid_zero_pass_finalization_evidence",
    finalization_uses_process=True,
    descriptor_family="sdxl",
    log_label="INSTANTID-ZERO-PASS",
)


def _completion_prompt_node(prompt, node_id):
    if not isinstance(prompt, dict):
        return None
    node = prompt.get(node_id)
    if not isinstance(node, dict):
        node = prompt.get(str(node_id))
    return node if isinstance(node, dict) else None


def _completion_prompt_link(prompt, value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], int)
        and _completion_prompt_node(prompt, value[0]) is not None
    )


def _completion_mode_roots(prompt, current_inputs, contract):
    roots = {}
    for role in contract.historical_root_roles:
        link = None
        for port_name in contract.root_ports[role]:
            candidate = current_inputs.get(port_name)
            if _completion_prompt_link(prompt, candidate):
                link = candidate
                break
        if link is not None:
            roots[role] = link
        elif role in contract.required_root_roles:
            return None, f"required {contract.name} generation root {role!r} is unavailable"
    return roots, "generation roots complete"


def _completion_key_for_graph(prompt, graph, contract):
    if not isinstance(graph, dict) or not isinstance(graph.get("roots"), dict):
        return None, f"{contract.name} completion graph is unavailable"
    return build_completion_key(
        prompt,
        graph["roots"],
        model_family=contract.cache_family,
        generation_mode=contract.generation_mode,
    )


def _completion_graph(owner, contract, prompt, unique_id):
    validator = getattr(owner, contract.graph_validator)
    return validator(prompt, unique_id)


def _completion_preflight(owner, contract, prompt, unique_id, graph=None, detail=None):
    if graph is None:
        graph, detail = _completion_graph(owner, contract, prompt, unique_id)
    if graph is None:
        return {
            "contract": contract,
            "graph": None,
            "key": None,
            "detail": detail,
            "cached": None,
        }
    cache_key, key_detail = _completion_key_for_graph(prompt, graph, contract)
    hit_validator = getattr(owner, contract.hit_validator)
    return {
        "contract": contract,
        "graph": graph,
        "key": cache_key,
        "detail": key_detail if cache_key else key_detail or detail,
        "cached": hit_validator(cache_key),
    }


def _completion_active_contract(contracts, prompt, unique_id):
    """Select the contract from the authoritative START source configuration."""
    current = _completion_prompt_node(prompt, unique_id)
    if not isinstance(current, dict):
        return None, "completion owner is unavailable in the expanded prompt"

    pending = [
        value[0]
        for value in (current.get("inputs", {}) or {}).values()
        if _completion_prompt_link(prompt, value)
    ]
    visited = set()
    sources = {}
    while pending:
        node_id = pending.pop()
        node_key = str(node_id)
        if node_key in visited:
            continue
        visited.add(node_key)
        node = _completion_prompt_node(prompt, node_id)
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")) == "CMKPipeCreateImage":
            sources[node_key] = node
            continue
        for value in (node.get("inputs", {}) or {}).values():
            if _completion_prompt_link(prompt, value):
                pending.append(value[0])

    if len(sources) != 1:
        return None, "expanded graph does not identify exactly one active generation source"

    source_inputs = next(iter(sources.values())).get("inputs", {}) or {}
    raw_family = str(source_inputs.get("model_family", "")).strip().lower()
    if raw_family == "hybrid":
        contract_name = "SDXL_ZIT_HYBRID"
    elif raw_family in {"z-image turbo", "z image turbo", "z_image_turbo"}:
        contract_name = "Z_IMAGE_TURBO"
    elif raw_family == "sdxl":
        if source_inputs.get("instantid_enabled") is True:
            instantid_mode = str(
                source_inputs.get("instantid_mode", "")
            ).strip().upper()
            if instantid_mode == "HYBRID":
                contract_name = "INSTANTID_HYBRID"
            elif instantid_mode == "ZERO-PASS":
                contract_name = "INSTANTID_ZERO_PASS"
            else:
                return None, "active source selects an unsupported InstantID mode"
        else:
            contract_name = "SDXL"
    else:
        return None, "active source selects an unknown generation family"

    matches = [contract for contract in contracts if contract.name == contract_name]
    if len(matches) != 1:
        return None, f"active source selects unavailable contract {contract_name}"
    return matches[0], f"active source selects {contract_name}"


def _resolve_completion_preflight(owner, contracts, prompt, unique_id):
    contract, selection_detail = _completion_active_contract(
        contracts, prompt, unique_id
    )
    if contract is None:
        return {
            "contract": None,
            "graph": None,
            "key": None,
            "detail": selection_detail,
            "cached": None,
        }
    graph, detail = _completion_graph(owner, contract, prompt, unique_id)
    if graph is None:
        return {
            "contract": None,
            "graph": None,
            "key": None,
            "detail": f"{selection_detail}; {contract.name}: {detail}",
            "cached": None,
        }
    return _completion_preflight(owner, contract, prompt, unique_id, graph, detail)


def _completion_finalization_evidence(owner, preflight, prompt, unique_id, process):
    contract = preflight["contract"]
    if contract.finalization_validator is None:
        return {}, "existing SDXL completion contract"
    validator = getattr(owner, contract.finalization_validator)
    if contract.finalization_uses_process:
        return validator(prompt, unique_id, process)
    return validator(prompt, unique_id)


def _completion_released_descriptor(contract, model, completion_metadata, evidence):
    if contract.descriptor_family == "hybrid":
        descriptor = evidence.get("model_descriptor") if isinstance(evidence, dict) else None
        try:
            released = _released_generation_model_descriptor(descriptor)
        except (TypeError, ValueError):
            return None
        return released if released is not None and released["model_family"] == "z_image_turbo" else None
    descriptor = _released_generation_model_descriptor(model)
    if descriptor is not None:
        return descriptor if descriptor["model_family"] == contract.descriptor_family else None
    if completion_metadata.get("generation_unload") != "UNLOADED":
        return None
    if contract.descriptor_family == "sdxl":
        model_identity = str(model.get("ckpt_name", "")).strip()
        vae_identity = str(
            model.get("vae_selection_name") or model.get("vae_name") or ""
        ).strip()
    else:
        model_identity = str(model.get("diffusion_model_name", "")).strip()
        vae_identity = str(model.get("vae_name", "")).strip()
    if not model_identity or not vae_identity:
        return None
    return make_released_generation_model_descriptor(
        contract.descriptor_family,
        model_identity,
        vae_identity,
        "unloaded",
    )


def _store_completion_result(preflight, *, image, process, log, visual, metadata, descriptor):
    contract = preflight["contract"]
    store_completion(
        preflight["key"],
        image=image,
        process=process,
        log=log,
        visual=visual,
        model_family=contract.cache_family,
        generation_mode=contract.generation_mode,
        fingerprint=preflight["key"],
        metadata=metadata,
        model_descriptor=descriptor,
    )


def _pending_connected_input(name, prompt, unique_id, inputs):
    current = prompt.get(str(unique_id), {}) if isinstance(prompt, dict) else {}
    connected = name in ((current.get("inputs") or {}) if isinstance(current, dict) else {})
    return [name] if connected and inputs.get(name) is None else []


def _module_gate_diagnostic(value):
    """Keep diagnostics auxiliary when a lazy producer did not materialize them."""
    if value is not None:
        return value
    return {
        "type": "CMK_DIAGNOSTIC",
        "version": 1,
        "title": "Module Result",
        "node": "CMK Module Bypass Gate",
        "mode": "Active / Diagnostic unavailable",
        "summary": "Active module result materialized without a diagnostic payload.",
        "details": (
            "MODEL, IMAGE and LOG were forwarded normally. The optional diagnostic "
            "output was not materialized by the lazy upstream path."
        ),
        "metadata": {"diagnostic_fallback": True},
        "metrics": {},
        "warnings": [],
        "preview": [],
        "images": [],
        "stages": [],
    }


_FACEPROCESS_RESULT_PHASE = "faceprocess.final"
_FACEPROCESS_RESULT_SCHEMA = "cmk_postprocess_faceprocess_result_v1"


def _prompt_node(prompt, node_id):
    if not isinstance(prompt, dict):
        return None
    node = prompt.get(node_id)
    if node is None:
        node = prompt.get(str(node_id))
    if isinstance(node, dict):
        return node
    node_text = str(node_id) if node_id is not None else ""
    if not node_text:
        return None
    matches = [
        candidate
        for key, candidate in prompt.items()
        if isinstance(candidate, dict)
        and (
            str(key).endswith(f":{node_text}")
            or str(key).endswith(f"/{node_text}")
            or str(key).endswith(f".{node_text}")
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _faceprocess_result_gate(prompt, unique_id):
    current = _prompt_node(prompt, unique_id)
    if current is None or current.get("class_type") != "CMKModuleBypassGate":
        return False
    current_inputs = current.get("inputs", {}) or {}
    boundary_ids = set()
    for name in ("MODEL ACTIVE", "IMAGE ACTIVE", "LOG ACTIVE"):
        link = current_inputs.get(name)
        if not isinstance(link, (list, tuple)) or len(link) != 2:
            return False
        boundary = _prompt_node(prompt, link[0])
        if boundary is None or boundary.get("class_type") != "CMKFaceBoundaryCache":
            return False
        boundary_ids.add(str(link[0]))
    return len(boundary_ids) == 1


def _faceprocess_state_key(prompt, unique_id):
    if not _faceprocess_result_gate(prompt, unique_id):
        return None
    fingerprint, _ = build_node_fingerprint(
        prompt,
        unique_id,
        ("CMKModuleBypassGate",),
        _FACEPROCESS_RESULT_SCHEMA,
        include_node_identity=True,
    )
    if not fingerprint:
        return None
    try:
        return make_key(
            node_id=unique_id,
            model_family="postprocess",
            phase=_FACEPROCESS_RESULT_PHASE,
            fingerprint=fingerprint,
            schema_version=_FACEPROCESS_RESULT_SCHEMA,
        )
    except PromptStateUnavailableError:
        return None


def _faceprocess_hit(state_key):
    if state_key is None:
        return None
    result = get_result(state_key)
    if (
        result is MISSING
        or not isinstance(result, dict)
        or result.get("schema_version") != _FACEPROCESS_RESULT_SCHEMA
    ):
        return None
    resources = get_resource(state_key)
    if (
        resources is MISSING
        or not isinstance(resources, dict)
        or "model" not in resources
    ):
        return None
    return resources["model"], result


def _publish_faceprocess(state_key, model, image, log, diagnostic, visual):
    if state_key is None or any(
        value is None for value in (model, image, log, diagnostic)
    ):
        return
    result = {
        "schema_version": _FACEPROCESS_RESULT_SCHEMA,
        "image": image,
        "log": log,
        "visual": visual,
        "diagnostic": diagnostic,
    }
    try:
        store_resource(state_key, {"model": model})
        publish_result(state_key, result)
    except (IncompleteResultError, UnsafeSnapshotTypeError) as exc:
        print(f"[CMK FaceProcess Result-State] NOT PUBLISHED: {exc}", flush=True)


class CMKImageCompareEnableGate:
    """Prevent embedded ImageCompare UI output while a module is disabled."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"ENABLE": ("BOOLEAN", {"default": True})},
            "optional": {
                "IMAGE A": ("IMAGE", {"lazy": True}),
                "IMAGE B": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE A", "IMAGE B")
    FUNCTION = "gate"
    CATEGORY = "CMK/Developer/Legacy Gates"
    DEV_ONLY = True

    def check_lazy_status(self, ENABLE=True, **inputs):
        if not bool(ENABLE):
            return []
        if inputs.get("IMAGE A") is None:
            return ["IMAGE A"]
        if inputs.get("IMAGE B") is None:
            return ["IMAGE B"]
        return []

    @staticmethod
    def gate(ENABLE=True, **inputs):
        if not bool(ENABLE):
            blocked = ExecutionBlocker(None)
            return blocked, blocked
        if inputs.get("IMAGE A") is None or inputs.get("IMAGE B") is None:
            raise ValueError("CMK Image Compare Enable Gate requires both images when enabled")
        return inputs["IMAGE A"], inputs["IMAGE B"]


class CMKModuleBypassGate:
    """Select a complete module result without evaluating the inactive path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"ENABLE": ("BOOLEAN", {"default": False})},
            "optional": {
                "MODEL BYPASS": (CMK_FINISH_INPUT, {"lazy": True}),
                "IMAGE BYPASS": (CMK_FINISH_INPUT, {"lazy": True}),
                "LOG BYPASS": (CMK_FINISH_INPUT, {"lazy": True}),
                "MODEL ACTIVE": (CMK_FINISH_INPUT, {"lazy": True}),
                "IMAGE ACTIVE": (CMK_FINISH_INPUT, {"lazy": True}),
                "LOG ACTIVE": (CMK_FINISH_INPUT, {"lazy": True}),
                "DIAGNOSTIC ACTIVE": ("CMK_DIAGNOSTIC", {"lazy": True}),
                "VISUAL BYPASS": ("CMK_VISUAL_PIPE", {"lazy": True}),
                "VISUAL ACTIVE": ("CMK_VISUAL_PIPE", {"lazy": True}),
                "PROCESS BYPASS": (CMK_FINISH_INPUT, {"lazy": True}),
                "PROCESS ACTIVE": (CMK_FINISH_INPUT, {"lazy": True}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE", "IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC", "CMK_VISUAL_PIPE", CMK_FINISH_INPUT)
    RETURN_NAMES = ("MODEL", "IMAGE", "LOG", "diagnostic", "VISUAL", "PROCESS")
    FUNCTION = "gate"
    CATEGORY = "CMK/Internal/Gates"
    DEV_ONLY = True

    def check_lazy_status(self, ENABLE=False, prompt=None, unique_id=None, **inputs):
        if bool(ENABLE):
            state_key = _faceprocess_state_key(prompt, unique_id)
            if _faceprocess_hit(state_key) is not None:
                return []
        prefix = "ACTIVE" if bool(ENABLE) else "BYPASS"
        current = prompt.get(str(unique_id), {}) if isinstance(prompt, dict) else {}
        connected = (current.get("inputs", {}) or {})
        for name in (f"MODEL {prefix}", f"IMAGE {prefix}", f"LOG {prefix}"):
            if name in connected and inputs.get(name) is None:
                return [name]
        if bool(ENABLE) and "DIAGNOSTIC ACTIVE" in connected and inputs.get("DIAGNOSTIC ACTIVE") is None:
            return ["DIAGNOSTIC ACTIVE"]
        visual_name = f"VISUAL {prefix}"
        if visual_name in connected and inputs.get(visual_name) is None:
            return [visual_name]
        process_name = f"PROCESS {prefix}"
        if process_name in connected and inputs.get(process_name) is None:
            return [process_name]
        return []

    @staticmethod
    def gate(ENABLE=False, prompt=None, unique_id=None, **inputs):
        state_key = None
        if bool(ENABLE):
            state_key = _faceprocess_state_key(prompt, unique_id)
            hit = _faceprocess_hit(state_key)
            if hit is not None:
                model, result = hit
                return (
                    model,
                    result["image"],
                    result["log"],
                    result["diagnostic"],
                    result["visual"],
                    inputs.get("PROCESS ACTIVE"),
                )
        prefix = "ACTIVE" if bool(ENABLE) else "BYPASS"
        values = tuple(inputs.get(f"{name} {prefix}") for name in ("MODEL", "IMAGE", "LOG"))
        if values[1] is None or values[2] is None:
            raise ValueError(f"CMK Module Bypass Gate is missing the {prefix} result")
        diagnostic = (
            _module_gate_diagnostic(inputs.get("DIAGNOSTIC ACTIVE"))
            if bool(ENABLE)
            else ExecutionBlocker(None)
        )
        result = (
            *values,
            diagnostic,
            inputs.get(f"VISUAL {prefix}"),
            inputs.get(f"PROCESS {prefix}"),
        )
        if bool(ENABLE):
            _publish_faceprocess(
                state_key,
                result[0],
                result[1],
                result[2],
                result[3],
                result[4],
            )
        return result


class CMKControlNetBypassGate:
    """Select ControlNet output or its unchanged SDXL input lazily."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"ENABLE": ("BOOLEAN", {"default": False})},
            "optional": {
                "PROCESS BYPASS": ("CMK_PROCESS_SDXL", {"lazy": True}),
                "IMAGE BYPASS": ("IMAGE", {"lazy": True}),
                "LOG BYPASS": ("CMK_LOG_PIPE", {"lazy": True}),
                "PROCESS ACTIVE": ("CMK_PROCESS_SDXL", {"lazy": True}),
                "IMAGE ACTIVE": ("IMAGE", {"lazy": True}),
                "LOG ACTIVE": ("CMK_LOG_PIPE", {"lazy": True}),
                "DIAGNOSTIC ACTIVE": ("CMK_DIAGNOSTIC", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("CMK_PROCESS_SDXL", "IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("PROCESS", "IMAGE", "LOG", "diagnostic")
    FUNCTION = "gate"
    CATEGORY = "CMK/Developer/Legacy Gates"
    DEV_ONLY = True

    def check_lazy_status(self, ENABLE=False, **inputs):
        prefix = "ACTIVE" if bool(ENABLE) else "BYPASS"
        # IMAGE may legitimately be None for Text2Image.  PROCESS and LOG are
        # the authoritative evidence that the selected branch was evaluated.
        for name in (f"PROCESS {prefix}", f"LOG {prefix}"):
            if inputs.get(name) is None:
                return [name]
        if bool(ENABLE) and inputs.get("DIAGNOSTIC ACTIVE") is None:
            return ["DIAGNOSTIC ACTIVE"]
        return []

    @staticmethod
    def gate(ENABLE=False, **inputs):
        prefix = "ACTIVE" if bool(ENABLE) else "BYPASS"
        values = tuple(inputs.get(f"{name} {prefix}") for name in ("PROCESS", "IMAGE", "LOG"))
        if values[0] is None or values[2] is None:
            raise ValueError(f"CMK ControlNet Bypass Gate is missing the {prefix} result")
        diagnostic = inputs.get("DIAGNOSTIC ACTIVE") if bool(ENABLE) else ExecutionBlocker(None)
        if bool(ENABLE) and diagnostic is None:
            raise ValueError("CMK ControlNet Bypass Gate is missing the active diagnostic")
        return (*values, diagnostic)


class CMKZITControlNetBypassGate(CMKControlNetBypassGate):
    """ZIT-typed counterpart of the established ControlNet bypass gate."""

    @classmethod
    def INPUT_TYPES(cls):
        contract = super().INPUT_TYPES()
        for name in ("PROCESS BYPASS", "PROCESS ACTIVE"):
            contract["optional"][name] = ("CMK_PROCESS_Z_IMAGE", {"lazy": True})
        return contract

    RETURN_TYPES = ("CMK_PROCESS_Z_IMAGE", "IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")


class CMKCombinedControlNetBypassGate:
    """Select the active Combined ControlNet result lazily."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"ENABLE": ("BOOLEAN", {"default": False})},
            "optional": {
                "PROCESS SDXL BYPASS": ("CMK_PROCESS_SDXL", {"lazy": True}),
                "PROCESS ZIT BYPASS": ("CMK_PROCESS_Z_IMAGE", {"lazy": True}),
                "IMAGE BYPASS": ("IMAGE", {"lazy": True}),
                "LOG BYPASS": ("CMK_LOG_PIPE", {"lazy": True}),
                "PROCESS SDXL ACTIVE": ("CMK_PROCESS_SDXL", {"lazy": True}),
                "PROCESS ZIT ACTIVE": ("CMK_PROCESS_Z_IMAGE", {"lazy": True}),
                "IMAGE ACTIVE": ("IMAGE", {"lazy": True}),
                "LOG ACTIVE": ("CMK_LOG_PIPE", {"lazy": True}),
                "DIAGNOSTIC ACTIVE": ("CMK_DIAGNOSTIC", {"lazy": True}),
            },
        }

    RETURN_TYPES = (
        "CMK_PROCESS_SDXL", "CMK_PROCESS_Z_IMAGE", "IMAGE",
        "CMK_LOG_PIPE", "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("PROCESS SDXL", "PROCESS ZIT", "IMAGE", "LOG", "diagnostic")
    FUNCTION = "gate"
    CATEGORY = "CMK/Internal/Gates"
    DEV_ONLY = True

    @staticmethod
    def _hybrid_mode(inputs):
        return any(
            isinstance(inputs.get(name), dict)
            and bool(inputs[name].get("hybrid_mode", False))
            for name in (
                "PROCESS SDXL ACTIVE", "PROCESS ZIT ACTIVE",
                "PROCESS SDXL BYPASS", "PROCESS ZIT BYPASS",
            )
        )

    def check_lazy_status(self, ENABLE=False, **inputs):
        if not bool(ENABLE):
            for base in ("PROCESS SDXL", "PROCESS ZIT", "LOG"):
                name = f"{base} BYPASS"
                if inputs.get(name) is None:
                    return [name]
            return []

        # Resolve the SDXL ControlNet result first. Its PROCESS carries the
        # Hybrid marker while the prepare node itself already suppresses ZIT.
        # The gate can then request the untouched ZIT bypass instead of the
        # nominal active leg.
        if inputs.get("PROCESS SDXL ACTIVE") is None:
            return ["PROCESS SDXL ACTIVE"]
        required = ["LOG ACTIVE"]
        required.append(
            "PROCESS ZIT BYPASS"
            if self._hybrid_mode(inputs)
            else "PROCESS ZIT ACTIVE"
        )
        for name in required:
            if inputs.get(name) is None:
                return [name]
        if inputs.get("DIAGNOSTIC ACTIVE") is None:
            return ["DIAGNOSTIC ACTIVE"]
        return []

    @classmethod
    def gate(cls, ENABLE=False, **inputs):
        if not bool(ENABLE):
            values = tuple(inputs.get(f"{base} BYPASS") for base in (
                "PROCESS SDXL", "PROCESS ZIT", "IMAGE", "LOG",
            ))
            result_label = "BYPASS"
        elif cls._hybrid_mode(inputs):
            # Hybrid ControlNet belongs exclusively to the SDXL composition
            # phase. Keep the ZIT finisher on its untouched bypass contract.
            values = (
                inputs.get("PROCESS SDXL ACTIVE"),
                inputs.get("PROCESS ZIT BYPASS"),
                inputs.get("IMAGE ACTIVE"),
                inputs.get("LOG ACTIVE"),
            )
            result_label = "HYBRID SDXL ACTIVE / ZIT BYPASS"
        else:
            values = tuple(inputs.get(f"{base} ACTIVE") for base in (
                "PROCESS SDXL", "PROCESS ZIT", "IMAGE", "LOG",
            ))
            result_label = "ACTIVE"
        if values[0] is None or values[1] is None or values[3] is None:
            raise ValueError(
                f"CMK Combined ControlNet Bypass Gate is missing the {result_label} result"
            )
        diagnostic = inputs.get("DIAGNOSTIC ACTIVE") if bool(ENABLE) else ExecutionBlocker(None)
        return (*values, diagnostic)


_INSTANTID_RESULT_PHASE = "instantid.final"
_INSTANTID_RESULT_SCHEMA = "cmk_instantid_result_v1"
_INSTANTID_SAMPLED_RESOURCE_FIELDS = {
    "model",
    "model_base",
    "model_identity",
    "model_patched",
    "clip",
    "clip_base",
    "clip_patched",
    "vae",
    "control_net",
    "conditioning_pos",
    "conditioning_neg",
    "conditioning_identity_pos",
    "conditioning_identity_neg",
    "lora_stack",
}


def _instantid_state_key(prompt, unique_id):
    fingerprint, _ = build_node_fingerprint(
        prompt,
        unique_id,
        ("CMKSamplerBypassGate",),
        _INSTANTID_RESULT_SCHEMA,
        include_node_identity=True,
        extra_identity={"content_guard": content_guard_prompt_token(prompt)},
    )
    if not fingerprint:
        return None
    try:
        return make_key(
            node_id=unique_id,
            model_family="sdxl",
            phase=_INSTANTID_RESULT_PHASE,
            fingerprint=fingerprint,
            schema_version=_INSTANTID_RESULT_SCHEMA,
        )
    except PromptStateUnavailableError:
        return None


def _partition_instantid_sampled(sampled):
    result_fields = {
        name: value
        for name, value in sampled.items()
        if name not in _INSTANTID_SAMPLED_RESOURCE_FIELDS
    }
    resource_fields = {
        name: value
        for name, value in sampled.items()
        if name in _INSTANTID_SAMPLED_RESOURCE_FIELDS
    }
    return result_fields, resource_fields


def _instantid_hit(state_key):
    if state_key is None:
        return None
    result = get_result(state_key)
    if (
        result is MISSING
        or result.get("schema_version") != _INSTANTID_RESULT_SCHEMA
    ):
        return None
    resources = get_resource(state_key)
    if resources is MISSING:
        return None
    sampled = dict(result["sampled"])
    sampled.update(resources["sampled"])
    return resources["model"], sampled, result


def _publish_instantid(state_key, model, sampled, log, diagnostic, visual):
    if state_key is None or any(
        value is None for value in (model, sampled, log, diagnostic)
    ) or not isinstance(sampled, dict):
        return
    result_sampled, resource_sampled = _partition_instantid_sampled(sampled)
    result = {
        "schema_version": _INSTANTID_RESULT_SCHEMA,
        "sampled": result_sampled,
        "log": log,
        "diagnostic": diagnostic,
        "visual": visual,
    }
    try:
        store_resource(
            state_key,
            {"model": model, "sampled": resource_sampled},
        )
        publish_result(state_key, result)
    except (IncompleteResultError, UnsafeSnapshotTypeError) as exc:
        print(f"[CMK InstantID Result-State] NOT PUBLISHED: {exc}", flush=True)


class CMKSamplerBypassGate:
    """Select an InstantID sampler result or its unchanged sampled input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"ENABLE": ("BOOLEAN", {"default": False})},
            "optional": {
                "MODEL BYPASS": ("CMK_MODEL_PIPE", {"lazy": True}),
                "SAMPLED BYPASS": ("CMK_SAMPLED_PIPE", {"lazy": True}),
                "LOG BYPASS": ("CMK_LOG_PIPE", {"lazy": True}),
                "MODEL ACTIVE": ("CMK_MODEL_PIPE", {"lazy": True}),
                "SAMPLED ACTIVE": ("CMK_SAMPLED_PIPE", {"lazy": True}),
                "LOG ACTIVE": ("CMK_LOG_PIPE", {"lazy": True}),
                "DIAGNOSTIC ACTIVE": ("CMK_DIAGNOSTIC", {"lazy": True}),
                "VISUAL BYPASS": ("CMK_VISUAL_PIPE", {"lazy": True}),
                "VISUAL ACTIVE": ("CMK_VISUAL_PIPE", {"lazy": True}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE", "CMK_SAMPLED_PIPE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC", "CMK_VISUAL_PIPE")
    RETURN_NAMES = ("MODEL", "SAMPLED", "LOG", "diagnostic", "VISUAL")
    FUNCTION = "gate"
    CATEGORY = "CMK/Internal/Gates"
    DEV_ONLY = True

    def check_lazy_status(self, ENABLE=False, prompt=None, unique_id=None, **inputs):
        if bool(ENABLE):
            state_key = _instantid_state_key(prompt, unique_id)
            if _instantid_hit(state_key) is not None:
                return []
        prefix = "ACTIVE" if bool(ENABLE) else "BYPASS"
        for name in (f"MODEL {prefix}", f"SAMPLED {prefix}", f"LOG {prefix}"):
            if inputs.get(name) is None:
                return [name]
        if bool(ENABLE) and inputs.get("DIAGNOSTIC ACTIVE") is None:
            return ["DIAGNOSTIC ACTIVE"]
        return _pending_connected_input(
            f"VISUAL {prefix}", prompt, unique_id, inputs
        )

    @staticmethod
    def gate(ENABLE=False, prompt=None, unique_id=None, **inputs):
        if bool(ENABLE):
            state_key = _instantid_state_key(prompt, unique_id)
            hit = _instantid_hit(state_key)
            if hit is not None:
                model, sampled, result = hit
                return (
                    model,
                    sampled,
                    result["log"],
                    result["diagnostic"],
                    result["visual"],
                )
        prefix = "ACTIVE" if bool(ENABLE) else "BYPASS"
        values = tuple(inputs.get(f"{name} {prefix}") for name in ("MODEL", "SAMPLED", "LOG"))
        if any(value is None for value in values):
            raise ValueError(f"CMK Sampler Bypass Gate is missing the {prefix} result")
        diagnostic = inputs.get("DIAGNOSTIC ACTIVE") if bool(ENABLE) else ExecutionBlocker(None)
        if bool(ENABLE) and diagnostic is None:
            raise ValueError("CMK Sampler Bypass Gate is missing the active diagnostic")
        visual = inputs.get(f"VISUAL {prefix}")
        if bool(ENABLE):
            _publish_instantid(
                state_key,
                values[0],
                values[1],
                values[2],
                diagnostic,
                visual,
            )
        return (*values, diagnostic, visual)


class CMKProcessEnableFlag:
    """Expose a Boolean module switch already stored in PROCESS."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "PROCESS": ("CMK_PROCESS_SDXL",),
                "flag": ("STRING", {"default": "instantid_enabled"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("ENABLE",)
    FUNCTION = "read"
    CATEGORY = "CMK/Internal/Gates"
    DEV_ONLY = True

    @staticmethod
    def read(PROCESS, flag="instantid_enabled"):
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Process Enable Flag requires a PROCESS dictionary")
        return (bool(PROCESS.get(str(flag), False)),)


class CMKZImageProcessEnableFlag(CMKProcessEnableFlag):
    """Expose a Boolean module switch stored in a Z-Image PROCESS pipe."""

    CATEGORY = "CMK/Developer/Legacy Gates"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "PROCESS": ("CMK_PROCESS_Z_IMAGE",),
                "flag": ("STRING", {"default": "controlnet_enabled"}),
            },
        }


_ZIT_FINAL_PHASE = "z_image_turbo.final"
_ZIT_FINAL_RESULT_SCHEMA = "cmk_z_image_turbo_final_result_v1"


def _zit_final_state_key(prompt, unique_id):
    fingerprint, _ = build_node_fingerprint(
        prompt,
        unique_id,
        ("CMKFamilyBranchGateZImage",),
        _ZIT_FINAL_RESULT_SCHEMA,
        include_node_identity=True,
    )
    if not fingerprint:
        return None
    try:
        return make_key(
            node_id=unique_id,
            model_family="z_image_turbo",
            phase=_ZIT_FINAL_PHASE,
            fingerprint=fingerprint,
            schema_version=_ZIT_FINAL_RESULT_SCHEMA,
        )
    except PromptStateUnavailableError:
        return None


def _released_zit_descriptor(model, diagnostic):
    metadata = diagnostic.get("metadata", {}) if isinstance(diagnostic, dict) else {}
    release_status = str(metadata.get("zit_model_status", "")).strip()
    if release_status.lower() not in _GENERATION_RESOURCE_RELEASE_STATUSES:
        return None
    model_identity = str(model.get("diffusion_model_name", "")).strip()
    vae_identity = str(model.get("vae_name", "")).strip()
    if not model_identity or not vae_identity:
        return None
    return make_released_generation_model_descriptor(
        "z_image_turbo", model_identity, vae_identity, release_status
    )


def _zit_final_hit(state_key):
    if state_key is None:
        return None
    result = get_result(state_key)
    if result is MISSING:
        return None
    resources = get_resource(state_key)
    if resources is not MISSING:
        return resources["model"], result
    descriptor = _released_generation_model_descriptor(
        result.get("released_model_descriptor")
    )
    if descriptor is None:
        return None
    return descriptor, result


def _publish_zit_final(state_key, model, image, log, visual, diagnostic):
    if state_key is None or any(
        value is None for value in (model, image, log, visual, diagnostic)
    ):
        return
    descriptor = _released_zit_descriptor(model, diagnostic)
    result = {
        "image": image,
        "log": log,
        "visual": visual,
        "diagnostic": diagnostic,
        "released_model_descriptor": descriptor,
    }
    try:
        publish_result(state_key, result)
    except (IncompleteResultError, UnsafeSnapshotTypeError) as exc:
        print(f"[CMK ZIT Result-State] NOT PUBLISHED: {exc}", flush=True)
        return
    if descriptor is None:
        store_resource(state_key, {"model": model})


class _CMKFamilyBranchGate:
    """Keep an inactive global subgraph from evaluating its implementation.

    ComfyUI expands a global subgraph while resolving one of its outputs.  The
    PROCESS input is therefore used as the cheap family signal, while every
    computed result stays lazy until that signal proves the branch is active.
    """

    PROCESS_TYPE = "CMK_PROCESS_SDXL"
    FAMILY = "sdxl"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS": (cls.PROCESS_TYPE, {"lazy": True}),
                "IMAGE": ("IMAGE", {"lazy": True}),
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
                "VISUAL": ("CMK_VISUAL_PIPE", {"lazy": True}),
                "diagnostic": ("CMK_DIAGNOSTIC", {"lazy": True}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "IMAGE",
        "CMK_LOG_PIPE",
        "CMK_VISUAL_PIPE",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("MODEL", "IMAGE", "LOG", "VISUAL", "diagnostic")
    FUNCTION = "gate"
    CATEGORY = "CMK/Internal/Gates"
    DEV_ONLY = True

    @cmk_timed_call("LAZY FAMILY BRANCH GATE")
    def check_lazy_status(self, PROCESS=None, prompt=None, unique_id=None, **inputs):
        if PROCESS is None:
            return ["PROCESS"]
        if isinstance(PROCESS, dict) and not PROCESS.get("family_active", True):
            return []
        # Resolve converging global-subgraph outputs one at a time.  Asking
        # ComfyUI for several lazy outputs of the same boundary in one pass can
        # leave the outer output node blocked after the inner task completes.
        for name in ("MODEL", "IMAGE", "LOG"):
            if inputs.get(name) is None:
                return [name]
        for name in ("VISUAL", "diagnostic"):
            pending = _pending_connected_input(name, prompt, unique_id, inputs)
            if pending:
                return pending
        return []

    def gate(self, PROCESS=None, **inputs):
        if PROCESS is None or (
            isinstance(PROCESS, dict) and not PROCESS.get("family_active", True)
        ):
            blocked = ExecutionBlocker(None)
            return (blocked, blocked, blocked, blocked, blocked)
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Family Branch Gate requires a CMK process pipe")
        actual = str(PROCESS.get("model_family", self.FAMILY)).strip().lower()
        if actual != self.FAMILY:
            raise ValueError("CMK Family Branch Gate received the wrong model family")
        missing = [
            name for name in ("MODEL", "IMAGE", "LOG")
            if inputs.get(name) is None
        ]
        if missing:
            raise ValueError(
                "CMK Family Branch Gate is missing " + ", ".join(missing)
            )
        return (
            inputs["MODEL"],
            inputs["IMAGE"],
            inputs["LOG"],
            inputs.get("VISUAL"),
            inputs.get("diagnostic"),
        )


_REFINER_RESULT_PHASE = "refiner.final"
_REFINER_RESULT_SCHEMA = "cmk_sdxl_refiner_result_v1"


def _refiner_state_key(prompt, unique_id):
    fingerprint, _ = build_node_fingerprint(
        prompt,
        unique_id,
        ("CMKFamilyBranchGateSDXL",),
        _REFINER_RESULT_SCHEMA,
        include_node_identity=True,
    )
    if not fingerprint:
        return None
    try:
        return make_key(
            node_id=unique_id,
            model_family="sdxl",
            phase=_REFINER_RESULT_PHASE,
            fingerprint=fingerprint,
            schema_version=_REFINER_RESULT_SCHEMA,
        )
    except PromptStateUnavailableError:
        return None


def _refiner_hit(state_key):
    if state_key is None:
        return None
    result = get_result(state_key)
    if (
        result is MISSING
        or result.get("schema_version") != _REFINER_RESULT_SCHEMA
    ):
        return None
    resources = get_resource(state_key)
    if resources is MISSING:
        return None
    return resources["model"], result


def _publish_refiner(state_key, model, image, log, visual, diagnostic):
    if state_key is None or any(
        value is None for value in (model, image, log, visual, diagnostic)
    ):
        return
    result = {
        "schema_version": _REFINER_RESULT_SCHEMA,
        "image": image,
        "log": log,
        "visual": visual,
        "diagnostic": diagnostic,
    }
    try:
        store_resource(state_key, {"model": model})
        publish_result(state_key, result)
    except (IncompleteResultError, UnsafeSnapshotTypeError) as exc:
        print(f"[CMK Refiner Result-State] NOT PUBLISHED: {exc}", flush=True)


class CMKFamilyBranchGateSDXL(_CMKFamilyBranchGate):
    @cmk_timed_call("LAZY REFINER FINAL GATE")
    def check_lazy_status(self, PROCESS=None, prompt=None, unique_id=None, **inputs):
        if PROCESS is None:
            return ["PROCESS"]
        if isinstance(PROCESS, dict) and not PROCESS.get("family_active", True):
            return []
        state_key = _refiner_state_key(prompt, unique_id)
        if _refiner_hit(state_key) is not None:
            return []
        return super().check_lazy_status(
            PROCESS=PROCESS,
            prompt=prompt,
            unique_id=unique_id,
            **inputs,
        )

    def gate(self, PROCESS=None, prompt=None, unique_id=None, **inputs):
        if PROCESS is None or (
            isinstance(PROCESS, dict) and not PROCESS.get("family_active", True)
        ):
            return super().gate(PROCESS=PROCESS, **inputs)
        if not isinstance(PROCESS, dict):
            return super().gate(PROCESS=PROCESS, **inputs)
        actual = str(PROCESS.get("model_family", self.FAMILY)).strip().lower()
        if actual != self.FAMILY:
            return super().gate(PROCESS=PROCESS, **inputs)
        state_key = _refiner_state_key(prompt, unique_id)
        hit = _refiner_hit(state_key)
        if hit is not None:
            model, result = hit
            return (
                model,
                result["image"],
                result["log"],
                result["visual"],
                result["diagnostic"],
            )
        values = super().gate(PROCESS=PROCESS, **inputs)
        _publish_refiner(
            state_key,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
        )
        return values


class CMKFamilyBranchGateZImage(_CMKFamilyBranchGate):
    PROCESS_TYPE = "CMK_PROCESS_Z_IMAGE"
    FAMILY = "z_image_turbo"

    @cmk_timed_call("LAZY ZIT FINAL GATE")
    def check_lazy_status(self, PROCESS=None, prompt=None, unique_id=None, **inputs):
        if PROCESS is None:
            return ["PROCESS"]
        if isinstance(PROCESS, dict) and not PROCESS.get("family_active", True):
            return []
        state_key = _zit_final_state_key(prompt, unique_id)
        if _zit_final_hit(state_key) is not None:
            return []
        return super().check_lazy_status(
            PROCESS=PROCESS,
            prompt=prompt,
            unique_id=unique_id,
            **inputs,
        )

    def gate(self, PROCESS=None, prompt=None, unique_id=None, **inputs):
        if PROCESS is None or (
            isinstance(PROCESS, dict) and not PROCESS.get("family_active", True)
        ):
            return super().gate(PROCESS=PROCESS, **inputs)
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Family Branch Gate requires a CMK process pipe")
        actual = str(PROCESS.get("model_family", self.FAMILY)).strip().lower()
        if actual != self.FAMILY:
            raise ValueError("CMK Family Branch Gate received the wrong model family")
        state_key = _zit_final_state_key(prompt, unique_id)
        hit = _zit_final_hit(state_key)
        if hit is not None:
            model, result = hit
            return (
                model,
                result["image"],
                result["log"],
                result["visual"],
                result["diagnostic"],
            )
        values = super().gate(PROCESS=PROCESS, **inputs)
        _publish_zit_final(
            state_key,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
        )
        return values


_FIRST_PASS_STAGE_KEY = "sdxl.first_pass"
_FIRST_PASS_RESULT_SCHEMA = "cmk_sdxl_first_pass_result_v1"
_FIRST_PASS_RESOURCE_FIELDS = {
    "model",
    "model_base",
    "model_identity",
    "model_patched",
    "clip",
    "clip_base",
    "clip_patched",
    "vae",
    "control_net",
    "conditioning_pos",
    "conditioning_neg",
    "conditioning_identity_pos",
    "conditioning_identity_neg",
    "lora_stack",
}


def clear_sdxl_sampled_boundary_cache():
    try:
        return release_resources(model_family="sdxl", phase="first_pass")
    except PromptStateUnavailableError:
        return 0


def _sdxl_first_pass_state_key(cache_key, unique_id):
    if not cache_key or unique_id is None:
        return None
    try:
        return make_key(
            node_id=unique_id,
            model_family="sdxl",
            phase="first_pass",
            fingerprint=cache_key,
            schema_version=_FIRST_PASS_RESULT_SCHEMA,
        )
    except PromptStateUnavailableError:
        return None


def _partition_sdxl_sampled(sampled):
    result_fields = {
        name: value
        for name, value in sampled.items()
        if name not in _FIRST_PASS_RESOURCE_FIELDS
    }
    resource_fields = {
        name: value
        for name, value in sampled.items()
        if name in _FIRST_PASS_RESOURCE_FIELDS
    }
    return result_fields, resource_fields


def _released_sdxl_descriptor(model, sampled):
    release_status = str(sampled.get("generation_model_status", "")).strip()
    if release_status.lower() not in _GENERATION_RESOURCE_RELEASE_STATUSES:
        return None
    model_identity = str(
        model.get("ckpt_name")
        or sampled.get("checkpoint_name")
        or ""
    ).strip()
    vae_identity = str(
        model.get("vae_selection_name")
        or model.get("vae_name")
        or sampled.get("vae_name")
        or ""
    ).strip()
    if not model_identity or not vae_identity:
        return None
    return make_released_generation_model_descriptor(
        "sdxl", model_identity, vae_identity, release_status
    )


def _sdxl_first_pass_hit(state_key):
    if state_key is None:
        return None
    result = get_result(state_key)
    if result is MISSING:
        return None
    resources = get_resource(state_key)
    if resources is not MISSING:
        sampled = dict(result["sampled"])
        sampled.update(resources["sampled"])
        return resources["model"], sampled, result
    descriptor = result.get("released_model_descriptor")
    if descriptor is None:
        return None
    return descriptor, dict(result["sampled"]), result


def _publish_sdxl_first_pass(state_key, model, sampled, log, image, visual, diagnostic):
    if state_key is None:
        return
    result_sampled, resource_sampled = _partition_sdxl_sampled(sampled)
    result = {
        "sampled": result_sampled,
        "log": log,
        "image": image,
        "visual": visual,
        "diagnostic": diagnostic,
        "released_model_descriptor": _released_sdxl_descriptor(model, sampled),
    }
    resources = {"model": model, "sampled": resource_sampled}
    try:
        store_resource(state_key, resources)
        publish_result(state_key, result)
    except (IncompleteResultError, UnsafeSnapshotTypeError) as exc:
        print(f"[CMK SDXL Result-State] NOT PUBLISHED: {exc}", flush=True)


def _materialize_first_pass_image(sampled):
    if not isinstance(sampled, dict):
        raise TypeError("CMK SDXL Sampler Boundary requires a CMK sampled pipe")
    existing = sampled.get("image_1st_pass")
    if existing is not None:
        return sampled, existing
    # During a split InstantID run, latent_1st_pass is deliberately still
    # noisy because module 15 must continue sampling it.  Module 10 already
    # owns the clean x0 estimate used for its live preview; decode that estimate
    # for the stable visual artifact without changing the functional hand-off.
    latent = sampled.get("instantid_keypoints_latent")
    if latent is None:
        latent = sampled.get("latent_1st_pass", sampled.get("latent_image"))
    vae = sampled.get("vae")
    if latent is None or vae is None:
        raise ValueError("CMK SDXL Sampler Boundary requires latent_1st_pass and vae")
    try:
        from nodes import VAEDecode
    except Exception as exc:
        raise RuntimeError(f"CMK SDXL Sampler Boundary: VAE Decode unavailable: {exc}") from exc
    decoded = VAEDecode().decode(vae, latent)
    image = decoded[0] if isinstance(decoded, (tuple, list)) else decoded
    result = dict(sampled)
    result["image_1st_pass"] = image
    return result, image


def _sampled_boundary_key(prompt, unique_id, *, include_content_guard=False):
    if not isinstance(prompt, dict):
        return None
    from .cmk_refiner_boundary_cache import (
        _canonical_node,
        _is_link,
        _resolve_current_node,
    )
    import hashlib
    import json

    _, current = _resolve_current_node(prompt, unique_id)
    if not isinstance(current, dict):
        return None
    sampled_link = (current.get("inputs", {}) or {}).get("SAMPLED")
    if not _is_link(prompt, sampled_link):
        return None
    payload = {
        "schema": "cmk_sdxl_sampled_boundary_v1",
        "sampled": _canonical_node(prompt, sampled_link[0], {}, set()),
        "output": int(sampled_link[1]),
    }
    if include_content_guard:
        payload["content_guard"] = content_guard_prompt_token(prompt)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class CMKFamilyBranchGateSDXLSampled(_CMKFamilyBranchGate):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS": (cls.PROCESS_TYPE, {"lazy": True}),
                "SAMPLED": ("CMK_SAMPLED_PIPE", {"lazy": True}),
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
                "VISUAL": ("CMK_VISUAL_PIPE", {"lazy": True}),
                "diagnostic": ("CMK_DIAGNOSTIC", {"lazy": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "CMK_SAMPLED_PIPE",
        "CMK_LOG_PIPE",
        "CMK_PROCESS_SDXL",
        "IMAGE",
        "CMK_VISUAL_PIPE",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("MODEL", "SAMPLED", "LOG", "PROCESS", "IMAGE", "VISUAL", "diagnostic")

    @cmk_timed_call("LAZY SDXL SAMPLED GATE")
    def check_lazy_status(self, PROCESS=None, prompt=None, unique_id=None, **inputs):
        context = _detailer_diag_context(unique_id)
        print(f"[CMK DIAG DETAILER] SDXL_GATE LAZY_ENTER {context}")
        if PROCESS is None:
            print(f"[CMK DIAG DETAILER] SDXL_GATE REQUEST input=PROCESS {context}")
            return ["PROCESS"]
        if isinstance(PROCESS, dict) and not PROCESS.get("family_active", True):
            print(f"[CMK DIAG DETAILER] SDXL_GATE INACTIVE no_request {context}")
            return []
        cache_key = _sampled_boundary_key(prompt, unique_id)
        state_key = _sdxl_first_pass_state_key(cache_key, unique_id)
        state = _detailer_diag_sdxl_state(state_key)
        hit = _sdxl_first_pass_hit(state_key) is not None
        print(
            "[CMK DIAG DETAILER] SDXL_GATE STATE "
            f"result_state={'YES' if state['result'] else 'NO'} "
            f"resource_state={'YES' if state['resource'] else 'NO'} "
            f"released_model_descriptor={'YES' if state['descriptor'] else 'NO'} "
            f"result_hit={'YES' if hit else 'NO'} reason={state['reason']} {context}"
        )
        if hit:
            print(f"[CMK DIAG DETAILER] SDXL_GATE HIT no_request {context}")
            return []
        # Resolve converging branches one at a time. Requesting MODEL and
        # SAMPLED together lets ComfyUI open the shared sampler preparation
        # through two lazy paths after a process restart, executing module 10
        # twice. MODEL is cheap; SAMPLED is the expensive authoritative stage.
        if inputs.get("MODEL") is None:
            print(f"[CMK DIAG DETAILER] SDXL_GATE REQUEST input=MODEL {context}")
            return ["MODEL"]
        if inputs.get("SAMPLED") is None:
            print(f"[CMK DIAG DETAILER] SDXL_GATE REQUEST input=SAMPLED {context}")
            return ["SAMPLED"]
        if inputs.get("LOG") is None:
            print(f"[CMK DIAG DETAILER] SDXL_GATE REQUEST input=LOG {context}")
            return ["LOG"]
        for name in ("VISUAL", "diagnostic"):
            pending = _pending_connected_input(name, prompt, unique_id, inputs)
            if pending:
                print(f"[CMK DIAG DETAILER] SDXL_GATE REQUEST input={pending[0]} {context}")
                return pending
        print(f"[CMK DIAG DETAILER] SDXL_GATE READY no_request {context}")
        return []

    def gate(self, PROCESS=None, prompt=None, unique_id=None, **inputs):
        context = _detailer_diag_context(unique_id)
        print(f"[CMK DIAG DETAILER] SDXL_GATE EXECUTE {context}")
        if PROCESS is None or (
            isinstance(PROCESS, dict) and not PROCESS.get("family_active", True)
        ):
            blocked = ExecutionBlocker(None)
            return (blocked, blocked, blocked, blocked, blocked, blocked, blocked)
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK SDXL Sampler Branch Gate requires a CMK process pipe")
        actual = str(PROCESS.get("model_family", "sdxl")).strip().lower()
        if actual != "sdxl":
            raise ValueError("CMK SDXL Sampler Branch Gate received the wrong model family")
        cache_key = _sampled_boundary_key(prompt, unique_id)
        state_key = _sdxl_first_pass_state_key(cache_key, unique_id)
        result_process = (
            stamp_artifact(PROCESS, _FIRST_PASS_STAGE_KEY, cache_key)
            if cache_key
            else PROCESS
        )
        if bool(PROCESS.get("hybrid_mode", False)):
            # Module 10 must run for the hybrid pre-pass, but its public
            # PROCESS becomes inactive afterwards so Refiner 20 and the final
            # family merge cannot mistake the pre-pass for the final result.
            result_process = dict(result_process)
            result_process["family_active"] = False
            result_process["hybrid_prepass_complete"] = True
        hit = _sdxl_first_pass_hit(state_key)
        if hit is not None:
            print(f"[CMK DIAG DETAILER] SDXL_GATE EXECUTE_HIT {context}")
            model, sampled, result = hit
            return (
                model,
                sampled,
                result["log"],
                result_process,
                result["image"],
                result["visual"],
                result["diagnostic"],
            )
        missing = [
            name for name in ("MODEL", "SAMPLED", "LOG")
            if inputs.get(name) is None
        ]
        if missing:
            print(
                "[CMK DIAG DETAILER] SDXL_GATE EXECUTE_MISS "
                f"missing={','.join(missing)} {context}"
            )
            raise ValueError(
                "CMK SDXL Sampler Branch Gate is missing " + ", ".join(missing)
            )
        sampled, image = _materialize_first_pass_image(inputs["SAMPLED"])
        print(f"[CMK DIAG DETAILER] SDXL_GATE EXECUTE_MISS publish=YES {context}")
        _publish_sdxl_first_pass(
            state_key,
            inputs["MODEL"],
            sampled,
            inputs["LOG"],
            image,
            inputs.get("VISUAL"),
            inputs.get("diagnostic"),
        )
        return (
            inputs["MODEL"],
            sampled,
            inputs["LOG"],
            result_process,
            image,
            inputs.get("VISUAL"),
            inputs.get("diagnostic"),
        )


class CMKSDXLResultBridgePipe:
    """End the SDXL-only section and expose the legacy neutral result context.

    This bridge is intentionally internal to the family-neutral finish module.
    Its input preserves editor-level family safety; its output is consumed only
    by model-independent result handling such as Save.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"PROCESS": ("CMK_PROCESS_SDXL",)}}

    RETURN_TYPES = ("CMK_PIPE",)
    RETURN_NAMES = ("PROCESS",)
    FUNCTION = "bridge"
    CATEGORY = "CMK/Developer/Legacy Bridge"
    DEV_ONLY = True

    @staticmethod
    def bridge(PROCESS):
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK SDXL Result Bridge: PROCESS must be a CMK process pipe")
        family = str(PROCESS.get("model_family", "sdxl")).strip().lower()
        if family != "sdxl":
            raise ValueError("CMK SDXL Result Bridge accepts only PROCESS SDXL")
        result = dict(PROCESS)
        result["result_contract"] = "family_neutral"
        result["source_model_family"] = "sdxl"
        return (result,)


def _release_completed_generation_resources(family, model, process):
    """Release generation resources without changing the completed result."""
    from .loaders.checkpoint_vae_loader import unload_model_pipe

    if not isinstance(model, dict) or not isinstance(process, dict):
        raise TypeError("CMK Generation Completion requires valid MODEL and PROCESS pipes")
    actual = str(process.get("model_family", family)).strip().lower()
    if actual != family:
        raise ValueError("CMK Generation Completion received a PROCESS from the wrong family")

    released_descriptor = _released_generation_model_descriptor(model)
    if released_descriptor is not None and released_descriptor["model_family"] != family:
        raise ValueError("CMK Generation Completion received a MODEL descriptor from the wrong family")
    unload_requested = bool(process.get("unload_models_after_use", True))
    if released_descriptor is not None:
        generation_unload = (
            "ALREADY RELEASED ("
            + released_descriptor["resource_release_status"]
            + ")"
        )
    else:
        generation_unload = (
            unload_model_pipe(model) if unload_requested else "KEPT LOADED"
        )
    cleared_generation_entries = (
        clear_sdxl_sampled_boundary_cache() if unload_requested else 0
    )
    if unload_requested:
        try:
            from .instantid.cmk_instantid_sampler import clear_instantid_boundary_cache

            cleared_generation_entries += clear_instantid_boundary_cache()
        except Exception:
            pass
    return {
        "generation_unload": generation_unload,
        "unload_requested": unload_requested,
        "cleared_generation_entries": int(cleared_generation_entries),
    }, released_descriptor


class _CMKPostProcessBoundaryBase:
    """Shared generation-to-postprocess transition contract."""

    @staticmethod
    def _model_widgets():
        from .loaders.checkpoint_vae_loader import (
            CMKCheckpointVAELoaderPipe,
            POSTPROCESS_MODEL_NONE,
        )

        required = CMKCheckpointVAELoaderPipe.INPUT_TYPES()["required"]
        checkpoint_widget = required["ckpt_name"]
        checkpoint_choices = list(checkpoint_widget[0])
        checkpoint_choices = [
            choice for choice in checkpoint_choices
            if str(choice).strip().lower() != "none"
        ]
        checkpoint_widget = (
            [*checkpoint_choices, POSTPROCESS_MODEL_NONE],
            *checkpoint_widget[1:],
        )

        def advanced(widget):
            options = dict(widget[1]) if len(widget) > 1 else {}
            options["advanced"] = True
            return (widget[0], options)

        return {
            "postprocess_checkpoint": advanced(checkpoint_widget),
            "postprocess_vae": advanced(required["vae_name"]),
            "use_checkpoint_vae": advanced(required["checkpoint_vae"]),
        }

    @staticmethod
    def _finish(
        family,
        model,
        process,
        image,
        log,
        visual,
        checkpoint,
        vae,
        checkpoint_vae,
        completion_metadata=None,
    ):
        from .cmk_log_pipe import cmk_add_block
        from .loaders.checkpoint_vae_loader import (
            is_postprocess_model_disabled,
            make_postprocess_model_spec,
        )

        if not isinstance(model, dict) or not isinstance(process, dict) or not isinstance(log, dict):
            raise TypeError("CMK PostProcess Boundary requires valid MODEL, PROCESS and LOG pipes")
        if image is None:
            raise ValueError("CMK PostProcess Boundary requires the completed generation IMAGE")
        release_metadata, released_descriptor = _release_completed_generation_resources(
            family, model, process
        )
        generation_family = "hybrid" if bool(process.get("hybrid_mode", False)) else family
        generation_unload = release_metadata["generation_unload"]
        unload_requested = release_metadata["unload_requested"]
        cleared_generation_entries = release_metadata["cleared_generation_entries"]
        if completion_metadata is not None:
            if not isinstance(completion_metadata, dict):
                raise TypeError("CMK PostProcess Boundary completion metadata must be a dictionary")
            completion_metadata.update(release_metadata)
        model_spec = make_postprocess_model_spec(checkpoint, vae, checkpoint_vae)
        postprocess_disabled = is_postprocess_model_disabled(model_spec)
        try:
            prompt_id = active_prompt_id()
        except Exception:
            prompt_id = None
        print(
            "[CMK DIAG DETAILER] POSTPROCESS_SETUP FINISH "
            f"prompt_id={prompt_id if prompt_id is not None else 'UNAVAILABLE'} "
            f"family={family} "
            f"input_model={'RELEASED_DESCRIPTOR' if released_descriptor is not None else 'MATERIALIZED_PIPE'} "
            f"generation_unload={generation_unload} "
            f"cleared_generation_entries={cleared_generation_entries} "
            "output_model=NEW_POSTPROCESS_SPEC"
        )
        checkpoint_label = "NONE" if postprocess_disabled else str(checkpoint)
        vae_label = (
            "NOT USED"
            if postprocess_disabled
            else ("checkpoint" if bool(checkpoint_vae) else str(vae))
        )
        result = dict(process)
        result.update({
            "result_contract": "family_neutral",
            "source_model_family": family,
            "generation_family": generation_family,
            "postprocess_model_family": "sdxl",
            "postprocess_checkpoint": checkpoint_label,
            "postprocess_vae": vae_label,
            "postprocess_model_disabled": postprocess_disabled,
        })
        model_spec["source_model_family"] = family
        model_spec["generation_family"] = generation_family
        log = cmk_add_block(
            log,
            "PostProcess Boundary",
            55,
            [
                f"GENERATION FAMILY : {generation_family.upper()}",
                f"GENERATION MODEL  : {generation_unload}",
                f"GENERATION CACHE  : CLEARED ({cleared_generation_entries})" if unload_requested else "GENERATION CACHE  : KEPT",
                f"POSTPROCESS MODEL : {checkpoint_label}",
                f"POSTPROCESS VAE   : {vae_label}",
                (
                    "MODEL LOAD         : DISABLED"
                    if postprocess_disabled
                    else "MODEL LOAD         : DEFERRED UNTIL REQUIRED"
                ),
            ],
            True,
        )
        if visual is None:
            try:
                from .cmk_visual import empty_visual
            except ImportError:
                from pipe.cmk_visual import empty_visual
            visual = empty_visual()
        return model_spec, result, image, log, visual

    def _restore_completion(
        self,
        preflight,
        postprocess_checkpoint,
        postprocess_vae,
        use_checkpoint_vae,
    ):
        contract = preflight["contract"]
        cached = preflight["cached"]
        print(
            f"[CMK Generation Completion / {contract.log_label}] RESTORE "
            f"{preflight['key'][:12]} ({preflight['detail']})"
        )
        return self._finish(
            contract.runtime_family,
            cached["model_descriptor"],
            cached["process"],
            cached["image"],
            cached["log"],
            cached["visual"],
            postprocess_checkpoint,
            postprocess_vae,
            use_checkpoint_vae,
        )

    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "CMK_RESULT_PROCESS",
        "IMAGE",
        "CMK_LOG_PIPE",
        "CMK_VISUAL_PIPE",
    )
    RETURN_NAMES = ("MODEL", "PROCESS P", "IMAGE", "LOG", "VISUAL")
    CATEGORY = "CMK/Flow/Finish"


class _CMKFamilyResultMergePipe(_CMKPostProcessBoundaryBase):
    """Lazily select SDXL/ZIT and establish the Combined PostProcess boundary."""

    _HYBRID_COMPLETION_MODE = "sdxl_zit_hybrid"
    _INSTANTID_HYBRID_COMPLETION_MODE = "instantid_hybrid"
    _INSTANTID_ZERO_PASS_COMPLETION_MODE = "instantid_zero_pass"
    _HYBRID_ROOTS = _ZIT_MERGE_ROOT_ROLES
    _INSTANTID_HYBRID_ROOTS = _SDXL_MERGE_ROOT_ROLES
    _COMPLETION_CONTRACTS = (
        _SDXL_ZIT_HYBRID_COMPLETION,
        _INSTANTID_HYBRID_COMPLETION,
        _INSTANTID_ZERO_PASS_COMPLETION,
    )
    _HYBRID_HANDOFF_INPUTS = (
        "SAMPLED SDXL",
        "LOG SDXL",
        "VISUAL SDXL",
    )
    _HYBRID_EXCLUDED_CLASSES = {
        "CMKInstantIDBoundary",
        "CMKSamplerBypassGate",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": cls._model_widgets(),
            "optional": {
                "MODEL ZIT": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS ZIT": (
                    "CMK_PROCESS_Z_IMAGE",
                    {"lazy": True, "label": "PROCESS G"},
                ),
                "IMAGE ZIT": ("IMAGE", {"lazy": True}),
                "LOG ZIT": ("CMK_LOG_PIPE", {"lazy": True}),
                "VISUAL ZIT": ("CMK_VISUAL_PIPE", {"lazy": True}),
                "MODEL SDXL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS SDXL": (
                    "CMK_PROCESS_SDXL",
                    {"lazy": True, "label": "PROCESS G"},
                ),
                "IMAGE SDXL": ("IMAGE", {"lazy": True}),
                "LOG SDXL": ("CMK_LOG_PIPE", {"lazy": True}),
                "VISUAL SDXL": ("CMK_VISUAL_PIPE", {"lazy": True}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    FUNCTION = "merge"

    @staticmethod
    def _prompt_node(prompt, node_id):
        return _completion_prompt_node(prompt, node_id)

    @classmethod
    def _prompt_link(cls, prompt, value):
        return _completion_prompt_link(prompt, value)

    @classmethod
    def _hybrid_graph(cls, prompt, unique_id):
        if not isinstance(prompt, dict):
            return None, "PROMPT is unavailable"
        current = cls._prompt_node(prompt, unique_id)
        if not isinstance(current, dict) or current.get("class_type") != cls.__name__:
            return None, "Family Result Merge is unavailable in the expanded prompt"

        current_inputs = current.get("inputs", {}) or {}
        roots, detail = _completion_mode_roots(
            prompt,
            current_inputs,
            _SDXL_ZIT_HYBRID_COMPLETION,
        )
        if roots is None:
            return None, detail

        pending = [link[0] for link in roots.values()]
        visited = set()
        gate_ids = []
        handoff_ids = []
        source_nodes = []
        found_finalizer = False
        found_sdxl_sampled_gate = False
        while pending:
            node_id = pending.pop()
            node_key = str(node_id)
            if node_key in visited:
                continue
            visited.add(node_key)
            node = cls._prompt_node(prompt, node_id)
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type", ""))
            if (
                class_type in cls._HYBRID_EXCLUDED_CLASSES
                or class_type.startswith("CMKInstantID")
            ):
                return None, f"{class_type} is outside the Hybrid completion contract"
            if class_type == "CMKFamilyBranchGateZImage":
                gate_ids.append(node_id)
            elif class_type == "CMKHybridZITInputPipe":
                handoff_ids.append(node_id)
                node_inputs = node.get("inputs", {}) or {}
                missing = [
                    name
                    for name in cls._HYBRID_HANDOFF_INPUTS
                    if not cls._prompt_link(prompt, node_inputs.get(name))
                ]
                if missing:
                    return None, (
                        "Hybrid handoff is incomplete: " + ", ".join(missing)
                    )
            elif class_type == "CMKZImageTurboFinalizePipe":
                found_finalizer = True
            elif class_type == "CMKFamilyBranchGateSDXLSampled":
                found_sdxl_sampled_gate = True
            elif class_type == "CMKPipeCreateImage":
                source_nodes.append(node)
            for value in (node.get("inputs", {}) or {}).values():
                if cls._prompt_link(prompt, value):
                    pending.append(value[0])

        if (
            len(gate_ids) != 1
            or len(handoff_ids) != 1
            or not found_finalizer
            or not found_sdxl_sampled_gate
            or len(source_nodes) != 1
        ):
            return None, "expanded graph does not prove the current Hybrid contract"
        source_inputs = source_nodes[0].get("inputs", {}) or {}
        if str(source_inputs.get("model_family", "")).strip().lower() != "hybrid":
            return None, "active source does not select SDXL/ZIT Hybrid"
        return {
            "roots": roots,
            "zit_gate_id": gate_ids[0],
            "handoff_id": handoff_ids[0],
        }, "expanded SDXL/ZIT Hybrid graph complete"

    @classmethod
    def _hybrid_completion_key(cls, prompt, unique_id):
        graph, detail = cls._hybrid_graph(prompt, unique_id)
        if graph is None:
            return None, detail
        return _completion_key_for_graph(prompt, graph, _SDXL_ZIT_HYBRID_COMPLETION)

    @classmethod
    def _instantid_graph(cls, prompt, unique_id, expected_mode):
        expected_mode = str(expected_mode or "").strip().upper()
        if expected_mode not in {"HYBRID", "ZERO-PASS"}:
            return None, "unsupported InstantID completion mode"
        mode_label = (
            "InstantID-Hybrid"
            if expected_mode == "HYBRID"
            else "InstantID Zero-Pass"
        )
        if not isinstance(prompt, dict):
            return None, "PROMPT is unavailable"
        current = cls._prompt_node(prompt, unique_id)
        if not isinstance(current, dict) or current.get("class_type") != cls.__name__:
            return None, "Family Result Merge is unavailable in the expanded prompt"

        current_inputs = current.get("inputs", {}) or {}
        contract = (
            _INSTANTID_HYBRID_COMPLETION
            if expected_mode == "HYBRID"
            else _INSTANTID_ZERO_PASS_COMPLETION
        )
        roots, detail = _completion_mode_roots(prompt, current_inputs, contract)
        if roots is None:
            return None, detail

        pending = [link[0] for link in roots.values()]
        visited = set()
        instantid_gate_ids = []
        instantid_boundary_ids = []
        instantid_sampler_ids = []
        refiner_gate_ids = []
        source_nodes = []
        found_sdxl_sampled_gate = False
        found_refiner = False
        while pending:
            node_id = pending.pop()
            node_key = str(node_id)
            if node_key in visited:
                continue
            visited.add(node_key)
            node = cls._prompt_node(prompt, node_id)
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type", ""))
            if class_type in {
                "CMKFamilyBranchGateZImage",
                "CMKHybridZITInputPipe",
                "CMKZImageTurboFinalizePipe",
            }:
                return None, f"{class_type} is outside the InstantID-Hybrid completion contract"
            if class_type == "CMKSamplerBypassGate":
                instantid_gate_ids.append(node_id)
            elif class_type == "CMKInstantIDBoundary":
                instantid_boundary_ids.append(node_id)
            elif class_type == "CMKInstantIDSamplerSDXLPipe":
                instantid_sampler_ids.append(node_id)
            elif class_type == "CMKInstantIDRegionalAdvancedSDXLPipe":
                if expected_mode != "HYBRID":
                    return None, (
                        "CMKInstantIDRegionalAdvancedSDXLPipe requires "
                        "InstantID-Hybrid preliminary-layout sampling"
                    )
                instantid_sampler_ids.append(node_id)
            elif class_type == "CMKFamilyBranchGateSDXL":
                refiner_gate_ids.append(node_id)
            elif class_type == "CMKFamilyBranchGateSDXLSampled":
                found_sdxl_sampled_gate = True
            elif class_type == "CMKRefinerPipe":
                found_refiner = True
            elif class_type == "CMKPipeCreateImage":
                source_nodes.append(node)
            for value in (node.get("inputs", {}) or {}).values():
                if cls._prompt_link(prompt, value):
                    pending.append(value[0])

        if (
            len(instantid_gate_ids) != 1
            or len(instantid_boundary_ids) != 1
            or len(instantid_sampler_ids) != 1
            or len(refiner_gate_ids) != 1
            or len(source_nodes) != 1
            or not found_sdxl_sampled_gate
            or not found_refiner
        ):
            return None, "expanded graph does not prove the current InstantID-Hybrid contract"

        source_inputs = source_nodes[0].get("inputs", {}) or {}
        source_family = str(source_inputs.get("model_family", "")).strip().lower()
        instantid_mode = str(source_inputs.get("instantid_mode", "")).strip().upper()
        if (
            source_family != "sdxl"
            or source_inputs.get("instantid_enabled") is not True
            or instantid_mode != expected_mode
        ):
            return None, "active source does not select SDXL " + mode_label

        return {
            "roots": roots,
            "instantid_gate_id": instantid_gate_ids[0],
            "refiner_gate_id": refiner_gate_ids[0],
        }, f"expanded SDXL {mode_label} graph complete"

    @classmethod
    def _instantid_hybrid_graph(cls, prompt, unique_id):
        return cls._instantid_graph(prompt, unique_id, "HYBRID")

    @classmethod
    def _instantid_zero_pass_graph(cls, prompt, unique_id):
        return cls._instantid_graph(prompt, unique_id, "ZERO-PASS")

    @classmethod
    def _instantid_hybrid_completion_key(cls, prompt, unique_id):
        graph, detail = cls._instantid_hybrid_graph(prompt, unique_id)
        if graph is None:
            return None, detail
        return _completion_key_for_graph(prompt, graph, _INSTANTID_HYBRID_COMPLETION)

    @classmethod
    def _instantid_zero_pass_completion_key(cls, prompt, unique_id):
        graph, detail = cls._instantid_zero_pass_graph(prompt, unique_id)
        if graph is None:
            return None, detail
        return _completion_key_for_graph(prompt, graph, _INSTANTID_ZERO_PASS_COMPLETION)

    @classmethod
    def _instantid_hybrid_finalization_evidence(cls, prompt, unique_id, process):
        graph, detail = cls._instantid_hybrid_graph(prompt, unique_id)
        if graph is None:
            return None, detail
        if not isinstance(process, dict):
            return None, "active PROCESS is unavailable"
        if (
            str(process.get("model_family", "sdxl")).strip().lower() != "sdxl"
            or not bool(process.get("instantid_enabled", False))
            or str(process.get("instantid_mode", "")).strip().upper() != "HYBRID"
            or bool(process.get("hybrid_mode", False))
        ):
            return None, "active PROCESS does not identify SDXL InstantID-Hybrid"

        instantid_key = _instantid_state_key(prompt, graph["instantid_gate_id"])
        if instantid_key is None:
            return None, "InstantID Final Gate result state is unavailable"
        try:
            instantid_result = get_result(instantid_key)
        except PromptStateUnavailableError:
            return None, "InstantID Final Gate result state is unavailable"
        if not isinstance(instantid_result, dict):
            return None, "InstantID Final Gate has no completed result"
        instantid_sampled = instantid_result.get("sampled")
        instantid_diagnostic = instantid_result.get("diagnostic")
        instantid_metadata = (
            instantid_diagnostic.get("metadata", {})
            if isinstance(instantid_diagnostic, dict)
            else {}
        )
        instantid_workflow = str(
            instantid_sampled.get("instantid_workflow", "")
            if isinstance(instantid_sampled, dict)
            else ""
        ).strip()
        instantid_complete = (
            instantid_result.get("schema_version") == _INSTANTID_RESULT_SCHEMA
            and isinstance(instantid_sampled, dict)
            and instantid_sampled.get("identity_applied") is True
            and instantid_sampled.get("instantid_zero_pass") is False
            and instantid_workflow.upper().startswith("HYBRID")
            and isinstance(instantid_result.get("log"), dict)
            and isinstance(instantid_result.get("visual"), dict)
            and isinstance(instantid_diagnostic, dict)
            and instantid_metadata.get("identity_applied") is True
            and str(instantid_metadata.get("workflow", "")).strip().upper().startswith("HYBRID")
        )
        if not instantid_complete:
            return None, "InstantID-Hybrid result or diagnostic is incomplete"

        refiner_key = _refiner_state_key(prompt, graph["refiner_gate_id"])
        if refiner_key is None:
            return None, "Refiner Final Gate result state is unavailable"
        try:
            refiner_result = get_result(refiner_key)
        except PromptStateUnavailableError:
            return None, "Refiner Final Gate result state is unavailable"
        refiner_complete = (
            isinstance(refiner_result, dict)
            and refiner_result.get("schema_version") == _REFINER_RESULT_SCHEMA
            and refiner_result.get("image") is not None
            and isinstance(refiner_result.get("log"), dict)
            and isinstance(refiner_result.get("visual"), dict)
            and isinstance(refiner_result.get("diagnostic"), dict)
        )
        if not refiner_complete:
            return None, "Refiner Final Gate has no completed result"

        return {
            "instantid_workflow": instantid_workflow,
            "instantid_result_schema": _INSTANTID_RESULT_SCHEMA,
            "refiner_result_schema": _REFINER_RESULT_SCHEMA,
            "refiner_complete": True,
        }, "InstantID-Hybrid and Refiner Final Gates complete"

    @classmethod
    def _instantid_hybrid_completion_hit(cls, cache_key):
        if not cache_key:
            return None
        result = lookup_completion(cache_key)
        if result is None:
            return None
        process = result.get("process")
        metadata = result.get("metadata")
        descriptor = result.get("model_descriptor")
        valid = (
            result.get("model_family") == "hybrid"
            and result.get("generation_mode") == cls._INSTANTID_HYBRID_COMPLETION_MODE
            and isinstance(process, dict)
            and str(process.get("model_family", "sdxl")).strip().lower() == "sdxl"
            and bool(process.get("instantid_enabled", False))
            and str(process.get("instantid_mode", "")).strip().upper() == "HYBRID"
            and not bool(process.get("hybrid_mode", False))
            and process.get("family_active", True) is not False
            and result.get("image") is not None
            and isinstance(result.get("log"), dict)
            and isinstance(result.get("visual"), dict)
            and isinstance(metadata, dict)
            and str(metadata.get("instantid_workflow", "")).strip().upper().startswith("HYBRID")
            and metadata.get("instantid_result_schema") == _INSTANTID_RESULT_SCHEMA
            and metadata.get("refiner_result_schema") == _REFINER_RESULT_SCHEMA
            and metadata.get("refiner_complete") is True
        )
        try:
            released = _released_generation_model_descriptor(descriptor)
            valid = (
                valid
                and released is not None
                and released["model_family"] == "sdxl"
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            invalidate_completion(cache_key)
            return None
        return result

    @classmethod
    def _instantid_zero_pass_finalization_evidence(cls, prompt, unique_id, process):
        graph, detail = cls._instantid_zero_pass_graph(prompt, unique_id)
        if graph is None:
            return None, detail
        if not isinstance(process, dict):
            return None, "active PROCESS is unavailable"
        if (
            str(process.get("model_family", "sdxl")).strip().lower() != "sdxl"
            or not bool(process.get("instantid_enabled", False))
            or str(process.get("instantid_mode", "")).strip().upper() != "ZERO-PASS"
            or bool(process.get("hybrid_mode", False))
        ):
            return None, "active PROCESS does not identify SDXL InstantID Zero-Pass"

        instantid_key = _instantid_state_key(prompt, graph["instantid_gate_id"])
        if instantid_key is None:
            return None, "InstantID Final Gate result state is unavailable"
        try:
            instantid_result = get_result(instantid_key)
        except PromptStateUnavailableError:
            return None, "InstantID Final Gate result state is unavailable"
        if not isinstance(instantid_result, dict):
            return None, "InstantID Final Gate has no completed result"
        instantid_sampled = instantid_result.get("sampled")
        instantid_diagnostic = instantid_result.get("diagnostic")
        instantid_metadata = (
            instantid_diagnostic.get("metadata", {})
            if isinstance(instantid_diagnostic, dict)
            else {}
        )
        instantid_workflow = str(
            instantid_sampled.get("instantid_workflow", "")
            if isinstance(instantid_sampled, dict)
            else ""
        ).strip()
        instantid_complete = (
            instantid_result.get("schema_version") == _INSTANTID_RESULT_SCHEMA
            and isinstance(instantid_sampled, dict)
            and instantid_sampled.get("identity_applied") is True
            and instantid_sampled.get("instantid_zero_pass") is True
            and instantid_workflow.upper().startswith("ZERO-PASS")
            and isinstance(instantid_result.get("log"), dict)
            and isinstance(instantid_result.get("visual"), dict)
            and isinstance(instantid_diagnostic, dict)
            and instantid_metadata.get("identity_applied") is True
            and str(instantid_metadata.get("workflow", "")).strip().upper().startswith("ZERO-PASS")
        )
        if not instantid_complete:
            return None, "InstantID Zero-Pass result or diagnostic is incomplete"

        refiner_key = _refiner_state_key(prompt, graph["refiner_gate_id"])
        if refiner_key is None:
            return None, "Refiner Final Gate result state is unavailable"
        try:
            refiner_result = get_result(refiner_key)
        except PromptStateUnavailableError:
            return None, "Refiner Final Gate result state is unavailable"
        refiner_complete = (
            isinstance(refiner_result, dict)
            and refiner_result.get("schema_version") == _REFINER_RESULT_SCHEMA
            and refiner_result.get("image") is not None
            and isinstance(refiner_result.get("log"), dict)
            and isinstance(refiner_result.get("visual"), dict)
            and isinstance(refiner_result.get("diagnostic"), dict)
        )
        if not refiner_complete:
            return None, "Refiner Final Gate has no completed result"

        return {
            "instantid_workflow": instantid_workflow,
            "instantid_result_schema": _INSTANTID_RESULT_SCHEMA,
            "refiner_result_schema": _REFINER_RESULT_SCHEMA,
            "refiner_complete": True,
        }, "InstantID Zero-Pass and Refiner Final Gates complete"

    @classmethod
    def _instantid_zero_pass_completion_hit(cls, cache_key):
        if not cache_key:
            return None
        result = lookup_completion(cache_key)
        if result is None:
            return None
        process = result.get("process")
        metadata = result.get("metadata")
        descriptor = result.get("model_descriptor")
        valid = (
            result.get("model_family") == "hybrid"
            and result.get("generation_mode") == cls._INSTANTID_ZERO_PASS_COMPLETION_MODE
            and isinstance(process, dict)
            and str(process.get("model_family", "sdxl")).strip().lower() == "sdxl"
            and bool(process.get("instantid_enabled", False))
            and str(process.get("instantid_mode", "")).strip().upper() == "ZERO-PASS"
            and not bool(process.get("hybrid_mode", False))
            and process.get("family_active", True) is not False
            and result.get("image") is not None
            and isinstance(result.get("log"), dict)
            and isinstance(result.get("visual"), dict)
            and isinstance(metadata, dict)
            and str(metadata.get("instantid_workflow", "")).strip().upper().startswith("ZERO-PASS")
            and metadata.get("instantid_result_schema") == _INSTANTID_RESULT_SCHEMA
            and metadata.get("refiner_result_schema") == _REFINER_RESULT_SCHEMA
            and metadata.get("refiner_complete") is True
        )
        try:
            released = _released_generation_model_descriptor(descriptor)
            valid = (
                valid
                and released is not None
                and released["model_family"] == "sdxl"
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            invalidate_completion(cache_key)
            return None
        return result

    @classmethod
    def _hybrid_finalization_evidence(cls, prompt, unique_id, process):
        graph, detail = cls._hybrid_graph(prompt, unique_id)
        if graph is None:
            return None, detail
        if not isinstance(process, dict) or not bool(process.get("hybrid_mode", False)):
            return None, "active PROCESS does not identify Hybrid mode"

        try:
            from .cmk_hybrid_bridge import _hybrid_handoff_state_key
        except ImportError:
            from pipe.cmk_hybrid_bridge import _hybrid_handoff_state_key

        handoff_key = _hybrid_handoff_state_key(prompt, graph["handoff_id"])
        if handoff_key is None:
            return None, "Hybrid handoff result state is unavailable"
        try:
            handoff = get_result(handoff_key)
        except PromptStateUnavailableError:
            return None, "Hybrid handoff result state is unavailable"
        if not isinstance(handoff, dict):
            return None, "Hybrid handoff has no completed result"
        handoff_metadata = handoff.get("metadata")
        handoff_complete = (
            handoff.get("image") is not None
            and isinstance(handoff.get("log"), dict)
            and isinstance(handoff.get("visual"), dict)
            and isinstance(handoff_metadata, dict)
            and handoff_metadata.get("handoff_percent")
            == int(process.get("hybrid_sdxl_handoff", 90))
            and float(handoff_metadata.get("zit_denoise", -1.0))
            == float(process.get("hybrid_zit_denoise", 0.20))
            and handoff_metadata.get("hybrid_inpaint_mode")
            is bool(process.get("hybrid_inpaint_mode", False))
        )
        if not handoff_complete:
            return None, "Hybrid handoff result or metadata is incomplete"

        zit_state_key = _zit_final_state_key(prompt, graph["zit_gate_id"])
        if zit_state_key is None:
            return None, "Hybrid ZIT Final Gate result state is unavailable"
        try:
            zit_result = get_result(zit_state_key)
        except PromptStateUnavailableError:
            return None, "Hybrid ZIT Final Gate result state is unavailable"
        if not isinstance(zit_result, dict):
            return None, "Hybrid ZIT Final Gate has no completed result"
        diagnostic = zit_result.get("diagnostic")
        diagnostic_metadata = (
            diagnostic.get("metadata", {}) if isinstance(diagnostic, dict) else {}
        )
        zit_descriptor = zit_result.get("released_model_descriptor")
        try:
            released = _released_generation_model_descriptor(zit_descriptor)
        except (TypeError, ValueError):
            released = None
        zit_complete = (
            zit_result.get("image") is not None
            and isinstance(zit_result.get("log"), dict)
            and isinstance(zit_result.get("visual"), dict)
            and isinstance(diagnostic, dict)
            and str(diagnostic_metadata.get("model_family", "")).strip().lower()
            == "z_image_turbo"
            and str(diagnostic_metadata.get("zit_model_status", "")).strip()
            == "UNLOADED"
            and released is not None
            and released["model_family"] == "z_image_turbo"
        )
        if not zit_complete:
            return None, "Hybrid ZIT finalization or global model release is incomplete"
        return {
            "handoff_percent": int(handoff_metadata["handoff_percent"]),
            "zit_denoise": float(handoff_metadata["zit_denoise"]),
            "hybrid_inpaint_mode": bool(handoff_metadata["hybrid_inpaint_mode"]),
            "zit_model_status": "UNLOADED",
            "model_descriptor": released,
        }, "Hybrid handoff, ZIT finalization and global model release complete"

    @classmethod
    def _hybrid_completion_hit(cls, cache_key):
        if not cache_key:
            return None
        result = lookup_completion(cache_key)
        if result is None:
            return None
        process = result.get("process")
        metadata = result.get("metadata")
        descriptor = result.get("model_descriptor")
        valid = (
            result.get("model_family") == "hybrid"
            and result.get("generation_mode") == cls._HYBRID_COMPLETION_MODE
            and isinstance(process, dict)
            and str(process.get("model_family", "")).strip().lower()
            == "z_image_turbo"
            and bool(process.get("hybrid_mode", False))
            and process.get("family_active", True) is not False
            and isinstance(result.get("log"), dict)
            and isinstance(result.get("visual"), dict)
            and isinstance(metadata, dict)
            and metadata.get("zit_model_status") == "UNLOADED"
            and isinstance(metadata.get("handoff_percent"), int)
            and isinstance(metadata.get("zit_denoise"), float)
            and isinstance(metadata.get("hybrid_inpaint_mode"), bool)
            and result.get("image") is not None
        )
        try:
            released = _released_generation_model_descriptor(descriptor)
            valid = (
                valid
                and released is not None
                and released["model_family"] == "z_image_turbo"
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            invalidate_completion(cache_key)
            return None
        return result

    @staticmethod
    def _pending_visual(suffix, inputs):
        name = f"VISUAL {suffix}"
        prompt = inputs.get("prompt") or {}
        entry = prompt.get(str(inputs.get("unique_id")), {})
        connected = name in (entry.get("inputs") or {}) or name in inputs
        return [name] if connected and inputs.get(name) is None else []

    @cmk_timed_call("LAZY FAMILY RESULT MERGE")
    def check_lazy_status(self, **inputs):
        preflight = _resolve_completion_preflight(
            type(self),
            self._COMPLETION_CONTRACTS,
            inputs.get("prompt"),
            inputs.get("unique_id"),
        )
        if preflight["cached"] is not None:
            contract = preflight["contract"]
            print(
                f"[CMK Generation Completion / {contract.log_label}] "
                "HIT before lazy inputs "
                f"{preflight['key'][:12]} ({preflight['detail']})"
            )
            return []
        contract = preflight["contract"]
        if contract is None:
            raise RuntimeError(
                "CMK PostProcess Setup cannot resolve the active generation mode: "
                + str(preflight["detail"])
            )
        suffix = "SDXL" if contract.runtime_family == "sdxl" else "ZIT"
        process_name = f"PROCESS {suffix}"
        process = inputs.get(process_name)
        if process is None:
            return [process_name]
        if not isinstance(process, dict) or process.get("family_active", True) is False:
            raise RuntimeError(
                f"CMK PostProcess Setup resolved {contract.name}, but {process_name} "
                "does not provide an active generation contract"
            )
        for name in ("MODEL", "IMAGE", "LOG"):
            input_name = f"{name} {suffix}"
            if inputs.get(input_name) is None:
                return [input_name]
        return self._pending_visual(suffix, inputs)

    def merge(self, postprocess_checkpoint, postprocess_vae, use_checkpoint_vae, **inputs):
        preflight = _resolve_completion_preflight(
            type(self),
            self._COMPLETION_CONTRACTS,
            inputs.get("prompt"),
            inputs.get("unique_id"),
        )
        if preflight["cached"] is not None:
            return self._restore_completion(
                preflight,
                postprocess_checkpoint,
                postprocess_vae,
                use_checkpoint_vae,
            )

        contract = preflight["contract"]
        if contract is None or not preflight["key"]:
            raise RuntimeError(
                "CMK PostProcess Setup cannot resolve a safe generation completion "
                "contract: " + str(preflight["detail"])
            )

        process_sdxl = inputs.get("PROCESS SDXL")
        process_z = inputs.get("PROCESS ZIT")
        active_sdxl = isinstance(process_sdxl, dict) and process_sdxl.get("family_active", True)
        active_z = isinstance(process_z, dict) and process_z.get("family_active", True)
        if active_sdxl and active_z:
            raise ValueError("CMK Active Family Result received two active PROCESS paths")
        if not active_sdxl and not active_z:
            raise ValueError("CMK Active Family Result received no active PROCESS path")
        family = "sdxl" if active_sdxl else "z_image_turbo"
        if family != contract.runtime_family:
            raise ValueError(
                f"CMK PostProcess Setup resolved {contract.name}, but the active "
                f"PROCESS family is {family}"
            )
        suffix = "SDXL" if family == "sdxl" else "ZIT"
        model = inputs.get(f"MODEL {suffix}")
        process = inputs.get(f"PROCESS {suffix}")
        image = inputs.get(f"IMAGE {suffix}")
        log = inputs.get(f"LOG {suffix}")
        missing = [
            name for name, value in (
                ("MODEL", model), ("PROCESS", process), ("IMAGE", image), ("LOG", log)
            ) if value is None
        ]
        if missing:
            raise ValueError(
                f"CMK Active Family Result: {suffix} is selected but "
                + ", ".join(missing)
                + " is not connected."
            )
        visual = inputs.get(f"VISUAL {suffix}")
        completion_metadata = {}
        result = self._finish(
            family, model, process, image, log, visual,
            postprocess_checkpoint, postprocess_vae, use_checkpoint_vae,
            completion_metadata,
        )

        evidence, evidence_detail = _completion_finalization_evidence(
            type(self),
            preflight,
            inputs.get("prompt"),
            inputs.get("unique_id"),
            process,
        )
        if evidence is None:
            print(
                f"[CMK Generation Completion / {contract.log_label}] NOT STORED: "
                + evidence_detail
            )
            return result
        generation_unload = str(completion_metadata.get("generation_unload", ""))
        if generation_unload != "UNLOADED" and not generation_unload.startswith(
            "ALREADY RELEASED ("
        ):
            print(
                f"[CMK Generation Completion / {contract.log_label}] NOT STORED: "
                "Boundary 30 resource release was not proven"
            )
            return result
        evidence = dict(evidence)
        descriptor = _completion_released_descriptor(
            contract,
            model,
            completion_metadata,
            evidence,
        )
        if descriptor is None:
            descriptor_label = (
                "released SDXL MODEL descriptor is unavailable"
                if contract.descriptor_family == "sdxl"
                else "released ZIT MODEL descriptor is unavailable"
            )
            print(
                f"[CMK Generation Completion / {contract.log_label}] NOT STORED: "
                + descriptor_label
            )
            return result
        evidence.pop("model_descriptor", None)
        completion_metadata.update(evidence)
        _store_completion_result(
            preflight,
            image=image,
            process=process,
            log=log,
            visual=result[4],
            metadata=completion_metadata,
            descriptor=descriptor,
        )
        print(
            f"[CMK Generation Completion / {contract.log_label}] STORED "
            f"{preflight['key'][:12]} ({preflight['detail']})"
        )
        return result


class _CMKSinglePostProcessBoundary(_CMKPostProcessBoundaryBase):
    PROCESS_TYPE = "CMK_PROCESS_SDXL"
    FAMILY = "sdxl"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": cls._model_widgets(),
            "optional": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS": (
                    cls.PROCESS_TYPE,
                    {"lazy": True, "label": "PROCESS G"},
                ),
                "IMAGE": ("IMAGE", {"lazy": True}),
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
                "VISUAL": ("CMK_VISUAL_PIPE", {"lazy": True}),
            },
        }

    FUNCTION = "boundary"

    def check_lazy_status(self, **inputs):
        context = _detailer_diag_context()
        print(
            "[CMK DIAG DETAILER] POSTPROCESS_SETUP LAZY_ENTER "
            f"class={type(self).__name__} {context}"
        )
        process = inputs.get("PROCESS")
        if process is None:
            print(f"[CMK DIAG DETAILER] POSTPROCESS_SETUP REQUEST input=PROCESS {context}")
            return ["PROCESS"]
        # An inactive family must stop at the boundary. In particular, do not
        # resolve MODEL merely to discover later that this branch is blocked.
        if isinstance(process, dict) and not process.get("family_active", True):
            print(f"[CMK DIAG DETAILER] POSTPROCESS_SETUP INACTIVE no_request {context}")
            return []
        for name in ("MODEL", "IMAGE", "LOG"):
            if inputs.get(name) is None:
                print(
                    "[CMK DIAG DETAILER] POSTPROCESS_SETUP REQUEST "
                    f"input={name} upstream_generation_request=YES {context}"
                )
                return [name]
        if "VISUAL" in inputs and inputs.get("VISUAL") is None:
            print(f"[CMK DIAG DETAILER] POSTPROCESS_SETUP REQUEST input=VISUAL {context}")
            return ["VISUAL"]
        print(f"[CMK DIAG DETAILER] POSTPROCESS_SETUP READY no_request {context}")
        return []

    def boundary(self, postprocess_checkpoint, postprocess_vae, use_checkpoint_vae, **inputs):
        context = _detailer_diag_context()
        print(
            "[CMK DIAG DETAILER] POSTPROCESS_SETUP EXECUTE "
            f"class={type(self).__name__} "
            f"input_model={'PRESENT' if inputs.get('MODEL') is not None else 'MISSING'} {context}"
        )
        process = inputs.get("PROCESS")
        if isinstance(process, dict) and not process.get("family_active", True):
            blocked = ExecutionBlocker(None)
            return blocked, blocked, blocked, blocked, blocked
        return self._finish(
            self.FAMILY,
            inputs.get("MODEL"), process, inputs.get("IMAGE"), inputs.get("LOG"),
            inputs.get("VISUAL"), postprocess_checkpoint, postprocess_vae,
            use_checkpoint_vae,
        )


class _CMKPostProcessBoundarySDXLPipe(_CMKSinglePostProcessBoundary):
    _COMPLETION_MODE = "sdxl"
    _COMPLETION_ROOTS = ("MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL")
    _REQUIRED_COMPLETION_ROOTS = ("MODEL", "PROCESS", "IMAGE", "LOG")
    _SDXL_EXCLUDED_UPSTREAM_CLASSES = {
        "CMKFamilyBranchGateZImage",
        "CMKHybridZITInputPipe",
        "CMKInstantIDBoundary",
        "CMKSamplerBypassGate",
        "CMKZImageTurboFinalizePipe",
    }
    _COMPLETION_CONTRACT = _SDXL_COMPLETION

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["hidden"] = {
            "prompt": "PROMPT",
            "unique_id": "UNIQUE_ID",
        }
        return result

    @classmethod
    def _sdxl_completion_graph(cls, prompt, unique_id):
        if not isinstance(prompt, dict):
            return None, "PROMPT is unavailable"
        current = prompt.get(unique_id)
        if not isinstance(current, dict):
            current = prompt.get(str(unique_id))
        if not isinstance(current, dict) or current.get("class_type") != cls.__name__:
            return None, "SDXL PostProcess Boundary is unavailable in the expanded prompt"
        current_inputs = current.get("inputs", {}) or {}
        roots, detail = _completion_mode_roots(
            prompt,
            current_inputs,
            _SDXL_COMPLETION,
        )
        if roots is None:
            return None, detail

        def process_source_nodes(process_link):
            pending_process = (
                [process_link[0]]
                if _completion_prompt_link(prompt, process_link)
                else []
            )
            visited_process = set()
            sources = {}
            while pending_process:
                process_node_id = pending_process.pop()
                process_node_key = str(process_node_id)
                if process_node_key in visited_process:
                    continue
                visited_process.add(process_node_key)
                process_node = _completion_prompt_node(prompt, process_node_id)
                if not isinstance(process_node, dict):
                    continue
                if str(process_node.get("class_type", "")) == "CMKPipeCreateImage":
                    sources[process_node_key] = process_node
                    continue
                for value in (process_node.get("inputs", {}) or {}).values():
                    if _completion_prompt_link(prompt, value):
                        pending_process.append(value[0])
            return sources

        pending = [link[0] for link in roots.values()]
        visited = set()
        source_nodes = {}
        off_gate_source_id = None
        while pending:
            node_id = pending.pop()
            node_key = str(node_id)
            if node_key in visited:
                continue
            visited.add(node_key)
            node = prompt.get(node_id)
            if not isinstance(node, dict):
                node = prompt.get(node_key)
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type", ""))
            if class_type == "CMKSamplerBypassGate":
                if off_gate_source_id is not None:
                    return None, "multiple InstantID bypass gates are ambiguous"
                node_inputs = node.get("inputs", {}) or {}
                enable_link = node_inputs.get("ENABLE")
                if not _completion_prompt_link(prompt, enable_link):
                    return None, "InstantID bypass gate ENABLE is unavailable"
                enable_node = _completion_prompt_node(prompt, enable_link[0])
                if str((enable_node or {}).get("class_type", "")) != "CMKProcessEnableFlag":
                    return None, (
                        "InstantID bypass gate ENABLE does not originate from "
                        "CMKProcessEnableFlag"
                    )
                enable_inputs = enable_node.get("inputs", {}) or {}
                if enable_inputs.get("flag") != "instantid_enabled":
                    return None, (
                        "InstantID bypass gate ENABLE does not read "
                        "instantid_enabled"
                    )
                enable_sources = process_source_nodes(enable_inputs.get("PROCESS"))
                if len(enable_sources) != 1:
                    return None, (
                        "InstantID bypass gate PROCESS does not identify exactly "
                        "one authoritative source"
                    )
                gate_source_id, gate_source = next(iter(enable_sources.items()))
                gate_source_inputs = gate_source.get("inputs", {}) or {}
                if gate_source_inputs.get("instantid_enabled") is not False:
                    return None, (
                        "InstantID bypass gate source does not explicitly disable "
                        "InstantID"
                    )
                bypass_inputs = (
                    "MODEL BYPASS",
                    "SAMPLED BYPASS",
                    "LOG BYPASS",
                    "VISUAL BYPASS",
                )
                missing_bypass = [
                    name
                    for name in bypass_inputs
                    if not _completion_prompt_link(prompt, node_inputs.get(name))
                ]
                if missing_bypass:
                    return None, (
                        "InstantID bypass gate is missing required OFF inputs: "
                        + ", ".join(missing_bypass)
                    )
                off_gate_source_id = gate_source_id
                pending.extend(node_inputs[name][0] for name in bypass_inputs)
                continue
            if class_type in cls._SDXL_EXCLUDED_UPSTREAM_CLASSES:
                return None, f"{class_type} is outside the Phase-2 SDXL completion contract"
            if class_type == "CMKPipeCreateImage":
                source_nodes[node_key] = node
            for value in (node.get("inputs", {}) or {}).values():
                if (
                    isinstance(value, (list, tuple))
                    and len(value) == 2
                    and isinstance(value[1], int)
                    and (
                        isinstance(prompt.get(value[0]), dict)
                        or isinstance(prompt.get(str(value[0])), dict)
                    )
                ):
                    pending.append(value[0])

        if len(source_nodes) != 1:
            return None, "expanded graph does not identify exactly one SDXL source"
        source_id, source_node = next(iter(source_nodes.items()))
        if off_gate_source_id is not None and off_gate_source_id != source_id:
            return None, (
                "InstantID bypass gate PROCESS does not originate from the active "
                "SDXL source"
            )
        source_inputs = source_node.get("inputs", {}) or {}
        if (
            str(source_inputs.get("model_family", "")).strip().lower() != "sdxl"
            or source_inputs.get("instantid_enabled") is True
        ):
            return None, "active source does not select pure SDXL"

        return {"roots": roots}, "expanded pure-SDXL graph complete"

    @classmethod
    def _completion_key(cls, prompt, unique_id):
        graph, detail = cls._sdxl_completion_graph(prompt, unique_id)
        if graph is None:
            return None, detail
        return _completion_key_for_graph(prompt, graph, _SDXL_COMPLETION)

    @staticmethod
    def _sdxl_completion_hit(cache_key):
        if not cache_key:
            return None
        result = lookup_completion(cache_key)
        if result is None:
            return None
        process = result.get("process")
        descriptor = result.get("model_descriptor")
        valid = (
            result.get("model_family") == "sdxl"
            and result.get("generation_mode") == "sdxl"
            and isinstance(process, dict)
            and str(process.get("model_family", "sdxl")).strip().lower() == "sdxl"
            and not bool(process.get("hybrid_mode", False))
            and process.get("family_active", True) is not False
            and isinstance(result.get("log"), dict)
            and isinstance(result.get("visual"), dict)
            and result.get("image") is not None
        )
        try:
            valid = valid and _released_generation_model_descriptor(descriptor) is not None
        except (TypeError, ValueError):
            valid = False
        if not valid:
            invalidate_completion(cache_key)
            return None
        return result

    @staticmethod
    def _released_descriptor_after_finish(model, completion_metadata):
        return _completion_released_descriptor(
            _SDXL_COMPLETION,
            model,
            completion_metadata,
            {},
        )

    def check_lazy_status(self, **inputs):
        preflight = _completion_preflight(
            type(self),
            self._COMPLETION_CONTRACT,
            inputs.get("prompt"),
            inputs.get("unique_id"),
        )
        if preflight["cached"] is not None:
            print(
                "[CMK Generation Completion / SDXL] HIT before lazy inputs "
                f"{preflight['key'][:12]} ({preflight['detail']})"
            )
            return []
        return super().check_lazy_status(**inputs)

    def boundary(self, postprocess_checkpoint, postprocess_vae, use_checkpoint_vae, **inputs):
        preflight = _completion_preflight(
            type(self),
            self._COMPLETION_CONTRACT,
            inputs.get("prompt"),
            inputs.get("unique_id"),
        )
        if preflight["cached"] is not None:
            return self._restore_completion(
                preflight,
                postprocess_checkpoint,
                postprocess_vae,
                use_checkpoint_vae,
            )

        process = inputs.get("PROCESS")
        if isinstance(process, dict) and not process.get("family_active", True):
            return super().boundary(
                postprocess_checkpoint,
                postprocess_vae,
                use_checkpoint_vae,
                **inputs,
            )
        if not preflight["key"]:
            raise RuntimeError(
                "CMK SDXL Generation Completion Cache cannot build its key: "
                + str(preflight["detail"])
            )

        completion_metadata = {}
        result = self._finish(
            self.FAMILY,
            inputs.get("MODEL"),
            process,
            inputs.get("IMAGE"),
            inputs.get("LOG"),
            inputs.get("VISUAL"),
            postprocess_checkpoint,
            postprocess_vae,
            use_checkpoint_vae,
            completion_metadata,
        )
        descriptor = self._released_descriptor_after_finish(
            inputs.get("MODEL"),
            completion_metadata,
        )
        if descriptor is None:
            print(
                "[CMK Generation Completion / SDXL] NOT STORED: "
                "generation resource release was not proven"
            )
            return result
        _store_completion_result(
            preflight,
            image=inputs.get("IMAGE"),
            process=process,
            log=inputs.get("LOG"),
            visual=result[4],
            metadata=completion_metadata,
            descriptor=descriptor,
        )
        print(
            "[CMK Generation Completion / SDXL] STORED "
            f"{preflight['key'][:12]} ({preflight['detail']})"
        )
        return result


class _CMKPostProcessBoundaryZITPipe(_CMKSinglePostProcessBoundary):
    PROCESS_TYPE = "CMK_PROCESS_Z_IMAGE"
    FAMILY = "z_image_turbo"
    _COMPLETION_MODE = "z_image_turbo"
    _COMPLETION_ROOTS = ("MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL")
    _REQUIRED_COMPLETION_ROOTS = ("MODEL", "PROCESS", "IMAGE", "LOG")
    _HYBRID_INPUTS = ("SAMPLED SDXL", "LOG SDXL", "VISUAL SDXL")
    _ZIT_EXCLUDED_UPSTREAM_CLASSES = {
        "CMKFamilyBranchGateSDXL",
        "CMKFamilyBranchGateSDXLSampled",
        "CMKInstantIDBoundary",
        "CMKSamplerBypassGate",
    }
    _COMPLETION_CONTRACT = _ZIT_COMPLETION

    @classmethod
    def INPUT_TYPES(cls):
        result = super().INPUT_TYPES()
        result["hidden"] = {
            "prompt": "PROMPT",
            "unique_id": "UNIQUE_ID",
        }
        return result

    @staticmethod
    def _prompt_node(prompt, node_id):
        return _completion_prompt_node(prompt, node_id)

    @classmethod
    def _prompt_link(cls, prompt, value):
        return _completion_prompt_link(prompt, value)

    @classmethod
    def _zit_completion_graph(cls, prompt, unique_id):
        if not isinstance(prompt, dict):
            return None, "PROMPT is unavailable"
        current = cls._prompt_node(prompt, unique_id)
        if not isinstance(current, dict) or current.get("class_type") != cls.__name__:
            return None, "ZIT PostProcess Boundary is unavailable in the expanded prompt"

        current_inputs = current.get("inputs", {}) or {}
        roots, detail = _completion_mode_roots(
            prompt,
            current_inputs,
            _ZIT_COMPLETION,
        )
        if roots is None:
            return None, detail

        active_contract, active_detail = _completion_active_contract(
            (_ZIT_COMPLETION,), prompt, unique_id
        )
        if active_contract is not _ZIT_COMPLETION:
            return None, active_detail

        pending = [link[0] for link in roots.values()]
        visited = set()
        found_zit_gate = False
        found_zit_finalizer = False
        found_zit_input = False
        while pending:
            node_id = pending.pop()
            node_key = str(node_id)
            if node_key in visited:
                continue
            visited.add(node_key)
            node = cls._prompt_node(prompt, node_id)
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type", ""))
            if (
                class_type in cls._ZIT_EXCLUDED_UPSTREAM_CLASSES
                or class_type.startswith("CMKInstantID")
            ):
                return None, f"{class_type} is outside the pure-ZIT completion contract"
            if class_type == "CMKFamilyBranchGateZImage":
                found_zit_gate = True
            elif class_type == "CMKZImageTurboFinalizePipe":
                found_zit_finalizer = True
            elif class_type == "CMKHybridZITInputPipe":
                found_zit_input = True
            for input_name, value in (node.get("inputs", {}) or {}).items():
                if (
                    class_type == "CMKHybridZITInputPipe"
                    and input_name in cls._HYBRID_INPUTS
                ):
                    # The START source above proves pure ZIT. These sockets may
                    # remain wired for Hybrid without making that handoff active.
                    continue
                if cls._prompt_link(prompt, value):
                    pending.append(value[0])

        if not (found_zit_gate and found_zit_finalizer and found_zit_input):
            return None, "expanded graph does not prove the current pure-ZIT contract"

        return {"roots": roots}, "expanded pure-ZIT graph complete"

    @classmethod
    def _completion_key(cls, prompt, unique_id):
        graph, detail = cls._zit_completion_graph(prompt, unique_id)
        if graph is None:
            return None, detail
        return _completion_key_for_graph(prompt, graph, _ZIT_COMPLETION)

    @staticmethod
    def _valid_finalizer_model_status(value):
        status = str(value or "").strip()
        return status in {"UNLOADED", "KEPT LOADED", "NOT AVAILABLE"} or status.startswith(
            "UNLOAD FAILED:"
        )

    @classmethod
    def _zit_finalization_evidence(cls, prompt, unique_id):
        graph, detail = cls._zit_completion_graph(prompt, unique_id)
        if graph is None:
            return None, detail

        pending = [link[0] for link in graph["roots"].values()]
        visited = set()
        gate_ids = []
        while pending:
            node_id = pending.pop()
            node_key = str(node_id)
            if node_key in visited:
                continue
            visited.add(node_key)
            node = cls._prompt_node(prompt, node_id)
            if not isinstance(node, dict):
                continue
            if node.get("class_type") == "CMKFamilyBranchGateZImage":
                gate_ids.append(node_id)
            for value in (node.get("inputs", {}) or {}).values():
                if cls._prompt_link(prompt, value):
                    pending.append(value[0])

        if len(gate_ids) != 1:
            return None, "expanded graph does not identify exactly one ZIT Final Gate"
        state_key = _zit_final_state_key(prompt, gate_ids[0])
        if state_key is None:
            return None, "ZIT Final Gate result state is unavailable"
        try:
            final_result = get_result(state_key)
        except PromptStateUnavailableError:
            return None, "ZIT Final Gate result state is unavailable"
        if not isinstance(final_result, dict):
            return None, "ZIT Final Gate has no completed result"
        diagnostic = final_result.get("diagnostic")
        metadata = diagnostic.get("metadata", {}) if isinstance(diagnostic, dict) else {}
        model_status = str(metadata.get("zit_model_status", "")).strip()
        complete = (
            final_result.get("image") is not None
            and isinstance(final_result.get("log"), dict)
            and isinstance(final_result.get("visual"), dict)
            and isinstance(diagnostic, dict)
            and str(metadata.get("model_family", "")).strip().lower()
            == "z_image_turbo"
            and cls._valid_finalizer_model_status(model_status)
        )
        if not complete:
            return None, "ZIT Final Gate diagnostic does not prove completed finalization"
        return {"zit_model_status": model_status}, "ZIT Final Gate result and diagnostic complete"

    @classmethod
    def _zit_completion_hit(cls, cache_key):
        if not cache_key:
            return None
        result = lookup_completion(cache_key)
        if result is None:
            return None
        process = result.get("process")
        descriptor = result.get("model_descriptor")
        metadata = result.get("metadata")
        valid = (
            result.get("model_family") == "z_image_turbo"
            and result.get("generation_mode") == "z_image_turbo"
            and isinstance(process, dict)
            and str(process.get("model_family", "")).strip().lower()
            == "z_image_turbo"
            and not bool(process.get("hybrid_mode", False))
            and process.get("family_active", True) is not False
            and isinstance(result.get("log"), dict)
            and isinstance(result.get("visual"), dict)
            and isinstance(metadata, dict)
            and cls._valid_finalizer_model_status(
                metadata.get("zit_model_status")
            )
            and result.get("image") is not None
        )
        try:
            released = _released_generation_model_descriptor(descriptor)
            valid = (
                valid
                and released is not None
                and released["model_family"] == "z_image_turbo"
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            invalidate_completion(cache_key)
            return None
        return result

    @staticmethod
    def _released_descriptor_after_finish(model, completion_metadata):
        return _completion_released_descriptor(
            _ZIT_COMPLETION,
            model,
            completion_metadata,
            {},
        )

    def check_lazy_status(self, **inputs):
        preflight = _completion_preflight(
            type(self),
            self._COMPLETION_CONTRACT,
            inputs.get("prompt"),
            inputs.get("unique_id"),
        )
        if preflight["cached"] is not None:
            print(
                "[CMK Generation Completion / ZIT] HIT before lazy inputs "
                f"{preflight['key'][:12]} ({preflight['detail']})"
            )
            return []
        return super().check_lazy_status(**inputs)

    def boundary(self, postprocess_checkpoint, postprocess_vae, use_checkpoint_vae, **inputs):
        preflight = _completion_preflight(
            type(self),
            self._COMPLETION_CONTRACT,
            inputs.get("prompt"),
            inputs.get("unique_id"),
        )
        if preflight["cached"] is not None:
            return self._restore_completion(
                preflight,
                postprocess_checkpoint,
                postprocess_vae,
                use_checkpoint_vae,
            )

        process = inputs.get("PROCESS")
        if (
            not preflight["key"]
            or (isinstance(process, dict) and not process.get("family_active", True))
        ):
            return super().boundary(
                postprocess_checkpoint,
                postprocess_vae,
                use_checkpoint_vae,
                **inputs,
            )

        completion_metadata = {}
        result = self._finish(
            self.FAMILY,
            inputs.get("MODEL"),
            process,
            inputs.get("IMAGE"),
            inputs.get("LOG"),
            inputs.get("VISUAL"),
            postprocess_checkpoint,
            postprocess_vae,
            use_checkpoint_vae,
            completion_metadata,
        )
        finalization_evidence, finalization_detail = _completion_finalization_evidence(
            type(self),
            preflight,
            inputs.get("prompt"),
            inputs.get("unique_id"),
            process,
        )
        if finalization_evidence is None:
            print(
                "[CMK Generation Completion / ZIT] NOT STORED: "
                "ZIT finalization is not proven ("
                + finalization_detail
                + ")"
            )
            return result
        descriptor = self._released_descriptor_after_finish(
            inputs.get("MODEL"),
            completion_metadata,
        )
        if descriptor is None:
            print(
                "[CMK Generation Completion / ZIT] NOT STORED: "
                "generation resource release was not proven"
            )
            return result
        completion_metadata.update(finalization_evidence)
        _store_completion_result(
            preflight,
            image=inputs.get("IMAGE"),
            process=process,
            log=inputs.get("LOG"),
            visual=result[4],
            metadata=completion_metadata,
            descriptor=descriptor,
        )
        print(
            "[CMK Generation Completion / ZIT] STORED "
            f"{preflight['key'][:12]} ({preflight['detail']})"
        )
        return result


class CMKPostProcessSetupPipe(
    _CMKFamilyResultMergePipe,
    _CMKPostProcessBoundarySDXLPipe,
    _CMKPostProcessBoundaryZITPipe,
):
    """Complete any supported generation family at the PostProcess boundary."""

    _COMPLETION_CONTRACTS = (
        _SDXL_COMPLETION,
        _ZIT_COMPLETION,
        _SDXL_ZIT_HYBRID_COMPLETION,
        _INSTANTID_HYBRID_COMPLETION,
        _INSTANTID_ZERO_PASS_COMPLETION,
    )
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL")


_VISUALIZER_COMMON_INPUTS = {
    "MODEL": "MODEL (opt)",
    "PROCESS": "PROCESS",
    "IMAGE": "IMAGE",
    "LOG": "LOG",
    "VISUAL": "VISUAL",
}
_VISUALIZER_FAMILY_SUFFIXES = ("ZIT", "SDXL")
_POSTPROCESS_COMPLETION_NODES = {
    "CMKPostProcessSetupPipe",
    "CMKPostProcessBoundarySDXLPipe",
    "CMKPostProcessBoundaryZITPipe",
    "CMKFamilyResultMergePipe",
}


def _visualizer_upstream_contains_postprocess(prompt, current_inputs):
    pending = []
    for input_name in _VISUALIZER_COMMON_INPUTS.values():
        value = current_inputs.get(input_name)
        if _completion_prompt_link(prompt, value):
            pending.append(value[0])
    visited = set()
    while pending:
        node_id = pending.pop()
        node_key = str(node_id)
        if node_key in visited:
            continue
        visited.add(node_key)
        node = _completion_prompt_node(prompt, node_id)
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")) in _POSTPROCESS_COMPLETION_NODES:
            return True
        for value in (node.get("inputs", {}) or {}).values():
            if _completion_prompt_link(prompt, value):
                pending.append(value[0])
    return False


def _visualizer_completion_prompt(prompt, unique_id):
    """Adapt the Visualizer's generic roots to the existing mode contracts."""
    if not isinstance(prompt, dict):
        return None, "PROMPT is unavailable"
    current = _completion_prompt_node(prompt, unique_id)
    if not isinstance(current, dict) or current.get("class_type") != "CMKVisualizer":
        return None, "Visualizer is unavailable in the expanded prompt"
    current_inputs = dict(current.get("inputs", {}) or {})
    if _visualizer_upstream_contains_postprocess(prompt, current_inputs):
        return None, "PostProcess Setup already owns generation completion"

    adapted_inputs = dict(current_inputs)
    family_connected = any(
        _completion_prompt_link(prompt, current_inputs.get(f"{role} {suffix}"))
        for suffix in _VISUALIZER_FAMILY_SUFFIXES
        for role in _VISUALIZER_COMMON_INPUTS
    )
    if not family_connected:
        for role, input_name in _VISUALIZER_COMMON_INPUTS.items():
            value = current_inputs.get(input_name)
            if value is not None:
                adapted_inputs.setdefault(role, value)
        for suffix in _VISUALIZER_FAMILY_SUFFIXES:
            for role, input_name in _VISUALIZER_COMMON_INPUTS.items():
                value = current_inputs.get(input_name)
                if value is not None:
                    adapted_inputs.setdefault(f"{role} {suffix}", value)
    adapted_current = dict(current)
    adapted_current["class_type"] = "CMKPostProcessSetupPipe"
    adapted_current["inputs"] = adapted_inputs
    adapted_prompt = dict(prompt)
    prompt_key = unique_id if unique_id in prompt else str(unique_id)
    adapted_prompt[prompt_key] = adapted_current
    return adapted_prompt, "direct generation roots adapted"


def _visualizer_completion_input_group(preflight, current_inputs):
    contract = preflight.get("contract")
    graph = preflight.get("graph")
    roots = graph.get("roots") if isinstance(graph, dict) else None
    if contract is None or not isinstance(roots, dict):
        return None
    preferred = (
        "ZIT"
        if contract in (_ZIT_COMPLETION, _SDXL_ZIT_HYBRID_COMPLETION)
        else "SDXL"
    )

    def base_role(role):
        text = str(role)
        for suffix in _VISUALIZER_FAMILY_SUFFIXES:
            marker = f" {suffix}"
            if text.endswith(marker):
                return text[:-len(marker)]
        return text

    def matches(group):
        for role, link in roots.items():
            base = base_role(role)
            input_name = (
                _VISUALIZER_COMMON_INPUTS.get(base)
                if group == "COMMON"
                else f"{base} {group}"
            )
            if input_name is None or current_inputs.get(input_name) != link:
                return False
        return True

    for group in (preferred, "COMMON"):
        if matches(group):
            return group
    return None


def visualizer_completion_preflight(prompt, unique_id):
    adapted_prompt, detail = _visualizer_completion_prompt(prompt, unique_id)
    if adapted_prompt is None:
        return {
            "contract": None,
            "graph": None,
            "key": None,
            "detail": detail,
            "cached": None,
            "completion_prompt": None,
            "input_group": "COMMON",
            "path_kind": "postprocess" if "already owns" in detail else "common",
        }
    preflight = _resolve_completion_preflight(
        CMKPostProcessSetupPipe,
        CMKPostProcessSetupPipe._COMPLETION_CONTRACTS,
        adapted_prompt,
        unique_id,
    )
    preflight["completion_prompt"] = adapted_prompt
    current = _completion_prompt_node(prompt, unique_id) or {}
    current_inputs = current.get("inputs", {}) or {}
    input_group = _visualizer_completion_input_group(preflight, current_inputs)
    family_connected = any(
        _completion_prompt_link(prompt, current_inputs.get(f"{role} {suffix}"))
        for suffix in _VISUALIZER_FAMILY_SUFFIXES
        for role in _VISUALIZER_COMMON_INPUTS
    )
    preflight["input_group"] = input_group or "COMMON"
    preflight["path_kind"] = (
        "direct"
        if preflight["contract"] is not None and input_group is not None
        else ("invalid" if family_connected else "common")
    )
    if preflight["contract"] is not None and input_group is None:
        preflight["cached"] = None
        preflight["detail"] = "completion roots mix incompatible Visualizer input groups"
    return preflight


def restore_visualizer_completion(preflight):
    cached = preflight.get("cached") if isinstance(preflight, dict) else None
    if not isinstance(cached, dict):
        raise ValueError("CMK Visualizer Completion restore requires a valid cache hit")
    contract = preflight["contract"]
    print(
        f"[CMK Generation Completion / {contract.log_label}] RESTORE at Visualizer "
        f"{preflight['key'][:12]} ({preflight['detail']})"
    )
    return (
        cached["model_descriptor"],
        cached["process"],
        cached["image"],
        cached["log"],
        cached["visual"],
    )


def complete_generation_at_visualizer(
    preflight, *, model, process, image, log, visual, unique_id
):
    """Release and persist a proven direct-generation result at the Visualizer."""
    contract = preflight.get("contract") if isinstance(preflight, dict) else None
    if contract is None or not preflight.get("key"):
        return model, process, image, log, visual
    if not isinstance(model, dict) or not isinstance(process, dict) or not isinstance(log, dict):
        raise TypeError("CMK Visualizer Completion requires valid MODEL, PROCESS and LOG pipes")
    if image is None:
        raise ValueError("CMK Visualizer Completion requires the completed generation IMAGE")
    actual = str(process.get("model_family", contract.runtime_family)).strip().lower()
    if actual != contract.runtime_family:
        raise ValueError("CMK Visualizer Completion received a PROCESS from the wrong family")

    evidence, evidence_detail = _completion_finalization_evidence(
        CMKPostProcessSetupPipe,
        preflight,
        preflight.get("completion_prompt"),
        unique_id,
        process,
    )
    release_metadata, _released = _release_completed_generation_resources(
        contract.runtime_family, model, process
    )
    if evidence is None:
        print(
            f"[CMK Generation Completion / {contract.log_label}] NOT STORED at Visualizer: "
            + evidence_detail
        )
        return model, process, image, log, visual
    generation_unload = str(release_metadata.get("generation_unload", ""))
    if generation_unload != "UNLOADED" and not generation_unload.startswith(
        "ALREADY RELEASED ("
    ):
        print(
            f"[CMK Generation Completion / {contract.log_label}] NOT STORED at Visualizer: "
            "generation resource release was not proven"
        )
        return model, process, image, log, visual
    evidence = dict(evidence)
    descriptor = _completion_released_descriptor(
        contract, model, release_metadata, evidence
    )
    if descriptor is None:
        print(
            f"[CMK Generation Completion / {contract.log_label}] NOT STORED at Visualizer: "
            "released generation MODEL descriptor is unavailable"
        )
        return model, process, image, log, visual
    evidence.pop("model_descriptor", None)
    release_metadata.update(evidence)
    _store_completion_result(
        preflight,
        image=image,
        process=process,
        log=log,
        visual=visual,
        metadata=release_metadata,
        descriptor=descriptor,
    )
    print(
        f"[CMK Generation Completion / {contract.log_label}] STORED at Visualizer "
        f"{preflight['key'][:12]} ({preflight['detail']})"
    )
    return descriptor, process, image, log, visual


class CMKResultProcessForwardPipe:
    """Forward a family-neutral PROCESS without resolving image or model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"PROCESS": (CMK_FINISH_INPUT,)}}

    RETURN_TYPES = ("CMK_RESULT_PROCESS",)
    RETURN_NAMES = ("PROCESS",)
    FUNCTION = "forward"
    CATEGORY = "CMK/Internal/Forward"
    DEV_ONLY = True

    @staticmethod
    def forward(PROCESS):
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Result Process Forward requires a PROCESS dictionary")
        return (PROCESS,)


class CMKZImageProcessForwardPipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"PROCESS": ("CMK_PROCESS_Z_IMAGE",)}}

    RETURN_TYPES = ("CMK_PROCESS_Z_IMAGE",)
    RETURN_NAMES = ("PROCESS",)
    FUNCTION = "forward"
    CATEGORY = "CMK/Internal/Forward"
    DEV_ONLY = True

    def check_lazy_status(self, PROCESS=None, **inputs):
        # PROCESS is the cheap family selector.  It must remain independent of
        # the sampled result so the boundary can select ZIT first and only then
        # request MODEL / IMAGE / LOG through the guarded expensive path.
        # Waiting for RESULT PROCESS here can leave downstream OUTPUT_NODEs
        # blocked after a cold global-subgraph execution in ComfyUI.
        return []

    @staticmethod
    def forward(PROCESS, **inputs):
        if PROCESS is None:
            return (None,)
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Z-Image Process Forward requires a process pipe")
        return (PROCESS,)


class CMKResultToLegacyBridgePipe:
    """Internal adapter for existing model-independent consumers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"PROCESS": ("CMK_RESULT_PIPE",)}}

    RETURN_TYPES = ("CMK_PIPE",)
    RETURN_NAMES = ("PROCESS",)
    FUNCTION = "bridge"
    CATEGORY = "CMK/Developer/Legacy Bridge"
    DEV_ONLY = True

    @staticmethod
    def bridge(PROCESS):
        if not isinstance(PROCESS, dict) or PROCESS.get("result_contract") != "family_neutral":
            raise ValueError("CMK Result Bridge requires a family-neutral PROCESS")
        return (dict(PROCESS),)


class CMKResultUnpackPipe:
    """Normalize a legacy result or the five-part module-35 contract.

    MODEL is the postprocess working model after the boundary; PROCESS retains the
    generation provenance independently of that model role.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "MODEL (opt)": (CMK_FINISH_INPUT,),
                "PROCESS": (CMK_FINISH_INPUT,),
                "IMAGE": (CMK_FINISH_INPUT,),
                "LOG": (CMK_FINISH_INPUT,),
            },
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE", "CMK_PIPE", "IMAGE", "CMK_LOG_PIPE")
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG")
    FUNCTION = "unpack"
    CATEGORY = "CMK/Internal/Result"
    DEV_ONLY = True

    @staticmethod
    def unpack(MODEL=None, PROCESS=None, IMAGE=None, LOG=None, **kwargs):
        MODEL = kwargs.get("MODEL (opt)", MODEL)
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Result Unpack requires a CMK PROCESS from SDXL, ZIT or the PostProcess Boundary")
        if IMAGE is None:
            raise TypeError("CMK Result Unpack requires an IMAGE from the same active path")
        if not isinstance(LOG, dict):
            raise TypeError("CMK Result Unpack requires a CMK LOG from the same active path")

        if PROCESS.get("type") == "CMK_MASK_DETAILER_PROCESS":
            if not isinstance(MODEL, dict):
                raise TypeError("CMK Mask Detailer result requires its MODEL at the Visualizer")
            normalized = dict(PROCESS)
            normalized["result_contract"] = "mask_detailer"
            normalized.setdefault("source_model_family", "image")
            return (MODEL, normalized, IMAGE, LOG)

        family = str(
            PROCESS.get(
                "source_model_family",
                PROCESS.get("model_family", ""),
            )
        ).strip().lower()
        neutral_image_origins = {
            "CMK Image Load and Resize -Pipe-",
            "CMK Swap Image Loader -Pipe-",
        }
        neutral_image_path = (
            PROCESS.get("result_contract") == "family_neutral"
            and family == "image"
            and PROCESS.get("pipe_origin") in neutral_image_origins
        )
        if not neutral_image_path and not isinstance(MODEL, dict):
            raise TypeError("CMK Result Unpack requires a CMK MODEL from SDXL, ZIT or the PostProcess Boundary")
        if family not in {"sdxl", "z_image_turbo"} and not neutral_image_path:
            raise ValueError(
                "CMK Result Unpack accepts only a complete SDXL path, a complete ZIT path, "
                "the output of the PostProcess Boundary, or a complete CMK image-input path"
            )
        if isinstance(MODEL, dict):
            model_family = str(MODEL.get("model_family", family)).strip().lower()
            valid_neutral_model = neutral_image_path and model_family in {"", "sdxl"}
            valid_postprocess_model = (
                MODEL.get("model_role") == "postprocess"
                and model_family == "sdxl"
                and PROCESS.get("result_contract") == "family_neutral"
            )
            if (
                model_family
                and model_family != family
                and not valid_neutral_model
                and not valid_postprocess_model
            ):
                raise ValueError("CMK Result Unpack received MODEL and PROCESS from different families")
        if not neutral_image_path and PROCESS.get("family_active") is False:
            raise ValueError("CMK Result Unpack received the inactive family path")

        normalized = dict(PROCESS)
        normalized["result_contract"] = "family_neutral"
        normalized["source_model_family"] = family
        return (MODEL, normalized, IMAGE, LOG)


class CMKResultPackPipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "MODEL (opt)": ("CMK_MODEL_PIPE",),
                "PROCESS": ("CMK_PIPE",),
                "IMAGE": ("IMAGE",),
                "LOG": ("CMK_LOG_PIPE",),
            },
        }

    RETURN_TYPES = (
        "CMK_RESULT_MODEL",
        "CMK_RESULT_PROCESS",
        "CMK_RESULT_IMAGE",
        "CMK_RESULT_LOG",
    )
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG")
    FUNCTION = "pack"
    CATEGORY = "CMK/Internal/Result"
    DEV_ONLY = True

    @staticmethod
    def pack(MODEL=None, PROCESS=None, IMAGE=None, LOG=None, **kwargs):
        MODEL = kwargs.get("MODEL (opt)", MODEL)
        if not isinstance(PROCESS, dict) or PROCESS.get("result_contract") != "family_neutral":
            raise ValueError("CMK Result Pack requires a family-neutral PROCESS")
        if MODEL is not None and not isinstance(MODEL, dict):
            raise TypeError("CMK Result Pack received an invalid MODEL")
        if IMAGE is None or not isinstance(LOG, dict):
            raise TypeError("CMK Result Pack received an incomplete result")
        return (MODEL, dict(PROCESS), IMAGE, LOG)
