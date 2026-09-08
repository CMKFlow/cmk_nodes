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


class _CMKAnyType(str):
    def __ne__(self, other):
        return False


CMK_FINISH_INPUT = _CMKAnyType("*")


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
    CATEGORY = "CMK/Developer/Boundary & Cache"
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
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE", "IMAGE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("MODEL", "IMAGE", "LOG", "diagnostic")
    FUNCTION = "gate"
    CATEGORY = "CMK/Developer/Boundary & Cache"
    DEV_ONLY = True

    def check_lazy_status(self, ENABLE=False, prompt=None, unique_id=None, **inputs):
        prefix = "ACTIVE" if bool(ENABLE) else "BYPASS"
        current = prompt.get(str(unique_id), {}) if isinstance(prompt, dict) else {}
        connected = (current.get("inputs", {}) or {})
        for name in (f"MODEL {prefix}", f"IMAGE {prefix}", f"LOG {prefix}"):
            if name in connected and inputs.get(name) is None:
                return [name]
        if bool(ENABLE) and "DIAGNOSTIC ACTIVE" in connected and inputs.get("DIAGNOSTIC ACTIVE") is None:
            return ["DIAGNOSTIC ACTIVE"]
        return []

    @staticmethod
    def gate(ENABLE=False, **inputs):
        prefix = "ACTIVE" if bool(ENABLE) else "BYPASS"
        values = tuple(inputs.get(f"{name} {prefix}") for name in ("MODEL", "IMAGE", "LOG"))
        if values[1] is None or values[2] is None:
            raise ValueError(f"CMK Module Bypass Gate is missing the {prefix} result")
        diagnostic = (
            _module_gate_diagnostic(inputs.get("DIAGNOSTIC ACTIVE"))
            if bool(ENABLE)
            else ExecutionBlocker(None)
        )
        return (*values, diagnostic)


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
    CATEGORY = "CMK/Developer/Boundary & Cache"
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
    CATEGORY = "CMK/Developer/Boundary & Cache"
    DEV_ONLY = True

    def check_lazy_status(self, ENABLE=False, **inputs):
        suffix = "ACTIVE" if bool(ENABLE) else "BYPASS"
        for base in ("PROCESS SDXL", "PROCESS ZIT", "LOG"):
            name = f"{base} {suffix}"
            if inputs.get(name) is None:
                return [name]
        if bool(ENABLE) and inputs.get("DIAGNOSTIC ACTIVE") is None:
            return ["DIAGNOSTIC ACTIVE"]
        return []

    @staticmethod
    def gate(ENABLE=False, **inputs):
        suffix = "ACTIVE" if bool(ENABLE) else "BYPASS"
        values = tuple(inputs.get(f"{base} {suffix}") for base in (
            "PROCESS SDXL", "PROCESS ZIT", "IMAGE", "LOG",
        ))
        if values[0] is None or values[1] is None or values[3] is None:
            raise ValueError(
                f"CMK Combined ControlNet Bypass Gate is missing the {suffix} result"
            )
        diagnostic = inputs.get("DIAGNOSTIC ACTIVE") if bool(ENABLE) else ExecutionBlocker(None)
        return (*values, diagnostic)


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
            },
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE", "CMK_SAMPLED_PIPE", "CMK_LOG_PIPE", "CMK_DIAGNOSTIC")
    RETURN_NAMES = ("MODEL", "SAMPLED", "LOG", "diagnostic")
    FUNCTION = "gate"
    CATEGORY = "CMK/Developer/Boundary & Cache"
    DEV_ONLY = True

    def check_lazy_status(self, ENABLE=False, **inputs):
        prefix = "ACTIVE" if bool(ENABLE) else "BYPASS"
        for name in (f"MODEL {prefix}", f"SAMPLED {prefix}", f"LOG {prefix}"):
            if inputs.get(name) is None:
                return [name]
        if bool(ENABLE) and inputs.get("DIAGNOSTIC ACTIVE") is None:
            return ["DIAGNOSTIC ACTIVE"]
        return []

    @staticmethod
    def gate(ENABLE=False, **inputs):
        prefix = "ACTIVE" if bool(ENABLE) else "BYPASS"
        values = tuple(inputs.get(f"{name} {prefix}") for name in ("MODEL", "SAMPLED", "LOG"))
        if any(value is None for value in values):
            raise ValueError(f"CMK Sampler Bypass Gate is missing the {prefix} result")
        diagnostic = inputs.get("DIAGNOSTIC ACTIVE") if bool(ENABLE) else ExecutionBlocker(None)
        if bool(ENABLE) and diagnostic is None:
            raise ValueError("CMK Sampler Bypass Gate is missing the active diagnostic")
        return (*values, diagnostic)


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
    CATEGORY = "CMK/Developer/Pipe/Forward"
    DEV_ONLY = True

    @staticmethod
    def read(PROCESS, flag="instantid_enabled"):
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Process Enable Flag requires a PROCESS dictionary")
        return (bool(PROCESS.get(str(flag), False)),)


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
            },
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "IMAGE",
        "CMK_LOG_PIPE",
    )
    RETURN_NAMES = ("MODEL", "IMAGE", "LOG")
    FUNCTION = "gate"
    CATEGORY = "CMK/Developer/Boundary & Cache"
    DEV_ONLY = True

    @cmk_timed_call("LAZY FAMILY BRANCH GATE")
    def check_lazy_status(self, PROCESS=None, **inputs):
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
        return []

    def gate(self, PROCESS=None, **inputs):
        if PROCESS is None or (
            isinstance(PROCESS, dict) and not PROCESS.get("family_active", True)
        ):
            blocked = ExecutionBlocker(None)
            return (blocked, blocked, blocked)
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
        )


class CMKFamilyBranchGateSDXL(_CMKFamilyBranchGate):
    pass


class CMKFamilyBranchGateZImage(_CMKFamilyBranchGate):
    PROCESS_TYPE = "CMK_PROCESS_Z_IMAGE"
    FAMILY = "z_image_turbo"
    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "IMAGE",
        "CMK_LOG_PIPE",
    )


_SDXL_SAMPLED_BOUNDARY_CACHE = {}
_SDXL_SAMPLED_BOUNDARY_MAX = 8
_FIRST_PASS_STAGE_KEY = "sdxl.first_pass"


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


def _sampled_boundary_key(prompt, unique_id):
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
    )
    RETURN_NAMES = ("MODEL", "SAMPLED", "LOG", "PROCESS", "IMAGE")

    @cmk_timed_call("LAZY SDXL SAMPLED GATE")
    def check_lazy_status(self, PROCESS=None, prompt=None, unique_id=None, **inputs):
        if PROCESS is None:
            return ["PROCESS"]
        if isinstance(PROCESS, dict) and not PROCESS.get("family_active", True):
            return []
        cache_key = _sampled_boundary_key(prompt, unique_id)
        if cache_key in _SDXL_SAMPLED_BOUNDARY_CACHE:
            return []
        # Resolve converging branches one at a time. Requesting MODEL and
        # SAMPLED together lets ComfyUI open the shared sampler preparation
        # through two lazy paths after a process restart, executing module 10
        # twice. MODEL is cheap; SAMPLED is the expensive authoritative stage.
        if inputs.get("MODEL") is None:
            return ["MODEL"]
        if inputs.get("SAMPLED") is None:
            return ["SAMPLED"]
        if inputs.get("LOG") is None:
            return ["LOG"]
        return []

    def gate(self, PROCESS=None, prompt=None, unique_id=None, **inputs):
        if PROCESS is None or (
            isinstance(PROCESS, dict) and not PROCESS.get("family_active", True)
        ):
            blocked = ExecutionBlocker(None)
            return (blocked, blocked, blocked, blocked, blocked)
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK SDXL Sampler Branch Gate requires a CMK process pipe")
        actual = str(PROCESS.get("model_family", "sdxl")).strip().lower()
        if actual != "sdxl":
            raise ValueError("CMK SDXL Sampler Branch Gate received the wrong model family")
        cache_key = _sampled_boundary_key(prompt, unique_id)
        result_process = (
            stamp_artifact(PROCESS, _FIRST_PASS_STAGE_KEY, cache_key)
            if cache_key
            else PROCESS
        )
        if cache_key in _SDXL_SAMPLED_BOUNDARY_CACHE and any(
            inputs.get(name) is None for name in ("MODEL", "SAMPLED", "LOG")
        ):
            model, sampled, log = _SDXL_SAMPLED_BOUNDARY_CACHE[cache_key]
            sampled, image = _materialize_first_pass_image(sampled)
            return (model, sampled, log, result_process, image)
        missing = [
            name for name in ("MODEL", "SAMPLED", "LOG")
            if inputs.get(name) is None
        ]
        if missing:
            raise ValueError(
                "CMK SDXL Sampler Branch Gate is missing " + ", ".join(missing)
            )
        sampled, image = _materialize_first_pass_image(inputs["SAMPLED"])
        result = (inputs["MODEL"], sampled, inputs["LOG"])
        if cache_key:
            _SDXL_SAMPLED_BOUNDARY_CACHE[cache_key] = result
            while len(_SDXL_SAMPLED_BOUNDARY_CACHE) > _SDXL_SAMPLED_BOUNDARY_MAX:
                _SDXL_SAMPLED_BOUNDARY_CACHE.pop(next(iter(_SDXL_SAMPLED_BOUNDARY_CACHE)))
        return (*result, result_process, image)


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
    CATEGORY = "CMK/Developer/Boundary & Cache"
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


class CMKFamilyResultMergePipe:
    """Lazily select one complete family result and expose a neutral contract."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "MODEL SDXL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS SDXL": ("CMK_PROCESS_SDXL", {"lazy": True}),
                "IMAGE SDXL": ("IMAGE", {"lazy": True}),
                "LOG SDXL": ("CMK_LOG_PIPE", {"lazy": True}),
                "VISUAL SDXL": ("CMK_VISUAL_PIPE", {"lazy": True}),
                "MODEL ZIT": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS ZIT": ("CMK_PROCESS_Z_IMAGE", {"lazy": True}),
                "IMAGE ZIT": ("IMAGE", {"lazy": True}),
                "LOG ZIT": ("CMK_LOG_PIPE", {"lazy": True}),
                "VISUAL ZIT": ("CMK_VISUAL_PIPE", {"lazy": True}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "CMK_RESULT_MODEL",
        "CMK_RESULT_PROCESS",
        "CMK_RESULT_IMAGE",
        "CMK_RESULT_LOG",
        "CMK_VISUAL_PIPE",
    )
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG", "VISUAL")
    FUNCTION = "merge"
    CATEGORY = "CMK/Flow/Finish"

    @staticmethod
    def _pending_visual(suffix, inputs):
        name = f"VISUAL {suffix}"
        prompt = inputs.get("prompt") or {}
        entry = prompt.get(str(inputs.get("unique_id")), {})
        connected = name in (entry.get("inputs") or {}) or name in inputs
        return [name] if connected and inputs.get(name) is None else []

    @cmk_timed_call("LAZY FAMILY RESULT MERGE")
    def check_lazy_status(self, **inputs):
        process_sdxl = inputs.get("PROCESS SDXL")
        process_z = inputs.get("PROCESS ZIT")
        # Probe ZIT first. In a ZIT run the inactive global SDXL subgraph ends
        # in an ExecutionBlocker; requesting it first prevents ComfyUI from
        # ever returning here to resolve the active ZIT branch.
        if process_z is None:
            return ["PROCESS ZIT"]
        active_z = isinstance(process_z, dict) and process_z.get("family_active", True)
        if active_z:
            for name in ("MODEL", "IMAGE", "LOG"):
                input_name = f"{name} ZIT"
                if inputs.get(input_name) is None:
                    return [input_name]
            return self._pending_visual("ZIT", inputs)
        if process_sdxl is None:
            return ["PROCESS SDXL"]
        active_sdxl = isinstance(process_sdxl, dict) and process_sdxl.get("family_active", True)
        if not active_sdxl and not active_z:
            return []
        for name in ("MODEL", "IMAGE", "LOG"):
            input_name = f"{name} SDXL"
            if inputs.get(input_name) is None:
                return [input_name]
        return self._pending_visual("SDXL", inputs)

    def merge(self, **inputs):
        process_sdxl = inputs.get("PROCESS SDXL")
        process_z = inputs.get("PROCESS ZIT")
        active_sdxl = isinstance(process_sdxl, dict) and process_sdxl.get("family_active", True)
        active_z = isinstance(process_z, dict) and process_z.get("family_active", True)
        if active_sdxl and active_z:
            raise ValueError("CMK Active Family Result received two active PROCESS paths")
        if not active_sdxl and not active_z:
            raise ValueError("CMK Active Family Result received no active PROCESS path")
        family = "sdxl" if active_sdxl else "z_image_turbo"
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
        if not isinstance(model, dict) or not isinstance(process, dict) or not isinstance(log, dict):
            raise TypeError("CMK Active Family Result requires valid MODEL, PROCESS and LOG pipes")
        actual = str(process.get("model_family", family)).strip().lower()
        if actual != family:
            raise ValueError("CMK Active Family Result received a PROCESS from the wrong family")
        result = dict(process)
        result["result_contract"] = "family_neutral"
        result["source_model_family"] = family
        visual = inputs.get(f"VISUAL {suffix}")
        if visual is None:
            try:
                from .cmk_visual import empty_visual
            except ImportError:  # Direct source loading in contract tests.
                from pipe.cmk_visual import empty_visual
            visual = empty_visual()
        return (model, result, image, log, visual)


class CMKResultProcessForwardPipe:
    """Forward a family-neutral PROCESS without resolving image or model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"PROCESS": (CMK_FINISH_INPUT,)}}

    RETURN_TYPES = ("CMK_RESULT_PROCESS",)
    RETURN_NAMES = ("PROCESS",)
    FUNCTION = "forward"
    CATEGORY = "CMK/Developer/Pipe/Forward"
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
    CATEGORY = "CMK/Developer/Pipe"
    DEV_ONLY = True

    def check_lazy_status(self, PROCESS=None, **inputs):
        # PROCESS is the cheap family selector.  It must remain independent of
        # the sampled result so module 35 can select ZIT first and only then
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
    CATEGORY = "CMK/Developer/Boundary & Cache"
    DEV_ONLY = True

    @staticmethod
    def bridge(PROCESS):
        if not isinstance(PROCESS, dict) or PROCESS.get("result_contract") != "family_neutral":
            raise ValueError("CMK Result Bridge requires a family-neutral PROCESS")
        return (dict(PROCESS),)


class CMKResultUnpackPipe:
    """Normalize a direct family result or the output of module 35.

    The public finish module deliberately keeps four sockets. Their values are
    validated as one complete contract here, so a simple SDXL or ZIT flow does
    not need an otherwise redundant family merge.
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
    CATEGORY = "CMK/Developer/Boundary & Cache"
    DEV_ONLY = True

    @staticmethod
    def unpack(MODEL=None, PROCESS=None, IMAGE=None, LOG=None, **kwargs):
        MODEL = kwargs.get("MODEL (opt)", MODEL)
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK 90 requires a CMK PROCESS from SDXL, ZIT or module 35")
        if IMAGE is None:
            raise TypeError("CMK 90 requires an IMAGE from the same active path")
        if not isinstance(LOG, dict):
            raise TypeError("CMK 90 requires a CMK LOG from the same active path")

        family = str(
            PROCESS.get(
                "source_model_family",
                PROCESS.get("model_family", ""),
            )
        ).strip().lower()
        neutral_image_path = (
            PROCESS.get("result_contract") == "family_neutral"
            and family == "image"
            and PROCESS.get("pipe_origin") == "CMK Image Load and Resize -Pipe-"
        )
        if not neutral_image_path and not isinstance(MODEL, dict):
            raise TypeError("CMK 90 requires a CMK MODEL from SDXL, ZIT or module 35")
        if family not in {"sdxl", "z_image_turbo"} and not neutral_image_path:
            raise ValueError(
                "CMK 90 accepts only a complete SDXL path, a complete ZIT path, "
                "the output of module 35, or a complete CMK image-input path"
            )
        if isinstance(MODEL, dict):
            model_family = str(MODEL.get("model_family", family)).strip().lower()
            valid_neutral_model = neutral_image_path and model_family in {"", "sdxl"}
            if model_family and model_family != family and not valid_neutral_model:
                raise ValueError("CMK 90 received MODEL and PROCESS from different families")
        if not neutral_image_path and PROCESS.get("family_active") is False:
            raise ValueError("CMK 90 received the inactive family path")

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
    CATEGORY = "CMK/Developer/Boundary & Cache"
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
