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


class _CMKAnyType(str):
    def __ne__(self, other):
        return False


CMK_FINISH_INPUT = _CMKAnyType("*")


def _gate_diagnostic(value, image, family):
    """Return a renderable diagnostic without resolving an observational branch."""
    if isinstance(value, dict) and value.get("type") in {
        "CMK_DIAGNOSTIC",
        "CMK_PREVIEW",
    }:
        return value
    family_label = "SDXL" if family == "sdxl" else "Z-Image Turbo"
    return {
        "type": "CMK_DIAGNOSTIC",
        "version": 1,
        "title": f"{family_label} Result",
        "node": "CMK Family Branch Gate",
        "mode": "Result",
        "summary": f"{family_label} result path materialized",
        "details": (
            "Lightweight result preview generated at the family gate; "
            "the optional upstream diagnostic was not reopened."
        ),
        "metadata": {"model_family": family, "gate_fallback": True},
        "metrics": {},
        "warnings": [],
        "stages": [],
        "preview": [image],
        "images": [image],
    }


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
            "required": {"PROCESS": (cls.PROCESS_TYPE,)},
            "optional": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "IMAGE": ("IMAGE", {"lazy": True}),
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
                "diagnostic": ("CMK_DIAGNOSTIC", {"lazy": True}),
                "RESULT PROCESS": (cls.PROCESS_TYPE, {"lazy": True}),
            },
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "CMK_PROCESS_SDXL",
        "IMAGE",
        "CMK_LOG_PIPE",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG", "diagnostic")
    FUNCTION = "gate"
    CATEGORY = "CMK/Developer/Boundary & Cache"
    DEV_ONLY = True

    @cmk_timed_call("LAZY FAMILY BRANCH GATE")
    def check_lazy_status(self, PROCESS=None, **inputs):
        if PROCESS is None:
            return []
        if isinstance(PROCESS, dict) and not PROCESS.get("family_active", True):
            return []
        # diagnostic is observational only. Requesting it as a mandatory lazy
        # input can reopen the complete implementation behind a boundary hit
        # (for example Refiner Prepare -> SAMPLED -> module 10).
        needed = [
            name for name in ("MODEL", "IMAGE", "LOG")
            if inputs.get(name) is None
        ]
        if "RESULT PROCESS" in inputs and inputs.get("RESULT PROCESS") is None:
            needed.append("RESULT PROCESS")
        return needed

    def gate(self, PROCESS=None, **inputs):
        if PROCESS is None or (
            isinstance(PROCESS, dict) and not PROCESS.get("family_active", True)
        ):
            blocked = ExecutionBlocker(None)
            return (blocked, PROCESS, blocked, blocked, blocked)
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Family Branch Gate requires a CMK process pipe")
        result_process = inputs.get("RESULT PROCESS") or PROCESS
        if not isinstance(result_process, dict):
            raise TypeError("CMK Family Branch Gate requires a valid result PROCESS")
        actual = str(result_process.get("model_family", self.FAMILY)).strip().lower()
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
            result_process,
            inputs["IMAGE"],
            inputs["LOG"],
            _gate_diagnostic(
                inputs.get("diagnostic"), inputs["IMAGE"], self.FAMILY
            ),
        )


class CMKFamilyBranchGateSDXL(_CMKFamilyBranchGate):
    pass


class CMKFamilyBranchGateZImage(_CMKFamilyBranchGate):
    PROCESS_TYPE = "CMK_PROCESS_Z_IMAGE"
    FAMILY = "z_image_turbo"
    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "CMK_PROCESS_Z_IMAGE",
        "IMAGE",
        "CMK_LOG_PIPE",
        "CMK_DIAGNOSTIC",
    )


class CMKFamilyBranchGateSDXLSampled(_CMKFamilyBranchGate):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"PROCESS": (cls.PROCESS_TYPE,)},
            "optional": {
                "MODEL": ("CMK_MODEL_PIPE", {"lazy": True}),
                "SAMPLED": ("CMK_SAMPLED_PIPE", {"lazy": True}),
                "LOG": ("CMK_LOG_PIPE", {"lazy": True}),
                "diagnostic": ("CMK_DIAGNOSTIC", {"lazy": True}),
            },
        }

    RETURN_TYPES = (
        "CMK_MODEL_PIPE",
        "CMK_PROCESS_SDXL",
        "CMK_SAMPLED_PIPE",
        "CMK_LOG_PIPE",
        "CMK_DIAGNOSTIC",
    )
    RETURN_NAMES = ("MODEL", "PROCESS", "SAMPLED", "LOG", "diagnostic")

    @cmk_timed_call("LAZY SDXL SAMPLED GATE")
    def check_lazy_status(self, PROCESS=None, **inputs):
        if PROCESS is None:
            return []
        if isinstance(PROCESS, dict) and not PROCESS.get("family_active", True):
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

    def gate(self, PROCESS=None, **inputs):
        if PROCESS is None or (
            isinstance(PROCESS, dict) and not PROCESS.get("family_active", True)
        ):
            blocked = ExecutionBlocker(None)
            return (blocked, PROCESS, blocked, blocked, blocked)
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK SDXL Sampler Branch Gate requires a CMK process pipe")
        actual = str(PROCESS.get("model_family", "sdxl")).strip().lower()
        if actual != "sdxl":
            raise ValueError("CMK SDXL Sampler Branch Gate received the wrong model family")
        missing = [
            name for name in ("MODEL", "SAMPLED", "LOG")
            if inputs.get(name) is None
        ]
        if missing:
            raise ValueError(
                "CMK SDXL Sampler Branch Gate is missing " + ", ".join(missing)
            )
        return (
            inputs["MODEL"], PROCESS, inputs["SAMPLED"], inputs["LOG"],
            _gate_diagnostic(
                inputs.get("diagnostic"), inputs["SAMPLED"].get("image"), self.FAMILY
            ),
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
                "MODEL ZIT": ("CMK_MODEL_PIPE", {"lazy": True}),
                "PROCESS ZIT": ("CMK_PROCESS_Z_IMAGE", {"lazy": True}),
                "IMAGE ZIT": ("IMAGE", {"lazy": True}),
                "LOG ZIT": ("CMK_LOG_PIPE", {"lazy": True}),
            },
        }

    RETURN_TYPES = (
        "CMK_RESULT_MODEL",
        "CMK_RESULT_PROCESS",
        "CMK_RESULT_IMAGE",
        "CMK_RESULT_LOG",
    )
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG")
    FUNCTION = "merge"
    CATEGORY = "CMK/Flow/Finish"

    @cmk_timed_call("LAZY FAMILY RESULT MERGE")
    def check_lazy_status(self, **inputs):
        process_sdxl = inputs.get("PROCESS SDXL")
        process_z = inputs.get("PROCESS ZIT")
        missing_process = [
            name for name, value in (
                ("PROCESS SDXL", process_sdxl),
                ("PROCESS ZIT", process_z),
            ) if value is None
        ]
        if missing_process:
            return missing_process
        active_sdxl = isinstance(process_sdxl, dict) and process_sdxl.get("family_active", True)
        active_z = isinstance(process_z, dict) and process_z.get("family_active", True)
        if not active_sdxl and not active_z:
            return []
        suffix = "SDXL" if active_sdxl else "ZIT"
        return [
            f"{name} {suffix}"
            for name in ("MODEL", "IMAGE", "LOG")
            if inputs.get(f"{name} {suffix}") is None
        ]

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
        return (model, result, image, log)


class CMKZImageProcessForwardPipe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"PROCESS": ("CMK_PROCESS_Z_IMAGE",)},
            "optional": {
                "RESULT PROCESS": ("CMK_PROCESS_Z_IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("CMK_PROCESS_Z_IMAGE",)
    RETURN_NAMES = ("PROCESS",)
    FUNCTION = "forward"
    CATEGORY = "CMK/Developer/Pipe"
    DEV_ONLY = True

    def check_lazy_status(self, PROCESS=None, **inputs):
        if PROCESS is None or not isinstance(PROCESS, dict):
            return []
        if not PROCESS.get("family_active", True):
            return []
        if inputs.get("RESULT PROCESS") is None:
            return ["RESULT PROCESS"]
        return []

    @staticmethod
    def forward(PROCESS, **inputs):
        if PROCESS is None:
            return (None,)
        if not isinstance(PROCESS, dict):
            raise TypeError("CMK Z-Image Process Forward requires a process pipe")
        if not PROCESS.get("family_active", True):
            return (PROCESS,)
        result = inputs.get("RESULT PROCESS")
        if not isinstance(result, dict):
            raise ValueError("CMK Z-Image Process Forward requires RESULT PROCESS for the active ZIT path")
        return (result,)


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
            "required": {
                "PROCESS": (CMK_FINISH_INPUT,),
                "IMAGE": (CMK_FINISH_INPUT,),
                "LOG": (CMK_FINISH_INPUT,),
            },
            "optional": {
                "MODEL": (CMK_FINISH_INPUT,),
            },
        }

    RETURN_TYPES = ("CMK_MODEL_PIPE", "CMK_PIPE", "IMAGE", "CMK_LOG_PIPE")
    RETURN_NAMES = ("MODEL", "PROCESS", "IMAGE", "LOG")
    FUNCTION = "unpack"
    CATEGORY = "CMK/Developer/Boundary & Cache"
    DEV_ONLY = True

    @staticmethod
    def unpack(MODEL=None, PROCESS=None, IMAGE=None, LOG=None):
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
            "required": {
                "PROCESS": ("CMK_PIPE",),
                "IMAGE": ("IMAGE",),
                "LOG": ("CMK_LOG_PIPE",),
            },
            "optional": {
                "MODEL": ("CMK_MODEL_PIPE",),
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
    def pack(MODEL=None, PROCESS=None, IMAGE=None, LOG=None):
        if not isinstance(PROCESS, dict) or PROCESS.get("result_contract") != "family_neutral":
            raise ValueError("CMK Result Pack requires a family-neutral PROCESS")
        if MODEL is not None and not isinstance(MODEL, dict):
            raise TypeError("CMK Result Pack received an invalid MODEL")
        if IMAGE is None or not isinstance(LOG, dict):
            raise TypeError("CMK Result Pack received an incomplete result")
        return (MODEL, dict(PROCESS), IMAGE, LOG)
