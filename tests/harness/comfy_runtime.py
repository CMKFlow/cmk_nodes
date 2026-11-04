"""Boot ComfyUI in-process and execute API prompts with the real executor.

The runtime is isolated: input, output, temp and user directories point to a
private work directory, and the package-local generation completion cache is
redirected there as well.  The real caches of the installation are never read,
written or cleaned.  Execution uses ``execution.PromptExecutor`` unchanged.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent
TESTS_ROOT = HARNESS_ROOT.parent
PACKAGE_ROOT = TESTS_ROOT.parent
COMFY_ROOT = PACKAGE_ROOT.parents[1]

_RUNTIME = None


@dataclass
class RunRecord:
    prompt_id: str
    executed: list[str]
    cached: list[str]
    ui: dict[str, dict]
    errors: list[dict]
    events: list[tuple[str, object]] = field(repr=False)

    @property
    def succeeded(self) -> bool:
        return not self.errors

    @property
    def entered_nodes(self) -> list[str]:
        """Nodes the executor entered, in first-entry order.

        ``executed`` holds the raw ``executing`` events; lazy nodes re-enter
        once per pending input, so entries repeat.  Entering a node is not the
        same as running its FUNCTION; use the probes for actual executions.
        """
        return list(dict.fromkeys(self.executed))


class Runtime:
    def __init__(self, workdir: Path, server, cmk_modules):
        self.workdir = workdir
        self.server = server
        self.cmk_modules = cmk_modules
        self.executor = None

    def new_executor(self, mode: str = "classic"):
        """Create an executor configured exactly like ComfyUI's ``main.py``.

        ``mode`` is ``"classic"`` (--cache-classic) or ``"ram_pressure"`` (the
        ComfyUI default).  The cache arguments mirror ``main.prompt_worker``.
        """
        import comfy.model_management
        import execution

        if mode == "classic":
            cache_type = execution.CacheType.CLASSIC
            cache_ram, cache_ram_inactive = 0, 0
        elif mode == "ram_pressure":
            cache_type = execution.CacheType.RAM_PRESSURE
            total = comfy.model_management.total_ram
            cache_ram = min(10.0, max(2.0, total * 0.10 / 1024.0))
            cache_ram_inactive = min(128.0, total / 1024.0)
        else:
            raise ValueError(f"unknown cache mode {mode!r}")
        self.mode = mode
        self.executor = execution.PromptExecutor(
            self.server,
            cache_type=cache_type,
            cache_args={"lru": 0, "ram": cache_ram, "ram_inactive": cache_ram_inactive},
        )
        return self.executor

    def validate(self, prompt: dict):
        import execution

        return asyncio.run(execution.validate_prompt(str(uuid.uuid4()), prompt, None))

    def run(self, prompt: dict, outputs: list[str] | None = None) -> RunRecord:
        if self.executor is None:
            self.new_executor()
        # ComfyUI writes "is_changed" back into the prompt dict.  The frontend
        # sends a fresh prompt per queue, so every run gets its own deep copy.
        prompt = copy.deepcopy(prompt)
        valid = self.validate(prompt)
        if not valid[0]:
            raise AssertionError(f"prompt validation failed: {valid[1]} {valid[3]}")
        execute_outputs = outputs if outputs is not None else valid[2]
        prompt_id = str(uuid.uuid4())
        self.server.clear()
        self.executor.execute(
            prompt, prompt_id, {"client_id": "cmk-harness"}, execute_outputs
        )
        events = list(self.server.events)
        executed = [
            str(data["node"]) for name, data in events
            if name == "executing" and isinstance(data, dict) and data.get("node") is not None
        ]
        cached = []
        for data in self.server.of("execution_cached"):
            cached.extend(str(node) for node in data.get("nodes", []))
        ui = {
            str(data["node"]): data.get("output") or {}
            for data in self.server.of("executed")
        }
        errors = [data for data in self.server.of("execution_error")]
        return RunRecord(prompt_id, executed, cached, ui, errors, events)

    def output(self, node_id: str):
        """Return the cached output tuple of a node after the last run."""
        entry = asyncio.run(self.executor.caches.outputs.get(str(node_id)))
        return None if entry is None else entry.outputs

    def force_ram_release(self):
        """Evict every evictable output through ComfyUI's own RAM-pressure path."""
        release = getattr(self.executor.caches.outputs, "ram_release", None)
        if release is None:
            raise RuntimeError("executor does not use the RAM pressure cache")
        return release(float("inf"), free_active=True)

    def cmk_module(self, dotted_suffix: str):
        """Return the loaded CMK module whose name ends with ``dotted_suffix``."""
        for name, module in self.cmk_modules.items():
            if name.endswith(dotted_suffix):
                return module
        raise KeyError(dotted_suffix)


def tensor_hash(value) -> str:
    import torch

    if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
        value = value[0]
    tensor = value.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def boot(workdir: Path, extra_model_paths: str | None = None) -> Runtime:
    """Boot once per process; repeated calls return the same runtime.

    ``extra_model_paths`` is the YAML the target ComfyUI instance starts with
    (``--extra-model-paths-config``).  It is only read, so combo values such as
    checkpoint names validate against the same model lists as in that instance.
    """
    global _RUNTIME
    if _RUNTIME is not None:
        if _RUNTIME.workdir != workdir:
            raise RuntimeError("the harness runtime is already bound to another work directory")
        return _RUNTIME

    workdir = Path(workdir)
    for name in ("input", "output", "temp", "user", "package"):
        (workdir / name).mkdir(parents=True, exist_ok=True)

    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))

    from comfy.cli_args import args

    # Deterministic CPU execution; no GPU work and no model loading is intended.
    args.cpu = True
    args.disable_all_custom_nodes = False

    import folder_paths

    folder_paths.set_input_directory(str(workdir / "input"))
    folder_paths.set_output_directory(str(workdir / "output"))
    folder_paths.set_temp_directory(str(workdir / "temp"))
    folder_paths.set_user_directory(str(workdir / "user"))
    if extra_model_paths:
        from utils.extra_config import load_extra_path_config

        load_extra_path_config(extra_model_paths)

    import server

    from .fake_server import FakeServer

    fake = FakeServer()
    server.PromptServer.instance = fake

    import nodes

    asyncio.run(nodes.init_builtin_extra_nodes())
    before = set(sys.modules)
    loaded = asyncio.run(nodes.load_custom_node(str(PACKAGE_ROOT)))
    if not loaded:
        raise RuntimeError("cmk-flow could not be loaded as a custom node")
    package_prefix = str(PACKAGE_ROOT)
    cmk_modules = {
        name: module for name, module in sys.modules.items()
        if name not in before
        and str(getattr(module, "__file__", "") or "").startswith(package_prefix)
    }

    runtime = Runtime(workdir, fake, cmk_modules)
    # The generation completion cache is package-local; keep it inside workdir.
    completion = runtime.cmk_module("pipe.cmk_generation_completion_cache")
    completion._PACKAGE_ROOT = workdir / "package"
    _RUNTIME = runtime
    return runtime
