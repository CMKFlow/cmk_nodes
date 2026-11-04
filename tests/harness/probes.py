"""Non-invasive instrumentation for the execution harness.

Probes wrap existing callables, forward the identical arguments and return the
original result unchanged.  They never touch ``IS_CHANGED``,
``fingerprint_inputs``, ``INPUT_TYPES``, ``RETURN_TYPES`` or
``NOT_IDEMPOTENT`` and never add attributes ComfyUI probes with ``hasattr``,
so cache signatures and lazy behaviour stay exactly as in production.
"""

from __future__ import annotations

import functools
import inspect
import sys
from collections import defaultdict


_ABSENT = object()


class Probes:
    def __init__(self):
        self.function_calls = defaultdict(list)   # class_type -> [unique_id]
        self.lazy_requests = defaultdict(list)    # class_type -> [(unique_id, requested)]
        self.statuses = []                        # (scope, status, cache_key)
        self.seam_calls = defaultdict(int)        # seam name -> count
        self._restore = []

    # -- installation -----------------------------------------------------
    def _replace(self, owner, name, raw, replacement):
        # Remember whether the owner defined the attribute itself, so that
        # uninstall restores inheritance instead of pinning a copy.
        present = name in getattr(owner, "__dict__", {})
        self._restore.append((owner, name, owner.__dict__[name] if present else _ABSENT))
        setattr(owner, name, replacement)

    def wrap_node_class(self, class_type: str, cls) -> None:
        function_name = getattr(cls, "FUNCTION", None)
        if function_name:
            self._wrap_method(cls, function_name, self._record_function(class_type))
        if inspect.getattr_static(cls, "check_lazy_status", None) is not None:
            self._wrap_method(cls, "check_lazy_status", self._record_lazy(class_type))

    def _wrap_method(self, cls, name, recorder) -> None:
        raw = inspect.getattr_static(cls, name)
        # A subclass may inherit a method that a base-class probe already
        # wrapped; always wrap the unwrapped original so each call is
        # recorded exactly once, under the class that was executed.
        inner = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw
        original = getattr(inner, "_cmk_probe_original", None)
        if original is not None:
            raw = staticmethod(original) if isinstance(raw, staticmethod) else (
                classmethod(original) if isinstance(raw, classmethod) else original)
        owner = next((klass for klass in cls.__mro__ if name in klass.__dict__), cls)
        if owner is not cls:
            # Wrap on the concrete class only; do not alter shared base classes.
            owner = cls
        if isinstance(raw, staticmethod):
            target = raw.__func__

            @functools.wraps(target)
            def static_wrapper(*args, **kwargs):
                result = target(*args, **kwargs)
                recorder(kwargs, result)
                return result

            static_wrapper._cmk_probe_original = target
            self._replace(owner, name, cls.__dict__.get(name, raw), staticmethod(static_wrapper))
        elif isinstance(raw, classmethod):
            target = raw.__func__

            @functools.wraps(target)
            def class_wrapper(klass, *args, **kwargs):
                result = target(klass, *args, **kwargs)
                recorder(kwargs, result)
                return result

            class_wrapper._cmk_probe_original = target
            self._replace(owner, name, cls.__dict__.get(name, raw), classmethod(class_wrapper))
        else:
            target = raw

            @functools.wraps(target)
            def method_wrapper(instance, *args, **kwargs):
                result = target(instance, *args, **kwargs)
                recorder(kwargs, result)
                return result

            method_wrapper._cmk_probe_original = target
            self._replace(owner, name, cls.__dict__.get(name, raw), method_wrapper)

    def _record_function(self, class_type):
        def record(kwargs, _result):
            self.function_calls[class_type].append(str(kwargs.get("unique_id", "")))
        return record

    def _record_lazy(self, class_type):
        def record(kwargs, result):
            self.lazy_requests[class_type].append(
                (str(kwargs.get("unique_id", "")), list(result or []))
            )
        return record

    def wrap_module_function(self, modules, function_name: str, recorder_factory) -> None:
        """Wrap one module-level function in every module that bound it."""
        original = None
        for module in modules:
            candidate = getattr(module, function_name, None)
            if callable(candidate) and getattr(candidate, "__module__", "").endswith("cmk_persistent_cache"):
                original = candidate
                break
        if original is None:
            raise LookupError(function_name)
        wrapper = recorder_factory(original)
        for module in list(sys.modules.values()):
            namespace = getattr(module, "__dict__", None)
            if not namespace:
                continue
            for key, value in list(namespace.items()):
                if value is original:
                    self._replace(module, key, value, wrapper)

    def wrap_write_status(self, modules) -> None:
        def factory(original):
            @functools.wraps(original)
            def write_status(scope, status, **fields):
                self.statuses.append((str(scope), str(status), fields.get("cache_key")))
                return original(scope, status, **fields)
            return write_status
        self.wrap_module_function(modules, "write_status", factory)

    def wrap_seam(self, owner, name: str, seam: str) -> None:
        """Count calls of an expensive engine entry point without replacing it."""
        raw = inspect.getattr_static(owner, name)
        target = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw

        @functools.wraps(target)
        def seam_wrapper(*args, **kwargs):
            self.seam_calls[seam] += 1
            return target(*args, **kwargs)

        replacement = seam_wrapper
        if isinstance(raw, staticmethod):
            replacement = staticmethod(seam_wrapper)
        elif isinstance(raw, classmethod):
            replacement = classmethod(seam_wrapper)
        self._replace(owner, name, raw, replacement)

    def uninstall(self) -> None:
        while self._restore:
            owner, name, raw = self._restore.pop()
            if raw is _ABSENT:
                delattr(owner, name)
            else:
                setattr(owner, name, raw)

    # -- reporting --------------------------------------------------------
    def calls(self, class_type: str) -> int:
        return len(self.function_calls.get(class_type, []))

    def snapshot(self) -> dict:
        return {
            "function_calls": {key: list(value) for key, value in self.function_calls.items()},
            "lazy_requests": {key: [list(item) for item in value] for key, value in self.lazy_requests.items()},
            "statuses": [list(item) for item in self.statuses],
            "seam_calls": dict(self.seam_calls),
        }


def install_probes(runtime, class_types=None) -> Probes:
    import nodes

    probes = Probes()
    selected = class_types or [name for name in nodes.NODE_CLASS_MAPPINGS if name.startswith("CMK")]
    for class_type in selected:
        probes.wrap_node_class(class_type, nodes.NODE_CLASS_MAPPINGS[class_type])
    probes.wrap_write_status(runtime.cmk_modules.values())
    return probes
