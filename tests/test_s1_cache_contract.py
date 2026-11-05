"""S1 §3.B — cache contracts (first increment: persistence and scope registry).

Contract tests assert the target contract.  Known violations are marked with
``expectedFailure`` and the finding ID, and each has a ``test_baseline_*``
partner that pins the exact current misbehaviour, so the contract test cannot
fail for an undocumented reason.  A fixed defect turns the contract test into
an unexpected success, which unittest reports as a failure.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from harness.restart import run_scenario

CMK_TEST_TIER = "T1"


class _Workdir(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="cmk-s1-cache-"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


class BoundaryPersistence(_Workdir):
    """Finding H-1 (found by the S1 harness, fixed in the boundary serializer):
    the Upscale & Save diagnostic aliases its preview tensors (``preview`` and
    ``images``), which safetensors rejects as shared memory.  Before the fix the
    boundary reported STORE_FAILED and never produced a restart HIT."""

    def test_h1_boundary_stores_diagnostic_with_shared_tensors(self):
        result = run_scenario("selfcheck_chain", self.root / "h1", runs=1)
        self.assertTrue(result["runs"][0]["succeeded"])
        self.assertEqual(result["runs"][0]["boundary_status"], "MISS_STORED")

    def test_boundary_result_persists_across_restart(self):
        workdir = self.root / "persist"
        run_scenario("selfcheck_chain", workdir, runs=1)
        restarted = run_scenario("selfcheck_chain", workdir, runs=1)
        self.assertEqual(restarted["runs"][0]["boundary_status"], "HIT")


class ScopeRegistry(_Workdir):
    """Target A9 / finding B4 and Phase 3 A3.2: every persistent scope is
    registered (so Clear Persistent Cache reaches it) and no scope is shared by
    two independent caches."""

    @classmethod
    def setUpClass(cls):
        root = Path(tempfile.mkdtemp(prefix="cmk-s1-scope-"))
        try:
            cls.inventory = run_scenario("scope_inventory", root)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def _unregistered(self):
        declared = set(self.inventory["declared"]) - {self.inventory["completion_scope"]}
        return declared - set(self.inventory["registered"])

    def _shared(self):
        owners = {}
        for scope, names in self.inventory["declared"].items():
            # one owner = one module/class; aliases inside a module count once
            owners[scope] = sorted({name.split(".")[0] for name in names})
        return {scope: names for scope, names in owners.items() if len(names) > 1}

    def test_baseline_b4_unregistered_scopes(self):
        self.assertEqual(self._unregistered(), {"image_resize_boundary"})

    @unittest.expectedFailure  # B4
    def test_every_declared_scope_is_registered(self):
        self.assertEqual(self._unregistered(), set())

    # A3.2 resolved: the Hybrid SDXL boundary has its own scope
    # "sdxl_hybrid_boundary"; "sdxl_handoff_boundary" belongs to InstantID only.
    def test_no_scope_is_shared_between_caches(self):
        self.assertEqual(self._shared(), {})

    def test_hybrid_and_instantid_handoff_scopes_are_separate_and_registered(self):
        owners = {scope: sorted({name.split(".")[0] for name in names})
                  for scope, names in self.inventory["declared"].items()}
        self.assertEqual(owners.get("sdxl_hybrid_boundary"), ["cmk_hybrid_bridge"])
        self.assertEqual(owners.get("sdxl_handoff_boundary"), ["cmk_pipe_sampler"])
        self.assertIn("sdxl_hybrid_boundary", self.inventory["registered"])

    def test_registered_scopes_are_declared(self):
        # Every registered scope must belong to a cache that actually writes it.
        self.assertEqual(set(self.inventory["registered"]) - set(self.inventory["declared"]), set())


if __name__ == "__main__":
    unittest.main()
