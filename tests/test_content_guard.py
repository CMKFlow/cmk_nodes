import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "engine" / "content_guard.py"
SPEC = importlib.util.spec_from_file_location("cmk_content_guard_test", MODULE_PATH)
CONTENT_GUARD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTENT_GUARD
SPEC.loader.exec_module(CONTENT_GUARD)


class ContentGuardRoleAgeTests(unittest.TestCase):
    @staticmethod
    def face(age):
        return SimpleNamespace(age=age)

    def assert_blocked(self, age, role, code):
        with self.assertRaises(CONTENT_GUARD.ContentGuardBlocked) as caught:
            CONTENT_GUARD._estimated_age(self.face(age), role)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.role, role)

    def test_source_remains_conservative_at_25(self):
        self.assert_blocked(24, "source", "CG_AGE_UNCERTAIN")
        self.assertEqual(
            CONTENT_GUARD._estimated_age(self.face(25), "source"),
            25,
        )

    def test_adult_target_is_allowed_from_18(self):
        self.assertEqual(
            CONTENT_GUARD._estimated_age(self.face(18), "target"),
            18,
        )
        self.assertEqual(
            CONTENT_GUARD._estimated_age(self.face(24), "target"),
            24,
        )

    def test_minors_remain_blocked_for_both_roles(self):
        for role in ("source", "target"):
            with self.subTest(role=role):
                self.assert_blocked(17, role, "CG_AGE_MINOR")

    def test_missing_and_invalid_ages_remain_fail_closed(self):
        for value, code in (
            (None, "CG_AGE_UNAVAILABLE"),
            (-1, "CG_AGE_INVALID"),
            (121, "CG_AGE_INVALID"),
        ):
            with self.subTest(value=value):
                self.assert_blocked(value, "target", code)

    def test_guard_version_records_contract_change(self):
        self.assertIn("role-age-thresholds", CONTENT_GUARD.GUARD_VERSION)


if __name__ == "__main__":
    unittest.main()
