"""CMK Multi Image Loader: four references, four independent backend paths.

The node exists only in the frontend.  The fixtures are API prompts that the
real ComfyUI frontend produced from the instantid_inpaint workflow with one
Multi Image Loader: CN -> Load Image 8437 (module 01 source), ID A -> the
InstantID module (inner Load Image 8438:7807), ID B -> 8444, ID C -> 8445.
Variants change ID C, deactivate ID B, change ID A, or only the selected slot.
"""

from __future__ import annotations

import unittest

from harness.support import ScenarioCase

CMK_TEST_TIER = "T1"
CN, ID_A, ID_B, ID_C = "8437", "8438:7807", "8444", "8445"
BOUNDARIES = ("handoff", "instantid_boundary", "refiner")


class MultiImageLoader(ScenarioCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.keys = cls.run_case("multi_image_loader_keys")

    def changed(self, variant):
        # The Visualizer (8334) runs on every prompt by design (IS_CHANGED), so
        # its key never repeats and is left out.
        base, other = self.keys["base"]["comfy"], self.keys[variant]["comfy"]
        return sorted(node for node in base if node != "8334" and base[node] != other[node])

    def test_no_shared_ancestor_in_the_prompt(self):
        references = self.keys["base"]["references"]
        self.assertEqual(references, {
            CN: "CMK Package · inpaint_reference2.png",
            ID_A: "CMK Package · face_identity_reference.png",
            ID_B: "CMK Package · face_reference.png",
            ID_C: "CMK Package · portrait_reference_00002.png",
        })

    def test_id_c_changes_only_its_consumer(self):
        self.assertEqual(self.changed("id_c_changed"), [ID_C])
        for boundary in BOUNDARIES:
            self.assertEqual(self.keys["id_c_changed"][boundary], self.keys["base"][boundary], boundary)

    def test_selected_slot_changes_nothing(self):
        self.assertEqual(self.changed("ui_selected"), [])

    def test_disabled_slot_affects_only_its_consumer(self):
        self.assertEqual(self.changed("id_b_disabled"), [ID_B])
        self.assertEqual(self.keys["id_b_disabled"]["references"][ID_B], "")

    def test_id_a_changes_the_identity_path_only(self):
        changed = self.changed("id_a_changed")
        self.assertIn(ID_A, changed)
        for node in (CN, ID_B, ID_C, "8430:7415", "8332"):
            self.assertNotIn(node, changed)
        self.assertEqual(self.keys["id_a_changed"]["handoff"], self.keys["base"]["handoff"])
        self.assertNotEqual(self.keys["id_a_changed"]["instantid_boundary"], self.keys["base"]["instantid_boundary"])
        self.assertNotEqual(self.keys["id_a_changed"]["refiner"], self.keys["base"]["refiner"])

    def test_boundary_keys_are_computed(self):
        for boundary in BOUNDARIES:
            self.assertTrue(self.keys["base"][boundary], boundary)

    def test_file_content_still_reaches_the_consumer(self):
        first, second = self.keys["content_digests"]
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
