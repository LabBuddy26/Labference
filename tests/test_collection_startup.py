import unittest

import numpy as np

from utils.collection_startup import (
    CollectionStartupConfig,
    CollectionStartupGate,
    task_startup_readiness,
)


class CollectionStartupGateTest(unittest.TestCase):
    def test_waits_for_warmup_then_consecutive_stable_frames(self):
        gate = CollectionStartupGate(
            CollectionStartupConfig(
                warmup_steps=2,
                stable_frames=3,
                joint_velocity_threshold=0.02,
                object_displacement_threshold=0.001,
                max_wait_steps=10,
            )
        )
        self.assertFalse(gate.observe([0.0] * 9, [[0.0, 0.0, 0.0]]).ready)
        self.assertFalse(gate.observe([0.0] * 9, [[0.0, 0.0, 0.0]]).ready)
        self.assertFalse(gate.observe([0.0] * 9, [[0.0, 0.0, 0.0]]).ready)
        ready = gate.observe([0.0] * 9, [[0.0, 0.0, 0.0]])
        self.assertTrue(ready.ready)
        self.assertEqual(ready.stable_frames, 3)

    def test_motion_resets_stability_window(self):
        gate = CollectionStartupGate(
            CollectionStartupConfig(warmup_steps=0, stable_frames=2, joint_velocity_threshold=0.02, max_wait_steps=10)
        )
        gate.observe([0.0] * 9, [[0.0, 0.0, 0.0]])
        moving = gate.observe([0.05] + [0.0] * 8, [[0.0, 0.0, 0.0]])
        self.assertFalse(moving.ready)
        self.assertEqual(moving.stable_frames, 0)
        gate.observe([0.0] * 9, [[0.0, 0.0, 0.0]])
        self.assertTrue(gate.observe([0.0] * 9, [[0.0, 0.0, 0.0]]).ready)

    def test_object_motion_resets_stability_window_and_timeout_is_reported(self):
        gate = CollectionStartupGate(
            CollectionStartupConfig(
                warmup_steps=0,
                stable_frames=3,
                object_displacement_threshold=0.001,
                max_wait_steps=3,
            )
        )
        gate.observe(np.zeros(9), [[0.0, 0.0, 0.0]])
        moving = gate.observe(np.zeros(9), [[0.01, 0.0, 0.0]])
        self.assertEqual(moving.stable_frames, 0)
        timeout = gate.observe(np.zeros(9), [[0.01, 0.0, 0.0]])
        self.assertTrue(timeout.timed_out)

    def test_requires_nine_ready_joint_values(self):
        with self.assertRaises(ValueError):
            CollectionStartupConfig.from_mapping({"ready_joint_positions": [0.0] * 7})

    def test_task_specific_geometry_can_reject_stationary_startup(self):
        class Task:
            def get_startup_readiness(self):
                return False, "tube is outside source socket"

        ready, detail = task_startup_readiness(Task())
        self.assertFalse(ready)
        self.assertIn("outside source socket", detail)

    def test_task_without_geometry_predicate_remains_compatible(self):
        ready, detail = task_startup_readiness(object())
        self.assertTrue(ready)
        self.assertEqual(detail, "")


if __name__ == "__main__":
    unittest.main()
