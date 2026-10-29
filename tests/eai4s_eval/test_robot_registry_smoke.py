from __future__ import annotations

import unittest

from eai4s_eval.robots.base import RobotAdapterNotConfigured
from eai4s_eval.robots.registry import available_robots, get_robot_adapter


class RobotRegistrySmokeTest(unittest.TestCase):
    def test_franka_and_ur5_are_registered(self) -> None:
        robots = available_robots()
        self.assertIn("franka_panda", robots)
        self.assertIn("ur5_single", robots)

    def test_franka_adapter_is_configured(self) -> None:
        adapter = get_robot_adapter("franka_panda")
        self.assertEqual(adapter.robot_id, "franka_panda")
        self.assertTrue(adapter.is_configured)
        self.assertTrue(adapter.joint_names)
        self.assertTrue(adapter.ee_body_name)

    def test_ur5_stub_fails_with_clear_configuration_error(self) -> None:
        adapter = get_robot_adapter("ur5_single")
        self.assertFalse(adapter.is_configured)
        with self.assertRaises(RobotAdapterNotConfigured) as ctx:
            adapter.build_robot_cfg("{ENV_REGEX_NS}/Robot")
        message = str(ctx.exception)
        self.assertIn("ur5_single", message)
        self.assertIn("not configured", message)
        self.assertIn("ur5_single.py", message)


if __name__ == "__main__":
    unittest.main()
