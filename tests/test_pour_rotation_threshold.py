import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.task_utils import TaskUtils


class PourRotationThresholdTest(unittest.TestCase):
    def setUp(self):
        self.task_utils = TaskUtils.get_instance()
        self.upright = np.array([0.0, 0.0, 0.0, 1.0])

    def test_rotation_angle_is_quaternion_sign_invariant(self):
        tilted = R.from_euler("x", 70.0, degrees=True).as_quat()
        self.assertAlmostEqual(
            self.task_utils.rotation_angle_degrees(self.upright, tilted), 70.0, places=6
        )
        self.assertAlmostEqual(
            self.task_utils.rotation_angle_degrees(self.upright, -tilted), 70.0, places=6
        )

    def test_pour_and_return_threshold_boundaries(self):
        tilted = R.from_euler("x", 70.0, degrees=True).as_quat()
        nearly_upright = R.from_euler("x", 15.0, degrees=True).as_quat()
        self.assertTrue(self.task_utils.check_rotation_angle(self.upright, tilted, 69.9))
        self.assertFalse(self.task_utils.check_rotation_angle(self.upright, tilted, 70.1))
        self.assertLessEqual(
            self.task_utils.rotation_angle_degrees(self.upright, nearly_upright), 15.0
        )


if __name__ == "__main__":
    unittest.main()
