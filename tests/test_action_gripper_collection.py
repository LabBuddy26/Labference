import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from controllers.gripper_action_tracker import GripperActionTracker
from data_collectors.data_collector import DataCollector


class _Action:
    def __init__(self, joint_positions, joint_indices=None):
        self.joint_positions = joint_positions
        self.joint_indices = joint_indices


class GripperActionTrackerTest(unittest.TestCase):
    def test_explicit_commands_update_state_and_sparse_actions_hold_it(self):
        tracker = GripperActionTracker((7, 8))
        self.assertEqual(tracker.state, GripperActionTracker.OPEN)

        self.assertEqual(
            tracker.update_from_action(_Action([None] * 7 + [0.04, 0.04])),
            GripperActionTracker.OPEN,
        )
        self.assertEqual(
            tracker.update_from_action(_Action([None] * 9)),
            GripperActionTracker.OPEN,
        )
        self.assertEqual(
            tracker.update_from_action(_Action([None] * 7 + [0.02, 0.02])),
            GripperActionTracker.CLOSED,
        )
        self.assertEqual(
            tracker.update_from_action(_Action([None] * 9)),
            GripperActionTracker.CLOSED,
        )
        self.assertEqual(
            tracker.update_from_action(_Action([0.05, 0.05], joint_indices=[7, 8])),
            GripperActionTracker.OPEN,
        )


class DataCollectorActionGripperTest(unittest.TestCase):
    def test_hdf5_stores_shifted_binary_gripper_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = DataCollector(
                camera_configs=[{"name": "camera", "image_type": "rgb"}],
                save_dir=directory,
                max_episodes=1,
                max_workers=1,
            )
            poses = [
                np.full(8, 0.0, dtype=np.float32),
                np.full(8, 1.0, dtype=np.float32),
                np.full(8, 2.0, dtype=np.float32),
            ]
            subtasks = ["Pick up the vial", "Pick up the vial", "Place the vial"]
            for pose, action_gripper, subtask in zip(poses, (1, 1, 0), subtasks):
                collector.cache_step(
                    camera_images={"camera_rgb": np.zeros((3, 4, 5), dtype=np.uint8)},
                    joint_angles=pose,
                    action_gripper=action_gripper,
                    subtask=subtask,
                    language_instruction="Pick and place the vial",
                )

            final_joint_positions = np.full(8, 3.0, dtype=np.float32)
            collector.write_cached_data(final_joint_positions)
            collector.close()

            episode_path = Path(directory) / "dataset" / "episode_0000.h5"
            with h5py.File(episode_path, "r") as episode:
                np.testing.assert_array_equal(episode["actions"][:], np.vstack([poses[1:], final_joint_positions]))
                np.testing.assert_array_equal(episode["action_gripper"][:], np.array([1, 0, 0], dtype=np.uint8))
                self.assertEqual(episode["action_gripper"].dtype, np.dtype("uint8"))
                self.assertEqual(episode["action_gripper"].shape, (len(poses),))
                self.assertEqual(episode["agent_pose"].shape[0], episode["action_gripper"].shape[0])
                self.assertEqual(
                    episode["subtask"].asstr()[:].tolist(),
                    subtasks,
                )
                self.assertEqual(episode["subtask"].shape, (len(poses),))
                self.assertEqual(
                    episode["language_instruction"].asstr()[()],
                    "Pick and place the vial",
                )

    def test_rejects_non_binary_gripper_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = DataCollector(
                camera_configs=[],
                save_dir=directory,
                max_workers=1,
            )
            with self.assertRaises(ValueError):
                collector.cache_step(
                    {}, np.zeros(8, dtype=np.float32), action_gripper=2, subtask="Pick"
                )
            with self.assertRaises(ValueError):
                collector.cache_step(
                    {}, np.zeros(8, dtype=np.float32), action_gripper=1, subtask="  "
                )
            collector.close()


if __name__ == "__main__":
    unittest.main()
