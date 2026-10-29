import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from data_collectors.data_collector import BackgroundWriteError, DataCollector


def _cache_one_step(collector: DataCollector, value: float) -> None:
    collector.cache_step(
        camera_images={
            "camera_rgb": np.full((3, 4, 5), int(value), dtype=np.uint8)
        },
        joint_angles=np.full(8, value, dtype=np.float32),
        action_gripper=1,
        subtask=f"step {value}",
        language_instruction="background writer test",
    )


class BackgroundDataWriterTest(unittest.TestCase):
    def _collector(self, directory, *, max_episodes=5, queue_capacity=2):
        return DataCollector(
            camera_configs=[{"name": "camera", "image_type": "rgb"}],
            save_dir=directory,
            max_episodes=max_episodes,
            queue_capacity=queue_capacity,
        )

    def test_completed_episode_is_detached_from_next_active_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = self._collector(directory)
            writer_started = threading.Event()
            allow_writer = threading.Event()
            original_persist = collector._persist_snapshot

            def blocked_persist(snapshot):
                writer_started.set()
                self.assertTrue(allow_writer.wait(timeout=5))
                original_persist(snapshot)

            collector._persist_snapshot = blocked_persist
            _cache_one_step(collector, 1)
            collector.write_cached_data(np.full(8, 2, dtype=np.float32))
            self.assertTrue(writer_started.wait(timeout=2))

            _cache_one_step(collector, 9)
            self.assertEqual(len(collector.temp_agent_pose), 1)
            np.testing.assert_array_equal(
                collector.temp_agent_pose[0], np.full(8, 9, dtype=np.float32)
            )

            collector.clear_cache()
            allow_writer.set()
            collector.close()
            with h5py.File(Path(directory) / "dataset" / "episode_0000.h5", "r") as episode:
                np.testing.assert_array_equal(
                    episode["agent_pose"][0], np.full(8, 1, dtype=np.float32)
                )

    def test_jsonl_is_appended_only_after_final_hdf5_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = self._collector(directory, max_episodes=1)
            observed = []
            original_append = collector._append_episode_metadata

            def checked_append(metadata):
                dataset = Path(directory) / "dataset"
                observed.append(
                    (
                        (dataset / "episode_0000.h5").exists(),
                        (dataset / "episode_0000.h5.tmp").exists(),
                    )
                )
                original_append(metadata)

            collector._append_episode_metadata = checked_append
            _cache_one_step(collector, 1)
            collector.write_cached_data(np.full(8, 2, dtype=np.float32))
            collector.close()
            collector.close()

            self.assertEqual(observed, [(True, False)])
            metadata_path = Path(directory) / "dataset" / "meta" / "episode.jsonl"
            records = [json.loads(line) for line in metadata_path.read_text().splitlines()]
            self.assertEqual(records[0]["episode_index"], 0)

    def test_capacity_two_queue_blocks_until_writer_makes_space(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = self._collector(directory, max_episodes=4, queue_capacity=2)
            writer_started = threading.Event()
            allow_writer = threading.Event()
            original_persist = collector._persist_snapshot

            def blocked_first_persist(snapshot):
                if snapshot.episode_index == 0:
                    writer_started.set()
                    self.assertTrue(allow_writer.wait(timeout=5))
                original_persist(snapshot)

            collector._persist_snapshot = blocked_first_persist
            _cache_one_step(collector, 0)
            collector.write_cached_data(np.full(8, 1, dtype=np.float32))
            self.assertTrue(writer_started.wait(timeout=2))
            for value in (1, 2):
                _cache_one_step(collector, value)
                collector.write_cached_data(np.full(8, value + 1, dtype=np.float32))

            enqueue_finished = threading.Event()

            def enqueue_fourth():
                _cache_one_step(collector, 3)
                collector.write_cached_data(np.full(8, 4, dtype=np.float32))
                enqueue_finished.set()

            producer = threading.Thread(target=enqueue_fourth)
            producer.start()
            time.sleep(0.1)
            self.assertFalse(enqueue_finished.is_set())

            allow_writer.set()
            producer.join(timeout=5)
            self.assertTrue(enqueue_finished.is_set())
            collector.close()
            self.assertEqual(
                len(list((Path(directory) / "dataset").glob("episode_*.h5"))), 4
            )

    def test_writer_failure_is_raised_on_next_frame_and_cleans_tmp(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = self._collector(directory, max_episodes=2)

            def failing_write(episode_path, *args, **kwargs):
                Path(episode_path).write_bytes(b"partial")
                raise OSError("injected disk failure")

            with mock.patch(
                "data_collectors.data_collector._write_episode_data",
                side_effect=failing_write,
            ):
                _cache_one_step(collector, 1)
                collector.write_cached_data(np.full(8, 2, dtype=np.float32))
                collector._write_queue.join()

            with self.assertRaisesRegex(BackgroundWriteError, "injected disk failure"):
                _cache_one_step(collector, 2)
            with self.assertRaises(BackgroundWriteError):
                collector.close()

            dataset = Path(directory) / "dataset"
            self.assertFalse((dataset / "episode_0000.h5").exists())
            self.assertFalse((dataset / "episode_0000.h5.tmp").exists())
            self.assertFalse((dataset / "meta" / "episode.jsonl").exists())

    def test_actions_ee_dataset_is_written_next_step_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = self._collector(directory, max_episodes=1)
            for value in (0, 1, 2):
                collector.cache_step(
                    camera_images={
                        "camera_rgb": np.full((3, 4, 5), value, dtype=np.uint8)
                    },
                    joint_angles=np.full(8, value, dtype=np.float32),
                    action_gripper=1 if value < 2 else 0,
                    subtask=f"step {value}",
                    language_instruction="ee writer test",
                    ee_pose=np.full(8, value + 10, dtype=np.float32),
                )
            collector.write_cached_data(
                np.full(8, 3, dtype=np.float32),
                final_ee_pose=np.full(8, 99, dtype=np.float32),
            )
            collector.close()

            with h5py.File(Path(directory) / "dataset" / "episode_0000.h5", "r") as episode:
                # actions_ee is next-step aligned like actions: [ee0, ee1, final].
                np.testing.assert_array_equal(
                    episode["actions_ee"][:],
                    np.vstack(
                        [
                            np.full(8, 11, dtype=np.float32),
                            np.full(8, 12, dtype=np.float32),
                            np.full(8, 99, dtype=np.float32),
                        ]
                    ),
                )
                # The joint actions stream keeps its historical semantics.
                np.testing.assert_array_equal(
                    episode["actions"][:],
                    np.vstack(
                        [
                            np.full(8, 1, dtype=np.float32),
                            np.full(8, 2, dtype=np.float32),
                            np.full(8, 3, dtype=np.float32),
                        ]
                    ),
                )
                np.testing.assert_array_equal(
                    episode["action_gripper"][:], np.array([1, 0, 0], dtype=np.uint8)
                )

    def test_actions_ee_is_optional_and_requires_a_final_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = self._collector(directory, max_episodes=2)
            _cache_one_step(collector, 1)
            collector.write_cached_data(np.full(8, 2, dtype=np.float32))
            collector.close()
            with h5py.File(Path(directory) / "dataset" / "episode_0000.h5", "r") as episode:
                self.assertNotIn("actions_ee", episode)

        with tempfile.TemporaryDirectory() as directory:
            collector = self._collector(directory, max_episodes=2)
            collector.cache_step(
                camera_images={"camera_rgb": np.zeros((3, 4, 5), dtype=np.uint8)},
                joint_angles=np.full(8, 0, dtype=np.float32),
                action_gripper=1,
                subtask="step",
                language_instruction="ee writer test",
                ee_pose=np.full(8, 1, dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "final_ee_pose"):
                collector.write_cached_data(np.full(8, 2, dtype=np.float32))
            collector.clear_cache()
            collector.close()

    def test_actions_ee_rejects_invalid_shape_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = self._collector(directory, max_episodes=1)
            common = {
                "camera_images": {
                    "camera_rgb": np.zeros((3, 4, 5), dtype=np.uint8)
                },
                "joint_angles": np.zeros(8, dtype=np.float32),
                "action_gripper": 1,
                "subtask": "step",
                "language_instruction": "ee validation test",
            }
            with self.assertRaisesRegex(ValueError, "8 values"):
                collector.cache_step(**common, ee_pose=np.zeros(7, dtype=np.float32))
            invalid = np.zeros(8, dtype=np.float32)
            invalid[0] = np.nan
            with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                collector.cache_step(**common, ee_pose=invalid)
            collector.close()

    def test_action_path_uses_the_same_atomic_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = self._collector(directory, max_episodes=1)
            for value in (0, 1, 2):
                collector.cache_step_action(
                    camera_images={
                        "camera_rgb": np.full((3, 4, 5), value, dtype=np.uint8)
                    },
                    joint_angles=np.full(8, value, dtype=np.float32),
                    ee_pose=np.full(7, value + 10, dtype=np.float32),
                    action_gripper=1 if value < 2 else 0,
                    subtask="move",
                    language_instruction="action path",
                )
            final_action = np.full(7, 99, dtype=np.float32)
            collector.write_cached_data_action(final_action)
            collector.close()

            with h5py.File(Path(directory) / "dataset" / "episode_0000.h5", "r") as episode:
                np.testing.assert_array_equal(
                    episode["actions"][:],
                    np.vstack(
                        [
                            np.full(7, 11, dtype=np.float32),
                            np.full(7, 12, dtype=np.float32),
                            final_action,
                        ]
                    ),
                )
                np.testing.assert_array_equal(
                    episode["action_gripper"][:], np.array([1, 0, 0], dtype=np.uint8)
                )

    def test_default_collection_rate_is_sixty_hz(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = self._collector(directory, max_episodes=1)
            for sim_step in range(60):
                collector.cache_step(
                    camera_images={
                        "camera_rgb": np.full((3, 4, 5), sim_step, dtype=np.uint8)
                    },
                    joint_angles=np.full(8, sim_step, dtype=np.float32),
                    action_gripper=1,
                    subtask="move",
                    sim_step=sim_step,
                )
            collector.write_cached_data(np.full(8, 60, dtype=np.float32))
            collector.close()

            with h5py.File(Path(directory) / "dataset" / "episode_0000.h5", "r") as episode:
                self.assertEqual(len(episode["agent_pose"]), 60)
                self.assertEqual(int(episode.attrs["fps"]), 60)
                self.assertEqual(int(episode.attrs["sample_stride"]), 1)
                np.testing.assert_array_equal(episode["sim_step"][:], np.arange(60))
                np.testing.assert_allclose(episode["timestamp"][:], np.arange(60) / 60.0)

    def test_ten_hz_collection_samples_every_six_physics_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = DataCollector(
                camera_configs=[{"name": "camera", "image_type": "rgb"}],
                save_dir=directory,
                max_episodes=1,
                sample_hz=10,
                source_hz=60,
            )
            for sim_step in range(100, 160):
                collector.cache_step(
                    camera_images={
                        "camera_rgb": np.full((3, 4, 5), sim_step, dtype=np.uint8)
                    },
                    joint_angles=np.full(8, sim_step, dtype=np.float32),
                    action_gripper=1,
                    subtask="move",
                    sim_step=sim_step,
                )
            collector.write_cached_data(np.full(8, 160, dtype=np.float32))
            collector.close()

            with h5py.File(Path(directory) / "dataset" / "episode_0000.h5", "r") as episode:
                expected_steps = np.arange(100, 160, 6)
                self.assertEqual(len(episode["agent_pose"]), 10)
                self.assertEqual(int(episode.attrs["fps"]), 10)
                self.assertEqual(int(episode.attrs["physics_hz"]), 60)
                self.assertEqual(int(episode.attrs["sample_stride"]), 6)
                np.testing.assert_array_equal(episode["sim_step"][:], expected_steps)
                np.testing.assert_allclose(episode["timestamp"][:], np.arange(10) / 10.0)
                np.testing.assert_array_equal(
                    episode["agent_pose"][:, 0], expected_steps.astype(np.float32)
                )

            metadata_path = Path(directory) / "dataset" / "meta" / "episode.jsonl"
            metadata = json.loads(metadata_path.read_text().strip())
            self.assertEqual(metadata["fps"], 10)
            self.assertEqual(metadata["physics_hz"], 60)
            self.assertEqual(metadata["duration_seconds"], 1.0)

    def test_collection_rate_only_accepts_ten_or_sixty_hz(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "10 or 60"):
                DataCollector(camera_configs=[], save_dir=directory, sample_hz=30)


if __name__ == "__main__":
    unittest.main()
