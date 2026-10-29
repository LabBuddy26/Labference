"""Bridge release-protocol rollouts into the LabBench HDF5 schema.

The release evaluators own motion and physical success decisions.  This class
only samples the observation immediately before each reference command and
uses the existing DataCollector to write the next-step-aligned streams.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .data_collector import DataCollector


class ReleaseEpisodeRecorder:
    def __init__(self, output_dir: str, *, camera_names: tuple[str, ...],
                 source_hz: int, sample_hz: int = 10, max_episodes: int = 1,
                 task_name: str, task_properties: Mapping[str, Any] | None = None):
        configs = [{"name": name.rsplit("_", 1)[0],
                    "image_type": name.rsplit("_", 1)[1]} for name in camera_names]
        self.collector = DataCollector(
            configs, save_dir=output_dir, max_episodes=max_episodes,
            compression="gzip", sample_hz=sample_hz, source_hz=source_hz,
            background_write=False,
        )
        self.task_name = task_name
        self.collector.set_task_properties(dict(task_properties or {}))
        self._last = None
        # Isaac cameras can briefly return ``None`` while a render product is
        # being refreshed.  Keep the last valid frame so one transient sensor
        # gap cannot abort an otherwise valid physical demonstration.
        self._last_cameras: dict[str, np.ndarray] = {}

    @staticmethod
    def _compact_joints(joints: Any) -> np.ndarray:
        values = np.asarray(joints, dtype=np.float32).reshape(-1)
        if values.size < 8:
            raise ValueError("release recorder requires 7 arm joints and a finger state")
        return np.concatenate((values[:7], values[7:8])).astype(np.float32)

    @staticmethod
    def _ee_pose(tcp_pose_xyzw: Any, joints: Any) -> np.ndarray:
        pose = np.asarray(tcp_pose_xyzw, dtype=np.float32).reshape(-1)
        if pose.size < 7:
            raise ValueError("TCP pose must contain xyz and xyzw quaternion")
        # Current LabBench actions_ee use scalar-first Isaac convention.
        quat_xyzw = pose[3:7]
        return np.concatenate((pose[:3], quat_xyzw[[3, 0, 1, 2]],
                               [np.asarray(joints).reshape(-1)[7]])).astype(np.float32)

    @staticmethod
    def _gripper_open(command: Any, joints: Any) -> int:
        values = np.asarray(command if command is not None else joints, dtype=np.float32).reshape(-1)
        if command is None or values.size < 8:
            values = np.asarray(joints, dtype=np.float32).reshape(-1)
            finger = float(np.mean(values[7:9]))
        elif values.size >= 9:
            finger = float(np.mean(values[-2:]))
        else:
            finger = float(values[7])
        return int(finger > 0.02)

    def record(self, *, cameras: Mapping[str, Any], joints: Any,
               tcp_pose_xyzw: Any, command: Any = None, sim_step: int,
               subtask: str) -> None:
        configured = {"front": "camera_1_rgb", "wrist": "camera_3_rgb",
                      "display": "display_rgb", "weights": "weights_rgb"}
        supplied = {configured[key] for key in cameras if key in configured}
        if supplied != set(self.collector.temp_cameras):
            raise ValueError("release camera streams mismatch: missing or unconfigured camera key")
        mapped = {}
        for source, target in (("front", "camera_1_rgb"), ("wrist", "camera_3_rgb"),
                               ("display", "display_rgb"), ("weights", "weights_rgb")):
            if source in cameras:
                frame = cameras[source]
                if frame is None:
                    frame = self._last_cameras.get(source)
                if frame is not None:
                    image = np.asarray(frame, dtype=np.uint8)
                    if image.ndim == 3 and image.shape[0] > 0 and image.shape[1] > 0:
                        self._last_cameras[source] = np.array(image, copy=True, order="C")
                        mapped[target] = self._last_cameras[source]
        expected = set(self.collector.temp_cameras)
        if set(mapped) != expected:
            missing = expected - set(mapped)
            extra = set(mapped) - expected
            # A camera may not have produced its first render yet.  Skip this
            # sample and wait for the next simulation tick; aborting the
            # episode here would discard a valid robot trajectory.
            if missing:
                return
            raise ValueError(f"release camera streams mismatch; missing={missing}, extra={extra}")
        compact = self._compact_joints(joints)
        self.collector.cache_step(
            camera_images=mapped,
            joint_angles=compact,
            action_gripper=self._gripper_open(command, joints),
            subtask=subtask,
            language_instruction=self.task_name,
            ee_pose=self._ee_pose(tcp_pose_xyzw, joints),
            sim_step=sim_step,
        )
        self._last = (compact, self._ee_pose(tcp_pose_xyzw, joints))

    def finish(self, *, joints: Any, tcp_pose_xyzw: Any) -> None:
        # A very short diagnostic run may end before the reference controller
        # issues its first action.  Keep the run's time-limit score and close
        # the collector cleanly instead of masking it with a second exception.
        if self._last is None:
            return
        final_joints = self._compact_joints(joints)
        self.collector.write_cached_data(
            final_joints,
            final_ee_pose=self._ee_pose(tcp_pose_xyzw, joints),
        )

    def abort(self) -> None:
        self.collector.clear_cache()

    def close(self) -> None:
        self.collector.close()
