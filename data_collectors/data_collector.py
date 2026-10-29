"""Episode collection with bounded, atomic background persistence."""

from __future__ import annotations

import copy
import json
import os
import queue
import threading
import traceback
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional

import h5py
import numpy as np


def _validate_episode_streams(
    camera_data: dict,
    agent_pose_data: np.ndarray,
    actions_data: np.ndarray,
    action_gripper_data: np.ndarray,
    subtask_data: np.ndarray,
    actions_ee_data: Optional[np.ndarray] = None,
    timestamp_data: Optional[np.ndarray] = None,
    sim_step_data: Optional[np.ndarray] = None,
) -> None:
    lengths = {
        "agent_pose": len(agent_pose_data),
        "actions": len(actions_data),
        "action_gripper": len(action_gripper_data),
        "subtask": len(subtask_data),
        **{name: len(values) for name, values in camera_data.items()},
    }
    if actions_ee_data is not None:
        lengths["actions_ee"] = len(actions_ee_data)
        if actions_ee_data.ndim != 2 or actions_ee_data.shape[1] != 8:
            raise ValueError(
                f"actions_ee must have shape [T, 8], got {actions_ee_data.shape}"
            )
        if not np.isfinite(actions_ee_data).all():
            raise ValueError("actions_ee contains NaN or infinity")
    if timestamp_data is not None:
        lengths["timestamp"] = len(timestamp_data)
    if sim_step_data is not None:
        lengths["sim_step"] = len(sim_step_data)
    if len(set(lengths.values())) != 1:
        raise ValueError(f"episode streams must have equal lengths, got {lengths}")
    if not np.isin(action_gripper_data, (0, 1)).all():
        raise ValueError("action_gripper contains values outside {0, 1}")


def _write_episode_data(
    episode_path: str,
    episode_name: str,
    camera_data: dict,
    agent_pose_data: np.ndarray,
    actions_data: np.ndarray,
    action_gripper_data: np.ndarray,
    subtask_data: np.ndarray,
    task_properties: dict = None,
    language_instruction: Optional[str] = None,
    compression=None,
    actions_ee_data: Optional[np.ndarray] = None,
    timestamp_data: Optional[np.ndarray] = None,
    sim_step_data: Optional[np.ndarray] = None,
    sample_hz: int = 60,
    source_hz: int = 60,
) -> None:
    """Write one complete episode to an HDF5 file."""
    _validate_episode_streams(
        camera_data,
        agent_pose_data,
        actions_data,
        action_gripper_data,
        subtask_data,
        actions_ee_data,
        timestamp_data,
        sim_step_data,
    )

    with h5py.File(episode_path, "w") as h5_file:
        print(f"Writing episode {episode_name} to {episode_path}", flush=True)
        for camera_name, image_data in camera_data.items():
            chunk_size = (min(64, image_data.shape[0]),) + image_data.shape[1:]
            kwargs = {
                "data": image_data,
                "dtype": "uint8",
                "chunks": chunk_size,
            }
            if compression == "gzip":
                kwargs.update({"compression": "gzip", "compression_opts": 5})
            h5_file.create_dataset(camera_name, **kwargs)

        h5_file.create_dataset(
            "agent_pose", data=agent_pose_data, dtype="float32", chunks=True
        )
        h5_file.create_dataset(
            "actions", data=actions_data, dtype="float32", chunks=True
        )
        h5_file.create_dataset(
            "action_gripper", data=action_gripper_data, dtype="uint8", chunks=True
        )
        h5_file.create_dataset(
            "subtask",
            data=np.asarray(subtask_data, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        if actions_ee_data is not None:
            h5_file.create_dataset(
                "actions_ee", data=actions_ee_data, dtype="float32", chunks=True
            )
        if timestamp_data is not None:
            h5_file.create_dataset(
                "timestamp", data=timestamp_data, dtype="float64", chunks=True
            )
        if sim_step_data is not None:
            h5_file.create_dataset(
                "sim_step", data=sim_step_data, dtype="int64", chunks=True
            )

        h5_file.attrs["fps"] = int(sample_hz)
        h5_file.attrs["physics_hz"] = int(source_hz)
        h5_file.attrs["sample_stride"] = int(source_hz // sample_hz)

        if language_instruction is not None:
            h5_file.create_dataset(
                "language_instruction",
                data=language_instruction,
                dtype=h5py.special_dtype(vlen=str),
            )
        if task_properties:
            h5_file.create_dataset(
                "task_properties",
                data=json.dumps(task_properties, ensure_ascii=False),
                dtype=h5py.special_dtype(vlen=str),
            )
        h5_file.flush()
        print(f"Finished writing episode {episode_name}", flush=True)


@dataclass
class _ActiveEpisodeBuffer:
    cameras: Dict[str, list]
    agent_pose: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    actions_ee: list = field(default_factory=list)
    action_gripper: list = field(default_factory=list)
    subtasks: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    sim_steps: list = field(default_factory=list)
    language_instruction: Optional[str] = None
    task_properties: dict = field(default_factory=dict)
    task_instruction: Optional[str] = None


@dataclass(frozen=True)
class _EpisodeSnapshot:
    episode_index: int
    episode_name: str
    final_path: str
    temporary_path: str
    camera_data: Dict[str, np.ndarray]
    agent_pose_data: np.ndarray
    actions_data: np.ndarray
    action_gripper_data: np.ndarray
    subtask_data: np.ndarray
    task_properties: dict
    language_instruction: Optional[str]
    metadata: dict
    actions_ee_data: Optional[np.ndarray] = None
    timestamp_data: Optional[np.ndarray] = None
    sim_step_data: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _WriterFailure:
    episode_index: int
    episode_name: str
    error: BaseException
    formatted_traceback: str


class BackgroundWriteError(RuntimeError):
    """Raised in the collection thread after the HDF5 writer has failed."""

    def __init__(self, failure: _WriterFailure):
        self.episode_index = failure.episode_index
        self.episode_name = failure.episode_name
        self.original_error = failure.error
        self.writer_traceback = failure.formatted_traceback
        super().__init__(
            f"background write failed for {failure.episode_name} "
            f"(index {failure.episode_index}): {failure.error}\n"
            f"Writer traceback:\n{failure.formatted_traceback}"
        )


class DataCollector:
    """Collect episodes and persist them using one bounded writer thread."""

    _STOP = object()

    def __init__(
        self,
        camera_configs: List[dict],
        save_dir="output",
        max_episodes=10,
        max_workers=4,
        compression=None,
        queue_capacity=2,
        sample_hz=60,
        source_hz=60,
        background_write=True,
    ):
        """Initialize the collector.

        ``max_workers`` remains accepted for source compatibility. Persistence is
        intentionally serialized through one writer thread because h5py and the
        JSONL index must be committed in deterministic order.
        """
        del max_workers
        if int(queue_capacity) <= 0:
            raise ValueError("queue_capacity must be a positive integer")
        self.sample_hz = int(sample_hz)
        self.source_hz = int(source_hz)
        if self.sample_hz not in {10, 20, 60}:
            raise ValueError(
                f"sample_hz must be 10 or 60 (20 is supported for 120 Hz release tasks), got {self.sample_hz}"
            )
        if self.source_hz not in {60, 120}:
            raise ValueError(f"source_hz must be 60 or 120, got {self.source_hz}")
        if self.source_hz % self.sample_hz != 0:
            raise ValueError(
                f"source_hz={self.source_hz} must be divisible by sample_hz={self.sample_hz}"
            )
        self.sample_stride = self.source_hz // self.sample_hz

        self.save_dir = save_dir
        self.max_episodes = max_episodes
        self.compression = compression
        self.session_dir = os.path.join(save_dir, "dataset")
        self.mate_dir = os.path.join(self.session_dir, "meta")
        self.episode_file_path = os.path.join(self.mate_dir, "episode.jsonl")
        self.episode_count = 0
        self.camera_configs = camera_configs
        self._camera_stream_names = self._build_camera_stream_names(camera_configs)

        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.mate_dir, exist_ok=True)
        self._active = self._new_active_buffer()
        self._sampling_origin_step: Optional[int] = None
        self._fallback_source_step = 0
        self._write_queue: queue.Queue = queue.Queue(maxsize=int(queue_capacity))
        self._failure_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._writer_failure: Optional[_WriterFailure] = None
        self._closed = False
        self._background_write = bool(background_write)
        self._writer_thread = None
        if self._background_write:
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="labbench-hdf5-writer",
                daemon=False,
            )
            self._writer_thread.start()

    @staticmethod
    def _build_camera_stream_names(camera_configs: List[dict]) -> List[str]:
        names = []
        for config in camera_configs:
            image_types = str(config["image_type"]).split("+")
            names.extend(f"{config['name']}_{image_type}" for image_type in image_types)
        return names

    def _new_active_buffer(self) -> _ActiveEpisodeBuffer:
        return _ActiveEpisodeBuffer(cameras={name: [] for name in self._camera_stream_names})

    # Compatibility accessors for code that inspects the former public cache.
    @property
    def temp_cameras(self):
        return self._active.cameras

    @property
    def temp_agent_pose(self):
        return self._active.agent_pose

    @property
    def temp_actions(self):
        return self._active.actions

    @property
    def temp_action_gripper(self):
        return self._active.action_gripper

    @property
    def temp_subtasks(self):
        return self._active.subtasks

    @property
    def temp_language_instruction(self):
        return self._active.language_instruction

    @property
    def temp_task_properties(self):
        return self._active.task_properties

    @property
    def task_instructions(self):
        return self._active.task_instruction

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("data collector is already closed")

    def _raise_if_writer_failed(self) -> None:
        with self._failure_lock:
            failure = self._writer_failure
        if failure is not None:
            raise BackgroundWriteError(failure) from failure.error

    def _record_writer_failure(
        self, snapshot: _EpisodeSnapshot, error: BaseException
    ) -> None:
        failure = _WriterFailure(
            episode_index=snapshot.episode_index,
            episode_name=snapshot.episode_name,
            error=error,
            formatted_traceback=traceback.format_exc(),
        )
        with self._failure_lock:
            if self._writer_failure is None:
                self._writer_failure = failure

    def set_task_properties(self, properties: dict) -> None:
        self._ensure_open()
        self._raise_if_writer_failed()
        self._active.task_properties = properties

    def _cache_common(
        self,
        camera_images: dict,
        joint_angles: np.ndarray,
        action_gripper: int,
        subtask: str,
        language_instruction: Optional[str],
        ee_pose: Optional[np.ndarray] = None,
        sim_step: Optional[int] = None,
    ) -> bool:
        self._ensure_open()
        self._raise_if_writer_failed()
        if action_gripper not in (0, 1):
            raise ValueError(
                f"action_gripper must be 0 (closed) or 1 (open), got {action_gripper!r}"
            )
        normalized_ee_pose = None
        if ee_pose is not None:
            normalized_ee_pose = np.asarray(ee_pose, dtype=np.float32).reshape(-1)
            if normalized_ee_pose.shape != (8,):
                raise ValueError(
                    f"ee_pose must contain 8 values [xyz, qwxyz, gripper], got "
                    f"shape {normalized_ee_pose.shape}"
                )
            if not np.isfinite(normalized_ee_pose).all():
                raise ValueError("ee_pose contains NaN or infinity")
        if not isinstance(subtask, str) or not subtask.strip():
            raise ValueError("subtask must be a non-empty string")
        source_step = self._resolve_source_step(sim_step)
        if self._sampling_origin_step is None:
            self._sampling_origin_step = source_step
        relative_step = source_step - self._sampling_origin_step
        if relative_step < 0:
            raise ValueError(
                f"sim_step moved backwards within an episode: {source_step} < "
                f"{self._sampling_origin_step}"
            )
        if relative_step % self.sample_stride != 0:
            return False
        if self._active.task_instruction is None and language_instruction is not None:
            self._active.task_instruction = language_instruction
        for camera_name, image in camera_images.items():
            if camera_name not in self._active.cameras:
                raise KeyError(f"unconfigured camera stream: {camera_name}")
            self._active.cameras[camera_name].append(image)
        self._active.agent_pose.append(joint_angles)
        self._active.action_gripper.append(np.uint8(action_gripper))
        self._active.subtasks.append(subtask.strip())
        self._active.sim_steps.append(np.int64(source_step))
        self._active.timestamps.append(np.float64(relative_step / self.source_hz))
        if normalized_ee_pose is not None:
            self._active.actions_ee.append(normalized_ee_pose)
        if language_instruction is not None:
            self._active.language_instruction = language_instruction
        return True

    def _resolve_source_step(self, sim_step: Optional[int]) -> int:
        if sim_step is None:
            source_step = self._fallback_source_step
        else:
            source_step = int(sim_step)
            if source_step < 0:
                raise ValueError(f"sim_step must be non-negative, got {source_step}")
        self._fallback_source_step = max(self._fallback_source_step + 1, source_step + 1)
        return source_step

    def _reset_sampling_clock(self) -> None:
        self._sampling_origin_step = None
        self._fallback_source_step = 0

    def cache_step(
        self,
        camera_images: dict,
        joint_angles: np.ndarray,
        action_gripper: int,
        subtask: str,
        language_instruction: Optional[str] = None,
        ee_pose: Optional[np.ndarray] = None,
        sim_step: Optional[int] = None,
    ) -> bool:
        return self._cache_common(
            camera_images,
            joint_angles,
            action_gripper,
            subtask,
            language_instruction,
            ee_pose,
            sim_step,
        )

    def cache_step_action(
        self,
        camera_images: dict,
        joint_angles: np.ndarray,
        ee_pose: np.ndarray,
        action_gripper: int,
        subtask: str,
        language_instruction: Optional[str] = None,
        sim_step: Optional[int] = None,
    ) -> bool:
        sampled = self._cache_common(
            camera_images,
            joint_angles,
            action_gripper,
            subtask,
            language_instruction,
            sim_step=sim_step,
        )
        if sampled:
            self._active.actions.append(ee_pose)
        return sampled

    def write_cached_data(
        self, final_joint_positions, *, final_ee_pose: Optional[np.ndarray] = None
    ) -> None:
        actions = self._active.agent_pose[1:] + [final_joint_positions]
        actions_ee = None
        if self._active.actions_ee:
            if final_ee_pose is None:
                raise ValueError(
                    "final_ee_pose is required when the episode cached ee_pose frames"
                )
            normalized_final_ee_pose = np.asarray(final_ee_pose, dtype=np.float32).reshape(-1)
            if normalized_final_ee_pose.shape != (8,):
                raise ValueError(
                    f"final_ee_pose must contain 8 values, got shape "
                    f"{normalized_final_ee_pose.shape}"
                )
            if not np.isfinite(normalized_final_ee_pose).all():
                raise ValueError("final_ee_pose contains NaN or infinity")
            actions_ee = self._active.actions_ee[1:] + [normalized_final_ee_pose]
        self._finalize_and_enqueue(actions, actions_ee)

    def write_cached_data_action(self, final_action) -> None:
        actions = self._active.actions[1:] + [final_action]
        self._finalize_and_enqueue(actions)

    def _finalize_and_enqueue(self, actions: list, actions_ee: Optional[list] = None) -> None:
        self._ensure_open()
        self._raise_if_writer_failed()
        if self.episode_count >= self.max_episodes:
            return

        active = self._active
        if not active.agent_pose or not active.action_gripper or not active.subtasks:
            raise ValueError("cannot write an episode without recorded steps")

        # Detach the completed cache before preparing its immutable NumPy snapshot.
        # Restore it if local validation fails so callers can inspect/retry the
        # episode rather than silently losing the recorded frames.
        self._active = self._new_active_buffer()
        self._reset_sampling_clock()
        try:
            shifted_gripper = active.action_gripper[1:] + [active.action_gripper[-1]]
            camera_data = {
                name: np.asarray(images, dtype=np.uint8)
                for name, images in active.cameras.items()
            }
            agent_pose_data = np.asarray(active.agent_pose)
            actions_data = np.asarray(actions)
            actions_ee_data = (
                np.asarray(actions_ee, dtype=np.float32)
                if actions_ee is not None
                else None
            )
            action_gripper_data = np.asarray(shifted_gripper, dtype=np.uint8)
            subtask_data = np.asarray(active.subtasks, dtype=object)
            timestamp_data = np.asarray(active.timestamps, dtype=np.float64)
            sim_step_data = np.asarray(active.sim_steps, dtype=np.int64)
            _validate_episode_streams(
                camera_data,
                agent_pose_data,
                actions_data,
                action_gripper_data,
                subtask_data,
                actions_ee_data,
                timestamp_data,
                sim_step_data,
            )
        except BaseException:
            self._active = active
            if active.sim_steps:
                self._sampling_origin_step = int(active.sim_steps[0])
                self._fallback_source_step = int(active.sim_steps[-1]) + 1
            raise

        episode_index = self.episode_count
        episode_name = f"episode_{episode_index:04d}"
        final_path = os.path.join(self.session_dir, f"{episode_name}.h5")
        task_properties = copy.deepcopy(active.task_properties)
        snapshot = _EpisodeSnapshot(
            episode_index=episode_index,
            episode_name=episode_name,
            final_path=final_path,
            temporary_path=f"{final_path}.tmp",
            camera_data=camera_data,
            agent_pose_data=agent_pose_data,
            actions_data=actions_data,
            action_gripper_data=action_gripper_data,
            subtask_data=subtask_data,
            task_properties=task_properties,
            language_instruction=active.language_instruction,
            actions_ee_data=actions_ee_data,
            timestamp_data=timestamp_data,
            sim_step_data=sim_step_data,
            metadata={
                "episode_index": episode_index,
                "tasks": [active.task_instruction] if active.task_instruction else [],
                "length": len(active.agent_pose),
                "task_properties": task_properties,
                "fps": self.sample_hz,
                "physics_hz": self.source_hz,
                "sample_stride": self.sample_stride,
                "duration_seconds": (
                    float(timestamp_data[-1] + 1.0 / self.sample_hz)
                    if len(timestamp_data)
                    else 0.0
                ),
            },
        )

        if self._background_write:
            # A full queue intentionally blocks the simulation to bound image memory.
            self._write_queue.put(snapshot, block=True)
            self.episode_count += 1
            self._raise_if_writer_failed()
        else:
            # Some Isaac/Replicator builds are not stable while a native HDF5
            # compression thread overlaps world.step(). Persist on the Kit
            # main thread before the next episode starts.
            try:
                self._persist_snapshot(snapshot)
            except BaseException as error:
                try:
                    if os.path.exists(snapshot.temporary_path):
                        os.remove(snapshot.temporary_path)
                except OSError:
                    pass
                self._record_writer_failure(snapshot, error)
                self._raise_if_writer_failed()
            self.episode_count += 1

    @staticmethod
    def _fsync_file(path: str) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _fsync_directory(path: str) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _append_episode_metadata(self, metadata: dict) -> None:
        with open(self.episode_file_path, "a", encoding="utf-8") as metadata_file:
            metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            metadata_file.flush()
            os.fsync(metadata_file.fileno())

    def _persist_snapshot(self, snapshot: _EpisodeSnapshot) -> None:
        _write_episode_data(
            snapshot.temporary_path,
            snapshot.episode_name,
            snapshot.camera_data,
            snapshot.agent_pose_data,
            snapshot.actions_data,
            snapshot.action_gripper_data,
            snapshot.subtask_data,
            snapshot.task_properties,
            snapshot.language_instruction,
            self.compression,
            snapshot.actions_ee_data,
            snapshot.timestamp_data,
            snapshot.sim_step_data,
            self.sample_hz,
            self.source_hz,
        )
        self._fsync_file(snapshot.temporary_path)
        os.replace(snapshot.temporary_path, snapshot.final_path)
        self._fsync_directory(self.session_dir)
        self._append_episode_metadata(snapshot.metadata)

    def _writer_loop(self) -> None:
        while True:
            item = self._write_queue.get()
            try:
                if item is self._STOP:
                    return
                snapshot = item
                with self._failure_lock:
                    writer_failed = self._writer_failure is not None
                if writer_failed:
                    continue
                try:
                    self._persist_snapshot(snapshot)
                except BaseException as error:
                    try:
                        if os.path.exists(snapshot.temporary_path):
                            os.remove(snapshot.temporary_path)
                    except OSError:
                        pass
                    self._record_writer_failure(snapshot, error)
            finally:
                self._write_queue.task_done()

    def clear_cache(self) -> None:
        """Discard the active episode without touching queued snapshots."""
        self._ensure_open()
        self._raise_if_writer_failed()
        self._active = self._new_active_buffer()
        self._reset_sampling_clock()

    def _merge_episode_files(self) -> None:
        merged_path = os.path.join(self.session_dir, "merged_episodes.hdf5")
        final_merged_path = os.path.join(self.session_dir, "episode_data.hdf5")
        episode_files = sorted(glob(os.path.join(self.session_dir, "episode_*.h5")))
        if not episode_files:
            print("No episodes to merge")
            return

        with h5py.File(merged_path, "w") as merged_file:
            for episode_path in episode_files:
                episode_name = os.path.splitext(os.path.basename(episode_path))[0]
                with h5py.File(episode_path, "r") as episode_file:
                    episode_group = merged_file.create_group(episode_name)
                    for key in episode_file.keys():
                        episode_file.copy(key, episode_group)
        os.replace(merged_path, final_merged_path)
        for episode_path in episode_files:
            os.remove(episode_path)
        print(
            f"Successfully merged {len(episode_files)} episodes into {final_merged_path}"
        )

    def close(self, merge=False) -> None:
        """Drain all writes, stop the writer, and surface background failures."""
        with self._close_lock:
            if not self._closed:
                self._closed = True
                if self._background_write:
                    self._write_queue.put(self._STOP, block=True)
                    self._write_queue.join()
                    self._writer_thread.join()

            self._raise_if_writer_failed()
            if merge:
                self._merge_episode_files()
