"""Streaming collection and LeRobot export for rack-placement recall v3."""

from __future__ import annotations

from dataclasses import dataclass
import json
from importlib.metadata import PackageNotFoundError, version as package_version
import os
from pathlib import Path
from typing import Any, Iterable

import cv2
import h5py
import numpy as np

from .core import CAP_COLORS, SLOTS


DATASET_SCHEMA = "labference/rack-recall-dataset-v1"
COLLECTION_SCHEMA = "labference/rack-recall-collection-plan-v1"
EPISODE_SUFFIXES = frozenset({".h5", ".hdf5"})
ACTION_NAMES = tuple(f"panda_joint{i + 1}_target_rad" for i in range(7)) + (
    "panda_finger_target_m",
)
STATE_NAMES = tuple(f"panda_joint{i + 1}_rad" for i in range(7)) + (
    "panda_finger_joint1_m", "panda_finger_joint2_m",
)


def generate_collection_plan(configs: Iterable[tuple[str, str] | tuple[str, str, str]], *, split: str = "train",
                             max_attempts: int = 3) -> dict[str, Any]:
    """Build a paired 12-slot x 4-yaw collection plan for each condition."""
    if split not in {"train", "validation"}:
        raise ValueError("collection split must be train or validation")
    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    base = 10_000 if split == "train" else 20_000
    yaws = (0.0, 90.0, 180.0, 270.0)
    parsed_configs = []
    for entry in configs:
        if len(entry) == 2:
            config, condition = entry
            contract_hash = None
        elif len(entry) == 3:
            config, condition, contract_hash = entry
        else:
            raise ValueError("collection config entries require path, condition, and optional hash")
        parsed_configs.append((config, condition, contract_hash))
    configs = tuple(parsed_configs)
    if {condition for _, condition, _ in configs} != {"single", "same_cap", "distinct_cap"}:
        raise ValueError("collection plan requires single, same_cap, and distinct_cap configs")
    units = []
    for condition_index, (config, condition, contract_hash) in enumerate(configs):
        for yaw_index, yaw in enumerate(yaws):
            for slot_index, slot in enumerate(SLOTS):
                pair_seed = base + yaw_index * 100 + slot_index
                seed = pair_seed + condition_index * 1_000
                color = CAP_COLORS[(yaw_index * len(SLOTS) + slot_index) % len(CAP_COLORS)]
                unit = {
                    "unit_id": f"{split}-{condition}-{slot}-yaw{int(yaw):03d}",
                    "split": split,
                    "config": config,
                    "condition": condition,
                    "collection_slot": slot,
                    "initial_yaw_deg": yaw,
                    "seed": seed,
                    "layout_seed": pair_seed + 100_000,
                    "motion_seed": pair_seed + 200_000,
                    "policy_seed": pair_seed + 300_000,
                    "color_seed": pair_seed + 400_000,
                    "target_cap_color": color,
                    "max_attempts": max_attempts,
                }
                if contract_hash is not None:
                    unit["contract_hash"] = contract_hash
                units.append(unit)
    return {
        "schema_version": COLLECTION_SCHEMA,
        "split": split,
        "control_hz": 20,
        "physics_hz": 120,
        "fixed_denominator": len(units),
        "units": units,
    }


def _append(dataset: h5py.Dataset, value: Any) -> None:
    dataset.resize(dataset.shape[0] + 1, axis=0)
    dataset[-1] = value


def _jpeg(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("failed to encode dataset camera frame")
    return encoded.reshape(-1)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class EpisodeDatasetWriter:
    output_path: Path
    metadata: dict[str, Any]
    control_hz: int = 20
    physics_hz: int = 120

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.output_path.with_suffix(self.output_path.suffix + ".partial")
        if self.partial_path.exists():
            self.partial_path.unlink()
        self.file = h5py.File(self.partial_path, "w")
        self.file.attrs["schema_version"] = DATASET_SCHEMA
        self.file.attrs["control_hz"] = self.control_hz
        self.file.attrs["physics_hz"] = self.physics_hz
        self.file.attrs["action_representation"] = "absolute_joint_position"
        self.file.attrs["action_units"] = "radians+meter"
        self.file.attrs["action_names_json"] = _json(ACTION_NAMES)
        self.file.attrs["state_names_json"] = _json(STATE_NAMES)
        self.file.attrs["metadata_json"] = _json(self.metadata)
        self.file.attrs["complete"] = False
        obs = self.file.create_group("observation")
        images = obs.create_group("images")
        jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        self.front = images.create_dataset("front_jpeg", shape=(0,), maxshape=(None,),
                                           dtype=jpeg_dtype)
        self.wrist = images.create_dataset("wrist_jpeg", shape=(0,), maxshape=(None,),
                                           dtype=jpeg_dtype)
        self.state = obs.create_dataset("state", shape=(0, 9), maxshape=(None, 9),
                                        dtype="float32", chunks=(128, 9))
        self.tcp = obs.create_dataset("tcp_pose_xyzw", shape=(0, 7), maxshape=(None, 7),
                                      dtype="float32", chunks=(128, 7))
        self.action = self.file.create_dataset("action", shape=(0, 8), maxshape=(None, 8),
                                               dtype="float32", chunks=(128, 8))
        self.timestamp = self.file.create_dataset("timestamp", shape=(0,), maxshape=(None,),
                                                  dtype="float64", chunks=(256,))
        self.physics_step = self.file.create_dataset("physics_step", shape=(0,), maxshape=(None,),
                                                     dtype="int64", chunks=(256,))
        diagnostics = self.file.create_group("diagnostics")
        text_dtype = h5py.string_dtype("utf-8")
        self.evaluator_state = diagnostics.create_dataset(
            "evaluator_state_json", shape=(0,), maxshape=(None,), dtype=text_dtype)
        self.tube_evidence = diagnostics.create_dataset(
            "tube_evidence_json", shape=(0,), maxshape=(None,), dtype=text_dtype)
        self.controller = diagnostics.create_dataset(
            "controller_json", shape=(0,), maxshape=(None,), dtype=text_dtype)
        raw = diagnostics.create_group("physics")
        self.raw_step = raw.create_dataset("step", shape=(0,), maxshape=(None,), dtype="int64")
        self.raw_command_index = raw.create_dataset(
            "command_index", shape=(0,), maxshape=(None,), dtype="int64")
        self.raw_state = raw.create_dataset("joint_state", shape=(0, 9), maxshape=(None, 9),
                                            dtype="float32", chunks=(512, 9))
        self.raw_force = raw.create_dataset(
            "target_rack_force_axial_lateral_n", shape=(0, 2), maxshape=(None, 2),
            dtype="float32", chunks=(512, 2))
        self.raw_phase = raw.create_dataset(
            "phase", shape=(0,), maxshape=(None,), dtype=text_dtype)
        self.control_count = 0

    def add_physics_step(self, *, step: int, joints: Any, command_index: int,
                         force_axial_n: float, force_lateral_n: float,
                         phase: str) -> None:
        _append(self.raw_step, int(step))
        _append(self.raw_command_index, int(command_index))
        _append(self.raw_state, np.asarray(joints, dtype=np.float32))
        _append(self.raw_force, np.asarray([force_axial_n, force_lateral_n], dtype=np.float32))
        _append(self.raw_phase, phase)

    def add_control_frame(self, *, step: int, front: np.ndarray, wrist: np.ndarray,
                          joint_state: Any, tcp_pose_xyzw: Any, action: Any,
                          evaluator_state: dict[str, Any], tube_evidence: dict[str, Any],
                          controller_status: dict[str, Any]) -> None:
        _append(self.front, _jpeg(front))
        _append(self.wrist, _jpeg(wrist))
        _append(self.state, np.asarray(joint_state, dtype=np.float32))
        _append(self.tcp, np.asarray(tcp_pose_xyzw, dtype=np.float32))
        _append(self.action, np.asarray(action, dtype=np.float32))
        _append(self.timestamp, float(step) / self.physics_hz)
        _append(self.physics_step, int(step))
        _append(self.evaluator_state, _json(evaluator_state))
        _append(self.tube_evidence, _json(tube_evidence))
        _append(self.controller, _json(controller_status))
        self.control_count += 1
        if self.control_count % 100 == 0:
            self.file.flush()

    def set_terminal_observation(self, *, front: np.ndarray, wrist: np.ndarray,
                                 joint_state: Any, tcp_pose_xyzw: Any,
                                 evaluator_state: dict[str, Any]) -> None:
        terminal = self.file.create_group("terminal_observation")
        jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        terminal_front = terminal.create_dataset("front_jpeg", shape=(1,), dtype=jpeg_dtype)
        terminal_wrist = terminal.create_dataset("wrist_jpeg", shape=(1,), dtype=jpeg_dtype)
        terminal_front[0] = _jpeg(front)
        terminal_wrist[0] = _jpeg(wrist)
        terminal.create_dataset("state", data=np.asarray(joint_state, dtype=np.float32))
        terminal.create_dataset("tcp_pose_xyzw", data=np.asarray(tcp_pose_xyzw,
                                                                  dtype=np.float32))
        terminal.attrs["evaluator_state_json"] = _json(evaluator_state)

    def finalize(self, *, scores: dict[str, Any], runtime_status: str,
                 accepted_for_training: bool) -> None:
        self.file.attrs["scores_json"] = _json(scores)
        self.file.attrs["runtime_status"] = runtime_status
        self.file.attrs["accepted_for_training"] = bool(accepted_for_training)
        self.file.attrs["complete"] = True
        self.file.flush()
        self.file.close()
        os.replace(self.partial_path, self.output_path)

    def abort(self, reason: str) -> None:
        if getattr(self, "file", None) is not None:
            self.file.attrs["abort_reason"] = reason
            self.file.flush()
            self.file.close()


def validate_hdf5_episode(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    errors: list[str] = []
    with h5py.File(path, "r") as episode:
        if episode.attrs.get("schema_version") != DATASET_SCHEMA:
            errors.append("unknown schema version")
        if not bool(episode.attrs.get("complete", False)):
            errors.append("episode is incomplete")
        required = ("observation/images/front_jpeg", "observation/images/wrist_jpeg",
                    "observation/state", "observation/tcp_pose_xyzw", "action",
                    "timestamp", "physics_step", "terminal_observation")
        for key in required:
            if key not in episode:
                errors.append(f"missing {key}")
        if errors:
            return {"path": str(path), "valid": False, "errors": errors}
        lengths = {key: len(episode[key]) for key in required[:-1]}
        if len(set(lengths.values())) != 1 or not next(iter(lengths.values())):
            errors.append(f"control arrays are not synchronized: {lengths}")
        timestamps = np.asarray(episode["timestamp"])
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            errors.append("timestamps are not strictly increasing")
        if episode["action"].shape[1:] != (8,) or episode["observation/state"].shape[1:] != (9,):
            errors.append("state/action dimensions do not match the profile")
        for key in ("observation/images/front_jpeg", "observation/images/wrist_jpeg"):
            image = cv2.imdecode(np.asarray(episode[key][0], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (480, 640):
                errors.append(f"invalid first image in {key}")
        accepted = bool(episode.attrs.get("accepted_for_training", False))
    return {"path": str(path), "valid": not errors, "errors": errors,
            "frame_count": lengths.get("action", 0), "accepted_for_training": accepted}


def _episode_files(source: str | Path) -> list[Path]:
    source = Path(source)
    if source.is_file():
        return [source] if source.suffix.lower() in EPISODE_SUFFIXES else []
    return sorted(path for path in source.rglob("*")
                  if path.is_file() and path.suffix.lower() in EPISODE_SUFFIXES)


def validate_dataset(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    files = _episode_files(root)
    partials = ([str(root)] if root.is_file() and root.suffix == ".partial" else
                sorted(str(path) for path in root.rglob("*.partial")))
    rows = [validate_hdf5_episode(path) for path in files]
    return {"schema_version": DATASET_SCHEMA, "episode_count": len(rows),
            "valid_episode_count": sum(row["valid"] for row in rows),
            "accepted_episode_count": sum(row["valid"] and row["accepted_for_training"]
                                          for row in rows),
            "partial_files": partials, "episodes": rows,
            "valid": bool(rows) and all(row["valid"] for row in rows) and not partials}


def export_lerobot(source: str | Path, output: str | Path,
                   *, repo_id: str = "labference/rack_recall_v3") -> dict[str, Any]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot export requires the isolated conversion environment with lerobot==0.3.3") from exc
    try:
        version = package_version("lerobot")
    except PackageNotFoundError:
        version = "unknown"
    if version != "0.3.3":
        raise RuntimeError(f"LeRobot 0.3.3 is required, found {version}")
    episodes = []
    for path in _episode_files(source):
        report = validate_hdf5_episode(path)
        if report["valid"] and report["accepted_for_training"]:
            episodes.append(path)
    if not episodes:
        raise ValueError("no complete clean-success episodes are available for export")
    features = {
        "observation.images.front": {"dtype": "video", "shape": (480, 640, 3),
                                     "names": ["height", "width", "channels"]},
        "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3),
                                     "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (9,), "names": list(STATE_NAMES)},
        "action": {"dtype": "float32", "shape": (8,), "names": list(ACTION_NAMES)},
    }
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=20, features=features,
                                    root=Path(output), robot_type="franka", use_videos=True)
    for path in episodes:
        with h5py.File(path, "r") as episode:
            metadata = json.loads(episode.attrs["metadata_json"])
            task = metadata["instruction"]
            for index in range(len(episode["action"])):
                front = cv2.cvtColor(cv2.imdecode(
                    np.asarray(episode["observation/images/front_jpeg"][index], dtype=np.uint8),
                    cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                wrist = cv2.cvtColor(cv2.imdecode(
                    np.asarray(episode["observation/images/wrist_jpeg"][index], dtype=np.uint8),
                    cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                dataset.add_frame({
                    "observation.images.front": front,
                    "observation.images.wrist": wrist,
                    "observation.state": np.asarray(episode["observation/state"][index]),
                    "action": np.asarray(episode["action"][index]),
                }, task=task, timestamp=float(episode["timestamp"][index]))
            dataset.save_episode()
    loaded = LeRobotDataset(repo_id=repo_id, root=Path(output))
    sample = loaded[0]
    return {"output": str(Path(output).resolve()), "lerobot_version": version,
            "episode_count": len(episodes), "frame_count": len(loaded),
            "readback_keys": sorted(sample), "valid": len(loaded) > 0}
