from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ReplayStep:
    state: dict[str, Any] | None
    snapshot: dict[str, Any] | None
    legacy_done: bool = False
    legacy_success: bool = False
    termination_reason: str | None = None


class ReplayHDF5Backend:
    """No-Isaac backend for contract, aggregation and UI end-to-end tests.

    Real collection datasets provide cameras, ``agent_pose``, actions and the
    instruction.  A replay used for semantic evaluation additionally stores a
    ``snapshot_json`` string per frame, normally exported from runtime.snapshot().
    """

    backend_id = "replay_hdf5"

    def __init__(self, path: str | Path, *, policy: Any, camera_keys: tuple[str, ...] = ("camera_1_rgb", "camera_3_rgb")):
        self.path = Path(path).resolve()
        self.policy = policy
        self.camera_keys = camera_keys
        self.frames: list[dict[str, Any]] = []
        self.index = 0
        self._policy_calls = 0
        self._latencies: list[float] = []
        self._last_chunk: list[list[float]] | None = None
        self._load()

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.ndarray) and value.shape == ():
            return ReplayHDF5Backend._text(value.item())
        return str(value)

    def _load(self) -> None:
        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("h5py is required for replay evaluation") from exc
        with h5py.File(self.path, "r") as source:
            if "agent_pose" not in source or "snapshot_json" not in source:
                raise ValueError("replay HDF5 requires agent_pose and snapshot_json datasets")
            poses = np.asarray(source["agent_pose"], dtype=np.float32)
            if poses.ndim != 2 or poses.shape[1] != 8:
                raise ValueError(f"replay agent_pose must have shape [T, 8], got {poses.shape}")
            instruction = self._text(source["language_instruction"][()]) if "language_instruction" in source else ""
            snapshots = [json.loads(self._text(item)) for item in source["snapshot_json"]]
            length = len(poses)
            if len(snapshots) != length:
                raise ValueError("snapshot_json length does not match agent_pose")
            for key in self.camera_keys:
                if key not in source or len(source[key]) != length:
                    raise ValueError(f"replay camera dataset is missing or length-mismatched: {key}")
            self.frames = [
                {
                    "state": {
                        "camera_data": {key: np.asarray(source[key][index]) for key in self.camera_keys},
                        "camera_display": {key: np.asarray(source[key][index]) for key in self.camera_keys},
                        "agent_pose": poses[index],
                        "joint_positions": poses[index],
                        "language_instruction": instruction,
                    },
                    "snapshot": snapshots[index],
                }
                for index in range(length)
            ]

    def reset_episode(self, *, seed: int, split: str) -> dict[str, Any]:
        self.index = 0
        self._policy_calls = 0
        self._latencies = []
        self._last_chunk = None
        self.policy.reset()
        snapshot = self.frames[0]["snapshot"] if self.frames else {}
        return {
            "seed": int(seed),
            "split": split,
            "replay_path": str(self.path),
            "object_position": snapshot.get("initial_object_pos") or snapshot.get("object_pos"),
        }

    def step(self) -> ReplayStep:
        if self.index >= len(self.frames):
            return ReplayStep(None, None, legacy_done=True, termination_reason="timeout")
        frame = self.frames[self.index]
        chunk = self.policy.predict(frame["state"])
        self._policy_calls += 1
        self._latencies.append(float(chunk.latency_ms))
        self._last_chunk = np.asarray(chunk.joint_positions, dtype=np.float32).tolist()
        self.index += 1
        done = self.index >= len(self.frames)
        return ReplayStep(
            state=frame["state"],
            snapshot=frame["snapshot"],
            legacy_done=done,
            legacy_success=bool(frame["snapshot"].get("legacy_success", False)),
            termination_reason=None if not done else "replay_complete",
        )

    def actual_state(self) -> dict[str, Any]:
        if not self.frames:
            return {"ee_pos": [0.0, 0.0, 0.0], "joint_pos": []}
        frame = self.frames[max(0, min(self.index - 1, len(self.frames) - 1))]
        snapshot = frame["snapshot"]
        return {
            "ee_pos": snapshot.get("hand_pos", [0.0, 0.0, 0.0]),
            "ee_quat": snapshot.get("hand_quat", [0.0, 0.0, 0.0, 1.0]),
            "joint_pos": np.asarray(frame["state"]["agent_pose"]).tolist(),
            "gripper_pos": snapshot.get("gripper_pos", []),
        }

    def enter_terminal_hold(self) -> None:
        return None

    def policy_stats(self) -> dict[str, Any]:
        return {
            "policy_calls": self._policy_calls,
            "latencies_ms": list(self._latencies),
            "last_action_chunk": self._last_chunk,
        }

    def close(self) -> None:
        self.policy.close()


__all__ = ["ReplayHDF5Backend", "ReplayStep"]
