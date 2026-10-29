"""Readiness gate for simulation data collection episodes.

The gate is intentionally independent of Isaac Sim so its timing and
stability behaviour can be tested without launching the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


DEFAULT_READY_JOINT_POSITIONS = (
    0.0,
    -0.7854,
    0.0,
    -2.3562,
    0.0,
    1.5708,
    0.7854,
    0.04,
    0.04,
)


@dataclass(frozen=True)
class CollectionStartupConfig:
    """Configuration for one collection episode's readiness phase."""

    ready_joint_positions: tuple[float, ...] = DEFAULT_READY_JOINT_POSITIONS
    drive_robot: bool = True
    warmup_steps: int = 120
    stable_frames: int = 30
    joint_velocity_threshold: float = 0.02
    object_displacement_threshold: float = 0.0005
    max_wait_steps: int = 600

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "CollectionStartupConfig":
        data = {} if raw is None else dict(raw)
        ready_joint_positions = tuple(float(value) for value in data.get("ready_joint_positions", cls.ready_joint_positions))
        if len(ready_joint_positions) != 9:
            raise ValueError(
                "collection_startup.ready_joint_positions must contain 9 Franka joint values "
                f"(including two fingers), got {len(ready_joint_positions)}"
            )
        config = cls(
            ready_joint_positions=ready_joint_positions,
            drive_robot=bool(data.get("drive_robot", cls.drive_robot)),
            warmup_steps=int(data.get("warmup_steps", cls.warmup_steps)),
            stable_frames=int(data.get("stable_frames", cls.stable_frames)),
            joint_velocity_threshold=float(data.get("joint_velocity_threshold", cls.joint_velocity_threshold)),
            object_displacement_threshold=float(
                data.get("object_displacement_threshold", cls.object_displacement_threshold)
            ),
            max_wait_steps=int(data.get("max_wait_steps", cls.max_wait_steps)),
        )
        if config.warmup_steps < 0 or config.stable_frames < 1 or config.max_wait_steps < 1:
            raise ValueError("collection_startup frame counts must be non-negative and stable_frames/max_wait_steps positive")
        if config.joint_velocity_threshold < 0 or config.object_displacement_threshold < 0:
            raise ValueError("collection_startup stability thresholds must be non-negative")
        return config


@dataclass(frozen=True)
class StartupProgress:
    ready: bool
    timed_out: bool
    steps: int
    stable_frames: int
    max_joint_velocity: float
    max_object_displacement: float


class CollectionStartupGate:
    """Wait for a warm, stationary scene before recording an episode."""

    def __init__(self, config: CollectionStartupConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._steps = 0
        self._stable_frames = 0
        self._previous_object_positions: list[np.ndarray] | None = None
        self._ready = False

    @property
    def ready_joint_positions(self) -> np.ndarray:
        return np.asarray(self.config.ready_joint_positions, dtype=np.float32)

    @property
    def is_ready(self) -> bool:
        return self._ready

    def observe(
        self,
        joint_velocities: Any,
        object_positions: Iterable[Any],
    ) -> StartupProgress:
        """Consume one physics frame and report whether collection may start."""
        if self._ready:
            return StartupProgress(True, False, self._steps, self._stable_frames, 0.0, 0.0)

        self._steps += 1
        velocities = np.asarray(joint_velocities, dtype=np.float64).reshape(-1)
        max_joint_velocity = float(np.max(np.abs(velocities))) if velocities.size else float("inf")

        current_positions = [
            np.asarray(position, dtype=np.float64).reshape(-1)[:3]
            for position in object_positions
            if position is not None and np.asarray(position).size >= 3
        ]
        max_object_displacement = float("inf")
        if self._previous_object_positions is not None and len(current_positions) == len(self._previous_object_positions):
            displacements = [
                float(np.linalg.norm(current - previous))
                for current, previous in zip(current_positions, self._previous_object_positions)
            ]
            max_object_displacement = max(displacements, default=0.0)
        self._previous_object_positions = [position.copy() for position in current_positions]

        after_warmup = self._steps >= self.config.warmup_steps
        stationary = (
            np.isfinite(max_joint_velocity)
            and max_joint_velocity <= self.config.joint_velocity_threshold
            and np.isfinite(max_object_displacement)
            and max_object_displacement <= self.config.object_displacement_threshold
        )
        self._stable_frames = self._stable_frames + 1 if after_warmup and stationary else 0
        self._ready = self._stable_frames >= self.config.stable_frames
        timed_out = not self._ready and self._steps >= self.config.max_wait_steps
        return StartupProgress(
            ready=self._ready,
            timed_out=timed_out,
            steps=self._steps,
            stable_frames=self._stable_frames,
            max_joint_velocity=max_joint_velocity,
            max_object_displacement=max_object_displacement,
        )


def task_startup_readiness(task: Any) -> tuple[bool, str]:
    """Evaluate an optional task-specific geometric startup predicate."""
    predicate = getattr(task, "get_startup_readiness", None)
    if not callable(predicate):
        return True, ""
    result = predicate()
    if isinstance(result, tuple):
        ready, detail = result
        return bool(ready), str(detail)
    return bool(result), ""
