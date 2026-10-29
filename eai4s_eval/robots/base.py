from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class RobotAdapterError(RuntimeError):
    """Base robot adapter error."""


class RobotAdapterNotConfigured(RobotAdapterError):
    """Raised by registered robot stubs whose assets/config are not ready."""


@dataclass
class ResolvedRobot:
    robot: Any
    arm_entity_cfg: Any
    gripper_entity_cfg: Any


class RobotAdapter(ABC):
    robot_id: str
    action_space: str
    joint_names: list[str]
    gripper_joint_names: list[str]
    ee_body_name: str
    camera_mounts: dict[str, str]

    @property
    def is_configured(self) -> bool:
        return True

    @abstractmethod
    def build_robot_cfg(self, prim_path: str) -> Any:
        """Return an Isaac Lab robot config."""

    @abstractmethod
    def resolve_scene_entities(self, scene: Any) -> ResolvedRobot:
        """Resolve robot entities from an InteractiveScene."""

    @abstractmethod
    def create_controller(self, num_envs: int, device: str) -> Any:
        """Create the robot controller used by this adapter."""

    @abstractmethod
    def make_gripper_tensors(self, num_envs: int, joint_count: int, device: str) -> tuple[Any, Any]:
        """Return open/close gripper tensors for the robot."""

    @abstractmethod
    def map_gripper_command(self, raw_gripper: Any, open_tensor: Any, close_tensor: Any) -> Any:
        """Map model gripper output to robot joint targets."""

    @abstractmethod
    def apply_action(
        self,
        *,
        robot: Any,
        controller: Any,
        arm_entity_cfg: Any,
        gripper_entity_cfg: Any,
        ee_pos: Any,
        ee_quat: Any,
        gripper_cmd: Any,
    ) -> None:
        """Apply one already-validated action to the robot."""

    def reset_robot(self, robot: Any) -> None:
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos.clone(),
            robot.data.default_joint_vel.clone(),
        )
