from __future__ import annotations

from typing import Any

from .base import ResolvedRobot, RobotAdapter, RobotAdapterNotConfigured


class UR5SingleAdapter(RobotAdapter):
    """Registered UR5 extension point.

    TODO before enabling:
    - Provide the UR5 USD/URDF/MJCF asset path and load config.
    - Confirm the six arm joint names and ordering.
    - Confirm gripper type, gripper joint names, open/close target values.
    - Confirm the end-effector body/link name used by Isaac Lab.
    - Confirm wrist/head camera mount prim paths and extrinsics.
    """

    robot_id = "ur5_single"
    action_space = "single_arm_absolute_ee_8d"
    joint_names = ["ur5_.*"]
    gripper_joint_names: list[str] = []
    ee_body_name = "tool0"
    camera_mounts = {"wrist_camera": "{ENV_REGEX_NS}/Robot/tool0/wrist_camera"}

    @property
    def is_configured(self) -> bool:
        return False

    def _raise(self) -> None:
        raise RobotAdapterNotConfigured(
            "ur5_single is registered but not configured yet. Add the UR5 asset path, "
            "joint names, gripper mapping, end-effector link, and camera mount in "
            "eai4s_eval/robots/ur5_single.py."
        )

    def build_robot_cfg(self, prim_path: str) -> Any:
        self._raise()

    def resolve_scene_entities(self, scene: Any) -> ResolvedRobot:
        self._raise()

    def create_controller(self, num_envs: int, device: str) -> Any:
        self._raise()

    def make_gripper_tensors(self, num_envs: int, joint_count: int, device: str) -> tuple[Any, Any]:
        self._raise()

    def map_gripper_command(self, raw_gripper: Any, open_tensor: Any, close_tensor: Any) -> Any:
        self._raise()

    def apply_action(self, **kwargs: Any) -> None:
        self._raise()
