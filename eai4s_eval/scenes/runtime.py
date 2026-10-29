from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RuntimeResetResult:
    target_object: Any
    original_object_pos: Any
    original_object_quat: Any


class SceneRuntime:
    scene_id = "base"
    target_object_name = "vial"
    distractor_names: tuple[str, ...] = ()

    def build_scene_cfg(self, *, robot_adapter: Any, asset_root: str | None, num_envs: int) -> Any:
        raise NotImplementedError

    def resolve_target_object_name(self, requested_target: str | None = None) -> str:
        return requested_target or self.target_object_name

    def after_setup(self, *, scene: Any, sim: Any, device: str) -> None:
        return None

    def reset_episode(
        self,
        *,
        scene: Any,
        sim: Any,
        robot: Any,
        adapter: Any,
        target_object: Any,
        original_object_pos: Any,
        original_object_quat: Any,
        seed: int,
        split: str,
        device: str,
        step_sim: Any,
        distractor_objects: dict[str, Any],
        distractor_original_pos: dict[str, Any],
    ) -> RuntimeResetResult:
        raise NotImplementedError

    def build_success_snapshot(
        self,
        *,
        scene: Any,
        target_object: Any,
        initial_object_pos: Any,
        robot: Any,
        resolved_robot: Any,
        step_index: int | None,
        device: str,
    ) -> dict[str, Any]:
        root_state = target_object.data.root_state_w[0]
        hand_pos = robot.data.body_state_w[0, resolved_robot.arm_entity_cfg.body_ids[0], :3]
        gripper_pos = robot.data.joint_pos[0, resolved_robot.gripper_entity_cfg.joint_ids]
        root_np = root_state.detach().cpu().numpy()
        return {
            "step_index": step_index,
            "object_pos": root_np[:3],
            "object_quat": root_np[3:7],
            "initial_object_pos": initial_object_pos.detach().cpu().numpy(),
            "hand_pos": hand_pos.detach().cpu().numpy(),
            "gripper_pos": gripper_pos.detach().cpu().numpy(),
            "object_lin_vel": root_np[7:10] if root_np.shape[0] >= 10 else [0.0, 0.0, 0.0],
            "object_ang_vel": root_np[10:13] if root_np.shape[0] >= 13 else [0.0, 0.0, 0.0],
            "task_state": {},
        }

    def build_quality_metrics(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return dict(snapshot.get("task_state") or {})


def split_xy_range(split: str) -> float:
    if split == "sim_clean":
        return 0.01
    if split == "sim_dev":
        return 0.03
    return 0.05


def randomized_yaw_deg(split: str, rng: np.random.Generator) -> float:
    if split != "sim_randomized":
        return 0.0
    return float(rng.uniform(-30.0, 30.0))
