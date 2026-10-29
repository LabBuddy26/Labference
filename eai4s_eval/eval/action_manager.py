from __future__ import annotations

from typing import Any

import numpy as np

from eai4s_eval.policy import ActionBatch, PolicyOutputInvalid
from eai4s_eval.utils.math import normalize_quat_wxyz


class ActionManager:
    def __init__(
        self,
        *,
        adapter: Any,
        robot: Any,
        controller: Any,
        arm_entity_cfg: Any,
        gripper_entity_cfg: Any,
        num_envs: int,
        device: str,
        max_delta_xyz: float = 0.03,
    ):
        self.adapter = adapter
        self.robot = robot
        self.controller = controller
        self.arm_entity_cfg = arm_entity_cfg
        self.gripper_entity_cfg = gripper_entity_cfg
        self.num_envs = num_envs
        self.device = device
        self.max_delta_xyz = max_delta_xyz
        self.open_tensor, self.close_tensor = adapter.make_gripper_tensors(
            num_envs, len(gripper_entity_cfg.joint_ids), device
        )

    def validate(self, batch: ActionBatch) -> None:
        if batch.ee_pos.ndim != 2 or batch.ee_pos.shape[1] != 3:
            raise PolicyOutputInvalid("ee_pos must have shape [T, 3]")
        if batch.ee_quat.ndim != 2 or batch.ee_quat.shape[1] != 4:
            raise PolicyOutputInvalid("ee_quat must have shape [T, 4]")
        if batch.gripper.shape[0] != batch.ee_pos.shape[0]:
            raise PolicyOutputInvalid("gripper horizon must match ee_pos")
        if not np.isfinite(batch.ee_pos).all() or not np.isfinite(batch.ee_quat).all():
            raise PolicyOutputInvalid("policy output contains NaN or inf")

    def current_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        from isaaclab.utils.math import subtract_frame_transforms

        root_pose = self.robot.data.root_state_w
        current_ee = self.robot.data.body_state_w[:, self.arm_entity_cfg.body_ids[0]]
        current_pos_b, current_quat_b = subtract_frame_transforms(
            root_pose[:, :3],
            root_pose[:, 3:7],
            current_ee[:, :3],
            current_ee[:, 3:7],
        )
        current_pos = current_pos_b[0].detach().cpu().numpy().astype(np.float32)
        current_quat = current_quat_b[0].detach().cpu().numpy().astype(np.float32)
        return current_pos, current_quat

    def apply_index(self, batch: ActionBatch, index: int) -> dict[str, Any]:
        self.validate(batch)
        idx = min(index, batch.ee_pos.shape[0] - 1)
        return self.apply_target(
            target_pos=batch.ee_pos[idx],
            target_quat=batch.ee_quat[idx],
            raw_gripper=float(batch.gripper[idx]),
        )

    def apply_target(self, *, target_pos: Any, target_quat: Any, raw_gripper: float) -> dict[str, Any]:
        import torch

        current_pos, _ = self.current_ee_pose()
        target_pos = np.asarray(target_pos, dtype=np.float32)
        delta = target_pos - current_pos
        delta_norm = float(np.linalg.norm(delta))
        clipped = False
        if delta_norm > self.max_delta_xyz:
            target_pos = current_pos + delta / max(delta_norm, 1e-9) * self.max_delta_xyz
            clipped = True
        target_quat = normalize_quat_wxyz(target_quat).astype(np.float32)

        ee_pos = torch.tensor(target_pos, device=self.device).reshape(1, 3)
        ee_quat = torch.tensor(target_quat, device=self.device).reshape(1, 4)
        raw_gripper_tensor = torch.tensor([raw_gripper], device=self.device)
        gripper_cmd = self.adapter.map_gripper_command(raw_gripper_tensor, self.open_tensor, self.close_tensor)
        self.adapter.apply_action(
            robot=self.robot,
            controller=self.controller,
            arm_entity_cfg=self.arm_entity_cfg,
            gripper_entity_cfg=self.gripper_entity_cfg,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            gripper_cmd=gripper_cmd,
        )
        return {
            "ee_pos": target_pos.tolist(),
            "ee_quat": target_quat.tolist(),
            "gripper_raw": float(raw_gripper),
            "clipped": clipped,
        }
