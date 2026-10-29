from __future__ import annotations

from typing import Any

import numpy as np


class ObservationManager:
    def __init__(self, *, scene: Any, resolved_robot: Any, instruction: str):
        self.scene = scene
        self.robot = resolved_robot.robot
        self.arm_entity_cfg = resolved_robot.arm_entity_cfg
        self.gripper_entity_cfg = resolved_robot.gripper_entity_cfg
        self.instruction = instruction

    def get_observation(self) -> dict[str, Any]:
        rgb = self.scene["camera"].data.output["rgb"].cpu().numpy()
        rgb_wrist = self.scene["camera2"].data.output["rgb"].cpu().numpy()
        joint_pos_tensor = self.robot.data.joint_pos
        joint_vel_tensor = self.robot.data.joint_vel
        joint_pos = joint_pos_tensor.cpu().numpy().astype(np.float64)
        joint_vel = joint_vel_tensor.cpu().numpy().astype(np.float64)
        ee_pose = self.robot.data.body_state_w[0, self.arm_entity_cfg.body_ids[0]]
        ee_pos = ee_pose[:3].detach().cpu().numpy().astype(np.float64)
        ee_quat = ee_pose[3:7].detach().cpu().numpy().astype(np.float64)
        gripper = joint_pos[:, self.gripper_entity_cfg.joint_ids]

        legacy = {
            "video.main_camera": rgb,
            "video.wrist_camera": rgb_wrist,
            "state.single_arm": joint_pos[:, :7],
            "state.gripper": gripper,
            "annotation.human.task_description": [self.instruction],
        }
        standard = {
            "observation": {
                "images": {
                    "main_camera": rgb,
                    "wrist_fisheye": rgb_wrist,
                },
                "robot_state": {
                    "joint_positions": joint_pos[0].tolist(),
                    "joint_velocities": joint_vel[0].tolist(),
                    "eef_pose": [*ee_pos.tolist(), *ee_quat.tolist()],
                    "gripper_state": float(gripper[0, 0]) if gripper.size else 0.0,
                },
            }
        }
        return {"legacy": legacy, "standard": standard}
