from __future__ import annotations

from typing import Any

from .base import ResolvedRobot, RobotAdapter


class FrankaPandaAdapter(RobotAdapter):
    robot_id = "franka_panda"
    action_space = "single_arm_absolute_ee_8d"
    joint_names = ["panda_joint.*"]
    gripper_joint_names = ["panda_finger_joint.*"]
    ee_body_name = "panda_hand"
    camera_mounts = {"wrist_camera": "{ENV_REGEX_NS}/Robot/panda_hand/wrist_camera"}

    def build_robot_cfg(self, prim_path: str) -> Any:
        from isaaclab.assets import ArticulationCfg
        from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

        return FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path=prim_path,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.5, 0.0, 0.05),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.569,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.810,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3.037,
                    "panda_joint7": 0.741,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                },
            ),
        )

    def resolve_scene_entities(self, scene: Any) -> ResolvedRobot:
        from isaaclab.managers import SceneEntityCfg

        arm_cfg = SceneEntityCfg("robot", joint_names=self.joint_names, body_names=[self.ee_body_name])
        gripper_cfg = SceneEntityCfg("robot", joint_names=self.gripper_joint_names, body_names=[self.ee_body_name])
        arm_cfg.resolve(scene)
        gripper_cfg.resolve(scene)
        return ResolvedRobot(robot=scene["robot"], arm_entity_cfg=arm_cfg, gripper_entity_cfg=gripper_cfg)

    def create_controller(self, num_envs: int, device: str) -> Any:
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        return DifferentialIKController(cfg, num_envs=num_envs, device=device)

    def make_gripper_tensors(self, num_envs: int, joint_count: int, device: str) -> tuple[Any, Any]:
        import torch

        open_tensor = 0.04 * torch.ones((num_envs, joint_count), device=device)
        close_tensor = torch.zeros((num_envs, joint_count), device=device)
        return open_tensor, close_tensor

    def map_gripper_command(self, raw_gripper: Any, open_tensor: Any, close_tensor: Any) -> Any:
        import torch

        gripper = raw_gripper
        if not torch.is_tensor(gripper):
            gripper = torch.tensor(gripper, device=open_tensor.device)
        gripper = gripper.flatten()
        return torch.where(
            (gripper > 0.5).unsqueeze(1),
            open_tensor.expand(gripper.shape[0], open_tensor.shape[1]),
            close_tensor.expand(gripper.shape[0], close_tensor.shape[1]),
        )

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
        import torch
        from isaaclab.utils.math import subtract_frame_transforms

        controller.reset()
        root_pose = robot.data.root_state_w
        ee_pose = robot.data.body_state_w[:, arm_entity_cfg.body_ids[0]]
        ee_pos_w, ee_quat_w = ee_pose[:, :3], ee_pose[:, 3:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose[:, :3], root_pose[:, 3:7], ee_pos_w, ee_quat_w
        )

        command = torch.cat([ee_pos, ee_quat], dim=-1)
        controller.set_command(command)
        jacobian = robot.root_physx_view.get_jacobians()[
            :,
            arm_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else arm_entity_cfg.body_ids[0],
            :,
            arm_entity_cfg.joint_ids,
        ]
        joint_pos = robot.data.joint_pos[:, arm_entity_cfg.joint_ids]
        joint_pos_des = controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        robot.set_joint_position_target(joint_pos_des, joint_ids=arm_entity_cfg.joint_ids)
        robot.set_joint_position_target(gripper_cmd, joint_ids=gripper_entity_cfg.joint_ids)
