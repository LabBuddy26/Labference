# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import os
import carb
import numpy as np
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.physics import ContactSensor
from isaacsim.sensors.camera import Camera

from utils.object_utils import ObjectUtils

class OpenArm(Robot):
    """OpenArm bimanual robot class.

    Args:
        prim_path (str): Prim path of the robot.
        name (str, optional): Name of the robot. Defaults to "openarm".
        usd_path (Optional[str], optional): Path to the OpenArm USD. Defaults to None.
        position (Optional[np.ndarray], optional): World position. Defaults to None.
        orientation (Optional[np.ndarray], optional): World orientation (wxyz). Defaults to None.
        end_effector_prim_name_left (Optional[str], optional): Left end-effector prim name.
            Defaults to "openarm_left_right_finger".
        end_effector_prim_name_right (Optional[str], optional): Right end-effector prim name.
            Defaults to "openarm_right_hand_tcp".
        gripper_dof_names_left (Optional[List[str]], optional): Left gripper joint names.
        gripper_dof_names_right (Optional[List[str]], optional): Right gripper joint names.
        gripper_open_position (Optional[np.ndarray], optional): Open positions for both fingers.
        gripper_closed_position (Optional[np.ndarray], optional): Closed positions for both fingers.
        deltas (Optional[np.ndarray], optional): Action deltas for grippers.
    """

    def __init__(
        self,
        prim_path: str = "/World/OpenArm",
        name: str = "openarm",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name_left: Optional[str] = None,
        end_effector_prim_name_right: Optional[str] = None,
        gripper_dof_names_left: Optional[List[str]] = None,
        gripper_dof_names_right: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        deltas: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector_left = None
        self._end_effector_right = None
        self._gripper_left = None
        self._gripper_right = None
        self.prim_path_str = prim_path

        
        if end_effector_prim_name_left is None:
            end_effector_prim_name_left = "openarm_left_right_finger"
        if end_effector_prim_name_right is None:
            end_effector_prim_name_right = "openarm_right_right_finger"

        self._end_effector_prim_path_left = f"{prim_path}/{end_effector_prim_name_left}"
        self._end_effector_prim_path_right = f"{prim_path}/{end_effector_prim_name_right}"

        
        if gripper_dof_names_left is None:
            gripper_dof_names_left = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
        if gripper_dof_names_right is None:
            gripper_dof_names_right = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]

        if gripper_open_position is None:
            gripper_open_position = np.array([0.04, 0.04]) / get_stage_units()
        if gripper_closed_position is None:
            gripper_closed_position = np.array([0.0, 0.0])

        
        if not prim.IsValid():
            if usd_path:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                add_reference_to_stage(usd_path=os.path.join(current_dir, usd_path), prim_path=prim_path)
            else:
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                carb.log_warn(
                    "No usd_path provided and default OpenArm USD path is not set. "
                    "Please pass usd_path explicitly."
                )
        
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None,
        )

        if deltas is None:
            deltas = np.array([0.04, 0.04]) / get_stage_units()

        self._gripper_left = ParallelGripper(
            end_effector_prim_path=self._end_effector_prim_path_left,
            joint_prim_names=gripper_dof_names_left,
            joint_opened_positions=gripper_open_position,
            joint_closed_positions=gripper_closed_position,
            action_deltas=deltas,
        )

        self._gripper_right = ParallelGripper(
            end_effector_prim_path=self._end_effector_prim_path_right,
            joint_prim_names=gripper_dof_names_right,
            joint_opened_positions=gripper_open_position,
            joint_closed_positions=gripper_closed_position,
            action_deltas=deltas,
        )

        
        self.left_arm_left_contact_sensor = ContactSensor(
            prim_path=f"{prim_path}/openarm_left_left_finger/contact_sensor",
            name="left_arm_left_contact_sensor",
            min_threshold=0,
            max_threshold=10000000,
            radius=0.05,
        )
        self.left_arm_right_contact_sensor = ContactSensor(
            prim_path=f"{prim_path}/openarm_left_right_finger/contact_sensor",
            name="left_arm_right_contact_sensor",
            min_threshold=0,
            max_threshold=10000000,
            radius=0.05,
        )

        self.right_arm_left_contact_sensor = ContactSensor(
            prim_path=f"{prim_path}/openarm_right_left_finger/contact_sensor",
            name="right_arm_left_contact_sensor",
            min_threshold=0,
            max_threshold=10000000,
            radius=0.05,
        )
        self.right_arm_right_contact_sensor = ContactSensor(
            prim_path=f"{prim_path}/openarm_right_right_finger/contact_sensor",
            name="right_arm_right_contact_sensor",
            min_threshold=0,
            max_threshold=10000000,
            radius=0.05,
        )

        try:
            self.left_camera = Camera(
                prim_path=f"{prim_path}/openarm_left_link7/arm_camera",
                translation=np.array([0.21, 0.05, 0.05]),
                frequency=60,
                resolution=(640, 480),
                orientation=np.array([1.0, -0.0, 0.0, -0.0])
            )
            self.left_camera.set_local_pose(orientation=np.array([-0.12279, -0.69636, -0.69636, -0.12279]), camera_axes="usd")
            self.left_camera.set_clipping_range(near_distance=0.1, far_distance=100.0)
            self.left_camera.set_focal_length(6.)

        except Exception:
            self.left_camera = None
            carb.log_warn("Left wrist camera prim not found, skipping.")

        try:
            self.right_camera = Camera(
                prim_path=f"{prim_path}/openarm_right_link7/arm_camera",
                translation=np.array([0.7, -0.025, -0.35]),
                frequency=60,
                resolution=(640, 480),
                orientation=np.array([1.0, -0.0, 0.0, -0.0])
            )
            self.right_camera.set_local_pose(orientation=np.array([0.29884, 0.64086, 0.64086, 0.29884]), camera_axes="usd")
            self.right_camera.set_clipping_range(near_distance=0.1, far_distance=100.0)
            self.right_camera.set_focal_length(6.)
        except Exception:
            self.right_camera = None
            carb.log_warn("Right wrist camera prim not found, skipping.")

        return

    def get_contact_sensors(self) -> Tuple:
        """Get all contact sensors.

        Returns:
            tuple: (left_arm_left, left_arm_right, right_arm_left, right_arm_right)
        """
        return (
            self.left_arm_left_contact_sensor,
            self.left_arm_right_contact_sensor,
            self.right_arm_left_contact_sensor,
            self.right_arm_right_contact_sensor,
        )

    @property
    def end_effector_left(self) -> SingleRigidPrim:
        return self._end_effector_left

    @property
    def end_effector_right(self) -> SingleRigidPrim:
        return self._end_effector_right

    @property
    def gripper_left(self) -> ParallelGripper:
        return self._gripper_left

    @property
    def gripper_right(self) -> ParallelGripper:
        return self._gripper_right

    
    @property
    def end_effector(self) -> SingleRigidPrim:
        return self._end_effector_left

    @property
    def gripper(self) -> ParallelGripper:
        return self._gripper_left


    def set_default_arm_pose(self) -> None:
        joint_positions = self.get_joint_positions()
        joint_positions[2] = -1.5 
        joint_positions[3] = 1.5
        
        self.set_joint_positions(joint_positions)
        self.set_joint_velocities(np.zeros(self.num_dof))

    def initialize(self, physics_sim_view=None) -> None:
        """Initialize the robot and grippers."""
        desired_position, desired_orientation = self.get_world_pose()
        super().initialize(physics_sim_view)

        self._end_effector_left = SingleRigidPrim(
            prim_path=self._end_effector_prim_path_left,
            name=self.name + "_end_effector_left",
        )
        self._end_effector_left.initialize(physics_sim_view)

        self._end_effector_right = SingleRigidPrim(
            prim_path=self._end_effector_prim_path_right,
            name=self.name + "_end_effector_right",
        )
        self._end_effector_right.initialize(physics_sim_view)

        self._gripper_left.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )

        self._gripper_right.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )
        if desired_position is not None:
            self.set_world_pose(
                position=desired_position,
                orientation=desired_orientation
            )

        
        kps, kds = self._articulation_controller.get_gains()
        kps[-4] = 800
        kps[-2] = 800

        kds[-4] = 40
        kds[-2] = 40

        self._articulation_controller.set_gains(kps=kps, kds=kds)
        self.set_default_arm_pose()

        return

    def post_reset(self) -> None:
        """Reset grippers after simulation reset."""
        super().post_reset()
        self._gripper_left.post_reset()
        self._gripper_right.post_reset()

        for gripper in [self._gripper_left, self._gripper_right]:
            for idx in gripper.joint_dof_indicies:
                self._articulation_controller.switch_dof_control_mode(
                    dof_index=idx, mode="position"
                )
        return

    def get_gripper_position(self, arm: str = "left") -> np.ndarray:
        """Get the tool center position of the specified arm.

        Args:
            arm (str): "left" or "right". Defaults to "left".

        Returns:
            np.ndarray: World position of the TCP.
        """
        
        if arm.lower() == "left":
            tcp_path = self.prim_path_str + "/openarm_left_link7/gripper_center"
        else:
            tcp_path = self.prim_path_str + "/openarm_right_link7/gripper_center"

        return ObjectUtils.get_instance().get_object_xform_position(object_path=tcp_path)

    def get_left_gripper_position(self) -> np.ndarray:
        return self.get_gripper_position(arm="left")

    def get_right_gripper_position(self) -> np.ndarray:
        return self.get_gripper_position(arm="right")