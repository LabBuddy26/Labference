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

import os
from typing import Optional

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import SingleArticulation


class OpenArmRMPFlowController(mg.MotionPolicyController):
    """RMPFlow Controller for OpenArm (supports left or right arm).

    Args:
        name (str): Controller name.
        robot_articulation (SingleArticulation): The OpenArm articulation.
        arm (str): "left" or "right". Defaults to "left".
        physics_dt (float, optional): Physics timestep. Defaults to 1.0/60.0.
        robot_description_path (Optional[str], optional): Path to robot_descriptor.yaml.
        urdf_path (Optional[str], optional): Path to the OpenArm URDF.
        rmpflow_config_path (Optional[str], optional): Path to rmpflow_config.yaml.
        end_effector_frame_name (Optional[str], optional): End-effector frame name.
    """

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        arm: str = "left",
        physics_dt: float = 1.0 / 60.0,
        robot_description_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        rmpflow_config_path: Optional[str] = None,
        end_effector_frame_name: Optional[str] = None,
    ) -> None:

        self.arm = arm.lower()
        if self.arm not in ["left", "right"]:
            raise ValueError(f"arm must be 'left' or 'right', got {arm}")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        rmpflow_dir = os.path.join(current_dir, "rmpflow")
        
        if robot_description_path is None:
            robot_description_path = os.path.join(
                rmpflow_dir, f"openarm_v1_{self.arm}_robot_descriptor.yaml"
            )
        if urdf_path is None:
            urdf_path = os.path.join(rmpflow_dir, "openarm_v1.urdf")
        if rmpflow_config_path is None:
            rmpflow_config_path = os.path.join(
                rmpflow_dir, f"openarm_v1_{self.arm}_rmpflow_common.yaml"
            )
        if end_effector_frame_name is None:
            end_effector_frame_name = f"openarm_{self.arm}_hand_tcp"

        
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path,
            rmpflow_config_path=rmpflow_config_path,
            end_effector_frame_name=end_effector_frame_name,
            maximum_substep_size=0.00334,
        )

        self.articulation_rmp = mg.ArticulationMotionPolicy(
            robot_articulation, self.rmp_flow, physics_dt
        )

        mg.MotionPolicyController.__init__(
            self, name=name, articulation_motion_policy=self.articulation_rmp
        )

        
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()

        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )
        return

    def reset(self):
        """Reset the controller and restore base pose."""
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )