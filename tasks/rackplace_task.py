from typing import Dict, Any
from .base_task import BaseTask
from isaacsim.core.utils.prims import set_prim_visibility
import numpy as np
import random

class RackPlaceTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        """Initialize the Sample Transport task.

        Args:
            cfg: Configuration object for the task.
            world: The simulation world instance.
            stage: The USD stage for the simulation.
            robot: The robot instance used in the task.
        """
        super().__init__(cfg, world, stage, robot)
        self.sub_object = cfg.sub_object
        self.target_path = cfg.target_path
        self.hole_index = None
        self.add_collider_to_prim(self.target_path + "/geometry/mesh")

    def reset(self):
        """Reset the task state."""
        super().reset()
        self.robot.initialize()
        for obj_config in self.obj_configs:
            obj_path = obj_config['path']
            position_range = obj_config['position_range']
            prim = self.stage.GetPrimAtPath(obj_path)
            if prim.IsValid():
                self.randomize_object_position(obj_path, position_range)

        self.hole_index = random.randint(0, 2)

    def get_basic_state_info(self, joint_positions: np.ndarray = None, object_path: str = None,
                           target_path: str = None, additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get basic state information containing common state data for all tasks.

        Args:
            joint_positions: Robot joint positions
            object_path: Object path
            target_path: Target object path
            additional_info: Additional state information

        Returns:
            Dict[str, Any]: Basic state information dictionary
        """

        if joint_positions is None:
            joint_positions = self.robot.get_joint_positions()
            if joint_positions is None:
                return None


        camera_data, display_data = self.get_camera_data()
        state = {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': display_data,
            'done': self.reset_needed,
            'gripper_position': self.robot.get_gripper_position(),
        }
        if object_path:
            state.update({
                'object_position': self.object_utils.get_geometry_center(object_path=object_path),
                'object_size': self.object_utils.get_object_size(object_path=object_path),
                'object_path': object_path,
                'object_name': object_path.split("/")[-1]
            })


        if target_path:
            state.update({
                'target_position': self.object_utils.get_geometry_center(object_path=target_path),
                'target_size': self.object_utils.get_object_size(object_path=target_path),
                'target_path': target_path,
                'target_name': target_path.split("/")[-1]
            })

        if additional_info:
            state.update(additional_info)

        return state
    def step(self):
        """Execute one simulation step.

        Returns:
            dict: A dictionary containing simulation state data, or None if not ready.
        """
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        pb = np.array([ 0.22605764, 0.03896235, 0.73949984])
        pb1 = np.array([ 0.29105763, 0.03894274, 0.73949984])
        pb2 = np.array([ 0.22604255, -0.01103764,  0.73949984])
        pb3 = np.array([ 0.29104254, -0.01105725,  0.73949984])
        pb4 = np.array([ 0.22602746, -0.06103764,  0.73949984])
        pb5 = np.array([ 0.29102745, -0.06105725,  0.73949984])
        pb6 = np.array([ 0.22601237, -0.11103763,  0.73949984])
        pb7 = np.array([ 0.29101237, -0.11105725,  0.73949984])
        pb8 = np.array([ 0.22599728, -0.16103763,  0.73949984])
        pb9 = np.array([ 0.29099728, -0.16105725,  0.73949984])
        pb10 = np.array([ 0.22598219, -0.21103763,  0.73949984])
        pb11 = np.array([ 0.29098219, -0.21105725,  0.73949984])

        target_position = pb + np.array([0.0052, -0.0126, 0.2])
        target1_position = pb1 + np.array([-0.006, -0.012, 0.2])
        target2_position = pb5 + np.array([-0.005, -0.011, 0.2])
        pos_list = [target_position, target1_position, target2_position]

        return self.get_basic_state_info(
            object_path=self.sub_object,
            target_path=self.target_path,
            additional_info={
                'target1_position': pos_list[self.hole_index],
            }
        )


