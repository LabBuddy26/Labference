import numpy as np
from .base_task import BaseTask
from typing import Dict, Any

class ConditionModuleTask(BaseTask):
    """
    This task consists of three stages:
    1. Door Opening Stage: Open the specified device door (drying oven or muffle furnace)
    2. Transfer Stage: Transfer the beaker from the table to the target platform
    3. Stirring Stage: Use a glass rod to stir in the beaker
    """
    
    def __init__(self, cfg, world, stage, robot):
        """Initialize the composite task
        
        Args:
            cfg: Task configuration
            world: Simulation world instance
            stage: USD stage
            robot: Robot instance
        """
        super().__init__(cfg, world, stage, robot)

        # Beaker and target platform
        self.beaker_path = "/World/beaker_03"
        self.target_plat_path = "/World/heat_device"
        
    def reset(self):
        """Reset task state and scene"""
        super().reset()
        self.robot.initialize()

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
                'target_position': self.object_utils.get_geometry_center(object_path=target_path) + np.array([0.05, 0, 0.1]),
                'target_size': self.object_utils.get_object_size(object_path=target_path),
                'target_path': target_path,
                'target_name': target_path.split("/")[-1]
            })
        
        if additional_info:
            state.update(additional_info)
            
        return state    
    def step(self) -> Dict[str, Any]:
        """Execute one step of the task
        
        Returns:
            Dict[str, Any]: Dictionary containing current state information
        """
        self.frame_idx += 1
        
        if not self.check_frame_limits(max_steps=6000):
            return None
        
        return self.get_basic_state_info(
            object_path=self.beaker_path,
            target_path=self.target_plat_path,
            additional_info={
            }
        ) 