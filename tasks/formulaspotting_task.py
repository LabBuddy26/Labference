from typing import Dict, Any
from .base_task import BaseTask
from isaacsim.core.utils.prims import set_prim_visibility
import numpy as np
import random 
from pxr import UsdPhysics, PhysxSchema

class FormulaSpottingTask(BaseTask):
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
        self.add_collider_to_prim(self.target_path + "/geometry/mesh")

    def add_collider_to_prim(self, prim_path):
        stage = self.stage
        prim = stage.GetPrimAtPath(prim_path)
        
        mesh_col_api = UsdPhysics.MeshCollisionAPI(prim)
        mesh_col_api.GetApproximationAttr().Set("sdf")
        
        sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        sdf_api.CreateSdfResolutionAttr().Set(256)

        collision_api = UsdPhysics.CollisionAPI(prim)
        collision_api.GetCollisionEnabledAttr().Set(True)
        return True

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
        

                # 5c3r
        pb = np.array([0.05, -0.002, 0.1])
        # 4c3r
        pb1 = np.array([0.082, -0.002, 0.1])
        # 3c3r
        pb2 = np.array([0.114, -0.002, 0.1])
        # 5c2r
        pb3 = np.array([0.05, 0.02, 0.1])
        # 5c4r
        pb4 = np.array([0.05, -0.028, 0.1])
        
        
        return self.get_basic_state_info(
            object_path=self.sub_object,
            target_path=self.target_path,
            additional_info={
                'target1_position': self.object_utils.get_geometry_center(object_path=self.target_path) + pb1,
                'target2_position': self.object_utils.get_geometry_center(object_path=self.target_path) + pb2,
                'target3_position': self.object_utils.get_geometry_center(object_path=self.target_path) + pb3,
                'target4_position': self.object_utils.get_geometry_center(object_path=self.target_path) + pb4,


            }

        )

        
