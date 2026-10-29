from .base_task import BaseTask
from isaacsim.core.utils.prims import set_prim_visibility
import numpy as np

class PickPourV2Task(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        """Initialize the Pick and Pour task.

        Args:
            cfg: Configuration object for the task.
            world: The simulation world instance.
            stage: The USD stage for the simulation.
            robot: The robot instance used in the task.
        """
        super().__init__(cfg, world, stage, robot)
        self.sub_object = cfg.sub_object
        self.target_path = cfg.target_path

    def reset(self):
        """Reset the task state."""
        super().reset()
        self.robot.initialize()
        # target_beaker
        self.object_utils.set_object_position(object_path=self.target_path, position=np.array([0.24125 + np.random.uniform(-0.075, 0.075), -0.31358 + np.random.uniform(-0.075, 0.075), 0.77]))

        # test_tube_rack
        # rack_position = np.array([0.28421, 0.30755, 0.82291])
        rack_position = np.array([0.28421, 0.30755, 0.80])
        self.object_utils.set_object_position(object_path="/World/tube_rack", position=rack_position)

        # glass_tube
        self.object_utils.set_object_position(object_path=self.sub_object, position=rack_position + np.array([-0.02301, -0.1125, 0.03197]))
    
    def step(self):
        """Execute one simulation step.

        Returns:
            dict: A dictionary containing simulation state data, or None if not ready.
        """
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        source_quaternion = self.object_utils.get_transform_quat(object_path=self.sub_object)
        
        return self.get_basic_state_info(
            object_path=self.sub_object,
            target_path=self.target_path,
            additional_info={
                'object_quaternion': source_quaternion,
            }
        )

        
