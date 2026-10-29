import numpy as np
from .base_task import BaseTask
from typing import Dict, Any

class LiquidMixing(BaseTask):
    """Level 4 composite task: Open Door + Transfer Beaker + Stir
    
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
        self.beaker_path = "/World/beaker_4"
        self.target_plat_path = "/World/plat"
        
    def reset(self):
        """Reset task state and scene"""
        super().reset()
        self.robot.initialize()
        
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