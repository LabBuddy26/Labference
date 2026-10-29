import numpy as np
from .base_task import BaseTask

class StirTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.glass_rod = self.cfg.obj_path
        self.target_beaker = self.cfg.target_path
        self.glass_rod_mesh = self.cfg.sub_obj_path

    def reset(self):
        super().reset()
        self.robot.initialize()

        # target_beaker
        self.object_utils.set_object_position(object_path=self.target_beaker, position=np.array([0.24125 + np.random.uniform(-0.075, 0.075), -0.31358 + np.random.uniform(-0.075, 0.075), 0.77]))

        # test_tube_rack
        rack_position = np.asarray(
            self.cfg.get("stir_rack_position", [0.28421, 0.38755, 0.82291]),
            dtype=float,
        )
        self.object_utils.set_object_position(object_path="/World/tube_rack", position=rack_position)

        # glass_rod
        spawn_offset = np.asarray(
            self.cfg.get("rod_spawn_offset", [-0.01152, -0.1125, 0.03197]),
            dtype=float,
        )
        self.object_utils.set_object_position(
            object_path=self.glass_rod,
            position=rack_position + spawn_offset,
        )
        # Preserve the transform convention used by the successful collection
        # demonstrations: the referenced mesh is normalized under the public
        # root and is moved by the scripted gripper helper after acquisition.
        self.object_utils.set_object_position(self.glass_rod_mesh, [0.0, 0.0, 0.0])

    def get_startup_object_positions(self):
        """Use the authored scene roots during the pre-pick stability gate."""
        positions = []
        for path in (self.glass_rod, self.target_beaker):
            position = self.object_utils.get_object_xform_position(path)
            if position is not None:
                positions.append(position)
        return positions
            
    def step(self):
        self.frame_idx += 1
        
        if not self.check_frame_limits(max_steps=2000):
            return None

        root_position = self.object_utils.get_object_xform_position(self.glass_rod)
        grasp_position = None if root_position is None else root_position.copy()
        if grasp_position is not None:
            # The rod centre sits inside/below the rack opening.  Aim at the
            # exposed upper section so both fingers can reach it.
            grasp_position[2] += float(self.cfg.get("rod_grasp_height_offset", 0.08))
        return self.get_basic_state_info(
            object_path=self.glass_rod,
            target_path=self.target_beaker,
            additional_info={
                'target_beaker': self.target_beaker,
                # Use one world-space physical pose for scripted picking and
                # evaluation truth; mixing root and child transforms produced
                # an offset grasp target.
                'object_position': root_position,
                'grasp_position': grasp_position,
                'glass_rod_position': self.object_utils.get_object_xform_position(self.glass_rod_mesh),
            }
        )
