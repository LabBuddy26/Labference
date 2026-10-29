from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base_task import BaseTask


class NMRInjectionTask(BaseTask):
    """State provider for grasping the NMR machine and inserting it into the
    sample box.

    The ``target_position`` carries the LabBench offset into the box opening
    (``geometry center + [0.05, -0.002, 0]``); the box mesh gets a
    GPU-compatible SDF collider so the inset object rests on real geometry.
    """

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.sub_object = str(cfg.sub_object)
        self.target_path = str(cfg.target_path)
        self.add_collider_to_prim(self.target_path + "/geometry/mesh")

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        for obj_config in self.obj_configs:
            path = str(obj_config["path"])
            if self.stage.GetPrimAtPath(path).IsValid():
                self.randomize_object_position(path, obj_config["position_range"])

    def step(self) -> Dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        center = self.object_utils.get_geometry_center(object_path=self.target_path)
        return self.get_basic_state_info(
            object_path=self.sub_object,
            target_path=self.target_path,
            additional_info={
                "target_position": np.asarray(center, dtype=np.float64) + np.array([0.05, -0.002, 0.0]),
            },
        )
