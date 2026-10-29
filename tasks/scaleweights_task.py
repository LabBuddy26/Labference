from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base_task import BaseTask


class ScaleWeightsTask(BaseTask):
    """State provider for moving three weights onto the three scale pads.

    Additional-info keys are named ``weight_01/02_*`` instead of reusing the
    vial keys that LabBench copied from the cohort task.
    """

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.sub_object1 = str(cfg.sub_object1)
        self.sub_object2 = str(cfg.sub_object2)
        self.sub_object3 = str(cfg.sub_object3)
        self.target_path = str(cfg.target_path)

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        for obj_config in self.obj_configs:
            path = str(obj_config["path"])
            if self.stage.GetPrimAtPath(path).IsValid():
                self.randomize_object_position(path, obj_config["position_range"])

    def _slot_positions(self) -> list[np.ndarray]:
        center = np.asarray(
            self.object_utils.get_geometry_center(object_path=self.target_path),
            dtype=np.float64,
        )
        raw_offsets = self.cfg.scale_weights_success.get(
            "target_slot_offsets",
            [[-0.05, 0.0, 0.0], [0.0, 0.0, 0.0], [0.05, 0.0, 0.0]],
        )
        return [center + np.asarray(offset, dtype=np.float64) for offset in raw_offsets]

    def step(self) -> Dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        slots = self._slot_positions()
        return self.get_basic_state_info(
            object_path=self.sub_object1,
            target_path=self.target_path,
            additional_info={
                "weight_01_position": self.object_utils.get_geometry_center(object_path=self.sub_object2),
                "weight_02_position": self.object_utils.get_geometry_center(object_path=self.sub_object3),
                "weight_01_size": self.object_utils.get_object_size(object_path=self.sub_object2),
                "weight_02_size": self.object_utils.get_object_size(object_path=self.sub_object3),
                "weight_01_name": self.sub_object2.rsplit("/", 1)[-1],
                "weight_02_name": self.sub_object3.rsplit("/", 1)[-1],
                "target_position": slots[0],
                "target_01_position": slots[1],
                "target_02_position": slots[2],
            },
        )
