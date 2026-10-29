from __future__ import annotations

import random
from typing import Any, Dict, Sequence

import numpy as np

from .base_task import BaseTask


class PlatePositionTask(BaseTask):
    """R11: move the gripper above the instructed sample-box socket."""

    SLOT_OFFSETS: Sequence[tuple[float, float, float]] = (
        (0.050, -0.002, 0.050),
        (0.082, -0.002, 0.050),
        (0.114, -0.002, 0.050),
        (0.050, 0.020, 0.050),
        (0.050, -0.028, 0.050),
    )
    TARGET_LABELS = (
        "column 5 row 3",
        "column 4 row 3",
        "column 3 row 3",
        "column 5 row 2",
        "column 5 row 4",
    )

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.sub_object = str(cfg.sub_object)
        self.target_path = str(cfg.target_path)
        self.target_index = 0
        self.add_collider_to_prim(self.target_path + "/geometry/mesh")

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        self.target_index = random.randrange(len(self.SLOT_OFFSETS))

    def step(self) -> Dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None
        center = np.asarray(
            self.object_utils.get_geometry_center(object_path=self.target_path),
            dtype=np.float64,
        )
        target = center + np.asarray(self.SLOT_OFFSETS[self.target_index])
        label = self.TARGET_LABELS[self.target_index]
        return self.get_basic_state_info(
            object_path=self.sub_object,
            target_path=self.target_path,
            additional_info={
                "target_position": target,
                "target1_position": target,
                "target_label": label,
                "task_instruction": f"Move the gripper above sample-box socket {label}",
            },
        )
