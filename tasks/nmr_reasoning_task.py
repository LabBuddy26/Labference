from __future__ import annotations

import random
from typing import Any, Dict, Sequence

import numpy as np

from .nmrinjection_task import NMRInjectionTask


class NMRReasoningTask(NMRInjectionTask):
    """Shared state provider for the R08/R09 target-selection tasks."""

    SLOT_OFFSETS: Sequence[tuple[float, float, float]] = (
        (0.050, -0.002, 0.0),
        (0.082, -0.002, 0.0),
        (0.114, -0.002, 0.0),
        (0.050, 0.020, 0.0),
        (0.050, -0.028, 0.0),
    )
    TARGET_LABELS: Sequence[str] = ()
    TASK_TEMPLATE: str = "Insert the NMR tube into {target_label}"

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.target_index = 0

    def reset(self) -> None:
        super().reset()
        self.target_index = random.randrange(len(self.SLOT_OFFSETS))

    def step(self) -> Dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        center = np.asarray(
            self.object_utils.get_geometry_center(object_path=self.target_path),
            dtype=np.float64,
        )
        target = center + np.asarray(
            self.SLOT_OFFSETS[self.target_index], dtype=np.float64
        )
        target_label = self.TARGET_LABELS[self.target_index]
        return self.get_basic_state_info(
            object_path=self.sub_object,
            target_path=self.target_path,
            additional_info={
                "target_position": target,
                "target_label": target_label,
                "task_instruction": self.TASK_TEMPLATE.format(
                    target_label=target_label
                ),
            },
        )
