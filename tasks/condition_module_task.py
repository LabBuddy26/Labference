from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base_task import BaseTask


class ConditionModuleTask(BaseTask):
    """State provider for the condition-module task: place the beaker on the
    heat device, press its button, then return the beaker.

    All scene paths (beaker / device / button) come from ``cfg.task`` instead
    of the hard-coded LabBench paths.  The device-top target keeps the
    LabBench offset (``geometry center + [0.05, 0, 0.1]``); the button pose is
    captured at the first step so snapshot logic can measure displacement.
    """

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.beaker_path = str(cfg.task.beaker_path)
        self.device_path = str(cfg.task.device_path)
        self.button_path = str(cfg.task.button_path)
        self.source_obj = self.beaker_path
        self.target_path = self.device_path
        self._initial_button_position = None
        self._initial_beaker_position = None

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        self._initial_button_position = None
        self._initial_beaker_position = None
        for obj_config in self.obj_configs:
            path = str(obj_config["path"])
            if self.stage.GetPrimAtPath(path).IsValid():
                self.randomize_object_position(path, obj_config["position_range"])

    def step(self) -> Dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        beaker_position = self.object_utils.get_geometry_center(object_path=self.beaker_path)
        button_position = self.object_utils.get_geometry_center(object_path=self.button_path)
        if button_position is None:
            raise RuntimeError(
                f"ConditionModule button prim is missing or has no geometry: {self.button_path}"
            )
        if self._initial_button_position is None and button_position is not None:
            self._initial_button_position = np.asarray(button_position, dtype=np.float64).copy()
        if self._initial_beaker_position is None and beaker_position is not None:
            self._initial_beaker_position = np.asarray(beaker_position, dtype=np.float64).copy()

        device_center = np.asarray(
            self.object_utils.get_geometry_center(object_path=self.device_path),
            dtype=np.float64,
        )
        return self.get_basic_state_info(
            object_path=self.beaker_path,
            target_path=self.device_path,
            additional_info={
                "target_position": device_center + np.array([0.05, 0.0, 0.1], dtype=np.float64),
                "beaker_position": beaker_position,
                "beaker_size": self.object_utils.get_object_size(object_path=self.beaker_path),
                "beaker_initial_position": (
                    None if self._initial_beaker_position is None else self._initial_beaker_position.copy()
                ),
                "button_position": button_position,
                "initial_button_position": (
                    None if self._initial_button_position is None else self._initial_button_position.copy()
                ),
            },
        )
