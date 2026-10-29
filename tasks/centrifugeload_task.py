from __future__ import annotations

from typing import Any, Dict

import numpy as np
from pxr import Gf, UsdPhysics

from .base_task import BaseTask


class CentrifugeLoadTask(BaseTask):
    """State provider for loading two NMR tubes into the centrifuge.

    Exposes both tube poses and the two rotor-slot targets.  The first slot is
    the LabBench offset (``geometry center + [0.016, -0.03, 0]``); the second
    slot mirrors it so the two tubes land on opposite sides of the rotor.
    """

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.sub_object1 = str(cfg.sub_object1)
        self.sub_object2 = str(cfg.sub_object2)
        self.target_path = str(cfg.target_path)
        self.add_collider_to_prim(self.target_path + "/geometry/mesh")
        # The source USD incorrectly defines the complete benchtop centrifuge
        # as a free 0.5 kg rigid body. Robot contact then moves the machine and
        # its holes during placement. Keep the fixture rigid but kinematic so
        # collisions remain active without allowing the device to drift.
        rigid_body = UsdPhysics.RigidBodyAPI.Get(self.stage, self.target_path)
        if rigid_body:
            rigid_body.CreateKinematicEnabledAttr(True).Set(True)
            rigid_body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            rigid_body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        for obj_config in self.obj_configs:
            path = str(obj_config["path"])
            if self.stage.GetPrimAtPath(path).IsValid():
                self.randomize_object_position(path, obj_config["position_range"])
        self.object_utils.set_object_position(
            object_path=self.sub_object2,
            position=np.asarray(self.cfg.preloaded_position, dtype=np.float64),
        )

    def _slot_offsets(self) -> list[np.ndarray]:
        """Mirrored rotor-hole offsets around the centrifuge geometry center."""
        raw_offsets = self.cfg.centrifuge_success.get(
            "target_slot_offsets",
            [[0.016, -0.03, 0.0], [-0.016, 0.03, 0.0]],
        )
        return [np.asarray(offset, dtype=np.float64) for offset in raw_offsets]

    def step(self) -> Dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        slots = [
            np.asarray(self.cfg.target1_position, dtype=np.float64),
            np.asarray(self.cfg.preloaded_position, dtype=np.float64),
        ]
        return self.get_basic_state_info(
            object_path=self.sub_object1,
            target_path=self.target_path,
            additional_info={
                "nmr_02_position": self.object_utils.get_geometry_center(object_path=self.sub_object2),
                "nmr_02_size": self.object_utils.get_object_size(object_path=self.sub_object2),
                "nmr_02_name": self.sub_object2.rsplit("/", 1)[-1],
                "target_position": slots[0],
                "target_02_position": slots[1],
            },
        )
