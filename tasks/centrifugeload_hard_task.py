from __future__ import annotations

from typing import Any, Dict

import numpy as np
from pxr import Gf, UsdPhysics

from .base_task import BaseTask


class CentrifugeLoadHardTask(BaseTask):
    """State provider for the three-tube centrifuge loading task."""

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.sub_object1 = str(cfg.sub_object1)
        self.sub_object2 = str(cfg.sub_object2)
        self.sub_object3 = str(cfg.sub_object3)
        self.target_path = str(cfg.target_path)
        self.add_collider_to_prim(self.target_path + "/geometry/mesh")
        rigid_body = UsdPhysics.RigidBodyAPI.Get(self.stage, self.target_path)
        if rigid_body:
            rigid_body.CreateKinematicEnabledAttr().Set(True)
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
            object_path=self.sub_object3,
            position=np.asarray(self.cfg.preloaded_position, dtype=np.float64),
        )

    def step(self) -> Dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None
        target1 = np.asarray(self.cfg.target1_position, dtype=np.float64)
        target2 = np.asarray(self.cfg.target2_position, dtype=np.float64)
        return self.get_basic_state_info(
            object_path=self.sub_object1,
            target_path=self.target_path,
            additional_info={
                "target_position": target1,
                "target1_position": target1,
                "object2_position": self.object_utils.get_geometry_center(object_path=self.sub_object2),
                "object2_size": self.object_utils.get_object_size(object_path=self.sub_object2),
                "object2_name": self.sub_object2.rsplit("/", 1)[-1],
                "target2_position": target2,
                "preloaded_position": self.object_utils.get_geometry_center(object_path=self.sub_object3),
            },
        )
