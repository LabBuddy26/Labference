from __future__ import annotations

from typing import Any, Dict

import numpy as np
from pxr import Gf, Usd, UsdGeom

from .base_task import BaseTask


class SampleOrganitionTask(BaseTask):
    """State provider for placing the vial into one of two sample boxes.

    The target box is chosen deterministically in ``reset`` from the global
    NumPy RNG.  ``runtime.reset_episode`` calls ``seed_everything(seed)``
    before ``task.reset``, so the choice is fully determined by the episode
    seed.  ``np.random.randint(1, 10) < 5`` mirrors LabBench's
    ``random.randint(1, 10) < 5``: box A is the target with probability 0.4.
    """

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.sub_object = str(cfg.sub_object)
        self.target_path = str(cfg.target_path)
        self.target_path1 = str(cfg.target_path1)
        self.target_final = None
        self.add_collider_to_prim(self.target_path + "/geometry/mesh", sdf_resolution=512)
        self.add_collider_to_prim(self.target_path1 + "/geometry/mesh", sdf_resolution=512)

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        for obj_config in self.obj_configs:
            path = str(obj_config["path"])
            if self.stage.GetPrimAtPath(path).IsValid():
                self.randomize_object_position(path, obj_config["position_range"])
        self.target_final = self.target_path if np.random.randint(1, 10) < 5 else self.target_path1

    def _local_to_world(self, path: str, local_point) -> np.ndarray:
        prim = self.stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise ValueError(f"Sample box does not exist: {path}")
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.asarray(transform.Transform(Gf.Vec3d(*np.asarray(local_point, dtype=np.float64))), dtype=np.float64)

    def _box_points(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        return (self._local_to_world(path, [0.024, 0.020, 0.0]),
                self._local_to_world(path, [0.024, 0.010, 0.0]))

    def step(self) -> Dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        chosen = self.target_final or self.target_path
        chosen_socket, chosen_center = self._box_points(chosen)
        target_a_socket, target_a_center = self._box_points(self.target_path)
        target_b_socket, target_b_center = self._box_points(self.target_path1)
        return self.get_basic_state_info(
            object_path=self.sub_object,
            target_path=chosen,
            additional_info={
                "target_position": chosen_center,
                "target_socket_position": chosen_socket,
                "target_box_name": chosen.rsplit("/", 1)[-1],
                "target_a_position": target_a_center,
                "target_a_socket_position": target_a_socket,
                "target_b_position": target_b_center,
                "target_b_socket_position": target_b_socket,
            },
        )
