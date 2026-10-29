from __future__ import annotations

from typing import Any, Dict

import numpy as np
from pxr import Gf, Usd, UsdGeom

from .base_task import BaseTask


class ClearResetTask(BaseTask):
    """State provider for clearing the bench: put the vial into the sample box
    and the tube onto the tube rack.

    Both insertion targets are derived from exact asset-local socket points.
    The tube's final centre is computed from the lower support surface plus
    half of its measured world-space height.
    """

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.sub_object = str(cfg.sub_object)
        self.sub_object1 = str(cfg.sub_object1)
        self.target_path = str(cfg.target_path)
        self.target_path1 = str(cfg.target_path1)
        self.add_collider_to_prim(self.target_path + "/geometry/mesh")
        self.add_collider_to_prim(self.target_path1 + "/geometry/mesh")

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        for obj_config in self.obj_configs:
            path = str(obj_config["path"])
            if self.stage.GetPrimAtPath(path).IsValid():
                self.randomize_object_position(path, obj_config["position_range"])

    def _local_point_to_world(self, prim_path: str, local_point) -> np.ndarray:
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Insertion target prim does not exist: {prim_path}")
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        point = transform.Transform(
            Gf.Vec3d(*np.asarray(local_point, dtype=np.float64))
        )
        return np.asarray(point, dtype=np.float64)

    def step(self) -> Dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        # SamplePlaceBox.usd contains a 12x5 hole array. The selected centre is
        # local [0.024, *, 0], not bbox centre + [0.053, -0.001]: with the
        # configured 2x scale the old approximation was about 5.1 mm off-axis.
        vial_socket = self._local_point_to_world(
            self.target_path,
            self.cfg.clear_reset_success.vial_socket_local_position,
        )
        vial_settled_center = self._local_point_to_world(
            self.target_path,
            self.cfg.clear_reset_success.vial_settled_center_local_position,
        )
        tube_socket = self._local_point_to_world(
            self.target_path1,
            self.cfg.clear_reset_success.tube_socket_local_position,
        )
        tube_bottom_seat = self._local_point_to_world(
            self.target_path1,
            self.cfg.clear_reset_success.tube_bottom_seat_local_position,
        )
        tube_size = np.asarray(
            self.object_utils.get_object_size(object_path=self.sub_object1),
            dtype=np.float64,
        )
        # The lower plate has a small support opening rather than a through
        # hole. A vertical tube settles with its bottom on that plate, so its
        # geometry centre is one half-length above the seat.
        tube_settled_center = tube_bottom_seat + np.array(
            [0.0, 0.0, 0.5 * tube_size[2]], dtype=np.float64
        )
        return self.get_basic_state_info(
            object_path=self.sub_object,
            target_path=self.target_path,
            additional_info={
                "target_position": vial_settled_center,
                "vial_socket_position": vial_socket,
                "object1_position": self.object_utils.get_geometry_center(object_path=self.sub_object1),
                "object1_size": tube_size,
                "object1_name": self.sub_object1.rsplit("/", 1)[-1],
                "tube_socket_position": tube_socket,
                "target1_inserted_center": tube_settled_center,
                "target1_position": tube_settled_center,
            },
        )
