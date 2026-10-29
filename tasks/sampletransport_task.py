from __future__ import annotations

from typing import Any, Dict

import numpy as np
from pxr import Gf, Usd, UsdGeom

from .base_task import BaseTask


class SampleTransportTask(BaseTask):
    """Expose the three vial poses and three sample-box slot targets."""

    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.vial_paths = [str(cfg.sub_object1), str(cfg.sub_object2), str(cfg.sub_object3)]
        self.sub_object1, self.sub_object2, self.sub_object3 = self.vial_paths
        self.target_path = str(cfg.target_path)
        # The asset-authored convex decomposition can bridge across the small
        # vial holes and behave like a solid plate.  Match ClearResetTask by
        # using the visual mesh as an SDF collider so the openings remain
        # physically traversable.
        self.add_collider_to_prim(self.target_path + "/geometry/mesh", sdf_resolution=512)

    def reset(self) -> None:
        super().reset()
        self.robot.initialize()
        for obj_config in self.obj_configs:
            path = str(obj_config["path"])
            if self.stage.GetPrimAtPath(path).IsValid():
                self.randomize_object_position(path, obj_config["position_range"])

    def _slot_positions(self) -> list[np.ndarray]:
        local_positions = self.cfg.sample_transport_success.get(
            "slot_local_positions",
            [[0.024, 0.020, 0.0], [0.040, 0.020, 0.0], [0.056, 0.020, 0.0]],
        )
        prim = self.stage.GetPrimAtPath(self.target_path)
        if not prim.IsValid():
            raise ValueError(f"Sample box does not exist: {self.target_path}")
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return [np.asarray(transform.Transform(Gf.Vec3d(*np.asarray(point, dtype=np.float64))), dtype=np.float64)
                for point in local_positions]

    def _slot_inserted_centers(self) -> list[np.ndarray]:
        positions = self.cfg.sample_transport_success.get(
            "slot_local_positions",
            [[0.024, 0.020, 0.0], [0.040, 0.020, 0.0], [0.056, 0.020, 0.0]],
        )
        prim = self.stage.GetPrimAtPath(self.target_path)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return [np.asarray(transform.Transform(Gf.Vec3d(float(point[0]), 0.010, float(point[2]))), dtype=np.float64)
                for point in positions]

    def step(self) -> Dict[str, Any] | None:
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        vial_positions = [
            self.object_utils.get_geometry_center(object_path=path) for path in self.vial_paths
        ]
        vial_sizes = [self.object_utils.get_object_size(object_path=path) for path in self.vial_paths]
        slots = self._slot_positions()
        inserted_centers = self._slot_inserted_centers()
        return self.get_basic_state_info(
            object_path=self.vial_paths[0],
            target_path=self.target_path,
            additional_info={
                "vial_positions": vial_positions,
                "vial_sizes": vial_sizes,
                "vial_paths": list(self.vial_paths),
                "target_slot_positions": slots,
                "target_slot_inserted_centers": inserted_centers,
                # Compatibility keys used by the original LabBench controller.
                "vial_01_position": vial_positions[1],
                "vial_02_position": vial_positions[2],
                "vial_01_size": vial_sizes[1],
                "vial_02_size": vial_sizes[2],
                "target_position": slots[0],
                "target_01_position": slots[1],
                "target_02_position": slots[2],
            },
        )
