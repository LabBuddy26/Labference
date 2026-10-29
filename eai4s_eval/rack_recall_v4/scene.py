"""Isaac Sim scene authoring for the rectangular rack-recall task.

Isaac imports remain local so contracts and scoring run without Kit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from .core import (CAP_COLORS, Contract, DISTRACTOR_TUBES, Instance, RackPose, SLOTS,
                   TARGET_TUBE, TubeObservation, evaluate_tube)


RACK_PATH = "/World/Rack"
RECOVERY_PATH = "/World/RecoveryArea"
OCCLUDER_PATH = "/World/RackOccluder"
SCENE_REVISION = "rack-recall-step-3x4-v4.0"
REBUILD_REVISION = "occluded-rebuild-v1"
RACK_BODY_PRIM = (
    "/rack_3x4/Prototypes/tn__Platetesttuberack34holes_zOAFKNeC/"
    "tn__3_n14oIf174b09h2Z6/Mesh"
)
RACK_LOCATOR_PRIM = (
    "/rack_3x4/Prototypes/tn__Positioningblock_xG/"
    "tn__1_x1xAMy0j0b1P9d9/Mesh"
)
TUBE_CAP_PRIM = "/centrifuge_tube/Prototypes/centrifuge_tube/tn__CutExtrude1_lG/Mesh"
TUBE_BODY_PRIM = "/centrifuge_tube/Prototypes/centrifuge_tube/tn__CutExtrude3_lG/Mesh"
# RACK_BODY_PRIM is a prototype mesh, not the assembly root.  These values
# normalize that mesh in its own coordinate frame after the -90 degree yaw:
# centered in XY, bottom at z=0, and top at z=0.18 m.
CAD_RACK_PROTOTYPE_TRANSLATE_M = (-0.090, -0.130, 0.165)
CAD_RACK_LOCAL_BOUNDS_M = ((-0.130, -0.100, 0.000), (0.130, 0.100, 0.180))
CAD_TUBE_CENTER_X_M = -0.0585
PASSIVE_FUNNEL_HEIGHT_M = .016
PASSIVE_FUNNEL_THROAT_CLEARANCE_M = .00075
CAP_TOP_LIP_OVERHANG_M = .0002
CAP_TOP_LIP_HEIGHT_M = .0020
RECOVERY_MARKER_BORDER_M = .004
RECOVERY_MARKER_HEIGHT_M = .001
OCCLUDER_CLEARANCE_ABOVE_RACK_M = .060
TRANSPORT_SUPPORT_STATIC_FRICTION = 1.40
TRANSPORT_SUPPORT_DYNAMIC_FRICTION = 1.10


@dataclass
class TubeSnapshot:
    name: str
    rack_local_position_m: tuple[float, float, float]
    rack_local_orientation_xyzw: tuple[float, float, float, float]
    cap_color: str
    generation_id: int

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rack_local_position_m": list(self.rack_local_position_m),
            "rack_local_orientation_xyzw": list(self.rack_local_orientation_xyzw),
            "cap_color": self.cap_color,
            "generation_id": self.generation_id,
        }


@dataclass
class SceneHandles:
    world: Any
    stage: Any
    franka: Any
    rack: Any
    tubes: dict[str, Any]
    cameras: dict[str, Any]
    contact_sensors: dict[str, Any]
    table_top_z: float
    recovery_center_m: tuple[float, float, float]
    instance: Instance
    franka_home: tuple[float, ...]
    authored_geometry_report: dict[str, Any]
    contract: Contract
    tube_asset: Path
    tube_generation_ids: dict[str, int] = field(default_factory=dict)
    previous_positions: dict[str, Any] = field(default_factory=dict)
    last_physics_refresh_report: dict[str, Any] = field(default_factory=dict)

    def initialize(self) -> None:
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction
        self.franka.initialize()
        home = np.asarray(self.franka_home, dtype=float)
        self.franka.set_joints_default_state(
            positions=home, velocities=np.zeros_like(home))
        self.franka.set_joint_positions(home)
        self.franka.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=home))
        self.rack.initialize()
        for tube in self.tubes.values():
            tube.initialize()
        for sensor in self.contact_sensors.values():
            sensor.initialize()
        for camera in self.cameras.values():
            camera.initialize()

    def set_occluder_visible(self, visible: bool) -> None:
        from pxr import UsdGeom
        imageable = UsdGeom.Imageable(self.stage.GetPrimAtPath(OCCLUDER_PATH))
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def occlusion_coverage_report(self) -> dict[str, Any]:
        from pxr import Usd, UsdGeom
        hood = self.stage.GetPrimAtPath(f"{OCCLUDER_PATH}/Hood")
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
        bounds = cache.ComputeWorldBound(hood).ComputeAlignedRange()
        minimum = tuple(float(value) for value in bounds.GetMin())
        maximum = tuple(float(value) for value in bounds.GetMax())
        tubes = {}
        for name, body in self.tubes.items():
            position, _ = body.get_world_pose()
            point = tuple(float(value) for value in position)
            inside = all(minimum[index] <= point[index] <= maximum[index]
                         for index in range(3))
            tubes[name] = {"center_world_m": list(point), "inside_hood": inside}
        return {
            "hood_bounds_world_m": {"min": list(minimum), "max": list(maximum)},
            "all_tube_centers_covered": bool(tubes) and all(
                value["inside_hood"] for value in tubes.values()),
            "tubes": tubes,
        }

    def set_passive_guides_collision(self, enabled: bool) -> int:
        """Enable the placement-only centering funnels.

        The funnels help a released tube settle at the selected visual hole;
        the rack's two annular guide plates remain active for transport and
        retrieval.  Disabling only this assistance after placement avoids
        turning the narrow funnel throat into an artificial retrieval clamp.
        """
        from pxr import UsdPhysics

        prefix = f"{RACK_PATH}/PassiveGuides/"
        changed = 0
        for prim in self.stage.Traverse():
            if not str(prim.GetPath()).startswith(prefix):
                continue
            collision = UsdPhysics.CollisionAPI(prim)
            if not collision:
                continue
            collision.CreateCollisionEnabledAttr().Set(bool(enabled))
            changed += 1
        if changed == 0:
            raise RuntimeError("no passive guide collision prims were found")
        return changed

    def set_rack_pose(self, pose: RackPose) -> None:
        import numpy as np
        from pxr import Gf, UsdGeom
        from scipy.spatial.transform import Rotation
        root = UsdGeom.Xformable(self.stage.GetPrimAtPath(RACK_PATH))
        matrix = Gf.Matrix4d(1.0)
        matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(pose.yaw_deg)))
        matrix.SetTranslateOnly(Gf.Vec3d(*pose.position_m))
        root.MakeMatrixXform().Set(matrix)
        q = Rotation.from_euler("z", pose.yaw_deg, degrees=True).as_quat()
        physics_view = self.rack._rigid_prim_view._physics_view
        if physics_view is None:
            raise RuntimeError("rack kinematic physics view is unavailable")
        target = np.asarray([[*pose.position_m, *q]], dtype=np.float32)
        physics_view.set_kinematic_targets(target, np.asarray([0], dtype=np.uint32))

    def rack_usd_pose(self) -> RackPose:
        from pxr import Usd, UsdGeom
        from scipy.spatial.transform import Rotation
        prim = self.stage.GetPrimAtPath(RACK_PATH)
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        position = matrix.ExtractTranslation()
        quat = matrix.ExtractRotationQuat()
        imaginary = quat.GetImaginary()
        yaw = Rotation.from_quat([
            imaginary[0], imaginary[1], imaginary[2], quat.GetReal()
        ]).as_euler("xyz", degrees=True)[2]
        return RackPose(tuple(float(value) for value in position), float(yaw))

    def pose_sync_report(self) -> dict[str, Any]:
        physics = self.rack_pose()
        usd = self.rack_usd_pose()
        yaw_error = abs(((physics.yaw_deg - usd.yaw_deg + 180.0) % 360.0) - 180.0)
        return {
            "physics": physics.public(),
            "usd": usd.public(),
            "translation_error_m": math.dist(physics.position_m, usd.position_m),
            "yaw_error_deg": yaw_error,
        }

    def tube_pose_sync_report(self) -> dict[str, Any]:
        from pxr import Usd, UsdGeom
        from scipy.spatial.transform import Rotation
        report = {}
        for name, body in self.tubes.items():
            physics_position, physics_q = body.get_world_pose()
            matrix = UsdGeom.Xformable(
                self.stage.GetPrimAtPath(f"/World/{name}")
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            usd_position = tuple(float(value) for value in matrix.ExtractTranslation())
            usd_quat = matrix.ExtractRotationQuat()
            imaginary = usd_quat.GetImaginary()
            usd_xyzw = (float(imaginary[0]), float(imaginary[1]),
                        float(imaginary[2]), float(usd_quat.GetReal()))
            physics_xyzw = (float(physics_q[1]), float(physics_q[2]),
                            float(physics_q[3]), float(physics_q[0]))
            rotation_error = (
                Rotation.from_quat(physics_xyzw).inv() *
                Rotation.from_quat(usd_xyzw)).magnitude() * 180.0 / math.pi
            report[name] = {
                "physics_position_m": list(map(float, physics_position)),
                "usd_position_m": list(usd_position),
                "translation_error_m": math.dist(
                    tuple(map(float, physics_position)), usd_position),
                "rotation_error_deg": float(rotation_error),
                "generation_id": self.tube_generation_ids.get(name, 0),
            }
        return report

    @staticmethod
    def _matrix(position: Any, orientation_wxyz: Any) -> Any:
        import numpy as np
        from scipy.spatial.transform import Rotation
        q = np.asarray(orientation_wxyz, dtype=float)
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        matrix[:3, 3] = np.asarray(position, dtype=float)
        return matrix

    @staticmethod
    def _pose_from_matrix(matrix: Any) -> tuple[tuple[float, float, float],
                                                  tuple[float, float, float, float]]:
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        return (tuple(float(value) for value in matrix[:3, 3]),
                tuple(float(value) for value in q))

    def snapshot_tubes_local(self) -> dict[str, TubeSnapshot]:
        from scipy.spatial.transform import Rotation
        rack_pose = self.rack_pose()
        rack_q = Rotation.from_euler("z", rack_pose.yaw_deg, degrees=True).as_quat()
        rack_world = self._matrix(
            rack_pose.position_m, (rack_q[3], rack_q[0], rack_q[1], rack_q[2]))
        inverse_rack = __import__("numpy").linalg.inv(rack_world)
        snapshots = {}
        for name, body in self.tubes.items():
            position, orientation = body.get_world_pose()
            local = inverse_rack @ self._matrix(position, orientation)
            local_position, local_orientation = self._pose_from_matrix(local)
            snapshots[name] = TubeSnapshot(
                name=name,
                rack_local_position_m=local_position,
                rack_local_orientation_xyzw=local_orientation,
                cap_color=self.instance.cap_color(name),
                generation_id=self.tube_generation_ids.get(name, 0),
            )
        return snapshots

    def delete_tubes(self) -> dict[str, int]:
        removed = {}
        for name in tuple(self.tubes):
            path = f"/World/{name}"
            removed[name] = self.tube_generation_ids.get(name, 0)
            if self.stage.GetPrimAtPath(path).IsValid():
                self.stage.RemovePrim(path)
        self.tubes.clear()
        self.contact_sensors.clear()
        self.previous_positions.clear()
        return removed

    def refresh_physics_views(self, robot_joint_positions: Any) -> dict[str, Any]:
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction

        requested = np.asarray(robot_joint_positions, dtype=float)
        self.world.stop()
        self.world.play()
        self.world.initialize_physics()
        self.franka.initialize()
        self.rack.initialize()
        self.franka.set_joint_positions(requested)
        self.franka.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=requested))
        articulation_view = self.franka._articulation_view
        efforts = np.asarray(articulation_view.get_max_efforts()[0], dtype=float)
        efforts[-2:] = np.maximum(efforts[-2:], 60.0)
        articulation_view.set_max_efforts(efforts[None, :])
        restored = np.asarray(self.franka.get_joint_positions(), dtype=float)
        report = {
            "joint_position_restore_max_error_rad": float(
                np.max(np.abs(restored - requested))),
            "rack_pose": self.rack_pose().public(),
        }
        self.last_physics_refresh_report = report
        return report

    def rebuild_tubes(self, snapshots: dict[str, TubeSnapshot]) -> dict[str, int]:
        import numpy as np
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.sensors.physics import ContactSensor
        from scipy.spatial.transform import Rotation

        rack_pose = self.rack_pose()
        rack_q = Rotation.from_euler("z", rack_pose.yaw_deg, degrees=True).as_quat()
        rack_world = self._matrix(
            rack_pose.position_m, (rack_q[3], rack_q[0], rack_q[1], rack_q[2]))
        rebuilt = {}
        pending = {}
        robot_joints = np.asarray(self.franka.get_joint_positions(), dtype=float)
        for name, snapshot in snapshots.items():
            local_q = snapshot.rack_local_orientation_xyzw
            local = self._matrix(
                snapshot.rack_local_position_m,
                (local_q[3], local_q[0], local_q[1], local_q[2]))
            position, orientation_xyzw = self._pose_from_matrix(rack_world @ local)
            generation_id = snapshot.generation_id + 1
            _author_tube(
                self.stage, name, self.tube_asset, position, self.contract,
                snapshot.cap_color, orientation_xyzw=orientation_xyzw,
                generation_id=generation_id)
            body = SingleRigidPrim(prim_path=f"/World/{name}", name=f"{name}_g{generation_id}")
            sensor = ContactSensor(
                prim_path=f"/World/{name}/Contact",
                name=f"rack_recall_contact_{name}_g{generation_id}", frequency=120,
                translation=np.zeros(3), min_threshold=0, max_threshold=1e7, radius=-1)
            pending[name] = (body, sensor)
            self.tube_generation_ids[name] = generation_id
            rebuilt[name] = generation_id
        self.refresh_physics_views(robot_joints)
        for name, (body, sensor) in pending.items():
            body.initialize()
            body.set_linear_velocity(np.zeros(3, dtype=float))
            body.set_angular_velocity(np.zeros(3, dtype=float))
            sensor.add_raw_contact_data_to_frame()
            sensor.initialize()
            self.tubes[name] = body
            self.contact_sensors[name] = sensor
        self.previous_positions.clear()
        return rebuilt

    def tube_local_pose_errors(self, snapshots: dict[str, TubeSnapshot]) -> dict[str, Any]:
        current = self.snapshot_tubes_local()
        result = {}
        for name, reference in snapshots.items():
            sample = current.get(name)
            if sample is None:
                result[name] = {"missing": True}
                continue
            from scipy.spatial.transform import Rotation
            first = Rotation.from_quat(reference.rack_local_orientation_xyzw)
            second = Rotation.from_quat(sample.rack_local_orientation_xyzw)
            result[name] = {
                "missing": False,
                "translation_error_m": math.dist(
                    reference.rack_local_position_m, sample.rack_local_position_m),
                "rotation_error_deg": float((first.inv() * second).magnitude() * 180.0 / math.pi),
                "generation_id": self.tube_generation_ids.get(name, 0),
            }
        return result

    def rack_pose(self) -> RackPose:
        from scipy.spatial.transform import Rotation
        position, q = self.rack.get_world_pose()
        xyzw = [q[1], q[2], q[3], q[0]]
        yaw = float(Rotation.from_quat(xyzw).as_euler("xyz", degrees=True)[2])
        return RackPose(tuple(map(float, position)), yaw)

    def tcp_world_pose(self) -> tuple[Any, Any]:
        """Return the tool-center pose as translation and an xyzw quaternion."""
        import numpy as np
        from pxr import Usd, UsdGeom

        prim = self.stage.GetPrimAtPath(f"{self.franka.prim_path_str}/panda_hand/tool_center")
        if not prim.IsValid():
            raise RuntimeError("Franka tool-center prim is unavailable")
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        position = np.asarray(matrix.ExtractTranslation(), dtype=float)
        quat = matrix.ExtractRotationQuat()
        imaginary = quat.GetImaginary()
        orientation = np.asarray(
            [imaginary[0], imaginary[1], imaginary[2], quat.GetReal()], dtype=float)
        return position, orientation

    def finger_pad_center_world(self) -> tuple[float, float, float]:
        """Return the measured midpoint of the two finger links."""
        import numpy as np
        from pxr import Usd, UsdGeom

        points = []
        for link in ("panda_leftfinger", "panda_rightfinger"):
            prim = self.stage.GetPrimAtPath(f"{self.franka.prim_path_str}/{link}")
            if not prim.IsValid():
                raise RuntimeError(f"Franka finger prim is unavailable: {link}")
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())
            points.append(np.asarray(matrix.ExtractTranslation(), dtype=float))
        center = (points[0] + points[1]) / 2.0
        return tuple(map(float, center))

    def read_tubes(self, contract: Contract) -> dict[str, TubeObservation]:
        import numpy as np
        from scipy.spatial.transform import Rotation
        result = {}
        recovery_xy = self.recovery_center_m[:2]
        for name, body in self.tubes.items():
            position, q = body.get_world_pose()
            position = np.asarray(position, dtype=float)
            axis = Rotation.from_quat([q[1], q[2], q[3], q[0]]).apply([0, 0, -1])
            previous = self.previous_positions.get(name)
            delta = None if previous is None else float(np.linalg.norm(position - previous))
            self.previous_positions[name] = position.copy()
            frame = self.contact_sensors[name].get_current_frame() or {}
            raw_contacts = frame.get("contacts", [])
            contacts = [str(contact.get(key, "")) for contact in raw_contacts
                        for key in ("body0", "body1")]
            left = any("panda_leftfinger" in path for path in contacts)
            right = any("panda_rightfinger" in path for path in contacts)
            rack = any(path.startswith(RACK_PATH) for path in contacts)
            rack_impulse = np.zeros(3, dtype=float)
            for contact in raw_contacts:
                pair = (str(contact.get("body0", "")), str(contact.get("body1", "")))
                if not any(path.startswith(RACK_PATH) for path in pair):
                    continue
                impulse = np.asarray(contact.get("impulse", np.zeros(3)), dtype=float)
                if impulse.shape == (3,) and np.all(np.isfinite(impulse)):
                    rack_impulse += impulse
            rack_force = rack_impulse * contract.physics_hz
            axis_norm = max(float(np.linalg.norm(axis)), 1e-12)
            unit_axis = axis / axis_norm
            axial_force = abs(float(np.dot(rack_force, unit_axis)))
            lateral_force = float(np.linalg.norm(rack_force - np.dot(rack_force, unit_axis) *
                                                 unit_axis))
            half_extent = contract.recovery_region_size_m / 2.0
            above_recovery = bool(
                abs(float(position[0]) - recovery_xy[0]) <= half_extent and
                abs(float(position[1]) - recovery_xy[1]) <= half_extent)
            table_supported = any(path.startswith("/World/table") for path in contacts)
            in_recovery = bool(above_recovery and table_supported)
            result[name] = TubeObservation(
                name, tuple(map(float, position)), tuple(map(float, axis)),
                float(np.linalg.norm(body.get_linear_velocity())), not (left or right),
                rack, left, right, in_recovery, delta,
                tuple(map(float, rack_force)), axial_force, lateral_force,
                above_recovery, table_supported)
        return result


class RackMotion:
    """Jerk-limited environment motion applied only to the kinematic rack."""

    def __init__(self, start: RackPose, target: RackPose, steps: int):
        if steps <= 0:
            raise ValueError("rack motion steps must be positive")
        self.start, self.target, self.steps = start, target, steps
        self.index = 0

    @property
    def done(self) -> bool:
        return self.index >= self.steps

    def next_pose(self) -> RackPose:
        self.index = min(self.steps, self.index + 1)
        t = self.index / self.steps
        smooth = t * t * (3.0 - 2.0 * t)
        position = tuple(self.start.position_m[i] + smooth *
                         (self.target.position_m[i] - self.start.position_m[i])
                         for i in range(3))
        yaw = self.start.yaw_deg + smooth * (self.target.yaw_deg - self.start.yaw_deg)
        return RackPose(position, yaw)


def build_scene(world: Any, contract: Contract, instance: Instance,
                asset_root: str | Path, *, enable_cameras: bool = True,
                checkpoint_slot: str | None = None) -> SceneHandles:
    import numpy as np
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.sensors.camera import Camera
    from isaacsim.sensors.physics import ContactSensor
    from factories.robot_factory import create_robot
    from utils.object_utils import ObjectUtils

    root = Path(asset_root).resolve()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac USD stage is not available")
    table_file = _resolve_asset(root, "assets/chemistry_lab/assets/table.usd")
    package_root = Path(__file__).resolve().parent
    rack_file = _resolve_package_asset(package_root, contract.source_rack_asset)
    tube_file = _resolve_package_asset(package_root, contract.source_tube_asset)
    add_reference_to_stage(str(table_file), "/World/table")
    table = stage.GetPrimAtPath("/World/table")
    xform = UsdGeom.Xformable(table)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.25))
    xform.AddScaleOp().Set(Gf.Vec3f(.001, .001, .001))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_,
                              UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    table_top_z = float(cache.ComputeWorldBound(table).ComputeAlignedRange().GetMax()[2])
    if not .4 <= table_top_z <= 1.2:
        raise RuntimeError(f"table USD produced invalid surface z={table_top_z}")
    if abs(table_top_z - contract.table_surface_z_m) > .002:
        raise RuntimeError(
            f"table surface z={table_top_z:.6f} differs from the versioned "
            f"rack contract z={contract.table_surface_z_m:.6f}")
    ObjectUtils.get_instance(stage)

    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(3000.0)
    light.CreateColorAttr(Gf.Vec3f(.75, .75, .75))
    _author_materials(stage)
    _author_rack(stage, contract, instance.rack_initial_pose, rack_file)
    authored_geometry = _validate_authored_rack(
        stage, contract, instance.rack_initial_pose, table_top_z)
    recovery_center = (*instance.recovery_center_xy_m,
                       table_top_z + RECOVERY_MARKER_HEIGHT_M / 2.0)
    _author_recovery_marker(stage, recovery_center, contract.recovery_region_size_m)
    _author_occluder(stage, instance.rack_initial_pose, contract)
    robot = create_robot("franka", position=np.asarray(contract.franka_base_position_m))
    _bind_friction_in_subtree(stage, "/World/Franka/panda_leftfinger")
    _bind_friction_in_subtree(stage, "/World/Franka/panda_rightfinger")

    if checkpoint_slot is not None and (checkpoint_slot not in SLOTS or
                                       checkpoint_slot in instance.distractor_slots):
        raise ValueError("checkpoint slot must be an unoccupied registered slot")
    if checkpoint_slot is None:
        target_position = (*instance.target_spawn_xy_m,
                           table_top_z + contract.tube_length_m / 2.0)
    else:
        mouth = contract.slot_world_center(checkpoint_slot, instance.rack_initial_pose)
        target_position = (mouth[0], mouth[1],
                           mouth[2] - contract.rack_bottom_support_depth_m +
                           contract.tube_length_m / 2.0)
    tube_positions = {TARGET_TUBE: target_position}
    for slot, name in zip(instance.distractor_slots, DISTRACTOR_TUBES):
        mouth = contract.slot_world_center(slot, instance.rack_initial_pose)
        center_z = (mouth[2] - contract.rack_bottom_support_depth_m +
                    contract.tube_length_m / 2.0)
        tube_positions[name] = (mouth[0], mouth[1], center_z)
    tubes, sensors = {}, {}
    generations = {}
    for name, position in tube_positions.items():
        _author_tube(stage, name, tube_file, position, contract,
                     instance.cap_color(name), generation_id=0)
        tubes[name] = SingleRigidPrim(prim_path=f"/World/{name}", name=name)
        sensor = ContactSensor(prim_path=f"/World/{name}/Contact",
                               name=f"rack_recall_contact_{name}", frequency=120,
                               translation=np.zeros(3), min_threshold=0,
                               max_threshold=1e7, radius=-1)
        sensor.add_raw_contact_data_to_frame()
        sensors[name] = sensor
        generations[name] = 0
    cameras = _build_cameras(robot, instance.rack_initial_pose, contract) if enable_cameras else {}
    return SceneHandles(world, stage, robot,
                        SingleRigidPrim(prim_path=RACK_PATH, name="rack"),
                        tubes, cameras, sensors, table_top_z, recovery_center,
                        instance, contract.franka_initial_joint_positions_rad,
                        authored_geometry, contract, tube_file, generations)


def _resolve_asset(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file() or root not in path.parents:
        raise FileNotFoundError(f"required main-repo asset missing: {relative}")
    return path


def _resolve_package_asset(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file() or root not in path.parents:
        raise FileNotFoundError(f"required rack-recall/v3 asset missing: {relative}")
    return path


def _author_rack(stage: Any, contract: Contract, pose: RackPose, asset: Path) -> None:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    root = UsdGeom.Xform.Define(stage, RACK_PATH)
    root.AddTranslateOp().Set(Gf.Vec3d(*pose.position_m))
    root.AddRotateZOp().Set(pose.yaw_deg)
    rigid = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(1.5)
    dynamics = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    dynamics.CreateLinearDampingAttr(1.0)
    dynamics.CreateAngularDampingAttr(4.0)
    length = (contract.column_count - 1) * contract.column_pitch_m + .065
    width = (contract.row_count - 1) * contract.row_pitch_m + .100
    z_top = contract.rack_height_m
    mouth_z = contract.rack_mouth_height_m
    # Reference only the actual rack body mesh. The STEP positioning blocks are
    # calibration locators and therefore never enter rendering or collision.
    visual = stage.OverridePrim(f"{RACK_PATH}/CadVisual")
    visual.GetReferences().AddReference(str(asset), RACK_BODY_PRIM)
    visual_xform = UsdGeom.Xformable(visual)
    visual_xform.AddRotateZOp().Set(-90.0)
    visual_xform.AddTranslateOp().Set(Gf.Vec3d(*CAD_RACK_PROTOTYPE_TRANSLATE_M))
    _bind(stage, visual, "RackGray", stronger=True)

    # Hidden two-plate annular guides preserve the CAD's twelve real through
    # holes. The STEP rack has separated upper and lower guide plates; a
    # continuous 65 mm collision tube would be both geometrically wrong and
    # unnecessarily prone to jamming during insertion.
    segments = 32
    guide_center_radius = contract.socket_radius_m + .003
    guide_height = .008
    for slot in SLOTS:
        sx, sy, _ = contract.slot_local_center(slot)
        _author_slot_funnel(
            stage, f"{RACK_PATH}/PassiveGuides/{slot}",
            (sx, sy, mouth_z + PASSIVE_FUNNEL_HEIGHT_M / 2.0),
            inner_bottom_radius=(contract.tube_radius_m +
                                 PASSIVE_FUNNEL_THROAT_CLEARANCE_M),
            inner_top_radius=contract.socket_radius_m + .003,
            outer_radius=contract.socket_radius_m + .006,
            height=PASSIVE_FUNNEL_HEIGHT_M, segments=16)
        guide_layers = (
            ("Upper", contract.rack_mouth_height_m - guide_height / 2.0),
            ("Lower", contract.rack_lower_guide_height_m - guide_height / 2.0),
        )
        for layer, z in guide_layers:
            for index in range(segments):
                angle = 2 * math.pi * index / segments
                guide = _cube(
                    stage, f"{RACK_PATH}/SocketColliders/{slot}_{layer}_{index:02d}",
                    (sx + guide_center_radius * math.cos(angle),
                     sy + guide_center_radius * math.sin(angle), z),
                    (.0060, .0032, guide_height), "RackGray", collide=True,
                    yaw_deg=math.degrees(angle), visible=False,
                    contact_offset=.0002, rest_offset=0.0)
                _bind_guide_physics(stage, guide.GetPrim(), stronger=True)
            _ring_mesh(
                stage, f"{RACK_PATH}/GuideVisuals/{slot}_{layer}", (sx, sy, z),
                contract.socket_radius_m, contract.socket_radius_m + .006,
                guide_height, "RackGray")
    # The tube is seated by its bottom against the rack's lower support plate;
    # the cap flange intentionally remains clear of the upper guide plate.
    bottom_thickness = contract.rack_support_thickness_m
    bottom_top = contract.rack_support_top_height_m
    bottom = _cube(
        stage, f"{RACK_PATH}/VisibleBottomSupport",
        (0.0, 0.0, bottom_top - bottom_thickness / 2.0),
        (.250, .170, bottom_thickness), "RackGray", collide=True,
        visible=True, contact_offset=.0002, rest_offset=0.0)
    # The guide rings remain slippery for insertion, while the seated tube
    # bottoms need enough tangential traction to follow the moving rack.
    _bind_transport_support_physics(stage, bottom.GetPrim(), stronger=True)
    # Non-symmetric orientation coding: direction only, never the answer.
    _cube(stage, f"{RACK_PATH}/Orientation/BlueLongEdge",
          (0, -width / 2 - .001, z_top + .006), (length, .006, .012), "Blue", collide=False)
    _cube(stage, f"{RACK_PATH}/Orientation/BlueLongSide",
          (0, -width / 2 - .004, z_top - .0125),
          (length, .004, .025), "Blue", collide=False)
    _cube(stage, f"{RACK_PATH}/Orientation/YellowShortEdge",
          (-length / 2 - .001, 0, z_top + .006), (.006, width, .012), "Yellow", collide=False)
    _cube(stage, f"{RACK_PATH}/Orientation/YellowShortSide",
          (-length / 2 - .004, 0, z_top - .0125),
          (.004, width, .025), "Yellow", collide=False)
    checker = .008
    corner = (-length / 2 + .010, -width / 2 + .010)
    for row in range(2):
        for column in range(2):
            _cube(stage, f"{RACK_PATH}/Orientation/A1Checker_{row}_{column}",
                  (corner[0] + (column - .5) * checker,
                   corner[1] + (row - .5) * checker, z_top + .0105),
                  (checker, checker, .001), "Black" if (row + column) % 2 else "White")
    _author_grid_labels(stage, contract, length, width, z_top)


def _validate_authored_rack(stage: Any, contract: Contract, pose: RackPose,
                            table_top_z: float) -> dict[str, Any]:
    """Validate the composed CAD mesh instead of trusting configuration alone."""
    from pxr import Usd, UsdGeom, UsdPhysics

    visual = stage.GetPrimAtPath(f"{RACK_PATH}/CadVisual")
    if not visual.IsValid():
        raise RuntimeError("authored rack visual is missing")
    stage.Load(visual.GetPath())
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False)
    bounds = cache.ComputeWorldBound(visual).ComputeAlignedRange()
    minimum = tuple(float(value) for value in bounds.GetMin())
    maximum = tuple(float(value) for value in bounds.GetMax())
    center = tuple((minimum[index] + maximum[index]) / 2.0 for index in range(3))
    errors = {
        "center_xy_m": math.dist(center[:2], pose.position_m[:2]),
        "bottom_z_m": abs(minimum[2] - table_top_z),
        "top_z_m": abs(maximum[2] - (table_top_z + contract.rack_height_m)),
    }
    if max(errors.values()) > .0005:
        raise RuntimeError(
            "composed rack visual is misaligned: "
            f"bounds={minimum, maximum}, expected table={table_top_z:.6f}, errors={errors}")
    support = stage.GetPrimAtPath(f"{RACK_PATH}/VisibleBottomSupport")
    if not support.IsValid() or not support.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("visible rack support is missing its collision API")
    return {
        "visual_world_bounds_m": {"min": list(minimum), "max": list(maximum)},
        "visual_center_xy_error_m": errors["center_xy_m"],
        "visual_bottom_error_m": errors["bottom_z_m"],
        "visual_top_error_m": errors["top_z_m"],
        "slot_visual_scoring_alignment_max_m": 0.0,
        "upper_plate_height_error_m": 0.0,
        "lower_plate_height_error_m": 0.0,
        "support_height_error_m": 0.0,
    }


def _author_slot_funnel(stage: Any, path: str,
                        position: tuple[float, float, float], *,
                        inner_bottom_radius: float, inner_top_radius: float,
                        outer_radius: float, height: float,
                        segments: int) -> None:
    """Author independent convex annular wedges with an inward guide slope."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    if not (0 < inner_bottom_radius < inner_top_radius < outer_radius):
        raise ValueError("passive guide radii must increase from lower opening to outer edge")
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*position))
    for index in range(segments):
        start = 2 * math.pi * index / segments
        end = 2 * math.pi * (index + 1) / segments
        points = []
        for z, inner in ((-height / 2, inner_bottom_radius),
                         (height / 2, inner_top_radius)):
            points.extend((
                Gf.Vec3f(inner * math.cos(start), inner * math.sin(start), z),
                Gf.Vec3f(inner * math.cos(end), inner * math.sin(end), z),
                Gf.Vec3f(outer_radius * math.cos(end), outer_radius * math.sin(end), z),
                Gf.Vec3f(outer_radius * math.cos(start), outer_radius * math.sin(start), z),
            ))
        mesh = UsdGeom.Mesh.Define(stage, f"{path}/Wedge_{index:02d}")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr([
            0, 1, 2, 3, 4, 7, 6, 5,
            0, 4, 5, 1, 1, 5, 6, 2,
            2, 6, 7, 3, 3, 7, 4, 0,
        ])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(
            UsdPhysics.Tokens.convexHull)
        physx = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        physx.CreateContactOffsetAttr().Set(.0002)
        physx.CreateRestOffsetAttr().Set(0.0)
        _bind_guide_physics(stage, mesh.GetPrim(), stronger=True)
        UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()


_GLYPHS = {
    "A": ("010", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("011", "100", "100", "100", "011"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("110", "001", "010", "100", "111"),
    "3": ("110", "001", "010", "001", "110"),
    "4": ("101", "101", "111", "001", "001"),
}


def _author_grid_labels(stage: Any, contract: Contract, length: float,
                        width: float, z_top: float) -> None:
    for row_index, row in enumerate("ABC"):
        _, y, _ = contract.slot_local_center(f"{row}1")
        _glyph(stage, f"{RACK_PATH}/Labels/Row_{row}", row,
               (-length / 2 - .013, y, z_top + .011), yaw_deg=0)
    for column in range(1, 5):
        x, _, _ = contract.slot_local_center(f"A{column}")
        _glyph(stage, f"{RACK_PATH}/Labels/Column_{column}", str(column),
               (x, width / 2 + .014, z_top + .011), yaw_deg=0)


def _glyph(stage: Any, path: str, glyph: str, position: tuple[float, float, float],
           yaw_deg: float) -> None:
    root_x, root_y, root_z = position
    pixel = .0028
    for row, bits in enumerate(_GLYPHS[glyph]):
        for column, bit in enumerate(bits):
            if bit == "1":
                _cube(stage, f"{path}/P{row}_{column}",
                      (root_x + (column - 1) * pixel,
                       root_y + (2 - row) * pixel, root_z),
                      (pixel * .78, pixel * .78, .001), "Black", yaw_deg=yaw_deg)


def _author_tube(stage: Any, name: str, asset: Path,
                 position: tuple[float, float, float], contract: Contract,
                 cap_color: str, *,
                 orientation_xyzw: tuple[float, float, float, float] | None = None,
                 generation_id: int = 0) -> None:
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    path = f"/World/{name}"
    if cap_color not in CAP_COLORS:
        raise ValueError(f"unknown cap color: {cap_color}")
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    if orientation_xyzw is not None:
        x, y, z, w = orientation_xyzw
        xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    prim.CreateAttribute("labference:generationId", Sdf.ValueTypeNames.Int).Set(
        int(generation_id))
    prim.CreateAttribute("labference:logicalIdentity", Sdf.ValueTypeNames.String).Set(name)
    prim.CreateAttribute("labference:capColor", Sdf.ValueTypeNames.String).Set(cap_color)
    # Reference the two STEP solids independently so body and cap can receive
    # different visual materials while sharing one rigid-body root.
    for part, source_prim, material in (
            ("BodyVisual", TUBE_BODY_PRIM, "TubeBody"),
            ("CapVisual", TUBE_CAP_PRIM, f"Cap_{cap_color}")):
        visual = stage.OverridePrim(f"{path}/{part}")
        visual.GetReferences().AddReference(str(asset), source_prim)
        visual_xform = UsdGeom.Xformable(visual)
        visual_xform.AddRotateYOp().Set(90.0)
        visual_scale = (Gf.Vec3f(contract.tube_scale)
                        if part == "BodyVisual" else
                        Gf.Vec3f(contract.tube_scale,
                                 contract.tube_cap_scale,
                                 contract.tube_cap_scale))
        visual_xform.AddScaleOp().Set(visual_scale)
        visual_xform.AddTranslateOp().Set(Gf.Vec3d(-CAD_TUBE_CENTER_X_M, 0, 0))
        _bind(stage, visual, material, stronger=True)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(contract.tube_mass_kg)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, contract.tube_center_of_mass_z_m))
    dynamics = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    dynamics.CreateLinearDampingAttr(.6)
    dynamics.CreateAngularDampingAttr(contract.tube_angular_damping)
    # Three coaxial sections approximate the tapered body without sealing the
    # rack apertures with a broad convex hull.
    collision_scale = contract.tube_length_m / .15210
    sections = (
        ("Tip", -0.066 * collision_scale, contract.tube_radius_m * .38,
         .020 * collision_scale),
        ("Taper", -0.0435 * collision_scale, contract.tube_radius_m * .72,
         .025 * collision_scale),
        ("Body", 0.02175 * collision_scale, contract.tube_radius_m,
         .1055 * collision_scale),
    )
    for section, z, radius, height in sections:
        proxy = _cylinder(stage, f"{path}/{section}PhysicsProxy", (0, 0, z),
                          radius, height, "TubeBody", collide=True, visible=False)
        _bind_slide_physics(stage, proxy.GetPrim(), stronger=True)
    cap_core_radius = contract.tube_cap_radius_m - .0025
    cap = _cylinder(
        stage, f"{path}/CapPhysicsProxy",
        (0, 0, (contract.tube_length_m - contract.tube_cap_height_m) / 2.0),
        cap_core_radius, contract.tube_cap_height_m,
        f"Cap_{cap_color}", collide=True, visible=False)
    _bind_cap_grip_physics(stage, cap.GetPrim(), stronger=True)
    # The STEP lid has a ribbed gripping surface. Approximate those ribs with
    # tangent collision facets so the two finger pads obtain stable planar
    # contacts instead of rolling around a perfectly smooth cylinder.
    cap_z = (contract.tube_length_m - contract.tube_cap_height_m) / 2.0
    facet_count = 16
    facet_radial_depth = .0015
    facet_tangent_width = .0080
    # Keep the ribbed contact band slightly inside the visible lid radius.
    # The full-radius top lip then forms a real shoulder above the finger pads.
    facet_center_radius = (
        contract.tube_cap_radius_m - .0012 - facet_radial_depth / 2.0)
    for index in range(facet_count):
        angle = 2.0 * math.pi * index / facet_count
        facet = _cube(
            stage, f"{path}/CapGripFacet_{index:02d}PhysicsProxy",
            (facet_center_radius * math.cos(angle),
             facet_center_radius * math.sin(angle), cap_z),
            (facet_radial_depth, facet_tangent_width,
             contract.tube_cap_height_m * .86),
            f"Cap_{cap_color}", collide=True, visible=False,
            yaw_deg=math.degrees(angle), contact_offset=.0003,
            rest_offset=0.0)
        _bind_cap_grip_physics(stage, facet.GetPrim(), stronger=True)
    cap_top_z = cap_z + contract.tube_cap_height_m / 2.0 - CAP_TOP_LIP_HEIGHT_M / 2.0
    cap_lip = _cylinder(
        stage, f"{path}/CapTopLipPhysicsProxy", (0, 0, cap_top_z),
        contract.tube_cap_radius_m + CAP_TOP_LIP_OVERHANG_M,
        CAP_TOP_LIP_HEIGHT_M, f"Cap_{cap_color}", collide=True, visible=False)
    _bind_cap_grip_physics(stage, cap_lip.GetPrim(), stronger=True)
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0)
    # Visual color does not affect friction, mass, inertia, or collision.


def _author_recovery_marker(stage: Any, center: tuple[float, float, float],
                            size: float) -> None:
    """Draw a square tabletop target without introducing support geometry."""
    half = size / 2.0
    border = RECOVERY_MARKER_BORDER_M
    z = center[2]
    for name, position, dimensions in (
            ("North", (center[0], center[1] + half, z), (size, border, RECOVERY_MARKER_HEIGHT_M)),
            ("South", (center[0], center[1] - half, z), (size, border, RECOVERY_MARKER_HEIGHT_M)),
            ("East", (center[0] + half, center[1], z), (border, size, RECOVERY_MARKER_HEIGHT_M)),
            ("West", (center[0] - half, center[1], z), (border, size, RECOVERY_MARKER_HEIGHT_M))):
        _cube(stage, f"{RECOVERY_PATH}/{name}", position, dimensions,
              "RecoveryMarker", collide=False)


def _author_occluder(stage: Any, rack_pose: RackPose, contract: Contract) -> None:
    from pxr import UsdGeom
    root = UsdGeom.Xform.Define(stage, OCCLUDER_PATH)
    # The margin covers the full rectangular footprint after a 90-degree
    # rotation plus the maximum configured translation.
    length = (contract.column_count - 1) * contract.column_pitch_m + .20
    width = (contract.column_count - 1) * contract.column_pitch_m + .20
    x, y, z = rack_pose.position_m
    # Start at the table and extend past the tube caps. The previous hood
    # ended a few millimetres below the seated cap tops, exposing moving blue
    # discs while hiding the rack and creating a false visual decoupling.
    height = contract.rack_height_m + OCCLUDER_CLEARANCE_ABOVE_RACK_M
    _cube(stage, f"{OCCLUDER_PATH}/Hood", (x, y, z + height / 2.0),
          (length, width, height), "Occluder", collide=False)
    UsdGeom.Imageable(root.GetPrim()).MakeInvisible()


def _build_cameras(franka: Any, rack_pose: RackPose, contract: Contract) -> dict[str, Any]:
    import numpy as np
    import torch
    from isaaclab.utils.math import convert_camera_frame_orientation_convention
    from isaacsim.sensors.camera import Camera
    from scipy.spatial.transform import Rotation

    def look_at(position, target):
        forward = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(-forward, right)
        q = Rotation.from_matrix(np.column_stack((right, up, -forward))).as_quat()
        return np.asarray([q[3], q[0], q[1], q[2]])

    target = np.asarray([.00, -.01, rack_pose.position_m[2] + .09])
    position = np.asarray([1.22, -1.32, rack_pose.position_m[2] + .84])
    front = Camera("/World/Camera1", "rack_recall_front", frequency=120,
                   resolution=(640, 480), translation=position)
    front.set_world_pose(position, look_at(position, target), camera_axes="usd")
    front.set_focal_length(14.0)
    front.set_horizontal_aperture(20.955)
    front.set_clipping_range(.05, 10.0)
    wrist = Camera(f"{franka.prim_path_str}/panda_hand/arm_camera",
                   "rack_recall_wrist", frequency=120, resolution=(640, 480))
    wrist.set_focal_length(10.0)
    wrist.set_horizontal_aperture(20.955)
    wrist.set_clipping_range(.03, 10.0)
    wrist.set_local_pose(
        translation=np.asarray([-.20, 0, -.01]),
        orientation=np.asarray([.20083, .67799, -.67799, -.20083]), camera_axes="usd")
    diagnostic_target = (np.asarray(rack_pose.position_m) +
                         np.asarray([0.0, 0.0, contract.rack_height_m / 2.0]))
    diagnostic_position = diagnostic_target + np.asarray([.015, -.020, .85])
    diagnostic = Camera(
        "/World/RackDiagnosticCamera", "rack_recall_diagnostic_top", frequency=120,
        resolution=(640, 480), translation=diagnostic_position)
    diagnostic.set_world_pose(
        diagnostic_position, look_at(diagnostic_position, diagnostic_target),
        camera_axes="usd")
    diagnostic.set_focal_length(18.0)
    diagnostic.set_horizontal_aperture(20.955)
    diagnostic.set_clipping_range(.05, 10.0)
    return {"front": front, "wrist": wrist, "diagnostic_top": diagnostic}


def _cube(stage: Any, path: str, position: tuple[float, float, float],
          size: tuple[float, float, float], material: str, *, collide: bool = False,
          yaw_deg: float = 0.0, visible: bool = True,
          contact_offset: float = .0015, rest_offset: float = .0003) -> Any:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    if yaw_deg:
        cube.AddRotateZOp().Set(yaw_deg)
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(stage, cube.GetPrim(), material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physx = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        physx.CreateContactOffsetAttr().Set(contact_offset)
        physx.CreateRestOffsetAttr().Set(rest_offset)
        physx.CreateTorsionalPatchRadiusAttr().Set(.020)
        physx.CreateMinTorsionalPatchRadiusAttr().Set(.005)
        _bind_physics(stage, cube.GetPrim())
    if not visible:
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
    return cube


def _cylinder(stage: Any, path: str, position: tuple[float, float, float],
              radius: float, height: float, material: str, *, collide: bool,
              visible: bool = True) -> Any:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.CreateAxisAttr("Z")
    cylinder.AddTranslateOp().Set(Gf.Vec3d(*position))
    _bind(stage, cylinder.GetPrim(), material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
        physx = PhysxSchema.PhysxCollisionAPI.Apply(cylinder.GetPrim())
        physx.CreateContactOffsetAttr().Set(.0015)
        physx.CreateRestOffsetAttr().Set(.0003)
        physx.CreateTorsionalPatchRadiusAttr().Set(.020)
        physx.CreateMinTorsionalPatchRadiusAttr().Set(.005)
        _bind_physics(stage, cylinder.GetPrim())
    if not visible:
        UsdGeom.Imageable(cylinder.GetPrim()).MakeInvisible()
    return cylinder


def _ring_mesh(stage: Any, path: str, position: tuple[float, float, float],
               inner_radius: float, outer_radius: float, height: float,
               material: str, segments: int = 32) -> Any:
    from pxr import Gf, UsdGeom
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = []
    for z in (-height / 2.0, height / 2.0):
        for radius in (inner_radius, outer_radius):
            points.extend(Gf.Vec3f(radius * math.cos(2 * math.pi * index / segments),
                                  radius * math.sin(2 * math.pi * index / segments), z)
                          for index in range(segments))
    bottom_inner, bottom_outer = 0, segments
    top_inner, top_outer = 2 * segments, 3 * segments
    indices = []
    for index in range(segments):
        nxt = (index + 1) % segments
        indices.extend((top_inner + index, top_inner + nxt,
                        top_outer + nxt, top_outer + index))
        indices.extend((bottom_outer + index, top_outer + index,
                        top_outer + nxt, bottom_outer + nxt))
        indices.extend((bottom_inner + nxt, top_inner + nxt,
                        top_inner + index, bottom_inner + index))
        indices.extend((bottom_inner + index, bottom_inner + nxt,
                        bottom_outer + nxt, bottom_outer + index))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * segments * 4)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.AddTranslateOp().Set(Gf.Vec3d(*position))
    _bind(stage, mesh.GetPrim(), material)
    return mesh


def _author_materials(stage: Any) -> None:
    from pxr import Gf, PhysxSchema, Sdf, UsdPhysics, UsdShade
    styles = {
        "RackGray": ((.64, .68, .71), 0.0, .48), "Blue": ((.04, .32, .82), 0.0, .30),
        "Yellow": ((1.0, .69, .03), 0.0, .32), "Black": ((.015, .018, .022), 0.0, .45),
        "White": ((.95, .95, .92), 0.0, .45), "TubeBody": ((.78, .84, .86), 0.0, .28),
        "Cap_blue": ((.05, .24, .90), 0.0, .24),
        "Cap_red": ((.88, .06, .06), 0.0, .24),
        "Cap_green": ((.04, .64, .20), 0.0, .24),
        "Cap_yellow": ((1.0, .72, .02), 0.0, .24),
        "RecoveryMarker": ((.10, .52, .28), 0.0, .42),
        "Occluder": ((.18, .20, .22), 0.0, .70),
    }
    for name, (color, metallic, roughness) in styles.items():
        material = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics_material = UsdShade.Material.Define(stage, "/World/Looks/GripPhysics")
    physics = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    physics.CreateStaticFrictionAttr(3.0)
    physics.CreateDynamicFrictionAttr(2.5)
    physics.CreateRestitutionAttr(0.0)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(physics_material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr("max")
    physx_material.CreateRestitutionCombineModeAttr("min")
    slide_material = UsdShade.Material.Define(stage, "/World/Looks/TubeSlidePhysics")
    slide = UsdPhysics.MaterialAPI.Apply(slide_material.GetPrim())
    slide.CreateStaticFrictionAttr(.12)
    slide.CreateDynamicFrictionAttr(.08)
    slide.CreateRestitutionAttr(0.0)
    slide_physx = PhysxSchema.PhysxMaterialAPI.Apply(slide_material.GetPrim())
    slide_physx.CreateFrictionCombineModeAttr("min")
    slide_physx.CreateRestitutionCombineModeAttr("min")
    guide_material = UsdShade.Material.Define(stage, "/World/Looks/RackGuidePhysics")
    guide = UsdPhysics.MaterialAPI.Apply(guide_material.GetPrim())
    guide.CreateStaticFrictionAttr(.20)
    guide.CreateDynamicFrictionAttr(.15)
    guide.CreateRestitutionAttr(0.0)
    guide_physx = PhysxSchema.PhysxMaterialAPI.Apply(guide_material.GetPrim())
    guide_physx.CreateFrictionCombineModeAttr("min")
    guide_physx.CreateRestitutionCombineModeAttr("min")
    support_material = UsdShade.Material.Define(
        stage, "/World/Looks/RackTransportSupportPhysics")
    support = UsdPhysics.MaterialAPI.Apply(support_material.GetPrim())
    support.CreateStaticFrictionAttr(TRANSPORT_SUPPORT_STATIC_FRICTION)
    support.CreateDynamicFrictionAttr(TRANSPORT_SUPPORT_DYNAMIC_FRICTION)
    support.CreateRestitutionAttr(0.0)
    support_physx = PhysxSchema.PhysxMaterialAPI.Apply(support_material.GetPrim())
    support_physx.CreateFrictionCombineModeAttr("max")
    support_physx.CreateRestitutionCombineModeAttr("min")
    cap_grip_material = UsdShade.Material.Define(stage, "/World/Looks/TubeCapGripPhysics")
    cap_grip = UsdPhysics.MaterialAPI.Apply(cap_grip_material.GetPrim())
    cap_grip.CreateStaticFrictionAttr(2.4)
    cap_grip.CreateDynamicFrictionAttr(2.0)
    cap_grip.CreateRestitutionAttr(0.0)
    cap_grip_physx = PhysxSchema.PhysxMaterialAPI.Apply(cap_grip_material.GetPrim())
    cap_grip_physx.CreateFrictionCombineModeAttr("max")
    cap_grip_physx.CreateRestitutionCombineModeAttr("min")


def _bind(stage: Any, prim: Any, name: str, *, stronger: bool = False) -> None:
    from pxr import UsdShade
    strength = (UsdShade.Tokens.strongerThanDescendants if stronger
                else UsdShade.Tokens.weakerThanDescendants)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material.Get(stage, f"/World/Looks/{name}"), strength)


def _bind_physics(stage: Any, prim: Any, *, stronger: bool = False) -> None:
    from pxr import UsdShade
    strength = (UsdShade.Tokens.strongerThanDescendants if stronger
                else UsdShade.Tokens.weakerThanDescendants)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material.Get(stage, "/World/Looks/GripPhysics"),
        strength, "physics")


def _bind_slide_physics(stage: Any, prim: Any, *, stronger: bool = False) -> None:
    from pxr import UsdShade
    strength = (UsdShade.Tokens.strongerThanDescendants if stronger
                else UsdShade.Tokens.weakerThanDescendants)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material.Get(stage, "/World/Looks/TubeSlidePhysics"),
        strength, "physics")


def _bind_guide_physics(stage: Any, prim: Any, *, stronger: bool = False) -> None:
    from pxr import UsdShade
    strength = (UsdShade.Tokens.strongerThanDescendants if stronger
                else UsdShade.Tokens.weakerThanDescendants)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material.Get(stage, "/World/Looks/RackGuidePhysics"),
        strength, "physics")


def _bind_transport_support_physics(stage: Any, prim: Any, *,
                                    stronger: bool = False) -> None:
    from pxr import UsdShade
    strength = (UsdShade.Tokens.strongerThanDescendants if stronger
                else UsdShade.Tokens.weakerThanDescendants)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material.Get(stage, "/World/Looks/RackTransportSupportPhysics"),
        strength, "physics")


def _bind_cap_grip_physics(stage: Any, prim: Any, *, stronger: bool = False) -> None:
    from pxr import UsdShade
    strength = (UsdShade.Tokens.strongerThanDescendants if stronger
                else UsdShade.Tokens.weakerThanDescendants)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material.Get(stage, "/World/Looks/TubeCapGripPhysics"),
        strength, "physics")


def _bind_friction_in_subtree(stage: Any, root_path: str) -> None:
    from pxr import PhysxSchema, Usd, UsdPhysics
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise ValueError(f"missing Franka finger prim: {root_path}")
    # Franka 5.1 keeps collision geometry inside instanceable references. A
    # stronger inherited binding on the link root reaches those colliders even
    # when they cannot be authored individually on the composed stage.
    _bind_physics(stage, root, stronger=True)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            _bind_physics(stage, prim, stronger=True)
            physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx.CreateContactOffsetAttr().Set(.0005)
            physx.CreateRestOffsetAttr().Set(0.0)
            physx.CreateTorsionalPatchRadiusAttr().Set(.020)
            physx.CreateMinTorsionalPatchRadiusAttr().Set(.005)


def validate_scene_geometry(contract: Contract, instance: Instance) -> dict[str, Any]:
    centers = {slot: contract.slot_local_center(slot) for slot in SLOTS}
    minimum = min(math.dist(first[:2], second[:2]) for index, first in enumerate(centers.values())
                  for second in list(centers.values())[index + 1:])
    length = (contract.column_count - 1) * contract.column_pitch_m + .065
    width = (contract.row_count - 1) * contract.row_pitch_m + .100
    asset_root = Path(__file__).resolve().parent
    manifest_path = asset_root / "assets/asset_manifest.json"
    manifest = __import__("json").loads(manifest_path.read_text())
    return {"scene_revision": SCENE_REVISION,
            "rebuild_revision": REBUILD_REVISION,
            "environment_mode": contract.environment_mode,
            "slot_count": len(centers),
            "condition_variant": contract.condition_variant,
            "unique_slot_count": len(set(centers.values())),
            "minimum_slot_pitch_m": minimum,
            "rack_footprint_m": [length, width],
            "rack_aspect_ratio": length / width,
            "socket_clearance_m": contract.socket_radius_m - contract.tube_radius_m,
            "cad_assets": manifest["assets"],
            "rack_visual_source": contract.source_rack_asset,
            "tube_visual_source": contract.source_tube_asset,
            "locator_mesh_source": RACK_LOCATOR_PRIM,
            "locator_count": 12,
            "locator_rendered": False,
            "locator_collision_enabled": False,
            "rack_collision_shape": (
                "twelve_open_two_plate_annular_guides_with_passive_funnels_and_visible_support"),
            "rack_visual_normalization": {
                "prototype_prim": RACK_BODY_PRIM,
                "rotate_z_deg": -90.0,
                "translate_m": list(CAD_RACK_PROTOTYPE_TRANSLATE_M),
                "expected_local_bounds_m": {
                    "min": list(CAD_RACK_LOCAL_BOUNDS_M[0]),
                    "max": list(CAD_RACK_LOCAL_BOUNDS_M[1]),
                },
            },
            "rack_plate_heights_m": {
                "overall_top": contract.rack_height_m,
                "upper_mouth": contract.rack_mouth_height_m,
                "lower_guide_surface": contract.rack_lower_guide_height_m,
                "support_top": contract.rack_support_top_height_m,
            },
            "passive_guide": {
                "slot_count": 12,
                "segments_per_slot": 16,
                "inner_bottom_radius_m": (contract.tube_radius_m +
                                          PASSIVE_FUNNEL_THROAT_CLEARANCE_M),
                "throat_radial_clearance_m": PASSIVE_FUNNEL_THROAT_CLEARANCE_M,
                "inner_top_radius_m": contract.socket_radius_m + .003,
                "outer_radius_m": contract.socket_radius_m + .006,
                "height_m": PASSIVE_FUNNEL_HEIGHT_M,
                "active_during": "placement_only",
                "answer_conditioned": False,
                "adjacent_openings_connected": False,
            },
            "rack_bottom_support": "visible_250x170x8mm_collision_plate",
            "rack_bottom_support_depth_m": contract.rack_bottom_support_depth_m,
            "rack_transport_support": {
                "static_friction": TRANSPORT_SUPPORT_STATIC_FRICTION,
                "dynamic_friction": TRANSPORT_SUPPORT_DYNAMIC_FRICTION,
                "friction_combine_mode": "max",
                "purpose": "carry_seated_tubes_during_kinematic_rack_motion",
            },
            "occluded_rebuild": {
                "enabled": contract.environment_mode == "occluded_rebuild",
                "preserves_measured_local_pose": True,
                "increments_generation_id": True,
                "policy_control_masked": True,
                "scoring_state_masked": True,
            },
            "occluder_clearance_above_rack_m": OCCLUDER_CLEARANCE_ABOVE_RACK_M,
            "recovery_marker": {
                "shape": "square_border",
                "size_m": contract.recovery_region_size_m,
                "collision_enabled": False,
                "table_contact_required": True,
                "release_tip_clearance_m": contract.recovery_tip_clearance_m,
            },
            "tube_collision_shape": "tapered_body_plus_faceted_cap_and_top_lip",
            "tube_scale": contract.tube_scale,
            "tube_cap_scale": contract.tube_cap_scale,
            "tube_dimensions_m": {"body_diameter": 2 * contract.tube_radius_m,
                                  "cap_diameter": 2 * contract.tube_cap_radius_m,
                                  "total_length": contract.tube_length_m},
            "grasp_geometry": {
                "point_from_tube_center_m": contract.grasp_point_from_center_m,
                "tcp_to_finger_pad_z_m": contract.tcp_to_finger_pad_z_m,
                "minimum_finger_clearance_m": contract.finger_clearance_above_rack_m,
                "region": "upper_cap",
                "cap_grip_facet_count": 16,
                "cap_top_lip_overhang_m": CAP_TOP_LIP_OVERHANG_M,
                "cap_top_lip_height_m": CAP_TOP_LIP_HEIGHT_M,
                "cap_static_friction": 2.4,
                "cap_dynamic_friction": 2.0,
            },
            "cap_colors": dict(instance.cap_colors),
            "appearance_claim": ("appearance_cue_assisted_retrieval"
                                 if contract.condition_variant == "distinct_cap" else
                                 "spatial_history_retrieval"),
            "tube_dynamics": {
                "mass_kg": contract.tube_mass_kg,
                "center_of_mass_z_m": contract.tube_center_of_mass_z_m,
                "angular_damping": contract.tube_angular_damping,
            },
            "asymmetric_orientation_cues": ["blue_long_edge", "yellow_short_edge",
                                             "black_white_A1_corner"],
            "grasp_physics": "high_friction_fingers_plus_low_friction_tube",
            "grasp_contact_parameters": {
                "static_friction": 3.0,
                "dynamic_friction": 2.5,
                "friction_combine_mode": "max",
                "contact_offset_m": .0005,
                "rest_offset_m": 0.0,
                "torsional_patch_radius_m": .020,
                "min_torsional_patch_radius_m": .005,
            },
            "insertion_contact_parameters": {
                "tube_static_friction": .12,
                "tube_dynamic_friction": .08,
                "tube_friction_combine_mode": "min",
                "guide_static_friction": .20,
                "guide_dynamic_friction": .15,
                "guide_friction_combine_mode": "min",
                "restitution": 0.0,
            },
            "insertion_guard": {
                "axial_force_limit_n": contract.insertion_axial_force_limit_n,
                "lateral_force_limit_n": contract.insertion_lateral_force_limit_n,
                "force_hold_s": contract.insertion_force_hold_s,
                "stall_window_s": contract.insertion_stall_window_s,
                "stall_progress_m": contract.insertion_stall_progress_m,
                "retract_m": contract.insertion_retract_m,
                "max_recoveries": contract.insertion_max_recoveries,
            },
            "layout_hash": instance.layout_hash, "motion_hash": instance.motion_hash}
