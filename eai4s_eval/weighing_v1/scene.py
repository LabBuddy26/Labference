"""Core World scene helpers for feedback weighing (Isaac imports stay local).

Build an empty physical scale, calibrate it, then place instance loads once
before the scored rollout. The scale joint is internal to the instrument;
robot-object grasping remains ordinary contact physics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import random

from .loadcell import LoadCellReading, segment_state


WEIGHT_MASSES_G = (10, 20, 50, 100)
SCALE_PATH = "/World/Scale"
TRAY_PATH = f"{SCALE_PATH}/Tray"
LOADCELL_JOINT_PATH = f"{SCALE_PATH}/LoadCellJoint"
PAN_MASS_KG = 0.12
SCENE_LAYOUT_VERSION = "weighing-layout/v1"


@dataclass(frozen=True)
class SceneLayout:
    version: str
    scale_xy: tuple[float, float]
    sample_spawn_xy: tuple[float, float]
    weight_spawn_xy: tuple[tuple[float, float], ...]

    def public(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        payload = json.dumps(self.public(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_scene_layout(scene_seed: int) -> SceneLayout:
    """Vary reachable object bays independently of the mass/decision seed."""
    rng = random.Random(scene_seed)
    scale_xy = (round(0.23 + rng.uniform(-0.008, 0.008), 4),
                round(0.13 + rng.uniform(-0.008, 0.008), 4))
    bays = [0.08, 0.155, 0.23, 0.305]
    rng.shuffle(bays)
    weight_xy = tuple((round(x + rng.uniform(-0.006, 0.006), 4),
                       round(-0.19 + rng.uniform(-0.004, 0.004), 4)) for x in bays)
    sample_xy = (round(0.39 + rng.uniform(-0.006, 0.006), 4),
                 round(-0.20 + rng.uniform(-0.004, 0.004), 4))
    return SceneLayout(SCENE_LAYOUT_VERSION, scale_xy, sample_xy, weight_xy)


@dataclass
class SceneHandles:
    world: Any
    stage: Any
    franka: Any
    sample: Any
    weights: dict[int, Any]
    tray: Any
    display: str
    cameras: dict[str, Any]
    tray_top_z: float
    tray_center: tuple[float, float, float]
    table_top_z: float
    loadcell_joint_path: str
    object_start_poses: dict[str, tuple[float, float, float]]
    layout: SceneLayout
    instance: Any = field(repr=False)
    _last_display_text: str | None = field(default=None, repr=False)
    _last_stable: bool | None = field(default=None, repr=False)
    _loads_placed: bool = field(default=False, repr=False)

    def initialize(self) -> None:
        """Initialize wrappers and all cameras after world.reset()."""
        from .cameras import initialize_camera
        self.franka.initialize()
        self.sample.initialize()
        for weight in self.weights.values():
            weight.initialize()
        self.tray.initialize()
        for camera in self.cameras.values():
            initialize_camera(camera)

    def place_initial_loads(self) -> dict[str, tuple[float, float, float]]:
        """Pre-score one-time checkpoint setup, after empty-pan calibration."""
        if self._loads_placed:
            raise RuntimeError("initial loads have already been placed")
        positions = initial_load_positions(self.tray_center, self.tray_top_z,
                                           self.instance.initial_weights_g)
        self._set_initial_pose(self.sample, positions["sample"])
        for mass in self.instance.initial_weights_g:
            self._set_initial_pose(self.weights[int(mass)], positions[f"{int(mass)}g"])
        self._loads_placed = True
        return positions

    @staticmethod
    def _set_initial_pose(body: Any, position: tuple[float, float, float]) -> None:
        body.set_world_pose(position=position)
        body.set_linear_velocity((0.0, 0.0, 0.0))
        body.set_angular_velocity((0.0, 0.0, 0.0))


def initial_load_positions(tray_center: tuple[float, float, float], tray_top_z: float,
                           initial_weights_g: tuple[int, ...] | list[int]) -> dict[str, tuple[float, float, float]]:
    """Non-overlapping locations for a sample and up to four 36-mm weights."""
    x, y, _ = tray_center
    slots = {10: (0.00, 0.045), 20: (0.068, 0.045),
             50: (-0.020, -0.046), 100: (0.067, -0.046)}
    positions = {"sample": (x - 0.070, y + 0.038, tray_top_z + 0.022)}
    for mass in initial_weights_g:
        mass = int(mass)
        if mass not in slots or f"{mass}g" in positions:
            raise ValueError(f"invalid or repeated initial weight: {mass}")
        dx, dy = slots[mass]
        positions[f"{mass}g"] = (x + dx, y + dy, tray_top_z + 0.020)
    footprints = [("sample", positions["sample"], 0.024)]
    footprints += [(f"{int(mass)}g", positions[f"{int(mass)}g"], 0.018)
                   for mass in initial_weights_g]
    for index, (name, center, radius) in enumerate(footprints):
        if abs(center[0] - x) + radius > 0.118 or abs(center[1] - y) + radius > 0.088:
            raise ValueError(f"initial {name} footprint is outside the scale tray")
        for other_name, other_center, other_radius in footprints[:index]:
            if math.dist(center[:2], other_center[:2]) < radius + other_radius + 0.002:
                raise ValueError(f"initial {name} overlaps {other_name}")
    return positions


def build_scene(world: Any, contract: Any, instance: Any, asset_root: str | Path, *,
                enable_cameras: bool = True) -> SceneHandles:
    """Author table, Franka, physical scale, sample, weights and cameras.

    Caller starts SimulationApp and creates a 1-metre Core World at 120 Hz.
    All loads start off the scale for empty-pan calibration. Do not call
    world.reset() again after place_initial_loads(), because it restores the
    stage's pre-calibration initial positions.
    """
    import numpy as np
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from factories.robot_factory import create_robot

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac USD stage is not available")
    _validate_instance(instance)
    layout = generate_scene_layout(instance.scene_seed)
    physics_hz = int(contract.physics_hz)
    if physics_hz <= 0:
        raise ValueError("contract.physics_hz must be positive")
    world.get_physics_context().set_physics_dt(1.0 / physics_hz)
    table_file = _resolve_table(asset_root)
    table_path = "/World/table"
    if stage.GetPrimAtPath(table_path).IsValid():
        raise ValueError(f"scene already contains {table_path}")
    add_reference_to_stage(usd_path=str(table_file), prim_path=table_path)
    table = stage.GetPrimAtPath(table_path)
    table_xform = UsdGeom.Xformable(table)
    table_xform.ClearXformOpOrder()
    table_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.25))
    table_xform.AddScaleOp().Set(Gf.Vec3f(0.001, 0.001, 0.001))
    table_bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_,
                                      UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    table_top_z = float(table_bounds.ComputeWorldBound(table).ComputeAlignedRange().GetMax()[2])
    if not math.isfinite(table_top_z) or not 0.4 <= table_top_z <= 1.2:
        raise RuntimeError(f"table USD produced invalid top surface: z={table_top_z}")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(1000.0)
    dome.CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75))
    franka = create_robot("franka", position=np.array([-0.30, 0.0, table_top_z - 0.01]))

    scale_x, scale_y = layout.scale_xy
    base_z = table_top_z + 0.002
    root = UsdGeom.Xform.Define(stage, SCALE_PATH)
    root.AddTranslateOp().Set(Gf.Vec3d(scale_x, scale_y, base_z))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    _author_materials(stage)
    base = _rigid_box(stage, f"{SCALE_PATH}/Base", (0.0, 0.0, 0.014),
                      (0.30, 0.24, 0.028), "Body", mass_kg=1.2)
    tray_local = (0.0, 0.018, 0.044)
    tray = _rigid_box(stage, TRAY_PATH, tray_local, (0.24, 0.18, 0.012),
                       "Tray", mass_kg=PAN_MASS_KG)
    _author_support_joints(stage, base.GetPrim(), tray.GetPrim(), (scale_x, scale_y, base_z), tray_local)
    display_path = f"{SCALE_PATH}/Display"
    _author_display(stage, display_path)
    tray_center = (scale_x + tray_local[0], scale_y + tray_local[1], base_z + tray_local[2])
    tray_top_z = tray_center[2] + 0.006

    sample_spawn = (*layout.sample_spawn_xy, table_top_z + 0.022)
    sample = _author_object(stage, "/World/Sample", sample_spawn,
                            float(instance.sample_mass_g) / 1000.0, sample=True)
    start_poses = {"sample": sample_spawn}
    weights = {}
    for index, mass in enumerate(WEIGHT_MASSES_G):
        start = (*layout.weight_spawn_xy[index], table_top_z + 0.020)
        weights[mass] = _author_object(stage, f"/World/weight_{mass}g", start,
                                       mass / 1000.0, sample=False, label=str(mass))
        start_poses[f"{mass}g"] = start
    cameras = _build_cameras(franka, (scale_x, scale_y, table_top_z),
                             contract.physics_hz) if enable_cameras else {}
    handles = SceneHandles(world, stage, franka, sample, weights,
                           SingleRigidPrim(prim_path=TRAY_PATH, name="weighing_tray"),
                           display_path, cameras, tray_top_z, tray_center, table_top_z,
                           LOADCELL_JOINT_PATH, start_poses, layout, instance)
    update_display(handles, LoadCellReading(-1, False, "not_calibrated", "physx_loadcell_joint",
                                            None, None, None, None, "----", False, None))
    return handles


def update_display(handles: SceneHandles, reading: LoadCellReading) -> dict[str, Any]:
    """Update emissive seven-segment prims from measured data only."""
    from pxr import Sdf, UsdShade

    text = reading.display_text if reading.valid else "----"
    stable = bool(reading.valid and reading.stable)
    if handles._last_display_text == text and handles._last_stable == stable:
        return {"display_text": text, "stable": stable}
    root = SCALE_PATH
    on = UsdShade.Material.Get(handles.stage, Sdf.Path(f"{root}/Looks/SegmentOn"))
    off = UsdShade.Material.Get(handles.stage, Sdf.Path(f"{root}/Looks/SegmentOff"))
    state = segment_state(text)
    for name, active in state.items():
        prim = handles.stage.GetPrimAtPath(f"{handles.display}/{name}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(on if active else off)
    indicator = handles.stage.GetPrimAtPath(f"{handles.display}/StableIndicator")
    stable_mat = UsdShade.Material.Get(handles.stage, Sdf.Path(f"{root}/Looks/StableOn"))
    UsdShade.MaterialBindingAPI.Apply(indicator).Bind(stable_mat if stable else off)
    handles._last_display_text = text
    handles._last_stable = stable
    return {"display_text": text, "stable": stable}


def _validate_instance(instance: Any) -> None:
    mass = float(instance.sample_mass_g)
    if not math.isfinite(mass) or mass <= 0 or mass >= 1000:
        raise ValueError("sample_mass_g must be in (0, 1000)")
    target = float(instance.target_g)
    if not math.isfinite(target) or not 0 <= target <= 999:
        raise ValueError("target_g must be in [0, 999]")
    initial = tuple(int(m) for m in instance.initial_weights_g)
    if len(initial) != len(set(initial)) or any(m not in WEIGHT_MASSES_G for m in initial):
        raise ValueError("initial_weights_g must contain unique standard weights")


def _resolve_table(asset_root: str | Path) -> Path:
    root = Path(asset_root).expanduser().resolve()
    for path in (root / "assets/chemistry_lab/assets/table.usd",
                 root / "chemistry_lab/assets/table.usd"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"main-repo table asset not found under {root}")


def _author_support_joints(stage: Any, base: Any, tray: Any,
                           root_position: tuple[float, float, float],
                           tray_local: tuple[float, float, float]) -> None:
    from pxr import Gf, Sdf, UsdPhysics

    root_joint = UsdPhysics.FixedJoint.Define(stage, f"{SCALE_PATH}/WorldAnchorJoint")
    root_joint.CreateBody1Rel().SetTargets([base.GetPath()])
    root_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(root_position[0], root_position[1], root_position[2] + 0.014))
    root_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint = UsdPhysics.FixedJoint.Define(stage, LOADCELL_JOINT_PATH)
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([tray.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(tray_local[0], tray_local[1], tray_local[2] - 0.014))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))


def _author_object(stage: Any, path: str, position: tuple[float, float, float],
                   mass_kg: float, *, sample: bool, label: str = "") -> Any:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from isaacsim.core.prims import SingleRigidPrim

    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*position))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass_kg)
    if not sample:
        dynamics = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
        dynamics.CreateLinearDampingAttr(0.4)
        dynamics.CreateAngularDampingAttr(10.0)
    if sample:
        _cylinder(stage, f"{path}/Body", (0, 0, 0), 0.024, 0.044, "Sample", collide=True)
        _cylinder(stage, f"{path}/Cap", (0, 0, 0.023), 0.023, 0.004, "SampleCap", collide=False)
    else:
        _cylinder(stage, f"{path}/Body", (0, 0, 0), 0.018, 0.040, "Weight", collide=True)
        _cylinder(stage, f"{path}/GripNeck", (0, 0, 0.026), 0.006, 0.012, "Weight",
                  collide=True)
        _cylinder(stage, f"{path}/TopLip", (0, 0, 0.034), 0.012, 0.004, "Weight",
                  collide=True)
        _author_weight_label(stage, path, label)
    return SingleRigidPrim(prim_path=path, name=("sample" if sample else f"weight_{label}g"))


def _build_cameras(franka: Any, scale_position: tuple[float, float, float],
                   physics_hz: int) -> dict[str, Any]:
    import numpy as np
    from isaacsim.sensors.camera import Camera
    from scipy.spatial.transform import Rotation

    def look_at(position, target):
        forward = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(-forward, right)
        q = Rotation.from_matrix(np.column_stack((right, up, -forward))).as_quat()
        return np.asarray([q[3], q[0], q[1], q[2]], dtype=float)

    front_position = np.array([0.90, -1.80, scale_position[2] + 1.00])
    front_target = np.array([-0.02, -0.02, scale_position[2] + 0.24])
    front = Camera(prim_path="/World/Camera1", name="weighing_front", frequency=physics_hz,
                   resolution=(640, 480), translation=front_position)
    front.set_world_pose(position=front_position, orientation=look_at(front_position, front_target), camera_axes="usd")
    # Isaac Sim camera intrinsics use millimetres.  Keep the fixed front-view
    # calibration used by the weighing task (12.0 mm focal length,
    # 20.955 mm horizontal aperture); the wrist camera below retains the
    # historical 10.0 mm camera_3 calibration.
    front.set_focal_length(12.0)
    front.set_horizontal_aperture(20.955)
    front.set_clipping_range(near_distance=0.1, far_distance=10.0)
    display_position = np.array([scale_position[0], scale_position[1] - 0.69,
                                 scale_position[2] + 0.105])
    display_target = np.array([scale_position[0], scale_position[1] - 0.125,
                               scale_position[2] + 0.065])
    close = Camera(prim_path="/World/ScaleDisplayCamera", name="weighing_display", frequency=physics_hz,
                   resolution=(640, 480),
                   translation=display_position)
    close.set_world_pose(position=display_position,
                         orientation=look_at(display_position, display_target), camera_axes="usd")
    close.set_focal_length(20.0)
    close.set_horizontal_aperture(20.955)
    close.set_clipping_range(near_distance=0.05, far_distance=5.0)
    weights_position = np.array([0.19, -0.85, scale_position[2] + 0.52])
    weights_target = np.array([0.19, -0.19, scale_position[2] + 0.025])
    weights_close = Camera(prim_path="/World/WeightsCamera", name="weighing_weights",
                           frequency=physics_hz, resolution=(640, 480),
                           translation=weights_position)
    weights_close.set_world_pose(position=weights_position,
                                 orientation=look_at(weights_position, weights_target),
                                 camera_axes="usd")
    weights_close.set_focal_length(11.0)
    weights_close.set_horizontal_aperture(20.955)
    weights_close.set_clipping_range(near_distance=0.05, far_distance=5.0)

    # ``create_robot`` returns the Franka articulation; it does not attach a
    # camera object. Author the wrist camera under panda_hand explicitly.
    wrist = Camera(
        # Keep the historical LabBench camera_3 prim and calibration.
        prim_path=f"{franka.prim_path_str}/panda_hand/arm_camera",
        name="weighing_wrist", frequency=physics_hz, resolution=(640, 480))
    wrist.set_focal_length(10.0)
    wrist.set_horizontal_aperture(20.955)
    wrist.set_clipping_range(near_distance=0.03, far_distance=10.0)
    # Match the side-looking hand camera used by the existing pick tasks.
    wrist.set_local_pose(
        translation=np.array([-0.20, 0.0, -0.01]),
        orientation=np.array([0.20083, 0.67799, -0.67799, -0.20083]),
        camera_axes="usd")
    return {"front": front, "wrist": wrist, "display": close,
            "weights": weights_close}


def _author_display(stage: Any, path: str) -> None:
    from pxr import Gf, UsdGeom

    _box(stage, f"{SCALE_PATH}/DisplayPanel", (0, -0.121, 0.062),
         (0.286, 0.007, 0.068), "Panel")
    UsdGeom.Xform.Define(stage, path)
    start_x, pitch, width, height, y = -0.078, 0.052, 0.034, 0.046, -0.125
    directions = {
        "a": (0, height * .42, width * .72, .0023),
        "g": (0, 0, width * .72, .0023),
        "d": (0, -height * .42, width * .72, .0023),
        "f": (-width * .39, height * .21, .0023, height * .38),
        "b": (width * .39, height * .21, .0023, height * .38),
        "e": (-width * .39, -height * .21, .0023, height * .38),
        "c": (width * .39, -height * .21, .0023, height * .38),
    }
    for index in range(4):
        center_x = start_x + index * pitch
        for segment, (dx, dz, sx, sz) in directions.items():
            _box(stage, f"{path}/d{index}_{segment}", (center_x + dx, y, .063 + dz),
                 (sx, .0012, sz), "SegmentOff")
        _box(stage, f"{path}/d{index}_dp", (center_x + width * .56, y, .043),
             (.003, .0012, .003), "SegmentOff")
    _box(stage, f"{path}/StableIndicator", (0.122, y, .081),
         (.006, .0012, .006), "SegmentOff")


def _author_weight_label(stage: Any, root: str, text: str) -> None:
    from .loadcell import SEGMENTS

    pitch, y = 0.0085, -0.0187
    start = -pitch * (len(text) - 1) / 2
    parts = {"a": (0, .004, .005, .0006), "g": (0, 0, .005, .0006),
             "d": (0, -.004, .005, .0006), "f": (-.003, .002, .0006, .0035),
             "b": (.003, .002, .0006, .0035), "e": (-.003, -.002, .0006, .0035),
             "c": (.003, -.002, .0006, .0035)}
    for index, digit in enumerate(text):
        for segment in SEGMENTS[digit]:
            dx, dz, sx, sz = parts[segment]
            _box(stage, f"{root}/Label/D{index}_{segment}",
                 (start + index * pitch + dx, y, .004 + dz), (sx, .0005, sz), "Label")


def _box(stage: Any, path: str, position: tuple[float, float, float],
         size: tuple[float, float, float], material: str, *, collide: bool = False) -> Any:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    box.AddTranslateOp().Set(Gf.Vec3d(*position))
    box.AddScaleOp().Set(Gf.Vec3f(*size))
    _bind(stage, box.GetPrim(), material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        PhysxSchema.PhysxContactReportAPI.Apply(box.GetPrim()).CreateThresholdAttr().Set(0)
        _bind_physics(stage, box.GetPrim())
    return box


def _rigid_box(stage: Any, path: str, position: tuple[float, float, float],
               size: tuple[float, float, float], material: str, *, mass_kg: float) -> Any:
    """Keep joint frames on unscaled Xforms; scale only the collider child."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*position))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.CollisionAPI.Apply(root.GetPrim())
    PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim()).CreateThresholdAttr().Set(0)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass_kg)
    _box(stage, f"{path}/Collider", (0.0, 0.0, 0.0), size, material, collide=True)
    return root


def _cylinder(stage: Any, path: str, position: tuple[float, float, float],
              radius: float, height: float, material: str, *, collide: bool) -> Any:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.CreateAxisAttr("Z")
    cylinder.AddTranslateOp().Set(Gf.Vec3d(*position))
    _bind(stage, cylinder.GetPrim(), material)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
        PhysxSchema.PhysxContactReportAPI.Apply(cylinder.GetPrim()).CreateThresholdAttr().Set(0)
        _bind_physics(stage, cylinder.GetPrim())
    return cylinder


def _bind(stage: Any, prim: Any, name: str) -> None:
    from pxr import UsdShade

    material = UsdShade.Material.Get(stage, f"{SCALE_PATH}/Looks/{name}")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _bind_physics(stage: Any, prim: Any) -> None:
    from pxr import UsdShade

    material = UsdShade.Material.Get(stage, f"{SCALE_PATH}/Looks/GripPhysics")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,
                                                   UsdShade.Tokens.weakerThanDescendants, "physics")


def _author_materials(stage: Any) -> None:
    from pxr import Gf, Sdf, UsdPhysics, UsdShade

    styles = {
        "Body": ((0.07, 0.08, 0.09), (0, 0, 0)),
        "Tray": ((0.57, 0.60, 0.62), (0, 0, 0)),
        "Panel": ((0.004, 0.004, 0.005), (0, 0, 0)),
        "SegmentOff": ((0.035, 0.002, 0.002), (0, 0, 0)),
        "SegmentOn": ((0.98, 0.015, 0.008), (25, .13, .05)),
        "StableOn": ((.08, .7, .16), (.1, 6, .3)),
        "Sample": ((.35, .44, .48), (0, 0, 0)),
        "SampleCap": ((.20, .30, .34), (0, 0, 0)),
        "Weight": ((.73, .74, .75), (0, 0, 0)),
        "Label": ((.015, .02, .025), (0, 0, 0)),
    }
    for name, (color, emissive) in styles.items():
        path = f"{SCALE_PATH}/Looks/{name}"
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(.4 if name in {"Tray", "Weight"} else 0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.38)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    grip = UsdShade.Material.Define(stage, f"{SCALE_PATH}/Looks/GripPhysics")
    physics = UsdPhysics.MaterialAPI.Apply(grip.GetPrim())
    physics.CreateStaticFrictionAttr(1.6)
    physics.CreateDynamicFrictionAttr(1.2)
    physics.CreateRestitutionAttr(0.0)
