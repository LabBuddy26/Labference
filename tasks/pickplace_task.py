from typing import Dict, Any
from .base_task import BaseTask
from isaacsim.core.utils.prims import set_prim_visibility
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

class PickPlaceTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        """Initialize the Pick and Pour task.

        Args:
            cfg: Configuration object for the task.
            world: The simulation world instance.
            stage: The USD stage for the simulation.
            robot: The robot instance used in the task.
        """
        super().__init__(cfg, world, stage, robot)
        self.sub_object = cfg.sub_object
        self.target_path = cfg.target_path
        self.tube_insertion_cfg = getattr(cfg, "tube_insertion", None)

    def _target_rack_local_point_to_world(self, local_point: np.ndarray) -> np.ndarray:
        """Transform an authored test-tube-rack point into world coordinates.

        The rack USD is authored in millimetres and is scaled by its parent
        Xform.  Applying the target rack's complete local-to-world transform
        makes the socket target follow both the configured placement and any
        future reset-time randomisation of the rack.
        """
        rack_prim = self.stage.GetPrimAtPath(self.target_path)
        if not rack_prim.IsValid():
            raise ValueError(f"Target tube rack does not exist: {self.target_path}")
        transform = UsdGeom.Xformable(rack_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        point = transform.Transform(Gf.Vec3d(*np.asarray(local_point, dtype=np.float64)))
        return np.asarray(point, dtype=np.float64)

    def _target_rack_local_axis_to_world(self, local_axis: np.ndarray) -> np.ndarray:
        """Return a normalized rack-local direction in world coordinates."""
        rack_prim = self.stage.GetPrimAtPath(self.target_path)
        transform = UsdGeom.Xformable(rack_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        origin = transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        endpoint = transform.Transform(Gf.Vec3d(*np.asarray(local_axis, dtype=np.float64)))
        axis = np.asarray(endpoint - origin, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            raise ValueError("Target rack insertion axis is degenerate")
        return axis / norm

    def _get_tube_insertion_target(self) -> Dict[str, np.ndarray]:
        """Compute the selected physical socket and resting tube centre."""
        socket = self._target_rack_local_point_to_world(
            self.tube_insertion_cfg.socket_local_position
        )
        pre_release_center = self._target_rack_local_point_to_world(
            getattr(
                self.tube_insertion_cfg,
                "pre_release_tube_center_local_position",
                self.tube_insertion_cfg.inserted_tube_center_local_position,
            )
        )
        settled_center = self._target_rack_local_point_to_world(
            self.tube_insertion_cfg.inserted_tube_center_local_position
        )
        return {
            "tube_socket_position": socket,
            "tube_pre_release_center": pre_release_center,
            "tube_inserted_center": settled_center,
            "tube_axis": self._target_rack_local_axis_to_world([0.0, 0.0, 1.0]),
        }

    def _get_object_axis(self, object_path: str) -> np.ndarray:
        """Get the tube's authored +Z (its longitudinal axis) in world space."""
        prim = self.stage.GetPrimAtPath(object_path)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        origin = transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        endpoint = transform.Transform(Gf.Vec3d(0.0, 0.0, 1.0))
        axis = np.asarray(endpoint - origin, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            raise ValueError(f"Object axis is degenerate: {object_path}")
        return axis / norm

    def reset(self):
        """Reset the task state."""
        super().reset()
        self.robot.initialize()
        # target_tube_rack
        # for obj_config in self.obj_configs:
        #     obj_path = obj_config['path']
        #     position_range = obj_config['position_range']
        #     prim = self.stage.GetPrimAtPath(obj_path)
        #     if prim.IsValid():
        #         self.randomize_object_position(obj_path, position_range)

        # test_tube_rack
        # rack_position = np.array([0.28421, 0.30755, 0.82291])
        rack_position = np.array([0.28421, 0.30755, 0.80])
        self._source_rack_position = rack_position.copy()
        self.object_utils.set_object_position(object_path="/World/tube_rack", position=rack_position)

        # Keep the source tube sufficiently exposed above the rack grid for a
        # collision-free vertical grasp.  This is intentionally independent
        # from the controller's end-effector grasp offset.
        tube_spawn_offset = np.asarray(
            getattr(
                self.tube_insertion_cfg,
                "source_tube_spawn_offset",
                [-0.023, -0.1125, 0.03197],
            ),
            dtype=np.float64,
        )
        self.object_utils.set_object_position(
            object_path=self.sub_object,
            position=rack_position + tube_spawn_offset,
        )
        self.object_utils.set_object_orientation(
            object_path=self.sub_object,
            orientation=np.array([0.0, 1.0, 0.0, 0.0]),
        )
        # Teleporting a dynamic prim does not reliably clear the velocity left
        # by the preceding episode. Reset every rigid descendant so the tube
        # settles vertically in the source socket before startup validation.
        tube_prim = self.stage.GetPrimAtPath(self.sub_object)
        for prim in Usd.PrimRange(tube_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body = UsdPhysics.RigidBodyAPI(prim)
                rigid_body.GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                rigid_body.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        # Keep tube/rack contact moderately slippery so a tiny socket-edge
        # error cannot store enough tangential force to launch the tube.  Grip
        # friction is bound separately to the finger collision geometry below.
        self.add_friction_to_prim(
            prim_path=f"{self.sub_object}/geometry/mesh",
            static_friction=float(getattr(self.tube_insertion_cfg, "tube_static_friction", 0.8)),
            dynamic_friction=float(getattr(self.tube_insertion_cfg, "tube_dynamic_friction", 0.6)),
            material_path="/World/PhysicsMaterials/TubeRackContact",
        )
        for finger_path in (
            "/World/Franka/panda_leftfinger",
            "/World/Franka/panda_rightfinger",
        ):
            finger_prim = self.stage.GetPrimAtPath(finger_path)
            if not finger_prim.IsValid():
                continue
            for prim in Usd.PrimRange(finger_prim):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                self.add_friction_to_prim(
                    prim_path=str(prim.GetPath()),
                    static_friction=float(
                        getattr(self.tube_insertion_cfg, "grip_static_friction", 3.0)
                    ),
                    dynamic_friction=float(
                        getattr(self.tube_insertion_cfg, "grip_dynamic_friction", 2.5)
                    ),
                    material_path="/World/PhysicsMaterials/TubeGrip",
                    binding_strength=UsdShade.Tokens.strongerThanDescendants,
                )

    def get_startup_object_positions(self):
        """Track the tube itself during the generic stationary-scene gate."""
        return [self.object_utils.get_geometry_center(object_path=self.sub_object)]

    def get_startup_readiness(self):
        """Require the source tube to be settled in its authored rack socket."""
        object_position = self.object_utils.get_geometry_center(object_path=self.sub_object)
        if object_position is None:
            return False, "source tube geometry centre is unavailable"
        rest_offset = np.asarray(
            self.tube_insertion_cfg.source_tube_rest_center_offset,
            dtype=np.float64,
        )
        target_position = self._source_rack_position + rest_offset
        xy_error = float(np.linalg.norm(object_position[:2] - target_position[:2]))
        z_error = abs(float(object_position[2] - target_position[2]))
        object_axis = self._get_object_axis(self.sub_object)
        axis_alignment = abs(float(np.dot(object_axis, np.array([0.0, 0.0, 1.0]))))
        axis_error_degrees = float(
            np.degrees(np.arccos(np.clip(axis_alignment, 0.0, 1.0)))
        )
        xy_tolerance = float(self.tube_insertion_cfg.source_startup_xy_tolerance)
        z_tolerance = float(self.tube_insertion_cfg.source_startup_z_tolerance)
        axis_tolerance = float(
            self.tube_insertion_cfg.source_startup_axis_tolerance_degrees
        )
        ready = (
            xy_error <= xy_tolerance
            and z_error <= z_tolerance
            and axis_error_degrees <= axis_tolerance
        )
        detail = (
            f"source tube target={np.array2string(target_position, precision=6)}, "
            f"actual={np.array2string(object_position, precision=6)}, "
            f"xy_error={xy_error:.6f}/{xy_tolerance:.6f} m, "
            f"z_error={z_error:.6f}/{z_tolerance:.6f} m, "
            f"axis_error={axis_error_degrees:.3f}/{axis_tolerance:.3f} deg"
        )
        return ready, detail
        
    def get_basic_state_info(self, joint_positions: np.ndarray = None, object_path: str = None, 
                           target_path: str = None, additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get basic state information containing common state data for all tasks.
        
        Args:
            joint_positions: Robot joint positions
            object_path: Object path
            target_path: Target object path
            additional_info: Additional state information
            
        Returns:
            Dict[str, Any]: Basic state information dictionary
        """
        
        if joint_positions is None:
            joint_positions = self.robot.get_joint_positions()
            if joint_positions is None:
                return None

        
        camera_data, display_data = self.get_camera_data()
        state = {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': display_data,
            'done': self.reset_needed,
            'gripper_position': self.robot.get_gripper_position(),
        }
        
        if object_path:
            state.update({
                'object_position': self.object_utils.get_geometry_center(object_path=object_path),
                'object_size': self.object_utils.get_object_size(object_path=object_path),
                'object_path': object_path,
                'object_name': object_path.split("/")[-1]
            })
        
        
        if target_path:
            if self.tube_insertion_cfg is not None:
                insertion_target = self._get_tube_insertion_target()
                target_position = insertion_target['tube_inserted_center']
            else:
                insertion_target = {}
                target_position = (
                    self.object_utils.get_object_xform_position(object_path=target_path)
                    + np.array([+0.03, -0.1525, 0.03197])
                )
            state.update({
                'target_position': target_position,
                'target_size': self.object_utils.get_object_size(object_path=target_path),
                'target_path': target_path,
                'target_name': target_path.split("/")[-1],
                **insertion_target,
            })
        
        if additional_info:
            state.update(additional_info)
            
        return state
    def step(self):
        """Execute one simulation step.

        Returns:
            dict: A dictionary containing simulation state data, or None if not ready.
        """
        self.frame_idx += 1
        if not self.check_frame_limits():
            return None

        source_quaternion = self.object_utils.get_transform_quat(object_path=self.sub_object)
        
        return self.get_basic_state_info(
            object_path=self.sub_object,
            target_path=self.target_path,
            additional_info={
                'object_quaternion': source_quaternion,
                'object_axis': self._get_object_axis(self.sub_object),
            }
        )

        
