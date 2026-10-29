"""Hard drying-oven cycle task imported from main and adapted for evaluation."""

import numpy as np
from pxr import UsdPhysics
from isaacsim.core.prims import SingleRigidPrim
from scipy.spatial.transform import Rotation

from .base_task import BaseTask


class DeviceHardOperateTask(BaseTask):
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
        self.device_path = str(cfg.task.device_path)
        self.beaker_path = str(cfg.task.beaker_path)
        self.beaker3_path = str(cfg.task.beaker3_path)
        self.button_path = str(cfg.task.button_path)
        self.interior_position = np.asarray(cfg.task.device_interior_position, dtype=float)
        self.device_reference_position = np.asarray(
            cfg.task.get("device_reference_position", [0.55, 0.13, 0.976]), dtype=float
        )
        self.beaker3_hand_target = np.asarray(cfg.task.beaker3_target_position, dtype=float)
        self.beaker3_geometry_center_offset = np.asarray(
            cfg.task.get("beaker3_geometry_center_offset", [0.02, 0.0, 0.05]), dtype=float
        )
        self.beaker3_initial_orientation = np.asarray(
            cfg.task.get("beaker3_initial_orientation_wxyz", [0.707, 0.707, 0.0, 0.0]),
            dtype=float,
        )
        self.beaker3_target_position = (
            self.beaker3_hand_target + self.beaker3_geometry_center_offset
        )
        self._beaker3_rigid_prim = None
        self._beaker3_body_local_offset = None
        self._beaker3_body_initial_orientation = None
        self.drive_api = None

    @property
    def joint_path(self):
        return f"{self.device_path}/{self.device_path.split('/')[-1]}/RevoluteJoint"

    @property
    def handle_path(self):
        return f"{self.device_path}/{self.device_path.split('/')[-1]}/handle"

    def reset(self):
        self.world.reset()
        self.reset_needed = False
        self.episode_outcome = None
        self.frame_idx = 0
        self.robot.initialize()
        self.source_obj = str(self.cfg.task.obj_paths[0].path)
        self.beaker3_obj = str(self.cfg.task.obj_paths[1].path)
        self.target_obj = str(self.cfg.task.obj_paths[2].path)
        self.object_utils.set_object_position(self.source_obj, self.interior_position)
        self.randomize_object_position(self.beaker3_obj, self.cfg.task.obj_paths[1].position_range)
        device_spawn = self.randomize_object_position(
            self.target_obj, self.cfg.task.obj_paths[2].position_range
        )
        # The authored hand target is calibrated relative to the oven's
        # reference placement. The oven root is randomized in XY, so translate
        # both the task's physical success window and the hand target with it.
        device_delta = np.asarray(device_spawn, dtype=float) - self.device_reference_position
        self.interior_position = np.asarray(
            self.cfg.task.device_interior_position, dtype=float
        ) + device_delta
        self.object_utils.set_object_position(self.source_obj, self.interior_position)
        self.beaker3_hand_target = np.asarray(
            self.cfg.task.beaker3_target_position, dtype=float
        ) + device_delta
        self.beaker3_target_position = (
            self.beaker3_hand_target + self.beaker3_geometry_center_offset
        )
        print(
            "L09 reset targets: "
            f"device_spawn={np.asarray(device_spawn).round(4).tolist()} "
            f"device_delta={device_delta.round(4).tolist()} "
            f"oven_hand={self.beaker3_hand_target.round(4).tolist()} "
            f"oven_object={self.beaker3_target_position.round(4).tolist()}",
            flush=True,
        )
        # The task controller may have made the beaker kinematic for a
        # scripted grasp.  World.reset() restores rigid-body state but does
        # not reliably clear this USD attribute or restore a side-lying beaker
        # after a failed episode. Restore its authored pose and explicitly
        # return every beaker rigid body to dynamic mode with zero velocity.
        self.object_utils.set_object_orientation(
            self.beaker3_obj, self.beaker3_initial_orientation
        )
        beaker_root = self.stage.GetPrimAtPath(self.beaker3_obj)
        rigid_body_prims = [
            prim for prim in self.stage.Traverse()
            if str(prim.GetPath()).startswith(self.beaker3_obj + "/")
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if not rigid_body_prims:
            raise RuntimeError(f"Beaker rigid body is missing below {self.beaker3_obj}")
        if self._beaker3_rigid_prim is None:
            self._beaker3_rigid_prim = SingleRigidPrim(
                prim_path=str(rigid_body_prims[0].GetPath()),
                name="l09_beaker3_reset_body",
            )
            self._beaker3_rigid_prim.initialize()
        for object_path in (self.source_obj, self.beaker3_obj):
            root = self.stage.GetPrimAtPath(object_path)
            for prim in [root, *(p for p in self.stage.Traverse()
                                  if str(p.GetPath()).startswith(object_path + "/"))]:
                rigid_body = UsdPhysics.RigidBodyAPI.Get(self.stage, prim.GetPath())
                if rigid_body:
                    rigid_body.CreateKinematicEnabledAttr().Set(False)
                    rigid_body.CreateVelocityAttr().Set((0.0, 0.0, 0.0))
                    rigid_body.CreateAngularVelocityAttr().Set((0.0, 0.0, 0.0))
        beaker_root_position = self.object_utils.get_object_xform_position(self.beaker3_obj)
        if self._beaker3_body_local_offset is None:
            body_position, body_orientation = self._beaker3_rigid_prim.get_world_pose()
            self._beaker3_body_local_offset = (
                np.asarray(body_position, dtype=float)
                - np.asarray(beaker_root_position, dtype=float)
            )
            self._beaker3_body_initial_orientation = np.asarray(body_orientation, dtype=float)
        else:
            # World.reset() doesn't consistently restore transforms written by
            # PhysX/Fabric to the nested rigid body. Restore that body relative
            # to the newly randomized root so each episode starts upright.
            self._beaker3_rigid_prim.set_world_pose(
                position=np.asarray(beaker_root_position, dtype=float)
                + self._beaker3_body_local_offset,
                orientation=self._beaker3_body_initial_orientation,
            )
        joint = self.stage.GetPrimAtPath(self.joint_path)
        if not joint.IsValid():
            raise RuntimeError(f"Drying-oven revolute joint is missing: {self.joint_path}")
        body_targets = UsdPhysics.Joint(joint).GetBody1Rel().GetTargets()
        if not body_targets:
            raise RuntimeError(f"Drying-oven door body is missing: {self.joint_path}")
        # Resolve the hinge centre while the authored USD relationship is
        # guaranteed to be available.  Once PhysX/Fabric starts updating the
        # articulated door, querying the relationship every control frame can
        # transiently return no targets in Isaac Sim 5.1.  The legacy helper
        # called exit() in that case, terminating a collection worker with the
        # misleading success code 0 before an HDF5 episode was finalized.
        self.revolute_joint_position = self.object_utils.get_revolute_joint_positions(
            self.joint_path)
        self.door_body_path = str(body_targets[0])
        self.door_rigid = SingleRigidPrim(
            prim_path=self.door_body_path, name="drying_oven_door_readback")
        self.door_rigid.initialize()
        self.handle_body_path = f"{self.handle_path}/mesh"
        self.handle_rigid = SingleRigidPrim(
            prim_path=self.handle_body_path, name="drying_oven_handle_readback")
        self.handle_rigid.initialize()
        handle_position, handle_quat_wxyz = self.handle_rigid.get_world_pose()
        handle_quat_xyzw = np.asarray(handle_quat_wxyz)[[1, 2, 3, 0]]
        handle_center = self.object_utils.get_geometry_center(object_path=self.handle_path)
        self.handle_center_local = Rotation.from_quat(handle_quat_xyzw).inv().apply(
            np.asarray(handle_center) - np.asarray(handle_position))
        self.drive_api = UsdPhysics.DriveAPI.Apply(joint, "angular")
        self.drive_api.CreateTypeAttr().Set("force")
        # Return the hinge to its authored closed state between episodes.
        # The previous episode may have left the autonomous drive targeting
        # 90 degrees; disabling it alone only freezes the door open.
        self.drive_api.CreateStiffnessAttr().Set(10000.0)
        self.drive_api.CreateDampingAttr().Set(1000.0)
        self.drive_api.CreateTargetPositionAttr().Set(0.0)
        for _ in range(120):
            self.world.step(render=False)
        self.drive_api.CreateStiffnessAttr().Set(0.0)
        self.drive_api.CreateDampingAttr().Set(0.0)

    def step(self):
        self.frame_idx += 1
        if self.frame_idx < 5:
            return None
        if self.frame_idx > int(self.cfg.task.max_steps):
            self.on_task_complete(False)
            return None
        joints = self.robot.get_joint_positions()
        if joints is None:
            return None
        camera_data, display_data = self.get_camera_data()
        handle_position, handle_quat_wxyz = self.handle_rigid.get_world_pose()
        handle_quat_xyzw = np.asarray(handle_quat_wxyz)[[1, 2, 3, 0]]
        handle_center = (np.asarray(handle_position)
                         + Rotation.from_quat(handle_quat_xyzw).apply(self.handle_center_local))
        return self.get_basic_state_info(
            joint_positions=joints,
            object_path=self.beaker3_path,
            additional_info={
                "object_name": "beaker3",
                "door_handle_position": handle_center,
                "beaker_position": self.object_utils.get_geometry_center(object_path=self.beaker_path),
                "beaker_size": self.object_utils.get_object_size(object_path=self.beaker_path),
                "beaker3_position": self.object_utils.get_geometry_center(object_path=self.beaker3_path),
                "beaker3_size": self.object_utils.get_object_size(object_path=self.beaker3_path),
                "beaker3_target_position": self.beaker3_target_position,
                "beaker3_place_hand_position": self.beaker3_hand_target,
                "device_interior_position": self.interior_position,
                "button_position": self.object_utils.get_geometry_center(object_path=self.button_path),
                "revolute_joint_position": self.revolute_joint_position.copy(),
                # Isaac Core returns quaternions as WXYZ; expose XYZW to the
                # scipy-based controller predicates.
                "door_orientation_xyzw": np.asarray(
                    self.door_rigid.get_world_pose()[1])[[1, 2, 3, 0]],
                "drive_api": self.drive_api,
                "gripper_position": self.robot.get_gripper_position(),
                "camera_display": display_data,
                "camera_data": camera_data,
                "sim_step": self.frame_idx,
                "done": self.reset_needed,
            },
        )
