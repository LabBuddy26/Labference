from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing

class PickController(BaseController):
    """A state machine controller for picking up objects.

    Manages the process of picking an object through multiple phases:
    - Phase 0: Move end effector above the object.
    - Phase 1: Lower end effector closer to the object.
    - Phase 2: Position end effector for grasping.
    - Phase 3: Wait for robot dynamics to settle.
    - Phase 4: Close gripper to grasp the object.
    - Phase 5: Lift the object.
    - Phase 6: Complete the sequence.

    Args:
        name (str): Identifier for the controller.
        cspace_controller (BaseController): Cartesian space controller that returns ArticulationAction.
        events_dt (List[float], optional): Duration for each phase. Defaults to [0.004, 0.002, 0.01, 0.2, 0.05, 0.004, 0.006].

    Raises:
        Exception: If events_dt is not a list or numpy array, or if its length is not 7.
    """

    _MOTION_EVENTS = frozenset({0, 1, 2, 5})

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        events_dt: typing.Optional[typing.List[float]] = None,
        position_threshold: float = 0.01,
        lift_speed: float = 0.02,
        physics_dt: float = 1.0 / 60.0,
        require_pose_reached: bool = False,
        max_motion_phase_steps: int = 1200,
        enable_scripted_attachment: bool = False,
        require_attachment_proximity: bool = True,
        incremental_lift: bool = False,
        attachment_object_path: typing.Optional[str] = None,
        attachment_proximity_tolerance: typing.Optional[float] = None,
    ) -> None:
        super().__init__(name=name)
        self._event = 0
        self._t = 0

        if events_dt is None:
            self._events_dt = [0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.006]
        else:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 7:
                raise Exception(f"events_dt length must be 7, got {len(self._events_dt)}")

        self._cspace_controller = cspace_controller
        self._start = True
        self.object_size = None
        self._position_threshold = position_threshold
        self._grasp_tolerance = 0.005
        self._robot_position = None


        self._lift_speed = lift_speed       
        self._physics_dt = physics_dt
        self._require_pose_reached = bool(require_pose_reached)
        self._max_motion_phase_steps = max(1, int(max_motion_phase_steps))
        self._motion_phase_steps = 0
        self._failed = False
        self._last_failure_reason = ""
        self._last_target_position = None
        self._last_actual_position = None
        self._enable_scripted_attachment = bool(enable_scripted_attachment)
        self._require_attachment_proximity = bool(require_attachment_proximity)
        self._incremental_lift = bool(incremental_lift)
        self._attachment_object_path = attachment_object_path
        self._default_attachment_proximity_tolerance = (
            None if attachment_proximity_tolerance is None
            else float(attachment_proximity_tolerance)
        )
        self._attachment_proximity_tolerance = self._default_attachment_proximity_tolerance
        self._grasp_target_position = None
        self._current_target_z = None        
        self._final_target_z = None
        self._event_zero_target = None

    def set_robot_position(self, position: np.ndarray):
        self._robot_position = position

    def _calculate_approach_direction(self, picking_position: np.ndarray) -> np.ndarray:
        if self._robot_position is None:
            return np.array([-1, 0, 0])  

        relative_pos = picking_position - self._robot_position
        
        horizontal_vec = relative_pos.copy()
        horizontal_vec[2] = 0

        if np.linalg.norm(horizontal_vec) > 0:
            horizontal_vec = -horizontal_vec / np.linalg.norm(horizontal_vec)
        else:
            horizontal_vec = np.array([-1, 0, 0])
        return horizontal_vec

    def forward(
        self,
        picking_position: np.ndarray,
        current_joint_positions: np.ndarray,
        object_name: str,
        object_size: np.ndarray,
        gripper_control,
        gripper_position: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        pre_offset_z: float = 0.12,
        after_offset_z: float = 0.15,
        pre_offset_x: float = 0.1,
        gripper_distances: float = None,
        pick_z_offset: float = None,
        grasp_tolerance: float = 0.005,
        attachment_proximity_tolerance: typing.Optional[float] = None,
    ) -> ArticulationAction:
        """Computes the joint positions for the current picking phase.

        Args:
            picking_position (np.ndarray): Target position for picking.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            object_name (str): Name of the object to pick.
            object_size (np.ndarray): Size of the object.
            gripper_control: Gripper controller instance.
            gripper_position (np.ndarray): Current position of the gripper.
            end_effector_orientation (np.ndarray, optional): Target orientation for the end effector. Defaults to [0, pi, 0] Euler angles.

        Returns:
            ArticulationAction: Joint positions for the robot to execute.
        """
        self.object_size = object_size
        
        if self._start:
            return self._handle_start_state(current_joint_positions)
        if self._failed:
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        self.pre_offset_z = pre_offset_z
        self.after_offset_z = after_offset_z
        self.pre_offset_x = pre_offset_x
        self._grasp_tolerance = float(grasp_tolerance)
        self._attachment_proximity_tolerance = (
            self._default_attachment_proximity_tolerance
            if attachment_proximity_tolerance is None
            else float(attachment_proximity_tolerance)
        )

        event_before = self._event
        target_joint_positions = self._execute_phase(
            picking_position,
            end_effector_orientation,
            current_joint_positions,
            object_name,
            gripper_control,
            gripper_position,
            gripper_distances,
            pick_z_offset,
        )

        if self._event != event_before:
            self._motion_phase_steps = 0
        elif self._event in self._MOTION_EVENTS:
            self._motion_phase_steps += 1
            if self._require_pose_reached and self._motion_phase_steps >= self._max_motion_phase_steps:
                self._mark_motion_timeout(self._event)

        timed_phase = not self._require_pose_reached or self._event in {3, 4, 6}
        # An incremental lift completes according to travelled distance, not a
        # fixed number of controller calls.  Timing phase 5 would otherwise
        # finish it before the slowly moving target reaches its final height.
        if self._incremental_lift and self._event == 5:
            timed_phase = False
        if timed_phase and self._event < len(self._events_dt):
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0
            
        return target_joint_positions

    def _handle_start_state(self, current_joint_positions):
        """Handles the initial state by opening the gripper.

        Args:
            current_joint_positions (np.ndarray): Current joint positions of the robot.

        Returns:
            ArticulationAction: Joint positions with gripper opened.
        """
        self._start = False
        target_joint_positions = [None] * current_joint_positions.shape[0]
        target_joint_positions[7] = 0.04 / get_stage_units()
        target_joint_positions[8] = 0.04 / get_stage_units()
        
        return ArticulationAction(joint_positions=target_joint_positions)

    def _execute_phase(
        self,
        picking_position,
        end_effector_orientation,
        current_joint_positions,
        object_name,
        gripper_control,
        gripper_position,
        gripper_distances,
        pick_z_offset,
    ):
        """Executes the current phase of the picking sequence.

        Args:
            picking_position (np.ndarray): Target position for picking.
            end_effector_orientation (np.ndarray): Target orientation for end effector.
            current_joint_positions (np.ndarray): Current robot joint positions.
            object_name (str): Name of the target object.
            gripper_control: Gripper controller instance.
            gripper_position (np.ndarray): Current gripper position.

        Returns:
            ArticulationAction: Joint position targets for robot control.
        """
        
        approach_dir = self._calculate_approach_direction(picking_position)
        
        if self._event == 0:
            if self._event_zero_target is not None:
                picking_position = self._event_zero_target.copy()
            else:
                picking_position = picking_position + approach_dir * (self.pre_offset_x / get_stage_units())
                picking_position[2] += self.object_size[2] + self.pre_offset_z
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation
            )
            self._record_motion_target(picking_position, gripper_position)
            if self._target_reached(
                picking_position,
                gripper_position,
                self._position_threshold,
                require_xyz=self._require_pose_reached,
            ):
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 1:
            
            # Keep the second pre-grasp waypoint consistent with the caller's
            # configured approach distance. The previous hard-coded 0.10 m
            # moved small side-grasp targets far away again after event 0.
            picking_position = picking_position + approach_dir * (
                self.pre_offset_x / get_stage_units()
            )
            picking_position[2] += self.get_pickprez_offset(object_name) / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation
            )
            self._record_motion_target(picking_position, gripper_position)
            if self._target_reached(
                picking_position,
                gripper_position,
                self._position_threshold,
                require_xyz=self._require_pose_reached,
            ):
                self._event += 1
                self._t = 0
            return target_joint_positions
        elif self._event == 2:
            if pick_z_offset is None:
                pick_z_offset = self.get_pickz_offset(object_name)
            picking_position[2] += pick_z_offset / get_stage_units()
            self._grasp_target_position = np.asarray(picking_position, dtype=float).copy()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=picking_position,
                target_end_effector_orientation=end_effector_orientation
            )
            self._record_motion_target(picking_position, gripper_position)
            if self._target_reached(
                picking_position,
                gripper_position,
                self._grasp_tolerance,
                require_xyz=True,
            ):
                self._event += 1
                self._t = 0
            return target_joint_positions

        elif self._event == 3:
            
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        elif self._event == 4:

            target_joint_positions = [None] * current_joint_positions.shape[0]
            if gripper_distances is None:
                gripper_distances = self.get_gripper_distance(object_name)
            # Callers specify an opening in metres; convert it consistently
            # here, including task-specific tighter grasp targets.
            gripper_target = gripper_distances / get_stage_units()
            target_joint_positions[7] = gripper_target
            target_joint_positions[8] = gripper_target
            target_joint_positions = ArticulationAction(joint_positions=target_joint_positions)
            
            grasp_position = (
                self._grasp_target_position.copy()
                if (self._incremental_lift or self._enable_scripted_attachment)
                and self._grasp_target_position is not None
                else np.asarray(picking_position, dtype=float).copy()
            )
            self.target_position = grasp_position.copy()
            self.target_position[2] += self.after_offset_z / get_stage_units()
            self._current_target_z = float(grasp_position[2])
            self._final_target_z = float(self.target_position[2])
            #gripper_control.add_object_to_gripper("/World/tube", "/World/Franka/panda_hand/tool_center")

            if self._enable_scripted_attachment and (
                    self._attachment_object_path or "glass" in object_name.lower()):
                grasp_error = np.linalg.norm(np.asarray(gripper_position) - grasp_position)
                attachment_ready = (
                    not self._require_attachment_proximity
                    or grasp_error <= max(
                        self._position_threshold,
                        0.01,
                        self._attachment_proximity_tolerance or 0.0,
                    )
                )
                if attachment_ready and not gripper_control.grasped_object_path:
                    object_path = self._attachment_object_path or (
                        "/World/glass_rod/glass_rod/Cylinder")
                    gripper_control.add_object_to_gripper(
                        object_path, "/World/Franka/panda_hand/tool_center")
                    print(
                        f"Scripted attachment established for {object_path} "
                        f"(grasp error={grasp_error:.4f} m).", flush=True)

            return target_joint_positions

        elif self._event == 5:
            lift_target = self.target_position
            if self._incremental_lift:
                lift_step = self._lift_speed * self._physics_dt / get_stage_units()
                self._current_target_z = min(
                    self._current_target_z + lift_step,
                    self._final_target_z,
                )
                lift_target = self.target_position.copy()
                lift_target[2] = self._current_target_z

            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=lift_target,
                target_end_effector_orientation=end_effector_orientation
            )
            self._record_motion_target(self.target_position, gripper_position)

            xy_distance = np.linalg.norm(gripper_position[:2] - self.target_position[:2])
            z_distance = abs(gripper_position[2] - self.target_position[2])
            lift_target_complete = (
                not self._incremental_lift
                or self._current_target_z >= self._final_target_z - 1e-9
            )

            if (
                lift_target_complete
                and xy_distance < self._position_threshold
                and z_distance < self._position_threshold
            ):
                self._event += 1
                self._t = 0
            return target_joint_positions
        else:
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

    def reset(
        self,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Resets the controller to the initial phase.

        Args:
            events_dt (List[float], optional): New phase durations. Defaults to None.

        Raises:
            Exception: If events_dt is not a list or numpy array, or if its length is not 7.
        """
        super().reset()
        self._cspace_controller.reset()
        self.reset_phase_state(events_dt=events_dt)

    def reset_phase_state(
        self,
        events_dt: typing.Optional[typing.List[float]] = None,
        initial_event: int = 0,
        initial_pregrasp_target: typing.Optional[np.ndarray] = None,
    ) -> None:
        """Reset only this pick state machine, preserving shared RMPFlow state."""
        self._event = int(initial_event)
        self._t = 0

        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 7:
                raise Exception(f"events_dt length must be 7, got {len(self._events_dt)}")

        self._start = True
        self.object_size = None
        self._robot_position = None
        self._motion_phase_steps = 0
        self._failed = False
        self._last_failure_reason = ""
        self._last_target_position = None
        self._last_actual_position = None
        self._grasp_target_position = None
        self._current_target_z = None
        self._final_target_z = None
        self._event_zero_target = (
            None
            if initial_pregrasp_target is None
            else np.asarray(initial_pregrasp_target, dtype=np.float64).copy()
        )

    def is_done(self) -> bool:
        """Checks if the picking sequence is complete.

        Returns:
            bool: True if the final phase is reached, False otherwise.
        """
        return self._failed or self._event >= len(self._events_dt)

    def has_failed(self) -> bool:
        """Return whether a pose-gated motion phase could not converge."""
        return self._failed

    @property
    def last_failure_reason(self) -> str:
        return self._last_failure_reason

    @staticmethod
    def _target_reached(target, actual, tolerance: float, require_xyz: bool) -> bool:
        if actual is None:
            return False
        target_array = np.asarray(target, dtype=np.float64).reshape(-1)[:3]
        actual_array = np.asarray(actual, dtype=np.float64).reshape(-1)[:3]
        if target_array.size != 3 or actual_array.size != 3:
            return False
        if not np.isfinite(target_array).all() or not np.isfinite(actual_array).all():
            return False
        error = np.abs(actual_array - target_array)
        if require_xyz:
            return bool(np.max(error) <= float(tolerance))
        return bool(np.linalg.norm(error[:2]) <= float(tolerance))

    def _record_motion_target(self, target, actual) -> None:
        self._last_target_position = np.asarray(target, dtype=np.float64).reshape(-1)[:3].copy()
        self._last_actual_position = (
            None
            if actual is None
            else np.asarray(actual, dtype=np.float64).reshape(-1)[:3].copy()
        )

    def _mark_motion_timeout(self, event: int) -> None:
        target = self._last_target_position
        actual = self._last_actual_position
        error = None if target is None or actual is None else actual - target
        self._last_failure_reason = (
            f"pick motion event {event} timed out; "
            f"target={None if target is None else target.tolist()}, "
            f"actual={None if actual is None else actual.tolist()}, "
            f"xyz_error={None if error is None else error.tolist()}"
        )
        self._failed = True
        print(f"[PickController] {self._last_failure_reason}", flush=True)

    def get_gripper_distance(self, item_name):
        """Determines the gripper opening distance for the specified object.

        Args:
            item_name (str): Name of the object to be gripped.

        Returns:
            float: Gripper finger distance in meters.
        """
        gripper_distances = {
            "rod": 0.003,
            "glass_rod": 0.002,
            "tube": 0.008,
            "beaker": 0.025,
            "beaker1": 0.03,
            "beaker_04": 0.025,
            "beaker_05": 0.025,
            "beaker_03": 0.025,
            "Erlenmeyer flask": 0.018,
            "Petri dish": 0.005,
            "pipette": 0.008,
            "microscope slide": 0.002,
            "conical_bottle01": 0.01,
            "conical_bottle02": 0.023,
            "conical_bottle03": 0.03,
            "conical_bottle04": 0.03,
            "graduated_cylinder_01": 0.005,
            "graduated_cylinder_02": 0.018,
            "graduated_cylinder_03": 0.024,
            "graduated_cylinder_04": 0.030,
            "nmr": 0.002,
            "nmr_01": 0.002,
            "nmr_02": 0.002,
        }

        for key in gripper_distances:
            if key == item_name.lower():
                return gripper_distances[key]

        return 0.02

    def get_pickz_offset(self, item_name):
        """Calculates the vertical offset for the final grasp position.

        Args:
            item_name (str): Name of the object to be picked.

        Returns:
            float: Vertical offset in meters.
        """
        offsets = {
            "conical_bottle02": 0.04,
            "conical_bottle03": 0.07,
            "conical_bottle04": 0.08,
            "beaker": 0.0,
            "beaker_04": 0.0,
            "beaker_05": 0.0,
            "beaker_03": 0.0,
            "beaker2": 0.0,
            "beaker_2": 0.0,
            "beaker_l": 0.02,
            "graduated_cylinder_01": 0.0,
            "graduated_cylinder_02": 0.0,
            "graduated_cylinder_03": 0.0,
            "graduated_cylinder_04": 0.0,
            "volume_flask": 0.05,
            "glass_rod": 0.02,
        }

        for key in offsets:
            if key == item_name.lower():
                return offsets[key]

        return self.object_size[2] * 2 / 5

    def get_pickprez_offset(self, item_name):
        """Calculates the vertical offset for the pre-grasp position.

        Args:
            item_name (str): Name of the object to be picked.

        Returns:
            float: Vertical offset in meters.
        """
        offsets = {
            "volume_flask": 0,
            "beaker2": 0.05,
            "conical_bottle03": 0.07,
            "conical_bottle04": 0.08,
            "graduated_cylinder_01": 0.05,
            "graduated_cylinder_02": 0.03,
            "graduated_cylinder_03": 0.03,
            "graduated_cylinder_04": 0.03
        }

        for key in offsets:
            if key == item_name.lower():
                return offsets[key]

        return self.object_size[2] * 2 / 3
