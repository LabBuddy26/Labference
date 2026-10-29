"""A guarded, vertical insertion controller for placing a test tube in a rack."""

from enum import Enum, auto
import typing

import numpy as np

from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers.gripper import Gripper


class TubeInsertPhase(Enum):
    LIFT_CLEARANCE = auto()
    APPROACH = auto()
    FINE_ALIGN = auto()
    INSERT = auto()
    HOLD_BEFORE_RELEASE = auto()
    RELEASE = auto()
    RETREAT = auto()
    POST_RELEASE_SETTLE = auto()
    DONE = auto()


class TubeInsertController(BaseController):
    """Place a gripped vertical tube into one known rack socket.

    The controller works in object-centre coordinates, then converts every
    waypoint to the RMPFlow target frame using the measured object-to-gripper
    offset captured just after a successful grasp.  This avoids assuming that
    ``panda_hand/tool_center`` and the tube centre are coincident.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        insertion_cfg,
    ) -> None:
        super().__init__(name=name)
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._cfg = insertion_cfg
        self._phase = self._initial_phase()
        self._grasp_offset: typing.Optional[np.ndarray] = None
        self._transfer_start_center: typing.Optional[np.ndarray] = None
        self._lateral_command_bias = np.zeros(3, dtype=np.float64)
        self._inserted_distance = 0.0
        self._hold_counter = 0
        self._settle_counter = 0
        self._released = False
        self._phase_step = 0
        self._failed = False
        self._last_failure_reason = ""

    @property
    def object_released(self) -> bool:
        """Whether the tube has been released at its final insertion depth."""
        return self._released

    def has_failed(self) -> bool:
        return self._failed

    def _initial_phase(self) -> TubeInsertPhase:
        return (
            TubeInsertPhase.LIFT_CLEARANCE
            if bool(getattr(self._cfg, "lift_before_approach", False))
            else TubeInsertPhase.APPROACH
        )

    @property
    def last_failure_reason(self) -> str:
        return self._last_failure_reason

    def _phase_timeout_steps(self) -> int:
        configured = getattr(self._cfg, "phase_timeout_steps", {})
        phase_name = self._phase.name.lower()
        try:
            return int(configured.get(phase_name, 600))
        except AttributeError:
            return int(getattr(configured, phase_name, 600))

    @staticmethod
    def _format_vector(value: typing.Optional[np.ndarray]) -> str:
        if value is None:
            return "unavailable"
        return np.array2string(
            np.asarray(value, dtype=np.float64),
            precision=6,
            separator=", ",
            suppress_small=False,
        )

    def _transition(self, phase: TubeInsertPhase) -> None:
        self._phase = phase
        self._phase_step = 0

    def _wait_or_timeout(
        self,
        action: ArticulationAction,
        *,
        socket_position: np.ndarray,
        object_position: np.ndarray,
        target_position: typing.Optional[np.ndarray],
        gripper_position: np.ndarray,
        expected_tube_center: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        self._phase_step += 1
        timeout_steps = self._phase_timeout_steps()
        if self._phase_step < timeout_steps:
            return action

        self._mark_failed(
            f"tube insertion phase {self._phase.name.lower()} timed out after "
            f"{self._phase_step} steps",
            socket_position=socket_position,
            object_position=object_position,
            target_position=target_position,
            gripper_position=gripper_position,
            expected_tube_center=expected_tube_center,
        )
        return action

    def _mark_failed(
        self,
        reason: str,
        *,
        socket_position: np.ndarray,
        object_position: np.ndarray,
        target_position: typing.Optional[np.ndarray],
        gripper_position: np.ndarray,
        expected_tube_center: typing.Optional[np.ndarray] = None,
    ) -> None:
        xyz_error = None
        lateral_error = None
        axial_error = None
        if target_position is not None:
            xyz_error = np.asarray(target_position) - np.asarray(gripper_position)
            if hasattr(self, "_insertion_axis"):
                axial_signed = float(np.dot(xyz_error, self._insertion_axis))
                axial_error = abs(axial_signed)
                lateral_error = float(
                    np.linalg.norm(xyz_error - axial_signed * self._insertion_axis)
                )
        self._last_failure_reason = (
            f"{reason}; "
            f"socket={self._format_vector(socket_position)}, "
            f"tube_center={self._format_vector(object_position)}, "
            f"expected_tube_center={self._format_vector(expected_tube_center)}, "
            f"rmp_target={self._format_vector(target_position)}, "
            f"actual_end_effector={self._format_vector(gripper_position)}, "
            f"xyz_error={self._format_vector(xyz_error)}, "
            f"lateral_error={lateral_error if lateral_error is not None else 'unavailable'}, "
            f"axial_error={axial_error if axial_error is not None else 'unavailable'}"
        )
        self._failed = True
        self._transition(TubeInsertPhase.DONE)
        print(f"[TubeInsertController] {self._last_failure_reason}", flush=True)

    def _stage_distance(self, value: float) -> float:
        return float(value) / get_stage_units()

    def _motion_action(
        self,
        target_position: np.ndarray,
        end_effector_orientation: np.ndarray,
    ) -> ArticulationAction:
        return self._cspace_controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=end_effector_orientation,
        )

    @staticmethod
    def _xy_error(target: np.ndarray, current: np.ndarray) -> float:
        return float(np.linalg.norm(target[:2] - current[:2]))

    @staticmethod
    def _xyz_error(target: np.ndarray, current: np.ndarray) -> float:
        return float(np.linalg.norm(target - current))

    def _gripper_target_for_object_center(self, object_center: np.ndarray) -> np.ndarray:
        # ``object_center - gripper_center`` remains approximately constant
        # while the fingers stay closed, and was measured after the scripted
        # pick rather than inferred from an asset-specific hand offset.
        return object_center - self._grasp_offset

    def _axis_distance(self, value: float) -> np.ndarray:
        return self._insertion_axis * self._stage_distance(value)

    def _insertion_target_reached(
        self,
        target_position: np.ndarray,
        gripper_position: np.ndarray,
    ) -> bool:
        error = np.asarray(target_position) - np.asarray(gripper_position)
        axial_signed = float(np.dot(error, self._insertion_axis))
        axial_error = abs(axial_signed)
        lateral_error = float(
            np.linalg.norm(error - axial_signed * self._insertion_axis)
        )
        lateral_tolerance = self._stage_distance(
            getattr(
                self._cfg,
                "insertion_lateral_tolerance",
                self._cfg.insertion_tracking_tolerance,
            )
        )
        axial_tolerance = self._stage_distance(
            getattr(
                self._cfg,
                "insertion_axial_tolerance",
                self._cfg.insertion_tracking_tolerance,
            )
        )
        return lateral_error < lateral_tolerance and axial_error < axial_tolerance

    def _object_target_reached(
        self,
        target_center: np.ndarray,
        object_position: np.ndarray,
    ) -> bool:
        """Require the tube itself, rather than only the hand, to be on path."""
        error = np.asarray(target_center) - np.asarray(object_position)
        axial_signed = float(np.dot(error, self._insertion_axis))
        axial_error = abs(axial_signed)
        lateral_error = float(
            np.linalg.norm(error - axial_signed * self._insertion_axis)
        )
        lateral_tolerance = self._stage_distance(
            getattr(self._cfg, "object_lateral_tolerance", 0.0015)
        )
        axial_tolerance = self._stage_distance(
            getattr(self._cfg, "object_axial_tolerance", 0.006)
        )
        return lateral_error < lateral_tolerance and axial_error < axial_tolerance

    def _object_inserted_at_or_beyond(
        self,
        target_center: np.ndarray,
        object_position: np.ndarray,
    ) -> bool:
        """Accept a laterally aligned object that reached or passed the target depth."""
        error = np.asarray(target_center) - np.asarray(object_position)
        axial_signed = float(np.dot(error, self._insertion_axis))
        lateral_error = float(
            np.linalg.norm(error - axial_signed * self._insertion_axis)
        )
        lateral_tolerance = self._stage_distance(
            getattr(self._cfg, "object_lateral_tolerance", 0.0015)
        )
        axial_tolerance = self._stage_distance(
            getattr(self._cfg, "object_axial_tolerance", 0.006)
        )
        # Positive axial_signed means the object lies deeper than the target
        # along the downward insertion direction. Only reject objects still
        # above the target by more than the configured tolerance.
        return lateral_error < lateral_tolerance and axial_signed >= -axial_tolerance

    def _lateral_object_feedback(
        self,
        target_center: np.ndarray,
        object_position: np.ndarray,
    ) -> np.ndarray:
        """Return a bounded lateral correction for RMPFlow steady-state error."""
        error = np.asarray(target_center) - np.asarray(object_position)
        axial_error = float(np.dot(error, self._insertion_axis))
        lateral_error = error - axial_error * self._insertion_axis
        error_norm = float(np.linalg.norm(lateral_error))
        max_correction = self._stage_distance(
            getattr(self._cfg, "fine_alignment_max_correction", 0.004)
        )
        if error_norm > max_correction and error_norm > 1e-12:
            lateral_error *= max_correction / error_norm
        return lateral_error

    def _update_lateral_command_bias(
        self,
        target_center: np.ndarray,
        object_position: np.ndarray,
    ) -> np.ndarray:
        """Learn and retain the Cartesian bias needed to centre the tube."""
        feedback = self._lateral_object_feedback(target_center, object_position)
        gain = float(getattr(self._cfg, "fine_alignment_feedback_gain", 0.2))
        self._lateral_command_bias += gain * feedback
        max_bias = self._stage_distance(
            getattr(self._cfg, "fine_alignment_max_correction", 0.004)
        )
        bias_norm = float(np.linalg.norm(self._lateral_command_bias))
        if bias_norm > max_bias and bias_norm > 1e-12:
            self._lateral_command_bias *= max_bias / bias_norm
        return self._lateral_command_bias.copy()

    def forward(
        self,
        socket_position: np.ndarray,
        inserted_tube_center: np.ndarray,
        insertion_axis: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        object_position: np.ndarray,
        end_effector_orientation: np.ndarray,
    ) -> ArticulationAction:
        """Advance one phase of the insert/release/retract sequence."""
        if self.is_done():
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        if gripper_position is None or object_position is None:
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        self._insertion_axis = np.asarray(insertion_axis, dtype=np.float64)
        axis_norm = np.linalg.norm(self._insertion_axis)
        if axis_norm < 1e-8:
            raise ValueError("Tube insertion axis must be non-zero")
        self._insertion_axis /= axis_norm

        socket_position = np.asarray(socket_position, dtype=np.float64)
        inserted_tube_center = np.asarray(inserted_tube_center, dtype=np.float64)
        gripper_position = np.asarray(gripper_position, dtype=np.float64)
        object_position = np.asarray(object_position, dtype=np.float64)
        if self._grasp_offset is None:
            self._grasp_offset = object_position - gripper_position
            self._transfer_start_center = object_position.copy()
        elif self._phase.value < TubeInsertPhase.RELEASE.value:
            expected_tube_center = gripper_position + self._grasp_offset
            grasp_tracking_error = float(
                np.linalg.norm(object_position - expected_tube_center)
            )
            grasp_tracking_tolerance = self._stage_distance(
                getattr(self._cfg, "grasp_tracking_tolerance", 0.025)
            )
            if grasp_tracking_error > grasp_tracking_tolerance:
                self._mark_failed(
                    f"tube lost from gripper during {self._phase.name.lower()} "
                    f"(tracking_error={grasp_tracking_error:.6f} m, "
                    f"tolerance={grasp_tracking_tolerance:.6f} m)",
                    socket_position=socket_position,
                    object_position=object_position,
                    target_position=None,
                    gripper_position=gripper_position,
                    expected_tube_center=expected_tube_center,
                )
                return ArticulationAction(
                    joint_positions=[None] * current_joint_positions.shape[0]
                )

            # Compensate only small in-finger motion.  Updating the offset after
            # a large rack contact would make a displaced tube look correctly
            # grasped and let the controller continue loading the collision.
            offset_update_limit = self._stage_distance(
                getattr(self._cfg, "grasp_offset_update_limit", 0.002)
            )
            if grasp_tracking_error <= offset_update_limit:
                self._grasp_offset = object_position - gripper_position

        approach_center = inserted_tube_center + self._axis_distance(self._cfg.approach_height)
        align_center = inserted_tube_center + self._axis_distance(self._cfg.alignment_height)
        alignment_tolerance = self._stage_distance(self._cfg.xy_alignment_tolerance)

        # For long objects, never use one diagonal source-to-rack trajectory.
        # First lift vertically at the source, then translate horizontally at
        # the higher of the current and configured approach heights.
        start_center = (
            object_position
            if self._transfer_start_center is None
            else self._transfer_start_center
        )
        start_axial = float(np.dot(start_center, self._insertion_axis))
        approach_axial = float(np.dot(approach_center, self._insertion_axis))
        travel_axial = max(start_axial, approach_axial)
        lift_center = start_center + self._insertion_axis * (
            travel_axial - start_axial
        )
        travel_center = approach_center + self._insertion_axis * (
            travel_axial - approach_axial
        )

        if self._phase == TubeInsertPhase.LIFT_CLEARANCE:
            target = self._gripper_target_for_object_center(lift_center)
            action = self._motion_action(target, end_effector_orientation)
            if self._xyz_error(target, gripper_position) < 0.01 / get_stage_units():
                self._transition(TubeInsertPhase.APPROACH)
                return action
            return self._wait_or_timeout(
                action,
                socket_position=socket_position,
                object_position=object_position,
                target_position=target,
                gripper_position=gripper_position,
                expected_tube_center=lift_center,
            )

        if self._phase == TubeInsertPhase.APPROACH:
            target = self._gripper_target_for_object_center(travel_center)
            action = self._motion_action(target, end_effector_orientation)
            if self._xyz_error(target, gripper_position) < 0.01 / get_stage_units():
                self._transition(TubeInsertPhase.FINE_ALIGN)
                return action
            return self._wait_or_timeout(
                action,
                socket_position=socket_position,
                object_position=object_position,
                target_position=target,
                gripper_position=gripper_position,
                expected_tube_center=travel_center,
            )

        if self._phase == TubeInsertPhase.FINE_ALIGN:
            target = self._gripper_target_for_object_center(align_center)
            # RMPFlow can settle a few millimetres short of its Cartesian
            # command. At this collision-free height, feed the measured tube
            # error back into the command so the tube—not merely the hand
            # target—becomes coaxial with the selected socket.
            target += self._update_lateral_command_bias(
                align_center, object_position
            )
            action = self._motion_action(target, end_effector_orientation)
            fine_alignment_z_tolerance = self._stage_distance(
                getattr(
                    self._cfg,
                    "fine_alignment_z_tolerance",
                    self._cfg.insertion_tracking_tolerance,
                )
            )
            if (
                self._xy_error(target, gripper_position) < alignment_tolerance
                and abs(target[2] - gripper_position[2]) < fine_alignment_z_tolerance
                and self._object_target_reached(align_center, object_position)
            ):
                self._transition(TubeInsertPhase.INSERT)
                return action
            return self._wait_or_timeout(
                action,
                socket_position=socket_position,
                object_position=object_position,
                target_position=target,
                gripper_position=gripper_position,
                expected_tube_center=align_center,
            )

        if self._phase == TubeInsertPhase.INSERT:
            # _inserted_distance is already expressed in stage units.
            current_center = align_center - self._insertion_axis * self._inserted_distance
            # A vial can settle a few millimetres in the fingers when it first
            # contacts the guide plate.  For tasks that explicitly enable
            # recovery, feed that measured object error back into the lateral
            # command while keeping the object-centre gate strict.  This
            # recovers a transient grasp offset without accepting an object
            # that remains outside the selected socket.
            if not self._object_target_reached(current_center, object_position) and bool(
                getattr(self._cfg, "insertion_feedback_recovery", False)
            ):
                self._update_lateral_command_bias(current_center, object_position)
            target = self._gripper_target_for_object_center(current_center)
            # Preserve the bias learned above the rack. Dropping it here would
            # immediately restore RMPFlow's lateral residual and stop descent.
            target += self._lateral_command_bias
            action = self._motion_action(target, end_effector_orientation)
            # Do not advance because the hand reached its command while the
            # tube is caught on a socket edge.  The object-centre gate keeps
            # every incremental descent coaxial and turns a jam into a phase
            # timeout instead of a spring-loaded launch.
            if (
                self._insertion_target_reached(target, gripper_position)
                and self._object_target_reached(current_center, object_position)
            ):
                insertion_distance = self._stage_distance(self._cfg.insertion_step)
                total_distance = max(self._stage_distance(self._cfg.alignment_height), 1e-9)
                progress = min(max(self._inserted_distance / total_distance, 0.0), 1.0)
                # Move briskly while the tube is clear of the guide, then
                # linearly reduce the step as it approaches the final seat.
                # This avoids both collision-prone jumps and the old fixed
                # micro-step that could never finish before the timeout.
                min_factor = float(getattr(self._cfg, "insertion_min_step_factor", 0.25))
                min_factor = min(max(min_factor, 0.05), 1.0)
                insertion_distance *= 1.0 - (1.0 - min_factor) * progress
                self._inserted_distance = min(
                    self._inserted_distance + insertion_distance,
                    total_distance,
                )
                if self._inserted_distance >= self._stage_distance(self._cfg.alignment_height):
                    self._transition(TubeInsertPhase.HOLD_BEFORE_RELEASE)
                    return action
            return self._wait_or_timeout(
                action,
                socket_position=socket_position,
                object_position=object_position,
                target_position=target,
                gripper_position=gripper_position,
                expected_tube_center=current_center,
            )

        if self._phase == TubeInsertPhase.HOLD_BEFORE_RELEASE:
            target = self._gripper_target_for_object_center(inserted_tube_center)
            target += self._lateral_command_bias
            action = self._motion_action(target, end_effector_orientation)
            object_reached = self._object_target_reached(inserted_tube_center, object_position)
            accept_deeper = bool(getattr(self._cfg, "accept_deeper_insertion", False))
            object_inserted = object_reached or (
                accept_deeper
                and self._object_inserted_at_or_beyond(inserted_tube_center, object_position)
            )
            # Once the object is physically seated at or below the requested
            # depth, contact with the rack can keep the hand several
            # millimetres from its free-space RMP target. In that case the
            # object geometry is authoritative and it is safe to release.
            hand_reached = self._insertion_target_reached(target, gripper_position)
            if object_inserted and (hand_reached or accept_deeper):
                self._hold_counter += 1
            else:
                self._hold_counter = 0
            if self._hold_counter >= int(self._cfg.pre_release_hold_frames):
                self._transition(TubeInsertPhase.RELEASE)
                return action
            return self._wait_or_timeout(
                action,
                socket_position=socket_position,
                object_position=object_position,
                target_position=target,
                gripper_position=gripper_position,
                expected_tube_center=inserted_tube_center,
            )

        if self._phase == TubeInsertPhase.RELEASE:
            # Do not start lifting in the same frame as the release command.
            # This gives the tube one physics step to transfer its support to
            # the rack instead of being pulled upward by the fingers.
            self._released = True
            self._transition(TubeInsertPhase.RETREAT)
            return self._gripper.forward(action="open")

        retreat_center = inserted_tube_center + self._axis_distance(self._cfg.retreat_height)
        retreat_target = self._gripper_target_for_object_center(retreat_center)
        if self._phase == TubeInsertPhase.RETREAT:
            action = self._motion_action(retreat_target, end_effector_orientation)
            if self._xyz_error(retreat_target, gripper_position) < 0.015 / get_stage_units():
                self._transition(TubeInsertPhase.POST_RELEASE_SETTLE)
                return action
            return self._wait_or_timeout(
                action,
                socket_position=socket_position,
                object_position=object_position,
                target_position=retreat_target,
                gripper_position=gripper_position,
            )

        if self._phase == TubeInsertPhase.POST_RELEASE_SETTLE:
            action = self._motion_action(retreat_target, end_effector_orientation)
            self._settle_counter += 1
            if self._settle_counter >= int(self._cfg.post_release_settle_frames):
                self._transition(TubeInsertPhase.DONE)
                return action
            return self._wait_or_timeout(
                action,
                socket_position=socket_position,
                object_position=object_position,
                target_position=retreat_target,
                gripper_position=gripper_position,
            )

        return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

    def reset(self) -> None:
        super().reset()
        self._cspace_controller.reset()
        self._phase = self._initial_phase()
        self._grasp_offset = None
        self._transfer_start_center = None
        self._lateral_command_bias = np.zeros(3, dtype=np.float64)
        self._inserted_distance = 0.0
        self._hold_counter = 0
        self._settle_counter = 0
        self._released = False
        self._phase_step = 0
        self._failed = False
        self._last_failure_reason = ""

    def is_done(self) -> bool:
        return self._phase == TubeInsertPhase.DONE
