from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing


class PressZController(BaseController):
    """Z-axis press state machine: approach above the target, press down onto
    it, hold the press, then retract.

    Rewritten from the LabBench version which closed the gripper in place at a
    raised height (never pressing down) and mutated the caller's
    ``target_position`` array in place on every frame.

    Args:
        name (str): Identifier for the controller.
        cspace_controller (BaseController): Cartesian space controller that
            returns joint position arrays.
        initial_offset (float, optional): Kept for API compatibility.
        events_dt (list of float, optional): Per-event time budget, defaults to
            [0.01, 0.01, 0.02, 0.01].
        approach_height (float, optional): Height above the target to approach
            before pressing, in stage units (default 0.10 m).
        press_offset (float, optional): How far below the target center to
            descend when pressing, in stage units (default 0.005 m).
    """

    _MOTION_EVENTS = frozenset({0, 1, 3})

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        initial_offset: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        approach_height: typing.Optional[float] = None,
        press_offset: typing.Optional[float] = None,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._approach_height = (
            approach_height if approach_height is not None else 0.10 / get_stage_units()
        )
        self._press_offset = (
            press_offset if press_offset is not None else 0.005 / get_stage_units()
        )
        self._initial_offset = (
            initial_offset if initial_offset is not None else 0.10 / get_stage_units()
        )

        if events_dt is None:
            self._events_dt = [0.01, 0.01, 0.02, 0.01]
        else:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt should be a NumPy array or list")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 4:
                raise Exception("events_dt should have 4 elements")

        self._cspace_controller = cspace_controller
        self._start = True
        self._position_threshold = 0.01 / get_stage_units()
        self._approach_point = None
        self._timed_out = False
        self._last_failure_reason = ""
        self._event_reached = False
        self._last_target_position = None
        self._last_actual_position = None

    def get_current_event(self) -> int:
        return self._event

    def forward(
        self,
        target_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        gripper_position: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Execute one step of the pressing action."""

        if self._start:
            # Initial state: open the gripper before approaching.
            self._start = False
            target_joint_positions = [None] * current_joint_positions.shape[0]
            target_joint_positions[7] = 0.04 / get_stage_units()
            target_joint_positions[8] = 0.04 / get_stage_units()
            return ArticulationAction(joint_positions=target_joint_positions)

        if self._timed_out:
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        if self.is_done():
            # Done: maintain current joint positions.
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))

        # Work on a private copy: callers pass the live button position from
        # task state, and the LabBench implementation accumulated Z offsets in
        # that array on every frame.
        target = np.asarray(target_position, dtype=np.float64).copy()

        event_before = self._event
        self._event_reached = False
        target_joint_positions = self._execute_phase(
            target, end_effector_orientation, current_joint_positions, gripper_position
        )
        self._update_state(event_before)
        return target_joint_positions

    def _execute_phase(self, target, end_effector_orientation, current_joint_positions, gripper_position):
        if self._event == 0:
            # Approach: move to the point above the target.
            target[2] += self._approach_height
            self._approach_point = target.copy()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=end_effector_orientation,
            )
            self._record_motion_target(self._approach_point, gripper_position)
            if self._position_reached(
                self._approach_point, gripper_position, self._position_threshold
            ):
                self._mark_event_reached()
            return target_joint_positions

        elif self._event == 1:
            # Press down: descend onto the target and close the gripper.
            target[2] -= self._press_offset
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=end_effector_orientation,
            )
            self._record_motion_target(target, gripper_position)
            if self._position_reached(
                target, gripper_position, self._position_threshold
            ):
                self._mark_event_reached()
            return target_joint_positions

        elif self._event == 2:
            # Hold: keep the gripper closed at the pressed position.
            target_joint_positions = [None] * current_joint_positions.shape[0]
            gripper_distance = 0.0015 / get_stage_units()
            target_joint_positions[7] = gripper_distance
            target_joint_positions[8] = gripper_distance
            return ArticulationAction(joint_positions=target_joint_positions)

        else:
            # Retract: rise back to the approach height and open the gripper.
            if self._approach_point is not None:
                target = self._approach_point.copy()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=end_effector_orientation,
            )
            self._record_motion_target(target, gripper_position)
            target_joint_positions[7] = 0.04 / get_stage_units()
            target_joint_positions[8] = 0.04 / get_stage_units()
            if self._position_reached(target, gripper_position, self._position_threshold):
                self._mark_event_reached()
            return ArticulationAction(joint_positions=target_joint_positions)

    def _update_state(self, event_before: int):
        if self._event_reached or self._event != event_before:
            return
        self._t += self._events_dt[event_before]
        if self._t >= 1.0:
            if event_before in self._MOTION_EVENTS:
                self._mark_motion_timeout(event_before)
            else:
                self._event += 1
                self._t = 0

    def reset(
        self,
        initial_offset: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._approach_point = None
        self._timed_out = False
        self._last_failure_reason = ""
        self._event_reached = False
        self._last_target_position = None
        self._last_actual_position = None
        if initial_offset is not None:
            self._initial_offset = initial_offset
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt should be a NumPy array or list")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 4:
                raise Exception("events_dt should have 4 elements")
        self._start = True

    def is_done(self) -> bool:
        return self._timed_out or self._event >= len(self._events_dt)

    def has_failed(self) -> bool:
        return self._timed_out

    @property
    def last_failure_reason(self) -> str:
        return self._last_failure_reason

    @staticmethod
    def _position_reached(target, actual, tolerance: float) -> bool:
        if target is None or actual is None:
            return False
        target_array = np.asarray(target, dtype=np.float64).reshape(-1)[:3]
        actual_array = np.asarray(actual, dtype=np.float64).reshape(-1)[:3]
        return bool(
            target_array.size == 3
            and actual_array.size == 3
            and np.isfinite(target_array).all()
            and np.isfinite(actual_array).all()
            and np.max(np.abs(target_array - actual_array)) <= float(tolerance)
        )

    def _record_motion_target(self, target, actual) -> None:
        self._last_target_position = np.asarray(target, dtype=np.float64).reshape(-1)[:3].copy()
        self._last_actual_position = (
            None
            if actual is None
            else np.asarray(actual, dtype=np.float64).reshape(-1)[:3].copy()
        )

    def _mark_event_reached(self) -> None:
        self._event += 1
        self._t = 0
        self._event_reached = True

    def _mark_motion_timeout(self, event: int) -> None:
        target = self._last_target_position
        actual = self._last_actual_position
        error = None if target is None or actual is None else actual - target
        self._last_failure_reason = (
            f"press motion event {event} timed out; "
            f"target={None if target is None else target.tolist()}, "
            f"actual={None if actual is None else actual.tolist()}, "
            f"xyz_error={None if error is None else error.tolist()}"
        )
        self._timed_out = True
        self._event = len(self._events_dt)
        self._t = 0
        print(f"[PressZController] {self._last_failure_reason}", flush=True)
