from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.stage import get_stage_units, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from isaacsim.robot.manipulators.grippers.gripper import Gripper

class PlaceController(BaseController):
    _MOTION_EVENTS = frozenset({0, 1, 4})

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        _position_threshold: float = 0.01,
        descent_position_threshold: float = 0.02,
        legacy_timed_progression: bool = False,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0

        self._events_dt = events_dt
        if events_dt is None:
            self._events_dt = [0.005, 0.01, 0.08, 0.05, 0.01, 0.1]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt must be numpy ")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) != 6:
                raise Exception("events dt 6")
        self._position_threshold = _position_threshold
        self._descent_position_threshold = float(descent_position_threshold)
        self._legacy_timed_progression = bool(legacy_timed_progression)
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._start = True
        self.target_position = None
        self._timed_out = False
        self._last_failure_reason = ""
        self._event_reached = False
        self._last_target_position = None
        self._last_actual_position = None
        return

    def forward(
        self,
        place_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_control,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        descent_end_effector_orientation: typing.Optional[np.ndarray] = None,
        gripper_position: typing.Optional[np.ndarray] = None,
        pre_place_z: float = 0.2,
        place_offset_z: float = 0.05,
    ) -> ArticulationAction:

        if self._start:
            self._start = False
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self._timed_out:
            return ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        if self.is_done():
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        descent_orientation = (
            end_effector_orientation
            if descent_end_effector_orientation is None
            else descent_end_effector_orientation
        )
        event_before = self._event
        self._event_reached = False
        if self._event == 0:
            self.target_position = place_position.copy()
            self.target_position[2] += pre_place_z / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=end_effector_orientation
            )
            self._record_motion_target(self.target_position, gripper_position)
            if self._position_reached(
                self.target_position,
                gripper_position,
                self._position_threshold,
            ):
                self._mark_event_reached()
        elif self._event == 1:
            self.target_position = place_position.copy()
            self.target_position[2] += place_offset_z / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=descent_orientation
            )
            self._record_motion_target(self.target_position, gripper_position)
            if self._position_reached(
                self.target_position,
                gripper_position,
                self._descent_position_threshold,
            ):
                self._mark_event_reached()
        elif self._event == 2:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            target_joint_positions = self._gripper.forward(action="open")
            self.target_position = place_position.copy()
            self.target_position[2] += 0.15 / get_stage_units()
            self.target_position[0] -= 0.15 / get_stage_units()
            gripper_control.release_object()
        elif self._event == 4:
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self.target_position,
                target_end_effector_orientation=descent_orientation
            )
            self._record_motion_target(self.target_position, gripper_position)
            if self._position_reached(
                self.target_position,
                gripper_position,
                self._position_threshold,
            ):
                self._mark_event_reached()
        else:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])

        if not self._event_reached and self._event == event_before and self._event < len(self._events_dt):
            self._t += self._events_dt[event_before]
            if self._t >= 1.0:
                if event_before in self._MOTION_EVENTS:
                    if self._legacy_timed_progression:
                        self._event += 1
                        self._t = 0
                    else:
                        self._mark_motion_timeout(event_before)
                else:
                    self._event += 1
                    self._t = 0

        return target_joint_positions

    def reset(
        self,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._start = True
        self.target_position = None
        self._timed_out = False
        self._last_failure_reason = ""
        self._event_reached = False
        self._last_target_position = None
        self._last_actual_position = None
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("event velocities  numpy ")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) != 6:
                raise Exception("events dt  6")
        return

    def is_done(self) -> bool:
        return self._timed_out or self._event >= len(self._events_dt)

    def has_failed(self) -> bool:
        return self._timed_out

    @property
    def last_failure_reason(self) -> str:
        return self._last_failure_reason

    @staticmethod
    def _position_reached(target, actual, tolerance: float) -> bool:
        if actual is None:
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
            f"place motion event {event} timed out; "
            f"target={None if target is None else target.tolist()}, "
            f"actual={None if actual is None else actual.tolist()}, "
            f"xyz_error={None if error is None else error.tolist()}"
        )
        self._timed_out = True
        self._event = len(self._events_dt)
        self._t = 0
        print(f"[PlaceController] {self._last_failure_reason}", flush=True)
