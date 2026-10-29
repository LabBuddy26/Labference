from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from controllers.atomic_actions.pick_controller import PickController
from controllers.atomic_actions.place_controller import PlaceController
from controllers.atomic_actions.pressZ_controller import PressZController
from controllers.base_controller import BaseController
from robots.franka.rmpflow_controller import RMPFlowController


class TaskPhase(Enum):
    """Condition-module phases, matching the main-branch controller."""

    PICKING1 = "picking1"
    PLACEING1 = "placing1"
    PRESS = "press"
    PICKING2 = "picking2"
    PLACEING2 = "placing2"
    FINISHED = "finished"


_PHASE_SEQUENCE = (
    TaskPhase.PICKING1,
    TaskPhase.PLACEING1,
    TaskPhase.PRESS,
    TaskPhase.PICKING2,
    TaskPhase.PLACEING2,
)


class ConditionModuleTaskController(BaseController):
    """Run main's dedicated pick/place/press sequence with safe validation.

    Motion timing, waypoints, orientations, and the use of separate atomic
    controllers intentionally follow ``main``. The state/config interface and
    collection helpers remain those of the evaluation merge branch.
    """

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.current_phase = TaskPhase.PICKING1
        self.phase_index = 0
        self.initial_beaker_position = None
        self.initial_button_position = None
        self.max_button_press_displacement = 0.0

    def _init_collect_mode(self, cfg, robot) -> None:
        super()._init_collect_mode(cfg, robot)

        # Keep main's controller topology and event timings. Both pick
        # controllers share one RMPFlow instance, while pressing has its own.
        pick_rmp_controller = RMPFlowController(
            name="condition_module_pick_controller",
            robot_articulation=robot,
        )
        self.pick_controller1 = PickController(
            name="condition_module_pick_controller_1",
            cspace_controller=pick_rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 1, 0.05, 0.01, 1],
        )
        self.place_controller1 = PlaceController(
            name="condition_module_place_controller_1",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
        )
        self.pick_controller2 = PickController(
            name="condition_module_pick_controller_2",
            cspace_controller=pick_rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.2, 0.05, 0.01, 0.1],
        )
        self.place_controller2 = PlaceController(
            name="condition_module_place_controller_2",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
        )
        self.press_controller = PressZController(
            name="condition_module_press_controller",
            cspace_controller=RMPFlowController(
                name="condition_module_press_rmpflow",
                robot_articulation=robot,
            ),
            events_dt=[0.004, 0.01, 0.01],
        )
        self.active_controller = self.pick_controller1

    def _init_infer_mode(self, cfg, robot) -> None:
        super()._init_infer_mode(cfg, robot)

    def reset(self) -> None:
        super().reset()
        self.current_phase = TaskPhase.PICKING1
        self.phase_index = 0
        self.initial_beaker_position = None
        self.initial_button_position = None
        self.max_button_press_displacement = 0.0
        if self.mode == "collect":
            self.pick_controller1.reset()
            self.place_controller1.reset()
            self.pick_controller2.reset()
            self.place_controller2.reset()
            self.press_controller.reset()
            self.active_controller = self.pick_controller1
        else:
            self.inference_engine.reset()

    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        if self.initial_beaker_position is None:
            initial = state.get("beaker_initial_position", state.get("object_position"))
            if initial is not None:
                self.initial_beaker_position = np.asarray(initial, dtype=np.float64).copy()
        if self.initial_button_position is None:
            initial = state.get("initial_button_position", state.get("button_position"))
            if initial is not None:
                self.initial_button_position = np.asarray(initial, dtype=np.float64).copy()
        if (
            self.current_phase == TaskPhase.PRESS
            and self.initial_button_position is not None
            and state.get("button_position") is not None
        ):
            axis = int(self.cfg.condition_module_success.button_press_axis)
            current_button_position = np.asarray(
                state["button_position"], dtype=np.float64
            )
            displacement = float(
                current_button_position[axis] - self.initial_button_position[axis]
            )
            self.max_button_press_displacement = max(
                self.max_button_press_displacement, displacement
            )
        if self.mode == "collect":
            return self._step_collect(state)
        return self._step_infer(state)

    def _step_collect(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        if self.current_phase == TaskPhase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if self.active_controller.is_done():
            failure_reason = self._atomic_failure_reason()
            if failure_reason:
                # PlaceController event 4 is only the post-release retreat.
                # main advances this event by time; the stricter merged
                # controller can time out when the device blocks that free-
                # space retreat target even though the beaker is already
                # correctly released. In that one case, trust the placed
                # object's geometry and continue. Motion failures before
                # release, or a misplaced object, still fail the episode.
                if (
                    self.current_phase in {TaskPhase.PLACEING1, TaskPhase.PLACEING2}
                    and failure_reason.startswith("place motion event 4 timed out")
                    and self._check_phase_success(state)
                ):
                    print(
                        "[ConditionModuleTaskController] accepting post-release "
                        f"retreat timeout because {self.current_phase.value} "
                        "already satisfies geometric success",
                        flush=True,
                    )
                    return self._advance(state)
                return self._fail(failure_reason)
            if not self._check_phase_success(state):
                if self.current_phase == TaskPhase.PRESS:
                    required = float(
                        self.cfg.condition_module_success.button_press_displacement_m
                    )
                    return self._fail(
                        "Phase press did not satisfy geometric success; "
                        f"max_button_displacement={self.max_button_press_displacement:.6f} m, "
                        f"required={required:.6f} m"
                    )
                return self._fail(
                    f"Phase {self.current_phase.value} did not satisfy geometric success"
                )
            return self._advance(state)

        action = self._get_phase_action(state)
        if action is not None:
            self.cache_collection_step(state, action)
        return action, False, False

    def _get_phase_action(self, state: Dict[str, Any]):
        if self.current_phase == TaskPhase.PICKING1:
            return self.pick_controller1.forward(
                picking_position=np.asarray(state["object_position"], dtype=np.float64).copy(),
                current_joint_positions=state["joint_positions"],
                object_size=np.asarray(state["object_size"], dtype=np.float64),
                object_name="beaker_03",
                gripper_control=self.gripper_control,
                gripper_position=state["gripper_position"],
                end_effector_orientation=R.from_euler(
                    "xyz", np.radians([0, 90, 10])
                ).as_quat(),
                pre_offset_x=0.1,
                pre_offset_z=0.1,
                after_offset_z=0,
            )
        if self.current_phase == TaskPhase.PLACEING1:
            return self.place_controller1.forward(
                place_position=np.asarray(state["target_position"], dtype=np.float64).copy(),
                current_joint_positions=state["joint_positions"],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler(
                    "xyz", np.radians([0, 90, 30])
                ).as_quat(),
                gripper_position=state["gripper_position"],
                pre_place_z=0.3,
                place_offset_z=0.02,
            )
        if self.current_phase == TaskPhase.PRESS:
            if state.get("button_position") is None:
                raise RuntimeError("ConditionModule state is missing button_position")
            # PressZ's final waypoint is 25 mm above the supplied mesh center.
            # Raise only this task's target so the wrist does not drive as far
            # into the button/device. Do not alter the shared atomic controller.
            button_target = np.asarray(
                state["button_position"], dtype=np.float64
            ).copy()
            button_target[2] += float(
                self.cfg.condition_module_success.button_press_target_offset_z_m
            )
            return self.press_controller.forward(
                target_position=button_target,
                current_joint_positions=state["joint_positions"],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler(
                    "xyz", np.radians([0, 90, 40])
                ).as_quat(),
                gripper_position=state["gripper_position"],
            )
        if self.current_phase == TaskPhase.PICKING2:
            return self.pick_controller2.forward(
                picking_position=np.asarray(state["object_position"], dtype=np.float64).copy(),
                current_joint_positions=state["joint_positions"],
                object_size=np.asarray(state["object_size"], dtype=np.float64),
                object_name="beaker_03",
                gripper_control=self.gripper_control,
                gripper_position=state["gripper_position"],
                end_effector_orientation=R.from_euler(
                    "xyz", np.radians([0, 90, 30])
                ).as_quat(),
                pre_offset_x=0.07,
                pre_offset_z=0.1,
                after_offset_z=0,
            )
        if self.current_phase == TaskPhase.PLACEING2:
            return self.place_controller2.forward(
                place_position=self.initial_beaker_position.copy(),
                current_joint_positions=state["joint_positions"],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler(
                    "xyz", np.radians([0, 90, 20])
                ).as_quat(),
                gripper_position=state["gripper_position"],
                pre_place_z=0.3,
                place_offset_z=0.02,
            )
        return None

    def _check_phase_success(self, state: Dict[str, Any]) -> bool:
        section = self.cfg.condition_module_success
        object_position = np.asarray(state["object_position"], dtype=np.float64)
        if self.current_phase in {TaskPhase.PICKING1, TaskPhase.PICKING2}:
            # main does not lift during PickController itself; PlaceController
            # performs the clearance move next. Validate that the fingers
            # finished closing at the live beaker pose before advancing.
            gripper_position = np.asarray(state["gripper_position"], dtype=np.float64)
            object_size = np.asarray(state["object_size"], dtype=np.float64)
            grasp_tolerance = max(0.05, 0.5 * float(np.linalg.norm(object_size)))
            finger_positions = np.asarray(state["joint_positions"], dtype=np.float64)[7:9]
            fingers_closed = finger_positions.size == 2 and np.max(finger_positions) <= 0.035
            return bool(
                np.linalg.norm(object_position - gripper_position) <= grasp_tolerance
                and fingers_closed
            )
        if self.current_phase == TaskPhase.PLACEING1:
            return self._near_target(object_position, state["target_position"], section)
        if self.current_phase == TaskPhase.PRESS:
            if self.initial_button_position is None:
                return False
            return bool(
                self.max_button_press_displacement
                >= float(section.button_press_displacement_m)
            )
        if self.current_phase == TaskPhase.PLACEING2:
            return self._near_target(object_position, self.initial_beaker_position, section)
        return False

    @staticmethod
    def _near_target(position, target, section) -> bool:
        target = np.asarray(target, dtype=np.float64)
        return bool(
            np.linalg.norm(position[:2] - target[:2])
            <= float(section.place_xy_tolerance_m)
            and abs(float(position[2] - target[2]))
            <= float(section.place_z_tolerance_m)
        )

    def _atomic_failure_reason(self) -> str:
        failed = getattr(self.active_controller, "has_failed", lambda: False)()
        if not failed:
            return ""
        return str(
            getattr(self.active_controller, "last_failure_reason", "motion timeout")
        )

    def _advance(self, state: Dict[str, Any]) -> Tuple[None, bool, bool]:
        if self.phase_index == len(_PHASE_SEQUENCE) - 1:
            self.write_cached_episode(state)
            self._last_success = True
            self.current_phase = TaskPhase.FINISHED
            self.reset_needed = True
            print("All phases completed, task successful!")
            return None, True, True

        self.phase_index += 1
        self.current_phase = _PHASE_SEQUENCE[self.phase_index]
        self.active_controller = self._controller_for_phase(self.current_phase)
        self.active_controller.reset()
        detail = ""
        if self.current_phase == TaskPhase.PRESS:
            self.max_button_press_displacement = 0.0
            detail = (
                ", press_target_offset_z="
                f"{float(self.cfg.condition_module_success.button_press_target_offset_z_m):.3f} m"
            )
        print(
            f"Switching to {self.current_phase.value} at sim_step="
            f"{state.get('sim_step', 'unknown')}{detail}...",
            flush=True,
        )
        return None, False, False

    def _controller_for_phase(self, phase: TaskPhase):
        return {
            TaskPhase.PICKING1: self.pick_controller1,
            TaskPhase.PLACEING1: self.place_controller1,
            TaskPhase.PRESS: self.press_controller,
            TaskPhase.PICKING2: self.pick_controller2,
            TaskPhase.PLACEING2: self.place_controller2,
        }[phase]

    def _fail(self, reason: str) -> Tuple[None, bool, bool]:
        self._last_failure_reason = reason
        self.data_collector.clear_cache()
        self._last_success = False
        self.current_phase = TaskPhase.FINISHED
        self.reset_needed = True
        print(f"ConditionModuleTaskController failed: {reason}")
        return None, True, False

    def _step_infer(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        state["language_instruction"] = self.get_language_instruction()
        return self.inference_engine.step_inference(state), False, False

    def get_language_instruction(self) -> Optional[str]:
        return {
            TaskPhase.PICKING1: "Pick up the beaker from the table",
            TaskPhase.PLACEING1: "Place the beaker on the condition module",
            TaskPhase.PRESS: "Press the condition module button",
            TaskPhase.PICKING2: "Pick up the beaker from the condition module",
            TaskPhase.PLACEING2: "Return the beaker to its original place",
        }.get(self.current_phase, "Operate the condition module with the beaker")

    def is_success(self) -> bool:
        return self._last_success
