from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from .atomic_actions.pickr_controller import PickRController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.tube_insert_controller import TubeInsertController
from .atomic_actions.pour_controller import PourController
from .atomic_actions.pressz_controller import PressZController
from .base_controller import BaseController
from utils.rotation_utils import isaac_quaternion_from_euler_deg


@dataclass(frozen=True)
class PhaseSpec:
    """Declarative description of one phase in a multi-object task."""

    name: str
    kind: str  # "pick" | "place" | "insert" | "press" | "pour"
    object_key: str = ""
    size_key: str = "object_size"
    name_key: str = "object_name"
    target_key: Optional[str] = None
    inserted_center_key: Optional[str] = None
    socket_key: Optional[str] = None
    pre_release_offset_z: float = 0.0
    quat_key: Optional[str] = None
    initial_quat_key: Optional[str] = None
    orientation_deg: Tuple[float, float, float] = (0.0, 180.0, 0.0)
    pre_offset_x: float = 0.03
    pre_offset_z: float = 0.10
    after_offset_z: float = 0.30
    pre_place_z: float = 0.30
    place_offset_z: float = 0.05
    gripper_distance: Optional[float] = None
    pick_z_offset: Optional[float] = None
    pick_z_offset_delta: float = 0.0
    pick_position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Keep the side approach offset during the pre-grasp descent (event 1),
    # like PickController: the arm stays on the robot side instead of moving
    # directly over the object, keeping the elbow high.
    apply_approach_offset: bool = False
    place_position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class MultiObjectTaskController(BaseController):
    """Reusable multi-object phase machine for scripted collection.

    Each phase is declared as a :class:`PhaseSpec` in ``PHASE_SPECS``.  The
    machine picks and places objects one at a time, advancing only when the
    active atomic controller is done AND the geometric success check passes,
    and failing otherwise.  This structurally avoids the LabBench bugs where a
    finished-but-failed phase deadlocked forever, missing ``return`` statements
    skipped whole phases, and ``is_success`` referenced never-assigned state.

    Eval mode is fully policy driven: ``_step_infer`` delegates to the
    inference engine and ignores the phase machine.
    """

    PHASE_SPECS: Tuple[PhaseSpec, ...] = ()
    SUCCESS_CFG_SECTION: str = ""
    LANGUAGE_INSTRUCTION: str = ""

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.phase_index = 0
        self._initial_positions: Dict[str, np.ndarray] = {}
        self.state = None

    def _success_cfg(self):
        return getattr(self.cfg, self.SUCCESS_CFG_SECTION)

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        self.pick_controller = PickRController(
            name=f"{self.__class__.__name__}_pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.01, 0.004, 0.01],
            grasp_position_threshold=float(
                getattr(
                    getattr(cfg, self.SUCCESS_CFG_SECTION),
                    "grasp_position_tolerance_m",
                    0.005,
                )
            ),
            legacy_timed_progression=self._main_compatible_collection,
        )
        self.place_controller = PlaceController(
            name=f"{self.__class__.__name__}_place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            legacy_timed_progression=self._main_compatible_collection,
        )
        self.insert_controller = None
        if any(spec.kind == "insert" for spec in self.PHASE_SPECS):
            self.insert_controller = TubeInsertController(
                name=f"{self.__class__.__name__}_insert_controller",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
                insertion_cfg=cfg.guarded_insertion,
            )
        self.press_controller = None
        if any(spec.kind == "press" for spec in self.PHASE_SPECS):
            self.press_controller = PressZController(
                name=f"{self.__class__.__name__}_press_controller",
                cspace_controller=self.rmp_controller,
            )
        self.pour_controller = None
        if any(spec.kind == "pour" for spec in self.PHASE_SPECS):
            self.pour_controller = PourController(
                name=f"{self.__class__.__name__}_pour_controller",
                cspace_controller=self.rmp_controller,
                events_dt=[0.006, 0.002, 0.009, 0.01, 0.009, 0.01],
            )
        self.active_controller = self.pick_controller

    def _init_infer_mode(self, cfg, robot):
        super()._init_infer_mode(cfg, robot)

    def reset(self) -> None:
        super().reset()
        self.phase_index = 0
        self._initial_positions = {}
        self.state = None
        if self.mode == "collect":
            self.pick_controller.reset()
            self.place_controller.reset()
            if self.insert_controller is not None:
                self.insert_controller.reset()
            if self.press_controller is not None:
                self.press_controller.reset()
            if self.pour_controller is not None:
                self.pour_controller.reset()
            self.active_controller = self.pick_controller

    def _controller_for(self, spec: PhaseSpec):
        if spec.kind == "pick":
            return self.pick_controller
        if spec.kind == "place":
            return self.place_controller
        if spec.kind == "insert":
            return self.insert_controller
        if spec.kind == "press":
            return self.press_controller
        if spec.kind == "pour":
            return self.pour_controller
        raise ValueError(f"Unknown phase kind: {spec.kind}")

    def _check_phase_success(self, state, spec: PhaseSpec) -> bool:
        if spec.kind == "pick":
            initial = self._initial_positions.get(spec.name)
            if initial is None or spec.object_key not in state:
                return False
            position = np.asarray(state[spec.object_key], dtype=np.float64)
            section = self._success_cfg()
            return bool(position[2] >= initial[2] + float(section.pick_lift_height_m))
        if spec.kind in ("place", "insert"):
            initial = self._initial_positions.get(spec.name)
            target = np.asarray(state[spec.target_key], dtype=np.float64) if spec.target_key else None
            if initial is None or target is None or spec.object_key not in state:
                return False
            position = np.asarray(state[spec.object_key], dtype=np.float64)
            section = self._success_cfg()
            return bool(
                np.linalg.norm(position[:2] - target[:2]) <= float(section.place_xy_tolerance_m)
                and abs(float(position[2] - initial[2])) <= float(section.place_z_tolerance_m)
            )
        if spec.kind == "press":
            # Button displacement measured against the position captured at
            # episode start (see _capture_initial_positions).
            initial = self._initial_positions.get(spec.name)
            if initial is None or spec.target_key not in state:
                return False
            current = np.asarray(state[spec.target_key], dtype=np.float64)
            section = self._success_cfg()
            displacement = float(
                np.linalg.norm(current - initial[: len(current)])
                if current.shape == initial.shape
                else abs(current[2] - initial[2])
            )
            return bool(displacement >= float(getattr(section, "button_press_displacement_m", 0.005)))
        return False

    def _capture_initial_positions(self, state):
        for spec in self.PHASE_SPECS:
            if spec.object_key and spec.object_key in state:
                self._initial_positions[spec.name] = np.asarray(
                    state[spec.object_key], dtype=np.float64
                ).copy()
            elif spec.kind == "press" and spec.target_key and spec.target_key in state:
                self._initial_positions[spec.name] = np.asarray(
                    state[spec.target_key], dtype=np.float64
                ).copy()

    def step(self, state: Dict[str, Any]):
        self.state = state
        if not self._initial_positions:
            self._capture_initial_positions(state)
        if self.mode == "collect":
            return self._step_collect(state)
        return self._step_infer(state)

    def _step_collect(self, state: Dict[str, Any]):
        if self.phase_index >= len(self.PHASE_SPECS):
            return None, True, self._last_success
        spec = self.PHASE_SPECS[self.phase_index]
        if self.active_controller.is_done():
            self._on_phase_completion(state, spec)
            if getattr(self.active_controller, "has_failed", lambda: False)():
                detail = getattr(
                    self.active_controller, "last_failure_reason", "motion timeout"
                )
                if self._accept_atomic_failure(state, spec, detail):
                    return self._advance(state)
                return self._fail(
                    state,
                    spec,
                    detail,
                )
            if self._check_phase_success(state, spec):
                return self._advance(state)
            return self._fail(state, spec)

        action = self._action_for_phase(state, spec)
        if action is not None:
            self.cache_collection_step(state, action)
        return action, False, False

    def _on_phase_completion(self, state, spec: PhaseSpec) -> None:
        """Hook for task-specific diagnostics before completion is judged."""

    def _accept_atomic_failure(
        self, state, spec: PhaseSpec, detail: str
    ) -> bool:
        """Allow a task to accept a non-critical atomic-controller failure."""
        return False

    def _action_for_phase(self, state, spec: PhaseSpec):
        if spec.kind == "pick":
            # The object can move once a grasp/attachment is established.  A
            # pick waypoint must remain anchored to the episode-start object
            # centre; recomputing it from the carried object makes the arm
            # chase its own target and can time out (notably in R10 recall).
            picking_position = np.asarray(
                self._initial_positions.get(spec.name, state[spec.object_key]),
                dtype=np.float64,
            ).copy()
            picking_position += np.asarray(
                spec.pick_position_offset, dtype=np.float64
            )
            return self.pick_controller.forward(
                picking_position=picking_position,
                current_joint_positions=state["joint_positions"],
                object_size=np.asarray(state[spec.size_key], dtype=np.float64),
                object_name=str(state.get(spec.name_key, spec.name)),
                gripper_control=self.gripper_control,
                gripper_position=state["gripper_position"],
                end_effector_orientation=isaac_quaternion_from_euler_deg(
                    spec.orientation_deg
                ),
                pre_offset_x=spec.pre_offset_x,
                pre_offset_z=spec.pre_offset_z,
                after_offset_z=spec.after_offset_z,
                gripper_distances=spec.gripper_distance,
                pick_z_offset=spec.pick_z_offset,
                pick_z_offset_delta=spec.pick_z_offset_delta,
                apply_approach_offset=spec.apply_approach_offset,
            )
        if spec.kind == "place":
            place_position = np.asarray(
                state[spec.target_key], dtype=np.float64
            ).copy()
            place_position += np.asarray(
                spec.place_position_offset, dtype=np.float64
            )
            return self.place_controller.forward(
                place_position=place_position,
                current_joint_positions=state["joint_positions"],
                gripper_control=self.gripper_control,
                end_effector_orientation=isaac_quaternion_from_euler_deg(
                    spec.orientation_deg
                ),
                gripper_position=state["gripper_position"],
                pre_place_z=spec.pre_place_z,
                place_offset_z=spec.place_offset_z,
            )
        if spec.kind == "insert":
            initial = self._initial_positions.get(spec.name)
            if initial is None:
                return None
            # Use a physical object-centre target, separate from any legacy
            # end-effector Z compensation stored in ``target_key``.
            center_key = spec.inserted_center_key or spec.target_key
            target_center = np.asarray(
                state[center_key], dtype=np.float64
            ).copy()
            pre_release_center = target_center.copy()
            pre_release_center[2] += float(spec.pre_release_offset_z)
            socket_position = (
                np.asarray(state[spec.socket_key], dtype=np.float64).copy()
                if spec.socket_key
                else target_center
            )
            return self.insert_controller.forward(
                socket_position=socket_position,
                inserted_tube_center=pre_release_center,
                insertion_axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                current_joint_positions=state["joint_positions"],
                gripper_position=state["gripper_position"],
                object_position=np.asarray(
                    state[spec.object_key], dtype=np.float64
                ),
                end_effector_orientation=isaac_quaternion_from_euler_deg(
                    spec.orientation_deg
                ),
            )
        if spec.kind == "press":
            return self.press_controller.forward(
                target_position=np.asarray(state[spec.target_key], dtype=np.float64),
                current_joint_positions=state["joint_positions"],
                gripper_control=self.gripper_control,
                end_effector_orientation=isaac_quaternion_from_euler_deg(
                    spec.orientation_deg
                ),
                gripper_position=state["gripper_position"],
            )
        raise ValueError(f"Unknown phase kind: {spec.kind}")

    def _advance(self, state) -> Tuple[None, bool, bool]:
        index = self.phase_index
        if index == len(self.PHASE_SPECS) - 1:
            self.write_cached_episode(state)
            self._last_success = True
            self.phase_index = len(self.PHASE_SPECS)
            self.reset_needed = True
            print(f"{self.__class__.__name__} task success!")
            return None, True, True
        self.phase_index = index + 1
        next_spec = self.PHASE_SPECS[self.phase_index]
        self.active_controller = self._controller_for(next_spec)
        self.active_controller.reset()
        print(f"Switching to {next_spec.name}...")
        return None, False, False

    def _fail(
        self,
        state,
        spec: PhaseSpec,
        detail: str | None = None,
    ) -> Tuple[None, bool, bool]:
        self._last_failure_reason = detail or f"Phase {spec.name} did not satisfy geometric success"
        self.data_collector.clear_cache()
        self._last_success = False
        self.phase_index = len(self.PHASE_SPECS)
        self.reset_needed = True
        print(f"{self.__class__.__name__} failed: {self._last_failure_reason}")
        return None, True, False

    def _step_infer(self, state) -> Tuple[Any, bool, bool]:
        state["language_instruction"] = self.get_language_instruction()
        return self.inference_engine.step_inference(state), False, False

    def get_language_instruction(self) -> Optional[str]:
        return self.LANGUAGE_INSTRUCTION or None
