from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction

from .atomic_actions.pickr_controller import PickRController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.tube_insert_controller import TubeInsertController
from .base_controller import BaseController
from utils.rotation_utils import isaac_quaternion_from_euler_deg


class Phase(Enum):
    PICKING = "picking"
    PLACING = "placing"
    RELEASING = "releasing"
    STABILIZING = "stabilizing"
    FINISHED = "finished"


class SingleObjectPlaceTaskController(BaseController):
    """Scripted single-object "pick - place - release - stabilize" flow.

    After the object is placed and the gripper is explicitly opened, the
    controller waits for the object to settle (small displacement for a
    configurable number of frames) before declaring success.  Eval mode is
    fully policy driven via ``_step_infer``.
    """

    SUCCESS_CFG_SECTION: str = ""
    LANGUAGE_INSTRUCTION: str = ""
    PICK_ORIENTATION_DEG: Tuple[float, float, float] = (0.0, 180.0, 0.0)
    PLACE_ORIENTATION_DEG: Tuple[float, float, float] = (0.0, 180.0, 0.0)
    PRE_OFFSET_X: float = 0.03
    PRE_OFFSET_Z: float = 0.10
    AFTER_OFFSET_Z: float = 0.30
    PRE_PLACE_Z: float = 0.30
    PLACE_OFFSET_Z: float = 0.05
    GRIPPER_DISTANCE: Optional[float] = None

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.state = None
        self._release_counter = 0
        self._stability_counter = 0
        self._prev_object_position = None

    def _success_cfg(self):
        return getattr(self.cfg, self.SUCCESS_CFG_SECTION)

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        self.pick_controller = PickRController(
            name=f"{self.__class__.__name__}_pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.01, 0.004, 0.01],
        )
        self.place_controller = PlaceController(
            name=f"{self.__class__.__name__}_place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
        )
        self.insert_controller = None
        if hasattr(cfg, "guarded_insertion"):
            self.insert_controller = TubeInsertController(
                name=f"{self.__class__.__name__}_insert_controller",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
                insertion_cfg=cfg.guarded_insertion,
            )
        self.active_controller = self.pick_controller

    def _init_infer_mode(self, cfg, robot):
        super()._init_infer_mode(cfg, robot)

    def reset(self) -> None:
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.state = None
        self._release_counter = 0
        self._stability_counter = 0
        self._prev_object_position = None
        if self.mode == "collect":
            self.pick_controller.reset()
            self.place_controller.reset()
            if self.insert_controller is not None:
                self.insert_controller.reset()
            self.active_controller = self.pick_controller

    def step(self, state: Dict[str, Any]):
        self.state = state
        if self.initial_position is None and "object_position" in state:
            self.initial_position = np.asarray(state["object_position"], dtype=np.float64).copy()
        if self.mode == "collect":
            return self._step_collect(state)
        return self._step_infer(state)

    def _open_gripper_action(self, state) -> ArticulationAction:
        target_joint_positions = [None] * state["joint_positions"].shape[0]
        target_joint_positions[7] = 0.04 / get_stage_units()
        target_joint_positions[8] = 0.04 / get_stage_units()
        return ArticulationAction(joint_positions=target_joint_positions)

    def _step_collect(self, state) -> Tuple[Any, bool, bool]:
        if self.current_phase == Phase.FINISHED:
            return None, True, self._last_success
        cfg = self._success_cfg()
        action = None

        if self.current_phase == Phase.PICKING:
            if self.active_controller.is_done():
                if self.pick_controller.has_failed():
                    self._fail(state, "pick", self.pick_controller.last_failure_reason)
                    return None, True, False
                position = np.asarray(state["object_position"], dtype=np.float64)
                if position[2] >= self.initial_position[2] + float(cfg.pick_lift_height_m):
                    self.current_phase = Phase.PLACING
                    self.active_controller = self.insert_controller or self.place_controller
                    self.active_controller.reset()
                    print("Switching to PLACING...")
                    return None, False, False
                self._fail(state, "pick")
                return None, True, False
            action = self.pick_controller.forward(
                picking_position=np.asarray(state["object_position"], dtype=np.float64),
                current_joint_positions=state["joint_positions"],
                object_size=np.asarray(state["object_size"], dtype=np.float64),
                object_name=str(state.get("object_name", "object")),
                gripper_control=self.gripper_control,
                gripper_position=state["gripper_position"],
                end_effector_orientation=isaac_quaternion_from_euler_deg(
                    self.PICK_ORIENTATION_DEG
                ),
                pre_offset_x=self.PRE_OFFSET_X,
                pre_offset_z=self.PRE_OFFSET_Z,
                after_offset_z=self.AFTER_OFFSET_Z,
                gripper_distances=self.GRIPPER_DISTANCE,
            )
        elif self.current_phase == Phase.PLACING:
            if self.active_controller.is_done():
                if self.active_controller.has_failed():
                    self._fail(state, "place", self.active_controller.last_failure_reason)
                    return None, True, False
                position = np.asarray(state["object_position"], dtype=np.float64)
                target = np.asarray(state["target_position"], dtype=np.float64)
                z_reference = (
                    float(target[2])
                    if str(getattr(cfg, "place_z_reference", "initial")) == "target"
                    else float(self.initial_position[2])
                )
                placed = (
                    np.linalg.norm(position[:2] - target[:2]) <= float(cfg.place_xy_tolerance_m)
                    and abs(float(position[2] - z_reference))
                    <= float(cfg.place_z_tolerance_m)
                )
                if placed:
                    self.current_phase = Phase.RELEASING
                    self._release_counter = 0
                    print("Switching to RELEASING...")
                    return None, False, False
                self._fail(state, "place")
                return None, True, False
            if self.insert_controller is not None:
                inserted_center = np.asarray(state["target_position"], dtype=np.float64).copy()
                inserted_center[2] += float(getattr(self.cfg.guarded_insertion, "pre_release_offset_z", 0.025))
                action = self.insert_controller.forward(
                    socket_position=np.asarray(state["target_socket_position"], dtype=np.float64),
                    inserted_tube_center=inserted_center,
                    insertion_axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                    current_joint_positions=state["joint_positions"],
                    gripper_position=state["gripper_position"],
                    object_position=np.asarray(state["object_position"], dtype=np.float64),
                    end_effector_orientation=isaac_quaternion_from_euler_deg(self.PLACE_ORIENTATION_DEG),
                )
            else:
                action = self.place_controller.forward(
                    place_position=np.asarray(state["target_position"], dtype=np.float64).copy(),
                    current_joint_positions=state["joint_positions"],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=isaac_quaternion_from_euler_deg(
                        self.PLACE_ORIENTATION_DEG
                    ),
                    gripper_position=state["gripper_position"],
                    pre_place_z=self.PRE_PLACE_Z,
                    place_offset_z=self.PLACE_OFFSET_Z,
                )
        elif self.current_phase == Phase.RELEASING:
            action = self._open_gripper_action(state)
            self._release_counter += 1
            if self._release_counter >= int(getattr(cfg, "release_steps", 30)):
                self.current_phase = Phase.STABILIZING
                self._stability_counter = 0
                self._prev_object_position = np.asarray(
                    state["object_position"], dtype=np.float64
                ).copy()
                print("Switching to STABILIZING...")
        elif self.current_phase == Phase.STABILIZING:
            position = np.asarray(state["object_position"], dtype=np.float64)
            displacement = (
                np.linalg.norm(position - self._prev_object_position)
                if self._prev_object_position is not None
                else 0.0
            )
            self._prev_object_position = position.copy()
            if displacement < float(getattr(cfg, "stable_displacement_m", 0.001)):
                self._stability_counter += 1
            else:
                self._stability_counter = 0
            if self._stability_counter >= int(getattr(cfg, "stable_frames", 30)):
                self.write_cached_episode(state)
                self._last_success = True
                self.current_phase = Phase.FINISHED
                self.reset_needed = True
                print(f"{self.__class__.__name__} task success!")
                return None, True, True
            action = self._open_gripper_action(state)

        if action is not None:
            self.cache_collection_step(state, action)
        return action, False, False

    def _fail(self, state, phase_name: str, detail: str | None = None) -> None:
        if detail is None and phase_name == "place":
            position = np.asarray(state.get("object_position", []), dtype=np.float64)
            target = np.asarray(state.get("target_position", []), dtype=np.float64)
            if position.size >= 3 and target.size >= 3:
                cfg = self._success_cfg()
                z_reference = (
                    float(target[2])
                    if str(getattr(cfg, "place_z_reference", "initial")) == "target"
                    else float(self.initial_position[2])
                )
                detail = (
                    f"Phase place did not satisfy geometric success; "
                    f"object={position[:3].tolist()}, target={target[:3].tolist()}, "
                    f"xy_error={float(np.linalg.norm(position[:2] - target[:2])):.6f} m/"
                    f"{float(cfg.place_xy_tolerance_m):.6f} m, "
                    f"z_error={abs(float(position[2] - z_reference)):.6f} m/"
                    f"{float(cfg.place_z_tolerance_m):.6f} m"
                )
        self._last_failure_reason = detail or f"Phase {phase_name} did not satisfy geometric success"
        self.data_collector.clear_cache()
        self._last_success = False
        self.current_phase = Phase.FINISHED
        self.reset_needed = True
        print(f"{self.__class__.__name__} failed: {self._last_failure_reason}")

    def _step_infer(self, state) -> Tuple[Any, bool, bool]:
        state["language_instruction"] = self.get_language_instruction()
        return self.inference_engine.step_inference(state), False, False

    def get_language_instruction(self) -> Optional[str]:
        return self.LANGUAGE_INSTRUCTION or None
