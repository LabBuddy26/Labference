from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from .atomic_actions.pickr_controller import PickRController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.tube_insert_controller import TubeInsertController
from .base_controller import BaseController
from utils.rotation_utils import isaac_quaternion_from_euler_deg


class Phase(Enum):
    PICK_VIAL1 = "picking_vial_01"
    PLACE_VIAL1 = "placing_vial_01"
    PICK_VIAL2 = "picking_vial_02"
    PLACE_VIAL2 = "placing_vial_02"
    PICK_VIAL3 = "picking_vial_03"
    PLACE_VIAL3 = "placing_vial_03"
    FINISHED = "finished"


_PHASES = (
    Phase.PICK_VIAL1,
    Phase.PLACE_VIAL1,
    Phase.PICK_VIAL2,
    Phase.PLACE_VIAL2,
    Phase.PICK_VIAL3,
    Phase.PLACE_VIAL3,
)


class SampleTransportTaskController(BaseController):
    """Script three pick/place pairs while eval remains fully policy driven."""

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.current_phase = Phase.PICK_VIAL1
        self.initial_vial_positions: list[np.ndarray] | None = None
        self.state = None

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        self.pick_controller = PickRController(
            name="sample_pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.01, 0.004, 0.01],
        )
        self.place_controller = PlaceController(
            name="sample_place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
        )
        self.insert_controller = TubeInsertController(
            name="sample_insert_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            insertion_cfg=cfg.guarded_insertion,
        )
        self.active_controller = self.pick_controller

    def _init_infer_mode(self, cfg, robot):
        super()._init_infer_mode(cfg, robot)

    def reset(self) -> None:
        super().reset()
        self.current_phase = Phase.PICK_VIAL1
        self.initial_vial_positions = None
        self.state = None
        if self.mode == "collect":
            self.pick_controller.reset()
            self.place_controller.reset()
            self.insert_controller.reset()
            self.active_controller = self.pick_controller

    def _phase_index(self) -> int:
        return _PHASES.index(self.current_phase)

    def _vial_index(self) -> int:
        return self._phase_index() // 2

    def _is_pick_phase(self) -> bool:
        return self._phase_index() % 2 == 0

    def _check_phase_success(self, state) -> bool:
        index = self._vial_index()
        position = np.asarray(state["vial_positions"][index], dtype=np.float64)
        cfg = self.cfg.sample_transport_success
        if self._is_pick_phase():
            return bool(position[2] >= self.initial_vial_positions[index][2] + float(cfg.pick_lift_height_m))
        target = np.asarray(state.get("target_slot_inserted_centers", state["target_slot_positions"])[index], dtype=np.float64)
        return bool(
            np.linalg.norm(position[:2] - target[:2]) <= float(cfg.place_xy_tolerance_m)
            and abs(float(position[2] - target[2])) <= float(cfg.place_z_tolerance_m)
        )

    def _advance(self, state) -> tuple[None, bool, bool]:
        index = self._phase_index()
        if index == len(_PHASES) - 1:
            self.write_cached_episode(state)
            self._last_success = True
            self.current_phase = Phase.FINISHED
            self.reset_needed = True
            print("Sample transport task success!")
            return None, True, True
        self.current_phase = _PHASES[index + 1]
        self.active_controller = self.pick_controller if self._is_pick_phase() else self.insert_controller
        self.active_controller.reset()
        print(f"Switching to {self.current_phase.value}...")
        return None, False, False

    def _fail(self, detail: str | None = None) -> tuple[None, bool, bool]:
        self._last_failure_reason = detail or (
            f"Phase {self.current_phase.value} did not satisfy geometric success"
        )
        self.data_collector.clear_cache()
        self._last_success = False
        self.current_phase = Phase.FINISHED
        self.reset_needed = True
        print(f"Sample transport failed: {self._last_failure_reason}")
        return None, True, False

    def step(self, state):
        self.state = state
        if self.initial_vial_positions is None:
            self.initial_vial_positions = [
                np.asarray(position, dtype=np.float64).copy() for position in state["vial_positions"]
            ]
        if self.mode == "collect":
            return self._step_collect(state)
        return self._step_infer(state)

    def _step_collect(self, state):
        if self.current_phase == Phase.FINISHED:
            return None, True, self._last_success
        if self.active_controller.is_done():
            if self.active_controller.has_failed():
                return self._fail(self.active_controller.last_failure_reason)
            return self._advance(state) if self._check_phase_success(state) else self._fail()

        index = self._vial_index()
        orientation = isaac_quaternion_from_euler_deg((0.0, 180.0, 0.0))
        if self._is_pick_phase():
            action = self.pick_controller.forward(
                picking_position=np.asarray(state["vial_positions"][index]),
                current_joint_positions=state["joint_positions"],
                object_size=np.asarray(state["vial_sizes"][index]),
                object_name=str(state["vial_paths"][index]).rsplit("/", 1)[-1],
                gripper_control=self.gripper_control,
                gripper_position=state["gripper_position"],
                end_effector_orientation=orientation,
                pre_offset_x=0.03,
                pre_offset_z=0.10,
                after_offset_z=0.30,
                gripper_distances=0.002,
            )
        else:
            inserted_center = np.asarray(state["target_slot_inserted_centers"][index], dtype=np.float64).copy()
            inserted_center[2] += float(getattr(self.cfg.guarded_insertion, "pre_release_offset_z", 0.025))
            action = self.insert_controller.forward(
                socket_position=np.asarray(state["target_slot_positions"][index]),
                inserted_tube_center=inserted_center,
                insertion_axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                current_joint_positions=state["joint_positions"],
                gripper_position=state["gripper_position"],
                object_position=np.asarray(state["vial_positions"][index]),
                end_effector_orientation=orientation,
            )
        self.cache_collection_step(state, action)
        return action, False, False

    def _step_infer(self, state):
        state["language_instruction"] = self.get_language_instruction()
        return self.inference_engine.step_inference(state), False, False

    def is_success(self):
        return self._last_success

    def get_language_instruction(self) -> Optional[str]:
        return "Pick up the three vial bottles and place each one into a separate slot in the sample box"
