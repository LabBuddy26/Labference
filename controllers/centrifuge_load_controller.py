from __future__ import annotations

import re
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation as R

from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .base_controller import BaseController


class Phase(Enum):
    PICKING_NMR1 = "picking_nmr_01"
    PLACING_NMR1 = "placing_nmr_01"
    FINISHED = "finished"


class CentrifugeLoadTaskController(BaseController):
    """Main-compatible single-tube scripted collection for L07.

    Evaluation remains policy driven. Collection restores the original
    NMR_01 Pick -> Place sequence while retaining the current aligned HDF5
    recording contract (actions_ee, action_gripper and camera3).
    """

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_nmr_01_position = None
        self.current_phase = Phase.PICKING_NMR1
        self.state = None

    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        control_cfg = cfg.get("collection_control", {})
        self.grasp_width_m = float(control_cfg.get("grasp_width_m", 0.001))
        self.place_pre_z_m = float(control_cfg.get("place_pre_z_m", 0.20))
        self.place_offset_z_m = float(control_cfg.get("place_offset_z_m", 0.070))
        self.place_xy_compensation_m = np.asarray(
            control_cfg.get("place_xy_compensation_m", [0.0, 0.0]),
            dtype=np.float64,
        )
        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            descent_position_threshold=float(
                control_cfg.get("place_descent_tolerance_m", 0.004)
            ),
            legacy_timed_progression=False,
        )
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.01, 0.004, 0.01],
        )
        self.active_controller = self.pick_controller

    def _init_infer_mode(self, cfg, robot):
        super()._init_infer_mode(cfg, robot)

    def reset(self):
        super().reset()
        self.current_phase = Phase.PICKING_NMR1
        self.initial_nmr_01_position = None
        self.state = None
        if self.mode == "collect":
            self.pick_controller.reset()
            self.place_controller.reset()
            self.active_controller = self.pick_controller
        else:
            self.inference_engine.reset()

    def _check_phase_success(self, state) -> bool:
        object_pos = np.asarray(state["object_position"], dtype=np.float64)
        if self.current_phase == Phase.PICKING_NMR1:
            return bool(object_pos[2] > self.initial_nmr_01_position[2] + 0.12)
        if self.current_phase == Phase.PLACING_NMR1:
            target = np.asarray(state["target_position"], dtype=np.float64)
            success_cfg = self.cfg.get("centrifuge_success", {})
            return bool(
                np.linalg.norm(object_pos[:2] - target[:2])
                < float(success_cfg.get("place_xy_tolerance_m", 0.02))
                and abs(float(object_pos[2] - target[2]))
                < float(success_cfg.get("place_z_tolerance_m", 0.04))
            )
        return False

    def step(self, state):
        self.state = state
        if self.initial_nmr_01_position is None:
            self.initial_nmr_01_position = np.asarray(
                state["object_position"], dtype=np.float64
            ).copy()
        if self.mode == "collect":
            return self._step_collect(state)
        return self._step_infer(state)

    def _fail(self, reason: str):
        self._last_failure_reason = reason
        self.data_collector.clear_cache()
        self._last_success = False
        self.current_phase = Phase.FINISHED
        self.reset_needed = True
        return None, True, False

    def _step_collect(self, state):
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if self.active_controller.is_done():
            if getattr(self.active_controller, "has_failed", lambda: False)():
                return self._fail(
                    getattr(
                        self.active_controller,
                        "last_failure_reason",
                        f"{self.current_phase.value} atomic action failed",
                    )
                )
            if not self._check_phase_success(state):
                target = np.asarray(state["target_position"]).tolist()
                atomic_target = getattr(
                    self.active_controller, "_last_target_position", None
                )
                return self._fail(
                    f"{self.current_phase.value} finished without satisfying "
                    f"geometric success; object={np.asarray(state['object_position']).tolist()}, "
                    f"target={target}, "
                    f"gripper={np.asarray(state['gripper_position']).tolist()}, "
                    f"atomic_target={None if atomic_target is None else np.asarray(atomic_target).tolist()}"
                )
            if self.current_phase == Phase.PICKING_NMR1:
                print("Pick task success! Switching to place...", flush=True)
                self.current_phase = Phase.PLACING_NMR1
                self.place_controller.reset()
                self.active_controller = self.place_controller
                return None, False, False

            print("Place task success! Switching to finished", flush=True)
            self.write_cached_episode(state)
            self._last_success = True
            self.current_phase = Phase.FINISHED
            self.reset_needed = True
            return None, True, True

        if self.current_phase == Phase.PICKING_NMR1:
            action = self.pick_controller.forward(
                picking_position=state["object_position"],
                current_joint_positions=state["joint_positions"],
                object_size=state["object_size"],
                object_name=state["object_name"],
                gripper_control=self.gripper_control,
                gripper_position=state["gripper_position"],
                end_effector_orientation=R.from_euler(
                    "xyz", np.radians([0, 90, 30])
                ).as_quat(),
                pre_offset_x=0.03,
                pre_offset_z=0.10,
                after_offset_z=0.30,
                gripper_distances=self.grasp_width_m,
            )
        else:
            compensated_target = np.asarray(
                state["target_position"], dtype=np.float64
            ).copy()
            compensated_target[:2] += self.place_xy_compensation_m
            action = self.place_controller.forward(
                place_position=compensated_target,
                current_joint_positions=state["joint_positions"],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler(
                    "xyz", np.radians([0, 90, 80])
                ).as_quat(),
                gripper_position=state["gripper_position"],
                pre_place_z=self.place_pre_z_m,
                place_offset_z=self.place_offset_z_m,
            )

        if action is not None:
            self.cache_collection_step(state, action)
        return action, False, False

    def _step_infer(self, state):
        state["language_instruction"] = self.get_language_instruction()
        return self.inference_engine.step_inference(state), False, False

    def is_success(self):
        return bool(self._last_success)

    def get_language_instruction(self) -> str:
        if self.state is None:
            return "Pick up the NMR tube from the table and place it in the centrifuge"
        object_name = re.sub(r"\d+", "", self.state["object_name"])
        object_name = object_name.replace("_", " ").replace("  ", " ").lower()
        return f"Pick up the {object_name} from the table and place it at the target"
