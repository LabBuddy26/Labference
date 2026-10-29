from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from scipy.spatial.transform import Rotation

from controllers.base_controller import BaseController


class PressPhase(Enum):
    APPROACH = "approach"
    CLOSE = "close"
    PRESS = "press"
    HOLD = "hold"
    FINISHED = "finished"


class PressTaskController(BaseController):
    """Strict front-panel button controller with aligned dataset recording."""

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot, use_default_config=False)
        self.current_phase = PressPhase.APPROACH
        self.state = None
        self._phase_steps = 0
        self._hold_steps = 0
        self._last_target = None

    def _init_collect_mode(self, cfg, robot) -> None:
        super()._init_collect_mode(cfg, robot)

    def _init_infer_mode(self, cfg, robot) -> None:
        super()._init_infer_mode(cfg, robot)

    def reset(self) -> None:
        super().reset()
        self.current_phase = PressPhase.APPROACH
        self.state = None
        self._phase_steps = 0
        self._hold_steps = 0
        self._last_target = None
        if self.mode == "collect":
            self.rmp_controller.reset()
        else:
            self.inference_engine.reset()

    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        self.state = state
        if self.mode == "collect":
            return self._step_collect(state)
        return self._step_infer(state)

    def _motion_action(self, state, target, gripper_target: float):
        action = self.rmp_controller.forward(
            target_end_effector_position=np.asarray(target, dtype=np.float64),
            # This RMPFlow wrapper follows the existing atomic controllers and
            # accepts scipy's xyzw quaternion convention.
            target_end_effector_orientation=Rotation.from_euler(
                "xyz", np.radians(self.cfg.press.orientation_deg)
            ).as_quat(),
        )
        positions = getattr(action, "joint_positions", None)
        if positions is not None and len(positions) >= 9:
            positions[7] = gripper_target / get_stage_units()
            positions[8] = gripper_target / get_stage_units()
        return action

    @staticmethod
    def _gripper_action(state, target: float) -> ArticulationAction:
        positions = [None] * len(state["joint_positions"])
        positions[7] = target / get_stage_units()
        positions[8] = target / get_stage_units()
        return ArticulationAction(joint_positions=positions)

    def _reached(self, state, target, tolerance: float) -> bool:
        actual = np.asarray(state["gripper_position"], dtype=np.float64)
        return bool(np.max(np.abs(actual[:3] - target[:3])) <= tolerance)

    def _advance(self, phase: PressPhase) -> None:
        self.current_phase = phase
        self._phase_steps = 0
        print(f"Switching to {phase.value}...", flush=True)

    def _fail(self, state, reason: str):
        actual = np.asarray(state["gripper_position"], dtype=np.float64)
        target = None if self._last_target is None else self._last_target.tolist()
        self._last_failure_reason = f"{reason}; target={target}, actual={actual[:3].tolist()}"
        self.data_collector.clear_cache()
        self._last_success = False
        self.current_phase = PressPhase.FINISHED
        self.reset_needed = True
        print(f"PressTaskController failed: {self._last_failure_reason}", flush=True)
        return None, True, False

    def _step_collect(self, state):
        if self.current_phase == PressPhase.FINISHED:
            return None, True, self._last_success

        cfg = self.cfg.press
        button_target = np.asarray(state["object_position"], dtype=np.float64).copy()
        closed = float(cfg.closed_gripper_width_m)
        action = None

        if self.current_phase == PressPhase.APPROACH:
            target = button_target.copy()
            target[0] -= float(cfg.approach_offset_x_m)
            self._last_target = target
            action = self._motion_action(state, target, float(cfg.open_gripper_width_m))
            if self._reached(state, target, float(cfg.approach_tolerance_m)):
                self._advance(PressPhase.CLOSE)
        elif self.current_phase == PressPhase.CLOSE:
            action = self._gripper_action(state, closed)
            if self._phase_steps >= int(cfg.close_steps):
                self._advance(PressPhase.PRESS)
        elif self.current_phase == PressPhase.PRESS:
            target = button_target.copy()
            target[0] += float(cfg.press_offset_x_m)
            self._last_target = target
            action = self._motion_action(state, target, closed)
            if self._reached(state, target, float(cfg.press_tolerance_m)):
                self._advance(PressPhase.HOLD)
        elif self.current_phase == PressPhase.HOLD:
            target = button_target.copy()
            target[0] += float(cfg.press_offset_x_m)
            self._last_target = target
            action = self._motion_action(state, target, closed)
            hold_requires_reach = bool(cfg.get("hold_requires_reach", True))
            if not hold_requires_reach or self._reached(
                state, target, float(cfg.press_tolerance_m)
            ):
                self._hold_steps += 1
            else:
                self._hold_steps = 0
            if self._hold_steps >= int(cfg.hold_steps):
                self.cache_collection_step(state, action)
                self.write_cached_episode(state)
                self._last_success = True
                self.current_phase = PressPhase.FINISHED
                self.reset_needed = True
                print("PressTaskController task success!", flush=True)
                return None, True, True

        self._phase_steps += 1
        if self._phase_steps > int(cfg.phase_timeout_steps):
            return self._fail(
                state, f"press phase {self.current_phase.value} timed out"
            )
        if action is not None:
            self.cache_collection_step(state, action)
        return action, False, False

    def _step_infer(self, state):
        state["language_instruction"] = self.get_language_instruction()
        return self.inference_engine.step_inference(state), False, False

    def get_language_instruction(self) -> Optional[str]:
        if self.state and self.state.get("task_instruction"):
            return str(self.state["task_instruction"])
        return "Read the display and press the correct button"
