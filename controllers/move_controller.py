from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
from isaacsim.core.utils.stage import get_stage_units
from scipy.spatial.transform import Rotation

from .base_controller import BaseController


class MovePhase(Enum):
    MOVING = "moving"
    HOLDING = "holding"
    FINISHED = "finished"


class MoveTaskController(BaseController):
    """Strict Cartesian move controller with aligned dataset recording."""

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot, use_default_config=False)
        self.current_phase = MovePhase.MOVING
        self.state = None
        self._phase_steps = 0
        self._stable_steps = 0
        self._last_target = None

    def reset(self) -> None:
        super().reset()
        self.current_phase = MovePhase.MOVING
        self.state = None
        self._phase_steps = 0
        self._stable_steps = 0
        self._last_target = None
        if self.mode == "collect":
            self.rmp_controller.reset()
        else:
            self.inference_engine.reset()

    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        self.state = state
        if self.mode != "collect":
            state["language_instruction"] = self.get_language_instruction()
            return self.inference_engine.step_inference(state), False, False
        if self.current_phase == MovePhase.FINISHED:
            return None, True, self._last_success

        cfg = self.cfg.move
        target = np.asarray(state["target1_position"], dtype=np.float64)
        actual = np.asarray(state["gripper_position"], dtype=np.float64)
        self._last_target = target.copy()
        action = self.rmp_controller.forward(
            target_end_effector_position=target,
            target_end_effector_orientation=Rotation.from_euler(
                "xyz", np.radians(cfg.orientation_deg)
            ).as_quat(),
        )
        positions = getattr(action, "joint_positions", None)
        if positions is not None and len(positions) >= 9:
            width = float(cfg.gripper_width_m) / get_stage_units()
            positions[7] = width
            positions[8] = width

        reached = bool(
            np.max(np.abs(actual[:3] - target[:3]))
            <= float(cfg.position_tolerance_m)
        )
        if reached:
            if self.current_phase == MovePhase.MOVING:
                self.current_phase = MovePhase.HOLDING
                self._stable_steps = 0
                print("Switching to holding...", flush=True)
            self._stable_steps += 1
        else:
            self._stable_steps = 0

        self.cache_collection_step(state, action)
        if self._stable_steps >= int(cfg.stable_steps):
            self.write_cached_episode(state)
            self._last_success = True
            self.current_phase = MovePhase.FINISHED
            self.reset_needed = True
            print("MoveTaskController task success!", flush=True)
            return None, True, True

        self._phase_steps += 1
        if self._phase_steps > int(cfg.phase_timeout_steps):
            error = actual[:3] - target[:3]
            self._last_failure_reason = (
                f"move phase timed out; target={target[:3].tolist()}, "
                f"actual={actual[:3].tolist()}, xyz_error={error.tolist()}"
            )
            self.data_collector.clear_cache()
            self._last_success = False
            self.current_phase = MovePhase.FINISHED
            self.reset_needed = True
            print(f"MoveTaskController failed: {self._last_failure_reason}", flush=True)
            return None, True, False
        return action, False, False

    def get_language_instruction(self) -> Optional[str]:
        if self.state and self.state.get("task_instruction"):
            return str(self.state["task_instruction"])
        return "Move the gripper above the instructed sample-box socket"
