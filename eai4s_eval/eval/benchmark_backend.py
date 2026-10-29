from __future__ import annotations

from typing import Any

import numpy as np

from eai4s_eval.eval.interfaces import EvaluationBackend, PolicyAdapter


class PolicyTrajectoryInferenceEngine:
    """Execute 10 Hz policy waypoints over a 60 Hz physics clock."""

    def __init__(
        self,
        *,
        policy: PolicyAdapter,
        trajectory_controller: Any,
        execution_horizon: int | None = None,
        control_hz: int = 10,
        physics_hz: int = 60,
        sim_steps_per_action: int | None = None,
        **_legacy_options: Any,
    ) -> None:
        if execution_horizon is not None and int(execution_horizon) <= 0:
            raise ValueError("execution_horizon must be positive or null")
        self.policy = policy
        self.trajectory_controller = trajectory_controller
        self.execution_horizon = None if execution_horizon is None else int(execution_horizon)
        self.control_hz = int(control_hz)
        self.physics_hz = int(physics_hz)
        if self.control_hz <= 0 or self.physics_hz <= 0:
            raise ValueError("control_hz and physics_hz must be positive")
        if self.physics_hz % self.control_hz != 0:
            raise ValueError(
                f"physics_hz={self.physics_hz} must be divisible by control_hz={self.control_hz}"
            )
        expected_hold = self.physics_hz // self.control_hz
        if sim_steps_per_action is not None and int(sim_steps_per_action) != expected_hold:
            raise ValueError(
                f"sim_steps_per_action={sim_steps_per_action} conflicts with "
                f"physics_hz/control_hz={expected_hold}"
            )
        self.sim_steps_per_action = expected_hold
        profile_control_hz = int(
            getattr(getattr(policy, "profile", None), "control_hz", self.control_hz)
        )
        if profile_control_hz != self.control_hz:
            raise ValueError(
                f"policy profile control_hz={profile_control_hz} does not match "
                f"evaluation control_hz={self.control_hz}"
            )
        # "ee_pos" targets are end-effector poses; anything else is treated as
        # the joint-angle contract. Official entrypoints always load a profile.
        self._action_space = str(
            getattr(getattr(policy, "profile", None), "action_space", "joint")
        ).lower()
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self.trajectory_controller.reset()
        self._chunk: np.ndarray | None = None
        self._step_index = 0
        self._hold_step = 0
        self._total_steps = 0
        self._executed_action_steps = 0
        self._current_target: np.ndarray | None = None
        self.policy_calls = 0
        self.latencies_ms: list[float] = []
        self.last_chunk = None

    def _request_chunk(self, state: dict[str, Any]) -> None:
        chunk = self.policy.predict(state)
        actions = np.asarray(chunk.joint_positions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 8 or actions.shape[0] < 1:
            raise ValueError(f"policy action chunk must have shape [T, 8], got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("policy action chunk contains NaN or infinity")
        horizon = actions.shape[0] if self.execution_horizon is None else min(
            self.execution_horizon, actions.shape[0]
        )
        self._chunk = np.ascontiguousarray(actions[:horizon], dtype=np.float32)
        self._step_index = 0
        self.policy_calls += 1
        self.latencies_ms.append(float(chunk.latency_ms))
        self.last_chunk = chunk
        first_action_step = self._executed_action_steps + 1
        last_action_step = first_action_step + horizon - 1
        print(
            f"[policy] chunk #{self.policy_calls}: "
            f"action_steps={first_action_step}-{last_action_step}, "
            f"policy_physics_step={self._total_steps}, "
            f"model_horizon={actions.shape[0]}, executing={horizon}, "
            f"physics_steps_per_action={self.sim_steps_per_action}, "
            f"latency={chunk.latency_ms:.0f}ms",
            flush=True,
        )
        if self._action_space == "ee_pos":
            object_position = state.get("object_position")
            if object_position is not None:
                object_xyz = np.asarray(object_position, dtype=np.float64).reshape(-1)[:3]
                executed_xyz = np.asarray(self._chunk[:, :3], dtype=np.float64)
                actual_ee = state.get("gripper_position")
                actual_ee_text = "unavailable"
                if actual_ee is not None:
                    actual_ee_xyz = np.asarray(actual_ee, dtype=np.float64).reshape(-1)[:3]
                    actual_ee_text = str(np.round(actual_ee_xyz, 5).tolist())
                distances = np.linalg.norm(executed_xyz - object_xyz, axis=1)
                closest_index = int(np.argmin(distances))
                closest_offset = executed_xyz[closest_index] - object_xyz
                print(
                    "[policy][ee] "
                    f"object_xyz={np.round(object_xyz, 5).tolist()}, "
                    f"actual_ee={actual_ee_text}, "
                    f"first_target={np.round(executed_xyz[0], 5).tolist()}, "
                    f"last_target={np.round(executed_xyz[-1], 5).tolist()}, "
                    f"closest_index={closest_index}, "
                    f"closest_offset_xyz={np.round(closest_offset, 5).tolist()}, "
                    f"closest_distance={distances[closest_index]:.5f}m",
                    flush=True,
                )

    def step_inference(self, state: dict[str, Any]) -> Any:
        self._total_steps += 1
        if self._hold_step == 0:
            if self._chunk is None or self._step_index >= self._chunk.shape[0]:
                self._request_chunk(state)
            assert self._chunk is not None
            self._current_target = self._chunk[self._step_index].copy()
            self._executed_action_steps += 1
        assert self._chunk is not None
        assert self._current_target is not None
        target = self._current_target
        if self._action_space == "ee_pos":
            # Re-run RMPFlow every physics frame while holding one 10 Hz EE
            # waypoint, preserving feedback toward the fixed Cartesian target.
            self.trajectory_controller.generate_ee_trajectory(target[np.newaxis, :])
        else:
            # Reapply the same direct joint target for all six physics frames.
            self.trajectory_controller.generate_trajectory(target[np.newaxis, :])
        action = self.trajectory_controller.get_next_action()
        self._hold_step += 1
        if self._hold_step >= self.sim_steps_per_action:
            self._hold_step = 0
            self._step_index += 1
        return action

    def close(self) -> None:
        self.policy.close()


class EAI4SBenchmarkBackend(EvaluationBackend):
    backend_id = "eai4s_benchmark"

    def __init__(
        self,
        *,
        runtime: Any,
        policy: PolicyAdapter,
        execution_horizon: int | None = None,
        control_hz: int = 10,
        sim_steps_per_action: int | None = None,
        **legacy_options: Any,
    ) -> None:
        self.runtime = runtime
        self.policy = policy
        self.execution_horizon = execution_horizon
        physics_dt = float(runtime.world.get_physics_dt())
        physics_hz = int(round(1.0 / physics_dt))
        self.engine: PolicyTrajectoryInferenceEngine = runtime.bind_inference_engine(
            lambda trajectory_controller: PolicyTrajectoryInferenceEngine(
                policy=policy,
                trajectory_controller=trajectory_controller,
                execution_horizon=execution_horizon,
                control_hz=control_hz,
                physics_hz=physics_hz,
                sim_steps_per_action=sim_steps_per_action,
                **legacy_options,
            )
        )

    def reset_episode(self, *, seed: int, split: str) -> dict[str, Any]:
        initial_conditions = self.runtime.reset_episode(seed=seed, split=split)
        self.engine.reset()
        return initial_conditions

    def step(self) -> Any:
        return self.runtime.step()

    def actual_state(self) -> dict[str, Any]:
        return self.runtime.actual_state()

    def enter_terminal_hold(self) -> None:
        self.runtime.enter_terminal_hold()

    def policy_stats(self) -> dict[str, Any]:
        return {
            "policy_calls": self.engine.policy_calls,
            "executed_action_steps": self.engine._executed_action_steps,
            "sim_steps": self.engine._total_steps,
            "control_hz": self.engine.control_hz,
            "physics_hz": self.engine.physics_hz,
            "sim_steps_per_action": self.engine.sim_steps_per_action,
            "latencies_ms": list(self.engine.latencies_ms),
            "last_action_chunk": None
            if self.engine.last_chunk is None
            else self.engine.last_chunk.joint_positions.tolist(),
            "current_action_target": None
            if self.engine._current_target is None
            else self.engine._current_target.tolist(),
            "anchor_offsets": [],
            "anchor_offset_steps": [],
            "last_anchor_offset": None,
        }

    def close(self) -> None:
        self.runtime.close()


__all__ = ["EAI4SBenchmarkBackend", "PolicyTrajectoryInferenceEngine"]
