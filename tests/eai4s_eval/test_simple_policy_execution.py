from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest

import numpy as np

from eai4s_eval.eval.benchmark_backend import PolicyTrajectoryInferenceEngine
from eai4s_eval.policy import Gr00tPolicyClient
from eai4s_eval.eval.external_inference_engine import ExternalInferenceEngine


class _Policy:
    def __init__(self) -> None:
        self.calls = 0
        self.reset_calls = 0

    def predict(self, _state):
        self.calls += 1
        actions = np.arange(32, dtype=np.float32).reshape(4, 8)
        return SimpleNamespace(joint_positions=actions, latency_ms=2.5)

    def reset(self):
        self.reset_calls += 1

    def close(self):
        pass


class _Trajectory:
    def __init__(self) -> None:
        self.targets = []

    def reset(self):
        self.targets = []

    def generate_trajectory(self, targets):
        self.targets.append(np.asarray(targets).copy())

    def generate_ee_trajectory(self, targets):
        self.targets.append(("ee", np.asarray(targets).copy()))

    def get_next_action(self):
        last = self.targets[-1]
        if isinstance(last, tuple):
            return last[1][0]
        return last[0]


class _Gr00tClient:
    def __init__(self) -> None:
        self.closed = False
        self.reset_calls = 0

    def get_action(self, observation):
        return {"action.single_arm": [[0.0] * 7], "action.gripper": [[1.0]]}, {"latency_ms": 7.0}

    def get_modality_config(self):
        return {"action": "joint7+gripper1"}

    def reset(self, options=None):
        self.reset_calls += 1
        return {}

    def close(self):
        self.closed = True


class _OfficialGr00tClient:
    def get_action(self, observation):
        return {"action.single_arm": [[0.0] * 7], "action.gripper": [[1.0]]}

    def call_endpoint(self, endpoint, requires_input=True):
        assert endpoint == "get_modality_config"
        assert not requires_input
        return {"video": {"modality_keys": ["video.main_camera", "video.wrist_camera"]}}


class SimplePolicyExecutionTest(unittest.TestCase):
    def test_joint_waypoints_are_held_for_six_physics_steps(self):
        policy = _Policy()
        trajectory = _Trajectory()
        engine = PolicyTrajectoryInferenceEngine(
            policy=policy,
            trajectory_controller=trajectory,
            execution_horizon=2,
        )
        actions = [engine.step_inference({}) for _ in range(13)]
        for action in actions[:6]:
            np.testing.assert_array_equal(action, np.arange(8, dtype=np.float32))
        for action in actions[6:12]:
            np.testing.assert_array_equal(action, np.arange(8, 16, dtype=np.float32))
        np.testing.assert_array_equal(actions[12], np.arange(8, dtype=np.float32))
        self.assertEqual(policy.calls, 2)
        self.assertEqual(engine._executed_action_steps, 3)
        self.assertEqual(engine._total_steps, 13)
        self.assertEqual(engine.sim_steps_per_action, 6)
        self.assertTrue(all(target.shape == (1, 8) for target in trajectory.targets))

    def test_chunk_log_reports_action_and_physics_step_counters(self):
        policy = _Policy()
        trajectory = _Trajectory()
        engine = PolicyTrajectoryInferenceEngine(
            policy=policy,
            trajectory_controller=trajectory,
            execution_horizon=2,
        )
        output = StringIO()
        with redirect_stdout(output):
            for _ in range(13):
                engine.step_inference({})
        log = output.getvalue()
        self.assertIn("chunk #1: action_steps=1-2, policy_physics_step=1", log)
        self.assertIn("chunk #2: action_steps=3-4, policy_physics_step=13", log)
        self.assertIn("physics_steps_per_action=6", log)

    def test_ee_pos_profile_executes_ee_waypoints(self):
        policy = _Policy()
        policy.profile = SimpleNamespace(action_space="ee_pos")
        trajectory = _Trajectory()
        engine = PolicyTrajectoryInferenceEngine(
            policy=policy,
            trajectory_controller=trajectory,
            execution_horizon=2,
        )
        actions = [engine.step_inference({}) for _ in range(7)]
        for action in actions[:6]:
            np.testing.assert_array_equal(action, np.arange(8, dtype=np.float32))
        np.testing.assert_array_equal(actions[6], np.arange(8, 16, dtype=np.float32))
        self.assertEqual(policy.calls, 1)
        # RMPFlow is recomputed on every physics step while the EE target stays fixed.
        self.assertEqual(len(trajectory.targets), 7)
        self.assertTrue(all(kind == "ee" for kind, _ in trajectory.targets))
        self.assertTrue(
            all(target.shape == (1, 8) for _, target in trajectory.targets)
        )
        for _, target in trajectory.targets[:6]:
            np.testing.assert_array_equal(target[0], np.arange(8, dtype=np.float32))

    def test_rejects_a_hold_override_that_conflicts_with_frequencies(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            PolicyTrajectoryInferenceEngine(
                policy=_Policy(),
                trajectory_controller=_Trajectory(),
                control_hz=10,
                physics_hz=60,
                sim_steps_per_action=1,
            )

    def test_ee_chunk_diagnostics_do_not_change_the_selected_horizon(self):
        policy = _Policy()
        policy.profile = SimpleNamespace(action_space="ee_pos")
        trajectory = _Trajectory()
        engine = PolicyTrajectoryInferenceEngine(
            policy=policy,
            trajectory_controller=trajectory,
            execution_horizon=2,
        )
        engine.step_inference({"object_position": np.array([0.0, 0.0, 0.0])})
        self.assertEqual(engine._chunk.shape, (2, 8))
        np.testing.assert_array_equal(engine._chunk[0], np.arange(8, dtype=np.float32))

    def test_gr00t_client_normalizes_official_get_action_response(self):
        upstream = _Gr00tClient()
        client = Gr00tPolicyClient(client=upstream)
        result = client.infer({"video.main_camera": np.zeros((1, 2, 2, 3), dtype=np.uint8)})
        self.assertIn("action.single_arm", result)
        self.assertEqual(result["model_latency_ms"], 7.0)
        self.assertIn("modality_config", client.get_server_metadata())
        client.reset()
        client.close()
        self.assertEqual(upstream.reset_calls, 1)
        self.assertTrue(upstream.closed)

    def test_gr00t_client_accepts_robot_inference_mapping_response(self):
        client = Gr00tPolicyClient(client=_OfficialGr00tClient())
        result = client.infer({"video.main_camera": np.zeros((1, 2, 2, 3), dtype=np.uint8)})
        self.assertEqual(np.asarray(result["action.single_arm"]).shape, (1, 7))
        self.assertEqual(
            client.get_server_metadata()["modality_config"]["video"]["modality_keys"],
            ["video.main_camera", "video.wrist_camera"],
        )

    def test_external_engine_is_a_deferred_non_network_placeholder(self):
        trajectory = _Trajectory()
        engine = ExternalInferenceEngine(cfg=None, trajectory_controller=trajectory)
        with self.assertRaisesRegex(RuntimeError, "before the benchmark"):
            engine.step_inference({})
        engine.reset()
        self.assertEqual(trajectory.targets, [])


if __name__ == "__main__":
    unittest.main()
