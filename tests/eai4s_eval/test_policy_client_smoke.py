from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from eai4s_eval.policy import PolicyClientError, PolicyOutputInvalid, VLAHttpPolicyClient
from eai4s_eval.policy.openpi_client import OpenPIHttpPolicyClient, OpenPIJointPolicyAdapter, OpenPIProfile


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._data


class _FakeRequests:
    class exceptions:
        class ConnectionError(Exception):
            pass

        class Timeout(Exception):
            pass

        class RequestException(Exception):
            pass

    def __init__(self, responses):
        self.responses = list(responses)
        self.urls: list[str] = []

    def post(self, url, json=None, timeout=None, **kwargs):
        self.urls.append(url)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _FakeResponse(response)


def _observation():
    return {
        "legacy": {"video.main_camera": np.zeros((1, 4, 4, 3), dtype=np.uint8)},
        "standard": {
            "observation": {
                "robot_state": {
                    "eef_pose": [0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
                }
            }
        },
    }


class PolicyClientSmokeTest(unittest.TestCase):
    def test_act_endpoint_converts_action_batch(self) -> None:
        fake = _FakeRequests(
            [
                {
                    "action.single_arm": [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]],
                    "action.gripper": [1.0],
                }
            ]
        )
        with patch("eai4s_eval.policy.http_client.requests", fake):
            batch = VLAHttpPolicyClient("127.0.0.1", 8188, api="act").predict(
                observation=_observation(),
                run_id="run",
                episode_id="ep",
                task_id="task",
                instruction="pick",
                step_index=0,
                control_hz=10,
            )
        self.assertEqual(batch.endpoint, "/act")
        self.assertEqual(batch.ee_pos.shape, (1, 3))
        self.assertEqual(batch.ee_quat.shape, (1, 4))
        self.assertEqual(batch.gripper.tolist(), [1.0])

    def test_predict_action_accepts_absolute_8d_action(self) -> None:
        fake = _FakeRequests([{"action": [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, -1.0]], "model_latency_ms": 12.5}])
        with patch("eai4s_eval.policy.http_client.requests", fake):
            batch = VLAHttpPolicyClient("127.0.0.1", 8188, api="predict_action").predict(
                observation=_observation(),
                run_id="run",
                episode_id="ep",
                task_id="task",
                instruction="pick",
                step_index=0,
                control_hz=10,
            )
        self.assertEqual(batch.endpoint, "/predict_action")
        self.assertEqual(batch.action_horizon, 1)
        self.assertAlmostEqual(batch.latency_ms, 12.5)
        self.assertEqual(batch.gripper.tolist(), [-1.0])

    def test_predict_action_accepts_delta_7d_action(self) -> None:
        fake = _FakeRequests([{"action": [[0.01, 0.02, 0.03, 0.0, 0.0, 0.0, 1.0]]}])
        with patch("eai4s_eval.policy.http_client.requests", fake):
            batch = VLAHttpPolicyClient("127.0.0.1", 8188, api="predict_action").predict(
                observation=_observation(),
                run_id="run",
                episode_id="ep",
                task_id="task",
                instruction="pick",
                step_index=0,
                control_hz=10,
            )
        np.testing.assert_allclose(batch.ee_pos[0], [0.41, 0.02, 0.33], rtol=1e-5)
        self.assertEqual(batch.gripper.tolist(), [1.0])

    def test_auto_falls_back_from_act_to_predict_action(self) -> None:
        fake = _FakeRequests(
            [
                {"bad": "payload"},
                {"action": [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0]]},
            ]
        )
        with patch("eai4s_eval.policy.http_client.requests", fake):
            batch = VLAHttpPolicyClient("127.0.0.1", 8188, api="auto").predict(
                observation=_observation(),
                run_id="run",
                episode_id="ep",
                task_id="task",
                instruction="pick",
                step_index=0,
                control_hz=10,
            )
        self.assertEqual(batch.endpoint, "/predict_action")
        self.assertEqual([url.rsplit("/", 1)[-1] for url in fake.urls], ["act", "predict_action"])

    def test_invalid_action_payload_raises_clear_error(self) -> None:
        fake = _FakeRequests(
            [
                {
                    "action.single_arm": [[0.1, 0.2, 0.3]],
                    "action.gripper": [1.0],
                }
            ]
        )
        with patch("eai4s_eval.policy.http_client.requests", fake):
            with self.assertRaises(PolicyOutputInvalid):
                VLAHttpPolicyClient("127.0.0.1", 8188, api="act").predict(
                    observation=_observation(),
                    run_id="run",
                    episode_id="ep",
                    task_id="task",
                    instruction="pick",
                    step_index=0,
                    control_hz=10,
                )


class OpenPIHttpPolicyClientTest(unittest.TestCase):
    """Smoke tests for the OpenPI HTTP policy client."""

    def test_http_client_infer_returns_dict(self) -> None:
        """OpenPIHttpPolicyClient posts observation to /act and returns the parsed dict."""
        fake = _FakeRequests([{"actions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]}])
        with patch("eai4s_eval.policy.openpi_client.requests", fake):
            client = OpenPIHttpPolicyClient(host="127.0.0.1", port=8188)
            result = client.infer({"observation/image": np.zeros((4, 4, 3), dtype=np.uint8)})
        self.assertEqual(result["actions"], [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        self.assertIn("/act", fake.urls[0])

    def test_http_client_non_dict_raises_error(self) -> None:
        """OpenPIHttpPolicyClient raises PolicyOutputInvalid when the server returns a non-dict."""
        fake = _FakeRequests(["not_a_dict"])
        with patch("eai4s_eval.policy.openpi_client.requests", fake):
            client = OpenPIHttpPolicyClient(host="127.0.0.1", port=8188)
            with self.assertRaises(PolicyOutputInvalid):
                client.infer({"observation/image": np.zeros((4, 4, 3), dtype=np.uint8)})

    def test_http_backend_adapter_returns_joint_chunk(self) -> None:
        """OpenPIJointPolicyAdapter(backend='http') returns a valid JointActionChunk."""
        profile = OpenPIProfile(action_dim=8, max_horizon=16)
        fake = _FakeRequests([
            {"actions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                         [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]}
        ])
        with patch("eai4s_eval.policy.openpi_client.requests", fake):
            adapter = OpenPIJointPolicyAdapter(
                host="127.0.0.1",
                port=8188,
                profile=profile,
                backend="http",
            )
            self.assertEqual(adapter.health()["backend"], "http")

            state = {
                "camera_data": {
                    "camera_1_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                    "camera_3_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                },
                "joint_positions": np.zeros(8, dtype=np.float32),
                "language_instruction": "pick up the object",
            }
            chunk = adapter.predict(state)
            self.assertEqual(chunk.joint_positions.shape, (2, 8))
            self.assertEqual(chunk.action_horizon, 2)
            self.assertGreater(chunk.latency_ms, 0)

    def test_http_backend_health_includes_backend(self) -> None:
        """OpenPIJointPolicyAdapter health() includes backend field."""
        fake = _FakeRequests([{"actions": [[0.1] * 8]}])
        with patch("eai4s_eval.policy.openpi_client.requests", fake):
            adapter = OpenPIJointPolicyAdapter(
                host="127.0.0.1",
                port=8188,
                backend="http",
            )
            self.assertEqual(adapter.health()["backend"], "http")
            self.assertEqual(adapter.health()["host"], "127.0.0.1")
            self.assertEqual(adapter.health()["control_hz"], 10)

    def test_server_metadata_rejects_a_non_ten_hz_model(self) -> None:
        class _Client:
            def get_server_metadata(self):
                return {"control_hz": 60}

            def close(self):
                pass

        with self.assertRaisesRegex(PolicyClientError, "control_hz=60"):
            OpenPIJointPolicyAdapter(
                host="127.0.0.1",
                port=8188,
                client=_Client(),
                profile=OpenPIProfile(control_hz=10),
            )

    # ── gr00t observation mode tests ──────────────────────────────────────

    def _gr00t_state(self) -> dict:
        """Build a benchmark state dict compatible with gr00t observation mode."""
        return {
            "camera_data": {
                "camera_1_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "camera_3_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            },
            "agent_pose": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32),
            "language_instruction": "pick glass bottle",
        }

    def test_gr00t_observation_has_batch_dim_and_split_state(self) -> None:
        """gr00t mode adds batch dim to images, splits state, wraps language in list."""
        profile = OpenPIProfile(
            action_dim=8, max_horizon=16,
            observation_mode="gr00t",
            state_arm_key="state.single_arm", state_arm_dim=7,
            state_gripper_key="state.gripper",
            prompt_key="annotation.human.task_description",
            observation_map={
                "camera_1_rgb": "video.main_camera",
                "camera_3_rgb": "video.wrist_camera",
            },
        )
        # We need to inspect the observation before it is sent.
        # The adapter's client.infer is called from predict, so we patch it
        # and capture the observation.
        captured: dict = {}

        def _capture_infer(obs):
            captured.update(obs)
            return {
                "action.single_arm": [[0.0] * 7],
                "action.gripper": [[0.0]],
            }

        adapter = OpenPIJointPolicyAdapter(
            host="127.0.0.1", port=8188,
            profile=profile, backend="http",
        )
        with patch.object(adapter.client, "infer", _capture_infer):
            adapter.predict(self._gr00t_state())

        # Camera images with batch dim
        self.assertEqual(captured["video.main_camera"].shape, (1, 4, 4, 3))
        self.assertEqual(captured["video.wrist_camera"].shape, (1, 4, 4, 3))
        # State split: 7 arm + 1 gripper, both with batch dim
        np.testing.assert_array_equal(captured["state.single_arm"],
                                       np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], dtype=np.float32))
        np.testing.assert_array_equal(captured["state.gripper"],
                                       np.array([[0.8]], dtype=np.float32))
        # Language wrapped in list
        self.assertEqual(captured["annotation.human.task_description"], ["pick glass bottle"])

    def test_gr00t_observation_health_includes_observation_mode(self) -> None:
        """gr00t mode health() includes observation_mode field."""
        profile = OpenPIProfile(observation_mode="gr00t")
        fake = _FakeRequests([{"actions": [[0.0] * 8]}])
        with patch("eai4s_eval.policy.openpi_client.requests", fake):
            adapter = OpenPIJointPolicyAdapter(
                host="127.0.0.1", port=8188,
                profile=profile, backend="http",
            )
            self.assertEqual(adapter.health()["observation_mode"], "gr00t")

    def test_gr00t_action_squeezes_batch_dim(self) -> None:
        """gr00t mode squeezes (1, T, 7)+(1, T, 1) action responses to (T, 8)."""
        profile = OpenPIProfile(
            action_dim=8, max_horizon=16,
            observation_mode="gr00t",
            action_arm_key="action.single_arm",
            action_gripper_key="action.gripper",
        )
        # Batched response: (1, 2, 7) + (1, 2, 1)
        fake = _FakeRequests([
            {
                "action.single_arm": [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                       [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]],
                "action.gripper": [[[1.0], [0.0]]],
            }
        ])
        with patch("eai4s_eval.policy.openpi_client.requests", fake):
            adapter = OpenPIJointPolicyAdapter(
                host="127.0.0.1", port=8188,
                profile=profile, backend="http",
            )
            chunk = adapter.predict(self._gr00t_state())
        self.assertEqual(chunk.joint_positions.shape, (2, 8))
        expected_row0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.04], dtype=np.float32)
        expected_row1 = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(chunk.joint_positions[0], expected_row0)
        np.testing.assert_array_equal(chunk.joint_positions[1], expected_row1)

    def test_gr00t_action_split_concatenates_correctly(self) -> None:
        """gr00t mode concatenates action.single_arm (T, 7) + action.gripper (T, 1) → (T, 8)."""
        profile = OpenPIProfile(
            action_dim=8, max_horizon=16,
            observation_mode="gr00t",
            action_arm_key="action.single_arm",
            action_gripper_key="action.gripper",
        )
        # Non-batched response: (2, 7) + (2, 1)
        fake = _FakeRequests([
            {
                "action.single_arm": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                                      [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
                "action.gripper": [[1.0], [0.0]],
            }
        ])
        with patch("eai4s_eval.policy.openpi_client.requests", fake):
            adapter = OpenPIJointPolicyAdapter(
                host="127.0.0.1", port=8188,
                profile=profile, backend="http",
            )
            chunk = adapter.predict(self._gr00t_state())
        self.assertEqual(chunk.joint_positions.shape, (2, 8))
        expected_row0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.04], dtype=np.float32)
        expected_row1 = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(chunk.joint_positions[0], expected_row0)
        np.testing.assert_array_equal(chunk.joint_positions[1], expected_row1)

    def test_gr00t_action_broadcasts_gripper(self) -> None:
        """gr00t mode broadcasts gripper (1, 1) to match arm horizon (16, 7)."""
        profile = OpenPIProfile(
            action_dim=8, max_horizon=16,
            observation_mode="gr00t",
            action_arm_key="action.single_arm",
            action_gripper_key="action.gripper",
        )
        # arm: (16, 7), gripper: (1, 1) → broadcast to (16, 1)
        arm_actions = [[float(i + j) for j in range(7)] for i in range(16)]
        fake = _FakeRequests([{
            "action.single_arm": arm_actions,
            "action.gripper": [[1.0]],
        }])
        with patch("eai4s_eval.policy.openpi_client.requests", fake):
            adapter = OpenPIJointPolicyAdapter(
                host="127.0.0.1", port=8188,
                profile=profile, backend="http",
            )
            chunk = adapter.predict(self._gr00t_state())
        self.assertEqual(chunk.joint_positions.shape, (16, 8))
        np.testing.assert_array_equal(chunk.joint_positions[:, 7], np.float32(0.04))


if __name__ == "__main__":
    unittest.main()
