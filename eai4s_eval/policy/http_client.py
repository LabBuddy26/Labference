from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - exercised in lightweight local envs.
    requests = None  # type: ignore

from eai4s_eval.utils.math import quat_multiply_wxyz, rotvec_to_quat_wxyz


class PolicyClientError(RuntimeError):
    """Base policy client error."""


class PolicyOutputInvalid(PolicyClientError):
    """Raised when a model response cannot be converted to robot actions."""


@dataclass
class ActionBatch:
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    gripper: np.ndarray
    raw: dict[str, Any]
    action_horizon: int
    latency_ms: float
    endpoint: str


class VLAHttpPolicyClient:
    def __init__(self, host: str, port: int, api: str = "auto", timeout_sec: float = 30.0):
        if api not in {"auto", "act", "predict_action"}:
            raise ValueError("policy_api must be one of: auto, act, predict_action")
        self.base_url = f"http://{host}:{port}"
        self.api = api
        self.timeout_sec = timeout_sec
        try:
            import json_numpy  # type: ignore

            json_numpy.patch()
        except Exception:
            pass

    def predict(
        self,
        *,
        observation: dict[str, Any],
        run_id: str,
        episode_id: str,
        task_id: str,
        instruction: str,
        step_index: int,
        control_hz: int,
    ) -> ActionBatch:
        if self.api == "act":
            return self._act(observation["legacy"])
        if self.api == "auto":
            try:
                return self._act(observation["legacy"])
            except Exception:
                return self._predict_action(
                    observation=observation,
                    run_id=run_id,
                    episode_id=episode_id,
                    task_id=task_id,
                    instruction=instruction,
                    step_index=step_index,
                    control_hz=control_hz,
                )
        if self.api == "predict_action":
            try:
                return self._predict_action(
                    observation=observation,
                    run_id=run_id,
                    episode_id=episode_id,
                    task_id=task_id,
                    instruction=instruction,
                    step_index=step_index,
                    control_hz=control_hz,
                )
            except Exception:
                raise
        raise ValueError("policy_api must be one of: auto, act, predict_action")

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        if requests is None:
            raise PolicyClientError(
                "The 'requests' package is required for HTTP policy calls. "
                "Install it with: pip install requests"
            )
        url = f"{self.base_url}{endpoint}"
        start = time.time()
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_sec)
        except requests.exceptions.ConnectionError as exc:
            raise PolicyClientError(
                f"Cannot connect to VLA policy server at {url}. "
                f"Ensure the server is running and reachable. ({exc})"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise PolicyClientError(
                f"VLA policy server at {url} timed out after {self.timeout_sec}s. "
                f"Consider increasing --policy_timeout_sec or checking the server. ({exc})"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise PolicyClientError(
                f"HTTP request to VLA policy server at {url} failed: {exc}"
            ) from exc
        latency_ms = (time.time() - start) * 1000.0
        response.raise_for_status()
        data = response.json()
        if data is None:
            raise PolicyOutputInvalid(f"{endpoint} returned null")
        return data, latency_ms

    def _act(self, legacy_observation: dict[str, Any]) -> ActionBatch:
        data, latency_ms = self._post_json("/act", {"observation": legacy_observation})
        if "action.single_arm" not in data or "action.gripper" not in data:
            raise PolicyOutputInvalid("/act response must contain action.single_arm and action.gripper")

        action_ee = np.asarray(data["action.single_arm"], dtype=np.float32)
        gripper = np.asarray(data["action.gripper"], dtype=np.float32).reshape(-1)
        if action_ee.ndim != 2 or action_ee.shape[1] < 7:
            raise PolicyOutputInvalid("action.single_arm must have shape [T, >=7]")
        if gripper.shape[0] != action_ee.shape[0]:
            raise PolicyOutputInvalid("action.gripper horizon must match action.single_arm")

        return ActionBatch(
            ee_pos=action_ee[:, :3],
            ee_quat=action_ee[:, 3:7],
            gripper=gripper,
            raw={"ee_pos": action_ee[:, :3].tolist(), "ee_quat": action_ee[:, 3:7].tolist(), "gripper": gripper.tolist()},
            action_horizon=int(action_ee.shape[0]),
            latency_ms=latency_ms,
            endpoint="/act",
        )

    def _predict_action(
        self,
        *,
        observation: dict[str, Any],
        run_id: str,
        episode_id: str,
        task_id: str,
        instruction: str,
        step_index: int,
        control_hz: int,
    ) -> ActionBatch:
        standard = observation["standard"]
        payload = {
            "run_id": run_id,
            "episode_id": episode_id,
            "task_id": task_id,
            "instruction": instruction,
            "observation": standard["observation"],
            "metadata": {
                "control_hz": control_hz,
                "step_index": step_index,
                "observation_track": "rgb_ego",
            },
        }
        data, latency_ms = self._post_json("/predict_action", payload)
        if "model_latency_ms" in data:
            latency_ms = float(data["model_latency_ms"])
        action = np.asarray(data.get("action"), dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if action.ndim != 2 or action.shape[1] not in {7, 8}:
            raise PolicyOutputInvalid("/predict_action action must have shape [T, 7] or [T, 8]")

        current_pose = np.asarray(standard["observation"]["robot_state"]["eef_pose"], dtype=np.float64)
        if current_pose.shape[0] < 7:
            raise PolicyOutputInvalid("observation robot_state.eef_pose must contain [pos3, quat4]")

        if action.shape[1] == 8:
            ee_pos = action[:, :3]
            ee_quat = action[:, 3:7]
            gripper = action[:, 7]
        else:
            ee_pos = []
            ee_quat = []
            base_pos = current_pose[:3]
            base_quat = current_pose[3:7]
            for row in action:
                ee_pos.append(base_pos + row[:3])
                delta_quat = rotvec_to_quat_wxyz(row[3:6])
                ee_quat.append(quat_multiply_wxyz(delta_quat, base_quat))
            ee_pos = np.asarray(ee_pos, dtype=np.float32)
            ee_quat = np.asarray(ee_quat, dtype=np.float32)
            gripper = action[:, 6]

        return ActionBatch(
            ee_pos=np.asarray(ee_pos, dtype=np.float32),
            ee_quat=np.asarray(ee_quat, dtype=np.float32),
            gripper=np.asarray(gripper, dtype=np.float32).reshape(-1),
            raw={"response": data, "action": action.tolist()},
            action_horizon=int(action.shape[0]),
            latency_ms=latency_ms,
            endpoint="/predict_action",
        )
