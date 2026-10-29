from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - exercised in lightweight local envs.
    requests = None  # type: ignore

from .http_client import PolicyClientError, PolicyOutputInvalid


@dataclass(frozen=True)
class OpenPIProfile:
    profile_id: str = "openpi_franka_ee_pos_v1"
    observation_map: dict[str, str] = field(
        default_factory=lambda: {
            "camera_1_rgb": "observation/front_image",
            "camera_3_rgb": "observation/wrist_image",
        }
    )
    state_key: str = "observation/state"
    prompt_key: str = "prompt"
    action_keys: tuple[str, ...] = ("actions", "action")
    action_dim: int = 8
    action_space: str = "ee_pos"
    control_hz: int = 10
    max_horizon: int = 16
    n_obs_steps: int = 1
    observation_mode: str = "openpi"
    state_arm_key: str = "state.single_arm"
    state_arm_dim: int = 7
    state_gripper_key: str = "state.gripper"
    action_arm_key: str = "action.single_arm"
    action_gripper_key: str = "action.gripper"
    gripper_mode: str = "binary"
    gripper_closed_value: float = 0.0
    gripper_open_value: float = 0.04
    gripper_binary_threshold: float = 0.5

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OpenPIProfile":
        obs_map = dict(raw.get("observation_map") or {})
        return cls(
            profile_id=str(raw.get("profile_id") or cls.profile_id),
            observation_map={str(k): str(v) for k, v in obs_map.items()}
            or cls().observation_map,
            state_key=str(raw.get("state_key") or "observation/state"),
            prompt_key=str(raw.get("prompt_key") or "prompt"),
            action_keys=tuple(str(item) for item in (raw.get("action_keys") or ("actions", "action"))),
            action_dim=int(raw.get("action_dim", 8)),
            action_space=str(raw.get("action_space") or "ee_pos").lower(),
            control_hz=int(raw.get("control_hz", 10)),
            max_horizon=int(raw.get("max_horizon", 40)),
            n_obs_steps=int(raw.get("n_obs_steps", 1)),
            observation_mode=str(raw.get("observation_mode") or "openpi"),
            state_arm_key=str(raw.get("state_arm_key") or "state.single_arm"),
            state_arm_dim=int(raw.get("state_arm_dim", 7)),
            state_gripper_key=str(raw.get("state_gripper_key") or "state.gripper"),
            action_arm_key=str(raw.get("action_arm_key") or "action.single_arm"),
            action_gripper_key=str(raw.get("action_gripper_key") or "action.gripper"),
            gripper_mode=str(raw.get("gripper_mode") or "binary").lower(),
            gripper_closed_value=float(raw.get("gripper_closed_value", 0.0)),
            gripper_open_value=float(raw.get("gripper_open_value", 0.04)),
            gripper_binary_threshold=float(raw.get("gripper_binary_threshold", 0.5)),
        )


@dataclass
class JointActionChunk:
    joint_positions: np.ndarray
    latency_ms: float
    raw: dict[str, Any]

    @property
    def action_horizon(self) -> int:
        return int(self.joint_positions.shape[0])


def _to_hwc_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 4:
        array = array[-1]
    if array.ndim != 3:
        raise PolicyOutputInvalid(f"camera image must be CHW or HWC, got {array.shape}")
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    elif array.shape[-1] != 3:
        raise PolicyOutputInvalid(f"camera image channel dimension is invalid: {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        if array.size and float(np.max(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0)
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))


class OpenPIHttpPolicyClient:
    """HTTP client for OpenPI-format policies served over REST (e.g. GR00T /act).

    Implements the same duck-typed interface as ``WebsocketClientPolicy``
    (``infer`` / ``get_server_metadata`` / ``reset`` / ``close``) so that
    :class:`OpenPIJointPolicyAdapter` can use either transport.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        api_key: str | None = None,
        timeout_sec: float = 30.0,
    ):
        self.base_url = f"http://{host}:{port}"
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        try:
            import json_numpy  # type: ignore

            json_numpy.patch()
        except ImportError:
            pass

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if requests is None:
            raise PolicyClientError(
                "The 'requests' package is required for HTTP policy calls. "
                "Install it with: pip install requests"
            )
        import json_numpy  # type: ignore

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = json_numpy.dumps({"observation": observation})
        response = requests.post(
            f"{self.base_url}/act",
            data=payload,
            headers=headers,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise PolicyOutputInvalid(f"/act returned {type(data).__name__}, expected a mapping")
        return data

    def get_server_metadata(self) -> dict[str, Any]:
        # GR00T HTTP servers do not expose a metadata endpoint.
        return {}

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


def load_openpi_profile(path: str | Path | None) -> OpenPIProfile:
    if path is None:
        return OpenPIProfile()
    profile_path = Path(path)
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise PolicyClientError("PyYAML is required to load OpenPI policy profiles") from exc
    try:
        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8-sig")) or {}
    except OSError as exc:
        raise PolicyClientError(f"cannot read OpenPI profile {profile_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PolicyClientError(f"OpenPI profile must be a mapping: {profile_path}")
    return OpenPIProfile.from_mapping(payload)


class OpenPIJointPolicyAdapter:
    """Validated OpenPI policy adapter for benchmark joint action chunks.

    Supports two backends:
    - ``websocket`` (default) — connects to an OpenPI WebSocket server.
    - ``http`` — connects to an HTTP REST endpoint (``POST /act``).
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        profile: OpenPIProfile | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        backend: str = "websocket",
        default_instruction: str = "",
        timeout_sec: float = 30.0,
    ):
        self.host = str(host)
        self.port = int(port)
        self.backend = str(backend).lower()
        self.timeout_sec = float(timeout_sec)
        if self.timeout_sec <= 0:
            raise PolicyClientError("policy timeout_sec must be greater than zero")
        self.default_instruction = str(default_instruction)
        self.profile = profile or OpenPIProfile()
        if self.profile.n_obs_steps != 1:
            raise PolicyClientError("the benchmark adapter currently requires n_obs_steps=1")
        if client is None:
            if self.backend == "http":
                client = OpenPIHttpPolicyClient(
                    host=self.host,
                    port=self.port,
                    api_key=api_key,
                    timeout_sec=self.timeout_sec,
                )
            elif self.backend == "websocket":
                from .openpi_websocket_client import OpenPIWebSocketPolicyClient

                client = OpenPIWebSocketPolicyClient(
                    host=self.host,
                    port=self.port,
                    api_key=api_key,
                    timeout_sec=self.timeout_sec,
                )
            else:
                raise PolicyClientError(
                    f"unknown policy backend: {self.backend!r} (expected 'websocket' or 'http')"
                )
        self.client = client
        self.server_metadata = self._read_metadata()
        self._validate_metadata(self.server_metadata)
        self.last_observation: dict[str, Any] | None = None

    def _read_metadata(self) -> dict[str, Any]:
        getter = getattr(self.client, "get_server_metadata", None)
        if getter is None:
            return {}
        metadata = getter()
        return dict(metadata) if isinstance(metadata, Mapping) else {"value": metadata}

    def _validate_metadata(self, metadata: Mapping[str, Any]) -> None:
        advertised_control_hz = metadata.get("control_hz")
        if (
            advertised_control_hz is not None
            and int(advertised_control_hz) != self.profile.control_hz
        ):
            raise PolicyClientError(
                f"policy server advertises control_hz={advertised_control_hz}, "
                f"expected {self.profile.control_hz}"
            )
        advertised_dim = metadata.get("action_dim") or metadata.get("action_dimension")
        if advertised_dim is not None and int(advertised_dim) != self.profile.action_dim:
            raise PolicyClientError(
                f"OpenPI server advertises action_dim={advertised_dim}, expected {self.profile.action_dim}"
            )
        advertised_keys = metadata.get("observation_keys")
        if isinstance(advertised_keys, (list, tuple, set)):
            required = set(self.profile.observation_map.values()) | {self.profile.state_key, self.profile.prompt_key}
            missing = sorted(required.difference(str(item) for item in advertised_keys))
            if missing:
                raise PolicyClientError(f"OpenPI server metadata is missing observation keys: {', '.join(missing)}")

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "backend": self.backend,
            "observation_mode": self.profile.observation_mode,
            "host": self.host,
            "port": self.port,
            "profile_id": self.profile.profile_id,
            "action_dim": self.profile.action_dim,
            "action_space": self.profile.action_space,
            "control_hz": self.profile.control_hz,
            "timeout_sec": self.timeout_sec,
            "server_metadata": self.server_metadata,
        }

    def reset(self) -> None:
        self.last_observation = None
        reset = getattr(self.client, "reset", None)
        if callable(reset):
            reset()

    def _observation(self, state: Mapping[str, Any]) -> dict[str, Any]:
        camera_data = state.get("camera_data") or {}
        observation: dict[str, Any] = {}
        missing: list[str] = []
        for source, target in self.profile.observation_map.items():
            if source not in camera_data:
                missing.append(source)
                continue
            image = _to_hwc_uint8(camera_data[source])
            if self.profile.observation_mode == "gr00t":
                image = image[np.newaxis, ...]  # (H, W, 3) → (1, H, W, 3)
            observation[target] = image
        if missing:
            raise PolicyOutputInvalid(
                f"benchmark state is missing policy camera streams: {', '.join(missing)}; "
                f"available: {', '.join(sorted(camera_data)) or '(none)'}"
            )
        joint_positions = np.asarray(state.get("agent_pose", state.get("joint_positions")), dtype=np.float32).reshape(-1)
        if joint_positions.size < self.profile.action_dim:
            raise PolicyOutputInvalid(
                f"robot state has {joint_positions.size} joints, expected at least {self.profile.action_dim}"
            )
        if self.profile.observation_mode == "gr00t":
            arm = joint_positions[: self.profile.state_arm_dim][np.newaxis, ...]
            gripper = joint_positions[self.profile.state_arm_dim : self.profile.action_dim][np.newaxis, ...]
            observation[self.profile.state_arm_key] = arm
            observation[self.profile.state_gripper_key] = gripper
        else:
            observation[self.profile.state_key] = joint_positions[: self.profile.action_dim]
        instruction = str(state.get("language_instruction") or self.default_instruction)
        if self.profile.observation_mode == "gr00t":
            observation[self.profile.prompt_key] = [instruction]
        else:
            observation[self.profile.prompt_key] = instruction
        return observation

    def predict(self, state: Mapping[str, Any]) -> JointActionChunk:
        observation = self._observation(state)
        self.last_observation = observation
        started = time.perf_counter()
        try:
            result = self.client.infer(observation)
        except Exception as exc:
            raise PolicyClientError(f"{self.backend} policy inference failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not isinstance(result, Mapping):
            raise PolicyOutputInvalid(f"policy response must be a mapping, got {type(result).__name__}")
        if self.profile.observation_mode == "gr00t":
            arm = np.asarray(result.get(self.profile.action_arm_key), dtype=np.float32)
            gripper = np.asarray(result.get(self.profile.action_gripper_key), dtype=np.float32)
            if arm.ndim == 3 and arm.shape[0] == 1:
                arm = arm[0]
            if gripper.ndim == 3 and gripper.shape[0] == 1:
                gripper = gripper[0]
            if arm.ndim == 1:
                arm = arm.reshape(1, -1)
            if gripper.ndim == 1:
                # (T,) → (T, 1): keep timesteps, add feature dim
                gripper = gripper.reshape(-1, 1)
            if arm.ndim != 2 or gripper.ndim != 2:
                raise PolicyOutputInvalid(
                    f"gr00t action keys must be 2D, got arm={arm.ndim}D, gripper={gripper.ndim}D"
                )
            if arm.shape[0] != gripper.shape[0]:
                # Broadcast the shorter key to match the longer horizon.
                # GR00T models often emit a single gripper value for the
                # whole trajectory while arm actions have one per step.
                if gripper.shape[0] == 1 and arm.shape[0] > 1:
                    gripper = np.repeat(gripper, arm.shape[0], axis=0)
                elif arm.shape[0] == 1 and gripper.shape[0] > 1:
                    arm = np.repeat(arm, gripper.shape[0], axis=0)
                else:
                    raise PolicyOutputInvalid(
                        f"gr00t action horizon mismatch: arm={arm.shape[0]}, gripper={gripper.shape[0]}"
                    )
            if arm.shape[1] != self.profile.state_arm_dim or gripper.shape[1] != self.profile.action_dim - self.profile.state_arm_dim:
                raise PolicyOutputInvalid(
                    f"gr00t action dims must be arm={self.profile.state_arm_dim}, gripper={self.profile.action_dim - self.profile.state_arm_dim}, "
                    f"got arm={arm.shape[1]}, gripper={gripper.shape[1]}"
                )
            actions = np.concatenate([arm, gripper], axis=-1)
        else:
            raw_action = None
            for key in self.profile.action_keys:
                if key in result:
                    raw_action = result[key]
                    break
            if raw_action is None:
                raise PolicyOutputInvalid(
                    f"OpenPI response contains none of the configured action keys: {', '.join(self.profile.action_keys)}"
                )
            # OpenPI's MessagePack decoder can expose arrays backed directly
            # by the received bytes, which are read-only.  The binary gripper
            # conversion below intentionally updates the last column in place.
            actions = np.array(raw_action, dtype=np.float32, copy=True)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2 or actions.shape[1] != self.profile.action_dim:
            raise PolicyOutputInvalid(
                f"policy joint action must have shape [T, {self.profile.action_dim}], got {actions.shape}"
            )
        if actions.shape[0] < 1:
            raise PolicyOutputInvalid("policy returned an empty action chunk")
        if not np.isfinite(actions).all():
            raise PolicyOutputInvalid("policy joint action contains NaN or infinity")
        if self.profile.gripper_mode == "binary":
            binary = np.clip(actions[:, -1], 0.0, 1.0)
            actions[:, -1] = np.where(
                binary >= self.profile.gripper_binary_threshold,
                self.profile.gripper_open_value,
                self.profile.gripper_closed_value,
            )
        elif self.profile.gripper_mode != "raw":
            raise PolicyOutputInvalid(
                f"unsupported gripper_mode={self.profile.gripper_mode!r}; expected 'binary' or 'raw'"
            )
        actions = np.ascontiguousarray(actions[: self.profile.max_horizon], dtype=np.float32)
        reported_latency = result.get("model_latency_ms")
        if reported_latency is not None:
            latency_ms = float(reported_latency)
        return JointActionChunk(joint_positions=actions, latency_ms=latency_ms, raw=dict(result))

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


__all__ = [
    "JointActionChunk",
    "OpenPIHttpPolicyClient",
    "OpenPIJointPolicyAdapter",
    "OpenPIProfile",
    "load_openpi_profile",
]
