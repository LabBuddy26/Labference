"""Declared symbolic policy transport for diagnostic rollouts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.request import Request, urlopen

from eai4s_eval.oracle_runtime import (
    ActionSpec, InterventionSpec, PolicyProfile, PolicySession, content_digest,
)


class PolicyServiceError(RuntimeError):
    pass


class HttpSymbolicPolicy:
    def __init__(self, url: str, timeout_s: float = 30.0):
        if not url.startswith(("http://", "https://")) or timeout_s <= 0:
            raise ValueError("policy URL must be HTTP(S) and timeout positive")
        self.url = url
        self.timeout_s = timeout_s

    def _post(self, payload: dict) -> dict:
        request = Request(self.url, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                result = json.load(response)
        except Exception as exc:
            raise PolicyServiceError(str(exc)) from exc
        if not isinstance(result, dict):
            raise PolicyServiceError("policy response must be a JSON object")
        return result

    def reset(self, seed: int) -> None:
        response = self._post({"command": "reset", "seed": seed})
        if response.get("status") != "ok":
            raise PolicyServiceError("policy reset did not return status=ok")

    def decide(self, request: dict) -> dict:
        return self._post({"command": "decide", "request": request})


def validate_action_chunk(payload: object, action_spec: ActionSpec | None = None) -> list[list[float]]:
    """Validate against the declared convention; legacy callers retain 8D joints."""
    try:
        return (action_spec or ActionSpec.legacy("array")).validate_chunk(payload)
    except ValueError as exc:
        raise PolicyServiceError(str(exc)) from exc


class HttpArrayPolicy(HttpSymbolicPolicy):
    """JSON transport for testing the native 8D action contract online."""

    def infer_full(self, request: dict) -> dict:
        result = self._post({"command": "infer", "request": request})
        return {"action_chunk": validate_action_chunk(result.get("action", result.get("actions"))),
                **{key: result[key] for key in ("semantic_report", "committed_action", "geometry_report",
                                                "server_timing", "policy_timing")
                   if key in result}}

    def infer(self, request: dict) -> list[list[float]]:
        return self.infer_full(request)["action_chunk"]


class OpenPIArrayPolicy:
    """Use the same WebsocketClientPolicy and observation keys as the main repository."""

    def __init__(self, host: str, port: int, api_key: str | None = None,
                 *, profile: PolicyProfile | None = None,
                 intervention: InterventionSpec | None = None):
        try:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
        except ImportError as exc:
            raise PolicyServiceError("openpi_client is required for openpi_array") from exc
        self.client = WebsocketClientPolicy(host=host, port=port, api_key=api_key)
        self.server_metadata = self.client.get_server_metadata()
        self.profile = profile or PolicyProfile.legacy(
            {"name": "openpi_array", "action_space": "array", "oracle_transports": ["text"],
             "resettable": False}, ("front", "wrist"))
        self.intervention = intervention or InterventionSpec()
        self.session = PolicySession(
            self.profile, self.intervention, server_metadata=self.server_metadata,
            reset_callback=self._reset_client)
        self.last_delivery_audit = None

    def _reset_client(self, seed: int) -> Mapping[str, object]:
        """Require a verifiable server reset for stateful policy profiles."""
        for method_name in ("reset_with_receipt", "reset_session"):
            method = getattr(self.client, method_name, None)
            if callable(method):
                result = method(seed)
                if isinstance(result, Mapping):
                    return result
                raise PolicyServiceError(
                    f"OpenPI {method_name} did not return a reset receipt")
        result = self.client.reset()
        if isinstance(result, Mapping):
            return result
        raise PolicyServiceError(
            "stateful OpenPI evaluation requires a server wrapper exposing "
            "reset_with_receipt(seed) or reset_session(seed)")

    def reset(self, seed: int) -> None:
        try:
            self.session.reset(seed)
        except Exception as exc:
            raise PolicyServiceError(f"OpenPI reset failed: {exc}") from exc

    def infer_full(self, request: dict) -> dict:
        import base64
        from io import BytesIO
        import numpy as np
        from PIL import Image

        observation = {"state": np.asarray(request["joint_positions"][:8], dtype=np.float32)}
        mapping = dict(self.profile.input_mapping) or {"front": "camera_1_rgb", "wrist": "camera_3_rgb"}
        if set(mapping) != set(self.profile.sensors.rgb_channels):
            raise PolicyServiceError("OpenPI input mapping must cover every declared RGB channel exactly once")
        for camera, key in mapping.items():
            image = request["images"][camera]
            encoded = image if isinstance(image, str) else image["base64"]
            observation[key] = np.asarray(Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB"))
        instruction = request["task_rule"]
        if "oracle_prompt" in request:
            instruction += "\n" + request["oracle_prompt"]
        if len(instruction.encode("utf-8")) > self.profile.max_prompt_bytes:
            raise PolicyServiceError("serialized prompt exceeds the declared policy limit")
        observation["language_instruction"] = instruction
        observation["prompt"] = instruction
        observation["labference_request_digest"] = content_digest(request)
        observation["labference_prompt_digest"] = content_digest(instruction)
        observation["labference_intervention"] = getattr(
            self, "intervention", InterventionSpec()).__dict__
        if "oracle_context" in request:
            observation["oracle_context"] = request["oracle_context"]
        if "selected_operation" in request:
            observation["selected_operation"] = request["selected_operation"]
            observation["interaction_pose"] = request["interaction_pose"]
        model_input_digest = content_digest(observation)
        observation["labference_model_input_digest"] = model_input_digest
        try:
            result = self.client.infer(observation)
            if not isinstance(result, dict):
                raise PolicyServiceError("OpenPI response must be a mapping")
            action = result.get("action", result.get("actions"))
            receipt = result.get("_labference_receipt")
            session = getattr(self, "session", None)
            audit = None if session is None else session.audit_request(
                request, adapter_boundary="openpi_preprocess", receipt=receipt,
                serialized_prompt=instruction,
                model_input_digest=model_input_digest,
                adapter_details={
                    "input_mapping": mapping,
                    "image_shapes": {camera: list(observation[key].shape)
                                     for camera, key in mapping.items()},
                    "prompt_bytes": len(instruction.encode("utf-8")),
                    "episode_memory": self.profile.episode_memory,
                    "history_protocol": self.profile.history_protocol,
                    "observation_id": request.get("observation_id"),
                    "frame_ids": request.get("frame_ids"),
                    "sim_time_s": request.get("sim_time_s"),
                })
            self.last_delivery_audit = audit
            full = {"action_chunk": validate_action_chunk(np.asarray(action).tolist(), self.profile.action)}
            if audit is not None:
                full["input_delivery_audit"] = audit.public()
            full.update({key: result[key] for key in
                         ("semantic_report", "committed_action", "geometry_report",
                          "server_timing", "policy_timing") if key in result})
            return full
        except PolicyServiceError:
            raise
        except Exception as exc:
            raise PolicyServiceError(f"OpenPI inference failed: {exc}") from exc

    def infer(self, request: dict) -> list[list[float]]:
        return self.infer_full(request)["action_chunk"]
