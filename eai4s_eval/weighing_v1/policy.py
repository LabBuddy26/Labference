"""Declared policy transports for feedback weighing.

The shared transport remains simulator-independent.  Weighing requests are
validated before they reach HTTP or OpenPI services.
"""

from __future__ import annotations

import base64
from dataclasses import replace
from io import BytesIO
import json

from eai4s_eval.oracle_runtime import InterventionSpec, PolicyProfile, content_digest

from eai4s_eval.centrifuge_v2.policy import (
    HttpArrayPolicy as _HttpArrayPolicy,
    HttpSymbolicPolicy as _HttpSymbolicPolicy,
    OpenPIArrayPolicy as _OpenPIArrayPolicy,
    PolicyServiceError,
    validate_action_chunk,
)

from .diagnostics import check_policy_request


class HttpSymbolicPolicy(_HttpSymbolicPolicy):
    """HTTP symbolic policy with the weighing input boundary enforced."""

    def decide(self, request: dict) -> dict:
        check_policy_request(request, allow_checkpoint="checkpoint_context" in request)
        return super().decide(request)


class HttpArrayPolicy(_HttpArrayPolicy):
    """HTTP native 8D policy with optional declared diagnostic reports."""

    def infer_full(self, request: dict) -> dict:
        check_policy_request(request, allow_checkpoint="checkpoint_context" in request)
        return super().infer_full(request)


class OpenPIArrayPolicy(_OpenPIArrayPolicy):
    """OpenPI adapter that also delivers the public scale state to the model."""

    def __init__(self, host: str, port: int, api_key: str | None = None,
                 *, profile: PolicyProfile | None = None,
                 intervention: InterventionSpec | None = None):
        default = PolicyProfile.legacy(
            {"name": "openpi_array", "action_space": "array", "oracle_transports": ["text"],
             "resettable": False}, ("front", "wrist", "display", "weights"))
        default = replace(default, input_mapping={
            "front": "camera_1_rgb", "wrist": "camera_3_rgb",
            "display": "display_rgb", "weights": "weights_rgb"})
        super().__init__(host, port, api_key, profile=profile or default, intervention=intervention)

    def infer_full(self, request: dict) -> dict:
        check_policy_request(request, allow_checkpoint="checkpoint_context" in request)
        if "display" not in request["images"]:
            raise PolicyServiceError("weighing OpenPI policy requires the display camera")
        import numpy as np
        from PIL import Image

        public_state = {
            "visible_weight_labels_g": request["visible_weight_labels_g"],
            "display_reading_g": request["display_reading_g"],
            "stable_indicator": request["stable_indicator"],
            "history": request["history"],
            "calibration": request["calibration"],
        }
        if "checkpoint_context" in request:
            public_state["checkpoint_context"] = request["checkpoint_context"]
        instruction = request["task_rule"] + "\nScale observation: " + json.dumps(
            public_state, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if "oracle_prompt" in request:
            instruction += "\n" + request["oracle_prompt"]
        if len(instruction.encode("utf-8")) > self.profile.max_prompt_bytes:
            raise PolicyServiceError("serialized prompt exceeds the declared policy limit")
        observation = {"state": np.asarray(request["joint_positions"][:8], dtype=np.float32),
                       "language_instruction": instruction, "prompt": instruction}
        observation["labference_request_digest"] = content_digest(request)
        observation["labference_prompt_digest"] = content_digest(instruction)
        observation["labference_intervention"] = getattr(
            self, "intervention", InterventionSpec()).__dict__
        mapping = dict(self.profile.input_mapping) or {
            "front": "camera_1_rgb", "wrist": "camera_3_rgb",
            "display": "display_rgb", "weights": "weights_rgb"}
        if set(mapping) != set(self.profile.sensors.rgb_channels):
            raise PolicyServiceError("OpenPI input mapping must cover every declared RGB channel exactly once")
        for camera, key in mapping.items():
            encoded = request["images"][camera]
            image = encoded if isinstance(encoded, str) else encoded["base64"]
            observation[key] = np.asarray(Image.open(BytesIO(base64.b64decode(image))).convert("RGB"))
        if "oracle_context" in request:
            observation["oracle_context"] = request["oracle_context"]
        if "checkpoint_context" in request:
            observation["checkpoint_context"] = request["checkpoint_context"]
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
                request, adapter_boundary="openpi_weighing_preprocess", receipt=receipt,
                serialized_prompt=instruction,
                model_input_digest=model_input_digest,
                adapter_details={
                    "input_mapping": mapping,
                    "image_shapes": {camera: list(observation[key].shape)
                                     for camera, key in mapping.items()},
                    "prompt_bytes": len(instruction.encode("utf-8")),
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


__all__ = [
    "PolicyServiceError", "validate_action_chunk", "HttpSymbolicPolicy",
    "HttpArrayPolicy", "OpenPIArrayPolicy",
]
