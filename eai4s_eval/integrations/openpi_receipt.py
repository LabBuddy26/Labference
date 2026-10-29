"""Server-side OpenPI wrapper for capability and exact-input acknowledgement."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from eai4s_eval.oracle_runtime import PolicyProfile, content_digest


class LabferenceOpenPIWrapper:
    """Wrap an OpenPI policy before passing it to WebsocketPolicyServer.

    The wrapper acknowledges preprocessing inputs.  It establishes delivery to
    the declared model boundary; behavioral uptake remains an evaluation result.
    """

    def __init__(self, policy: Any, profile: PolicyProfile, *, policy_revision: str | None = None):
        if profile.adapter != "openpi":
            raise ValueError("OpenPI wrapper requires an openpi policy profile")
        if not callable(getattr(policy, "infer", None)):
            raise TypeError("wrapped OpenPI policy must implement infer(observation)")
        self.policy = policy
        self.profile = profile
        self.policy_revision = None if policy_revision is None else policy_revision.strip()
        if policy_revision is not None and not self.policy_revision:
            raise ValueError("policy_revision must be a non-empty checkpoint identifier")

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        request_digest = observation.get("labference_request_digest")
        prompt_digest = observation.get("labference_prompt_digest")
        if not isinstance(request_digest, str) or not isinstance(prompt_digest, str):
            raise ValueError("Labference request and prompt digests are required")
        prompt = observation.get("prompt")
        if not isinstance(prompt, str) or content_digest(prompt) != prompt_digest:
            raise ValueError("OpenPI prompt differs from the declared serialized prompt")
        if len(prompt.encode("utf-8")) > self.profile.max_prompt_bytes:
            raise ValueError("OpenPI prompt exceeds the declared policy limit")
        mapping = dict(self.profile.input_mapping) or {
            channel: f"camera_{index + 1}_rgb"
            for index, channel in enumerate(self.profile.sensors.rgb_channels)
        }
        missing = [model_key for model_key in mapping.values() if model_key not in observation]
        if missing:
            raise ValueError(f"OpenPI observation is missing mapped image inputs: {missing}")
        model_input_digest = observation.get("labference_model_input_digest")
        model_input = {key: value for key, value in observation.items()
                       if key != "labference_model_input_digest"}
        if not isinstance(model_input_digest, str) or content_digest(model_input) != model_input_digest:
            raise ValueError("OpenPI model input differs from the declared transformed input")
        result = self.policy.infer(dict(observation))
        if not isinstance(result, Mapping):
            raise TypeError("wrapped OpenPI policy must return a mapping")
        return {**dict(result), "_labference_receipt": {
            "request_digest": request_digest,
            "prompt_digest": prompt_digest,
            "model_input_digest": model_input_digest,
            "intervention": observation.get("labference_intervention"),
            "accepted_rgb_channels": list(self.profile.sensors.rgb_channels),
            "accepted_depth_channels": list(self.profile.sensors.depth_channels),
            "accepted_input_mapping": mapping,
        }}

    def reset_with_receipt(self, seed: int) -> dict[str, Any]:
        """Forward a state-clearing reset and require auditable confirmation."""
        method = getattr(self.policy, "reset_with_receipt", None)
        if not callable(method):
            method = getattr(self.policy, "reset_episode", None)
        if not callable(method):
            raise RuntimeError(
                "stateful OpenPI policy must implement reset_with_receipt(seed) "
                "or reset_episode(seed)")
        result = method(seed)
        if not isinstance(result, Mapping):
            raise RuntimeError("wrapped OpenPI policy reset did not return a receipt")
        return dict(result)

    def metadata(self) -> dict[str, Any]:
        capabilities = {
            "schema_version": self.profile.schema_version,
            "rgb_channels": list(self.profile.sensors.rgb_channels),
            "depth_channels": list(self.profile.sensors.depth_channels),
            "action": asdict(self.profile.action),
            "oracle_transports": sorted(self.profile.oracle_transports),
            "replacement_sites": sorted(self.profile.replacement_sites),
            "max_prompt_bytes": self.profile.max_prompt_bytes,
            "reset": asdict(self.profile.reset),
            "agent_tools": list(self.profile.agent_tools),
            "episode_memory": self.profile.episode_memory,
            "history_protocol": self.profile.history_protocol,
        }
        if self.policy_revision is not None:
            capabilities["policy_revision"] = self.policy_revision
        return {"labference": capabilities}
