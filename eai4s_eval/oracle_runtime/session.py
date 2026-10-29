"""Policy lifecycle and end-to-end input-delivery audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import hashlib
import json
from typing import Any, Callable, Mapping

from .contracts import CapabilityCheck, InterventionSpec, PolicyProfile, check_capabilities


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        raw = value.tobytes()
        return {"ndarray_sha256": hashlib.sha256(raw).hexdigest(),
                "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        raise ValueError("cannot hash non-finite runtime content")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def content_digest(value: Any) -> str:
    raw = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InputDeliveryAudit:
    request_digest: str
    prompt_digest: str | None
    oracle_digest: str | None
    rgb_channels: tuple[str, ...]
    depth_channels: tuple[str, ...]
    adapter_boundary: str
    receipt_status: str
    receipt_digest: str | None = None
    receipt_prompt_digest: str | None = None
    model_input_digest: str | None = None
    receipt_model_input_digest: str | None = None
    adapter_details: Mapping[str, Any] | None = None

    def public(self) -> dict[str, Any]:
        return asdict(self)


class PolicySession:
    """Wrap a policy service with capability, reset, and delivery checks."""

    def __init__(self, profile: PolicyProfile, intervention: InterventionSpec,
                 *, server_metadata: Mapping[str, Any] | None = None,
                 reset_callback: Callable[[int], Any] | None = None):
        self.profile = profile
        self.intervention = intervention
        self.server_metadata = dict(server_metadata or {})
        self.capability_check: CapabilityCheck = check_capabilities(
            profile, intervention, self.server_metadata)
        self.reset_callback = reset_callback
        self.action_buffer: list[Any] = []
        self.audits: list[InputDeliveryAudit] = []
        self.reset_receipts: list[dict[str, Any]] = []
        self._last_session_id: str | None = None
        if not self.capability_check.compatible:
            raise ValueError("; ".join(self.capability_check.errors))

    def reset(self, seed: int) -> dict[str, Any]:
        self.action_buffer.clear()
        if self.profile.reset.mode == "stateless":
            receipt = {"status": "stateless_declared", "seed": seed, "session_id": None,
                       "action_buffer_cleared": True,
                       "agent_context_cleared": self.profile.reset.clear_agent_context,
                       "episode_memory_cleared": self.profile.reset.clear_episode_memory}
            self.reset_receipts.append(receipt)
            return receipt
        if self.reset_callback is None:
            raise RuntimeError(f"{self.profile.reset.mode} reset requires an implementation")
        result = self.reset_callback(seed)
        if not isinstance(result, Mapping) or result.get("status") != "ok" or result.get("seed") != seed:
            raise RuntimeError("policy reset did not return a matching acknowledgement")
        session_id = result.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("policy reset acknowledgement needs a session_id")
        if self.profile.reset.mode == "restart" and session_id == self._last_session_id:
            raise RuntimeError("restart reset reused the previous policy session")
        requirements = {
            "action_buffer_cleared": self.profile.reset.clear_action_buffer,
            "agent_context_cleared": self.profile.reset.clear_agent_context,
            "episode_memory_cleared": self.profile.reset.clear_episode_memory,
        }
        for field, required in requirements.items():
            if required and result.get(field) is not True:
                raise RuntimeError(f"policy reset did not confirm {field}")
        receipt = dict(result)
        self._last_session_id = session_id
        self.reset_receipts.append(receipt)
        return receipt

    def audit_request(self, request: Mapping[str, Any], *, adapter_boundary: str,
                      receipt: Mapping[str, Any] | None = None,
                      serialized_prompt: str | None = None,
                      model_input_digest: str | None = None,
                      adapter_details: Mapping[str, Any] | None = None) -> InputDeliveryAudit:
        images = request.get("images", {})
        depth = request.get("depth", {})
        if not isinstance(images, Mapping) or set(self.profile.sensors.rgb_channels) != set(images):
            raise ValueError("request RGB channels differ from the policy profile")
        if not isinstance(depth, Mapping) or set(self.profile.sensors.depth_channels) != set(depth):
            raise ValueError("request depth channels differ from the policy profile")
        oracle = request.get("oracle_context", request.get("oracle_prompt", request.get("oracle_tool_result")))
        request_digest = content_digest(request)
        prompt_digest = None if serialized_prompt is None else content_digest(serialized_prompt)
        receipt_digest = None if receipt is None else receipt.get("request_digest")
        receipt_prompt_digest = None if receipt is None else receipt.get("prompt_digest")
        receipt_model_input_digest = None if receipt is None else receipt.get("model_input_digest")
        status = "not_required"
        if self.profile.receipt_required:
            if receipt_digest != request_digest:
                raise ValueError("policy service did not acknowledge the exact serialized request")
            if prompt_digest is not None and receipt_prompt_digest != prompt_digest:
                raise ValueError("policy service did not acknowledge the exact serialized prompt")
            if model_input_digest is not None and receipt_model_input_digest != model_input_digest:
                raise ValueError("policy service did not acknowledge the exact transformed model input")
            status = "verified"
        elif receipt is not None:
            request_matches = receipt_digest == request_digest
            prompt_matches = prompt_digest is None or receipt_prompt_digest == prompt_digest
            model_input_matches = (model_input_digest is None or
                                   receipt_model_input_digest == model_input_digest)
            status = "verified" if request_matches and prompt_matches and model_input_matches else "mismatch"
        audit = InputDeliveryAudit(
            request_digest, prompt_digest, None if oracle is None else content_digest(oracle),
            self.profile.sensors.rgb_channels, self.profile.sensors.depth_channels,
            adapter_boundary, status, receipt_digest, receipt_prompt_digest,
            model_input_digest, receipt_model_input_digest,
            None if adapter_details is None else dict(adapter_details))
        self.audits.append(audit)
        return audit
