"""Auditable information support and accessible-interface replacement hooks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

from .contracts import InterventionSpec, PolicyProfile, check_capabilities
from .session import content_digest


@dataclass(frozen=True)
class InterventionRecord:
    opportunity_id: str
    evidence_step: int
    profile: str
    mode: str
    site: str
    status: str
    oracle_digest: str | None
    original_digest: str | None = None
    applied_digest: str | None = None
    reason: str | None = None

    def public(self) -> dict[str, Any]:
        return asdict(self)


class OracleValidator(Protocol):
    """Task-owned validation at the evaluator side of an intervention hook."""

    def validate_assist(self, profile: str, context: Mapping[str, Any],
                        observation: Mapping[str, Any]) -> None: ...

    def validate_replacement(self, profile: str, site: str, original: Any,
                             replacement: Any,
                             runtime_context: Mapping[str, Any]) -> None: ...


class InterventionController:
    """Apply an intervention only at its registered, externally visible boundary."""

    def __init__(self, profile: PolicyProfile, spec: InterventionSpec,
                 validator: OracleValidator | None = None):
        check = check_capabilities(profile, spec)
        handshake_only = check.errors == ("server metadata lacks a labference capability handshake",)
        if not check.compatible and not handshake_only:
            raise ValueError("; ".join(check.errors))
        self.profile = profile
        self.spec = spec
        self.validator = validator
        self.records: list[InterventionRecord] = []
        self._consumed: dict[tuple[str, str, str], tuple[str, Any]] = {}
        if spec.profile != "none" and validator is None:
            raise ValueError("an active intervention requires a task-owned Oracle validator")

    def assist(self, observation: Mapping[str, Any], oracle_context: Mapping[str, Any], *,
               opportunity_id: str, evidence_step: int) -> dict[str, Any]:
        if self.spec.mode != "assist":
            return dict(observation)
        if evidence_step < 0 or not opportunity_id or opportunity_id == "unopened":
            raise ValueError("assistance requires a registered decision opportunity")
        self.validator.validate_assist(self.spec.profile, oracle_context, observation)
        key = (opportunity_id, self.spec.mode, self.spec.site)
        digest = content_digest(oracle_context)
        if key in self._consumed:
            previous_digest, _ = self._consumed[key]
            if previous_digest != digest:
                raise ValueError("Oracle context changed within one decision opportunity")
        else:
            self._consumed[key] = (digest, dict(oracle_context))
            self.records.append(InterventionRecord(
                opportunity_id, evidence_step, self.spec.profile, self.spec.mode, self.spec.site,
                "delivered", digest))
        result = dict(observation)
        if "oracle_context" in result or "oracle_prompt" in result or "oracle_tool_result" in result:
            raise ValueError("Oracle support was already delivered")
        if self.spec.transport == "structured":
            result["oracle_context"] = dict(oracle_context)
        elif self.spec.transport == "text":
            import json
            result["oracle_prompt"] = "<ORACLE>" + json.dumps(
                oracle_context, sort_keys=True, separators=(",", ":"), allow_nan=False) + "</ORACLE>"
        elif self.spec.transport == "tool":
            result["oracle_tool_result"] = dict(oracle_context)
        else:
            raise ValueError("assist intervention has no transport")
        return result

    def replace(self, original_output: Any, oracle_output: Any, *, site: str,
                opportunity_id: str, evidence_step: int,
                runtime_context: Mapping[str, Any] | None = None) -> Any:
        if self.spec.mode != "replace" or site != self.spec.site:
            return original_output
        if evidence_step < 0 or not opportunity_id:
            raise ValueError("replacement requires a registered opportunity and evidence step")
        context = dict(runtime_context or {})
        self.validator.validate_replacement(
            self.spec.profile, site, original_output, oracle_output, context)
        key = (opportunity_id, self.spec.mode, self.spec.site)
        digest = content_digest(oracle_output)
        if key in self._consumed:
            previous_digest, previous_output = self._consumed[key]
            if previous_digest != digest:
                raise ValueError("replacement changed within one decision opportunity")
            return previous_output
        self._consumed[key] = (digest, oracle_output)
        self.records.append(InterventionRecord(
            opportunity_id, evidence_step, self.spec.profile, self.spec.mode, self.spec.site,
            "applied", digest, content_digest(original_output), digest))
        return oracle_output

    def reject_stale(self, *, opportunity_id: str, evidence_step: int, reason: str) -> None:
        self.records.append(InterventionRecord(
            opportunity_id, evidence_step, self.spec.profile, self.spec.mode, self.spec.site,
            "stale", None, reason=reason))

    def public_records(self) -> list[dict[str, Any]]:
        return [record.public() for record in self.records]
