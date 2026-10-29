"""Versioned contracts shared by native VLA, agent, and harness adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "labference/policy-profile-v1"
INTERVENTION_VERSION = "labference/intervention-v1"
SYSTEM_TYPES = {"native_vla", "agent", "evaluation_harness", "symbolic"}
ADAPTERS = {"openpi", "http", "rpent", "vla_evaluation_harness", "rule_expert"}
ORACLE_PROFILES = {"none", "semantic", "decision", "geometry", "skill", "history"}
INTERVENTION_MODES = {"none", "assist", "replace"}
ORACLE_TRANSPORTS = {"none", "structured", "text", "tool"}
EPISODE_MEMORY_MODES = {"none", "model_state", "agent_context", "explicit_memory"}
HISTORY_PROTOCOLS = {"current_frame_only", "online_once"}
REPLACEMENT_SITES = {
    "semantic": "semantic_report",
    "decision": "committed_action",
    "geometry": "geometry_report",
    "skill": "executor",
    "history": "history",
}


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class ActionSpec:
    """The exact action convention consumed by the simulator adapter."""

    representation: str
    dimensions: int
    mode: str
    frame: str
    units: str
    gripper: str
    control_hz: float
    max_chunk: int = 40

    def __post_init__(self) -> None:
        if self.representation not in {"joint_position", "eef_pose", "eef_delta", "symbolic"}:
            raise ValueError("unknown action representation")
        if self.mode not in {"absolute", "delta", "symbolic"}:
            raise ValueError("unknown action mode")
        if type(self.dimensions) is not int or self.dimensions < 0:
            raise ValueError("action dimensions must be a non-negative integer")
        if self.representation == "symbolic" and (self.dimensions != 0 or self.mode != "symbolic"):
            raise ValueError("symbolic actions must have zero dimensions and symbolic mode")
        if self.representation != "symbolic" and self.dimensions <= 0:
            raise ValueError("numeric actions require positive dimensions")
        if not all(isinstance(value, str) and value.strip() for value in
                   (self.frame, self.units, self.gripper)):
            raise ValueError("action frame, units, and gripper convention are required")
        if not _finite(self.control_hz) or self.control_hz <= 0 or type(self.max_chunk) is not int or self.max_chunk <= 0:
            raise ValueError("action rate and chunk limit must be positive")

    @classmethod
    def legacy(cls, action_space: str) -> "ActionSpec":
        if action_space == "array":
            return cls("joint_position", 8, "absolute", "franka_joint",
                       "radians+normalized_gripper", "symmetric_finger_position", 120.0, 40)
        if action_space == "symbolic":
            return cls("symbolic", 0, "symbolic", "task", "operation", "operation", 1.0, 1)
        raise ValueError("legacy action space must be array or symbolic")

    def validate_chunk(self, payload: Any) -> list[list[float]]:
        if self.representation == "symbolic":
            raise ValueError("symbolic action specs do not accept numeric chunks")
        if not isinstance(payload, (list, tuple)) or not 1 <= len(payload) <= self.max_chunk:
            raise ValueError(f"action chunk must contain 1-{self.max_chunk} waypoints")
        result = []
        for waypoint in payload:
            if (not isinstance(waypoint, (list, tuple)) or len(waypoint) != self.dimensions or
                    any(not _finite(value) for value in waypoint)):
                raise ValueError(f"every action waypoint must contain {self.dimensions} finite numbers")
            result.append([float(value) for value in waypoint])
        return result

    @property
    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SensorSpec:
    rgb_channels: tuple[str, ...]
    depth_channels: tuple[str, ...] = ()
    state_format: str = "joint_positions"

    def __post_init__(self) -> None:
        channels = self.rgb_channels + self.depth_channels
        if not self.rgb_channels or len(set(channels)) != len(channels):
            raise ValueError("sensor channels must be non-empty and unique")
        if any(not isinstance(value, str) or not value.strip() for value in channels):
            raise ValueError("sensor channel names must be non-empty strings")
        if not isinstance(self.state_format, str) or not self.state_format.strip():
            raise ValueError("state format is required")


@dataclass(frozen=True)
class ResetSpec:
    mode: str = "stateless"
    clear_action_buffer: bool = True
    clear_agent_context: bool = True
    clear_episode_memory: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"stateless", "rpc", "restart"}:
            raise ValueError("reset mode must be stateless, rpc, or restart")


@dataclass(frozen=True)
class PolicyProfile:
    name: str
    system_type: str
    adapter: str
    action: ActionSpec
    sensors: SensorSpec
    reports: frozenset[str] = frozenset()
    oracle_transports: frozenset[str] = frozenset()
    replacement_sites: frozenset[str] = frozenset()
    reset: ResetSpec = field(default_factory=ResetSpec)
    input_mapping: Mapping[str, str] = field(default_factory=dict)
    agent_tools: tuple[str, ...] = ()
    strict_handshake: bool = True
    receipt_required: bool = False
    max_prompt_bytes: int = 32768
    episode_memory: str = "none"
    history_protocol: str = "current_frame_only"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.name.strip():
            raise ValueError("invalid policy profile schema or name")
        if self.system_type not in SYSTEM_TYPES or self.adapter not in ADAPTERS:
            raise ValueError("unknown policy system type or adapter")
        if not self.reports <= {"semantic", "decision", "geometry", "execution"}:
            raise ValueError("unknown policy report")
        if not self.oracle_transports <= ORACLE_TRANSPORTS - {"none"}:
            raise ValueError("unknown Oracle transport")
        if not self.replacement_sites <= set(REPLACEMENT_SITES.values()):
            raise ValueError("unknown replacement site")
        if self.system_type == "native_vla" and self.adapter not in {"openpi", "http", "vla_evaluation_harness"}:
            raise ValueError("native VLA requires an action-model adapter")
        if self.system_type == "agent" and self.adapter not in {"rpent", "http"}:
            raise ValueError("agent profile requires an agent adapter")
        if len(set(self.agent_tools)) != len(self.agent_tools):
            raise ValueError("agent tools must be unique")
        if type(self.max_prompt_bytes) is not int or self.max_prompt_bytes <= 0:
            raise ValueError("max_prompt_bytes must be a positive integer")
        if self.episode_memory not in EPISODE_MEMORY_MODES:
            raise ValueError("unknown episode memory mode")
        if self.history_protocol not in HISTORY_PROTOCOLS:
            raise ValueError("unknown history protocol")
        if self.episode_memory != "none" and self.reset.mode == "stateless":
            raise ValueError("stateful episode memory requires rpc or restart reset")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyProfile":
        if not isinstance(data, Mapping):
            raise ValueError("policy profile must be an object")
        known = {"schema_version", "name", "system_type", "adapter", "action", "sensors",
                 "reports", "oracle_transports", "replacement_sites", "reset", "input_mapping",
                 "agent_tools", "strict_handshake", "receipt_required", "max_prompt_bytes",
                 "episode_memory", "history_protocol"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown policy profile fields: {sorted(unknown)}")
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            name=str(data["name"]), system_type=str(data["system_type"]), adapter=str(data["adapter"]),
            action=ActionSpec(**dict(data["action"])), sensors=SensorSpec(
                tuple(data["sensors"]["rgb_channels"]), tuple(data["sensors"].get("depth_channels", [])),
                str(data["sensors"].get("state_format", "joint_positions"))),
            reports=frozenset(data.get("reports", [])),
            oracle_transports=frozenset(data.get("oracle_transports", [])),
            replacement_sites=frozenset(data.get("replacement_sites", [])),
            reset=ResetSpec(**dict(data.get("reset", {}))),
            input_mapping=dict(data.get("input_mapping", {})),
            agent_tools=tuple(data.get("agent_tools", [])),
            strict_handshake=bool(data.get("strict_handshake", True)),
            receipt_required=bool(data.get("receipt_required", False)),
            max_prompt_bytes=int(data.get("max_prompt_bytes", 32768)),
            episode_memory=str(data.get("episode_memory", "none")),
            history_protocol=str(data.get("history_protocol", "current_frame_only")),
        )

    @classmethod
    def legacy(cls, interface: Mapping[str, Any], rgb_channels: tuple[str, ...]) -> "PolicyProfile":
        action_space = str(interface.get("action_space", "symbolic"))
        name = str(interface.get("policy_id", interface.get("name", "legacy")))
        adapter_name = str(interface.get("name", "http"))
        adapter = "openpi" if adapter_name == "openpi_array" else (
            "rule_expert" if adapter_name == "diagnostic_rule_expert" else "http")
        return cls(
            name=name, system_type="symbolic" if action_space == "symbolic" else "native_vla",
            adapter=adapter, action=ActionSpec.legacy(action_space), sensors=SensorSpec(rgb_channels),
            reports=frozenset(interface.get("supported_reports", [])),
            oracle_transports=frozenset(interface.get("oracle_transports", [])),
            replacement_sites=frozenset(
                name for enabled, name in (
                    ("semantic" in interface.get("supported_reports", []), "semantic_report"),
                    ("decision" in interface.get("supported_reports", []), "committed_action"),
                    ("geometry" in interface.get("supported_reports", []), "geometry_report"),
                    (bool(interface.get("selected_operation_input")), "executor"),
                ) if enabled),
            reset=ResetSpec("rpc" if interface.get("resettable", False) else "stateless"),
            episode_memory=str(interface.get("episode_memory", "none")),
            history_protocol=str(interface.get("history_protocol", "current_frame_only")),
            strict_handshake=False,
        )

    def public(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("reports", "oracle_transports", "replacement_sites"):
            value[key] = sorted(value[key])
        value["agent_tools"] = list(self.agent_tools)
        value["sensors"]["rgb_channels"] = list(self.sensors.rgb_channels)
        value["sensors"]["depth_channels"] = list(self.sensors.depth_channels)
        return value

    @property
    def hash(self) -> str:
        payload = json.dumps(self.public(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InterventionSpec:
    profile: str = "none"
    mode: str = "none"
    site: str = "none"
    transport: str = "none"
    schema_version: str = INTERVENTION_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INTERVENTION_VERSION or self.profile not in ORACLE_PROFILES:
            raise ValueError("invalid intervention schema or profile")
        if self.mode not in INTERVENTION_MODES or self.transport not in ORACLE_TRANSPORTS:
            raise ValueError("invalid intervention mode or transport")
        if self.profile == "none":
            if (self.mode, self.site, self.transport) != ("none", "none", "none"):
                raise ValueError("none profile must not define an intervention")
            return
        expected = "policy_input" if self.mode == "assist" else REPLACEMENT_SITES[self.profile]
        if self.mode == "none" or self.site != expected:
            raise ValueError(f"{self.profile}/{self.mode} must use site {expected}")
        if self.mode == "assist" and self.transport not in {"structured", "text", "tool"}:
            raise ValueError("assist intervention requires a transport")
        if self.mode == "replace" and self.transport != "none":
            raise ValueError("replacement is an interface hook, not an input transport")
        if self.profile == "skill" and self.mode != "replace":
            raise ValueError("Skill is only available as an executor replacement")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InterventionSpec":
        if not isinstance(data, Mapping):
            raise ValueError("intervention spec must be an object")
        unknown = set(data) - {"schema_version", "profile", "mode", "site", "transport"}
        if unknown:
            raise ValueError(f"unknown intervention fields: {sorted(unknown)}")
        return cls(**dict(data))

    @classmethod
    def legacy(cls, profile: str, transport: str) -> "InterventionSpec":
        if profile == "none":
            return cls()
        if profile == "skill":
            return cls(profile="skill", mode="replace", site="executor", transport="none")
        return cls(profile=profile, mode="assist", site="policy_input", transport=transport)


@dataclass(frozen=True)
class CapabilityCheck:
    compatible: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    verified_from_server: bool


def ranking_eligibility(
    profile: PolicyProfile,
    intervention: InterventionSpec,
    interface: Mapping[str, Any],
    *,
    uses_evaluator_state: bool,
    server_metadata: Mapping[str, Any] | None = None,
) -> tuple[bool, str, tuple[str, ...]]:
    """Return a conservative leaderboard decision and its auditable reasons."""
    raw_revision = interface.get("policy_revision")
    capabilities = None if server_metadata is None else server_metadata.get("labference")
    if raw_revision is None and isinstance(capabilities, Mapping):
        raw_revision = capabilities.get("policy_revision")
    revision = str(raw_revision).strip() if raw_revision is not None else "unversioned"
    reasons: list[str] = []
    if uses_evaluator_state or profile.adapter == "rule_expert":
        reasons.append("evaluator_state_or_reference_policy")
    if not revision or revision.lower() in {"none", "null", "unknown", "unversioned"}:
        revision = "unversioned"
        reasons.append("unversioned_policy")
    if intervention.profile == "skill":
        reasons.append("skill_diagnostic_ceiling")
    return not reasons, revision, tuple(reasons)


def check_capabilities(profile: PolicyProfile, intervention: InterventionSpec,
                       server_metadata: Mapping[str, Any] | None = None) -> CapabilityCheck:
    errors: list[str] = []
    warnings: list[str] = []
    if intervention.mode == "assist" and intervention.transport not in profile.oracle_transports:
        errors.append(f"policy does not support {intervention.transport} Oracle transport")
    if intervention.mode == "replace" and intervention.site not in profile.replacement_sites:
        errors.append(f"policy does not expose replacement site {intervention.site}")
    if intervention.mode == "replace" and profile.system_type == "native_vla" and intervention.site in {
            "semantic_report", "committed_action", "geometry_report"}:
        errors.append("native action-only VLA has no declared module hook for this replacement")
    declared = None if server_metadata is None else server_metadata.get("labference")
    verified = isinstance(declared, Mapping)
    if profile.strict_handshake and not verified:
        errors.append("server metadata lacks a labference capability handshake")
    elif not verified:
        warnings.append("server handshake is optional for this profile; capabilities are configuration-declared")
    else:
        expected_rgb = set(profile.sensors.rgb_channels)
        accepted_rgb = set(declared.get("rgb_channels", []))
        if not expected_rgb <= accepted_rgb:
            errors.append(f"server does not accept RGB channels {sorted(expected_rgb - accepted_rgb)}")
        if not set(profile.sensors.depth_channels) <= set(declared.get("depth_channels", [])):
            errors.append("server does not accept all declared depth channels")
        if declared.get("action") != asdict(profile.action):
            errors.append("server action convention differs from policy profile")
        if declared.get("max_prompt_bytes") != profile.max_prompt_bytes:
            errors.append("server prompt limit differs from policy profile")
        if declared.get("reset") != asdict(profile.reset):
            errors.append("server reset lifecycle differs from policy profile")
        if profile.system_type == "agent" and set(declared.get("agent_tools", [])) != set(profile.agent_tools):
            errors.append("server agent tool inventory differs from policy profile")
        if declared.get("episode_memory", "none") != profile.episode_memory:
            errors.append("server episode memory mode differs from policy profile")
        if declared.get("history_protocol", "current_frame_only") != profile.history_protocol:
            errors.append("server history protocol differs from policy profile")
        if intervention.mode == "assist" and intervention.transport not in set(
                declared.get("oracle_transports", [])):
            errors.append("server handshake does not advertise the selected Oracle transport")
        if intervention.mode == "replace" and intervention.site not in set(
                declared.get("replacement_sites", [])):
            errors.append("server handshake does not advertise the selected replacement site")
    return CapabilityCheck(not errors, tuple(errors), tuple(warnings), verified)


def validate_connection_binding(profile: PolicyProfile, interface: Mapping[str, Any]) -> None:
    """Ensure legacy connection fields cannot silently override a policy profile."""
    expected_space = "symbolic" if profile.action.representation == "symbolic" else "array"
    if interface.get("action_space", expected_space) != expected_space:
        raise ValueError("policy_interface action_space conflicts with the policy profile")
    name = str(interface.get("name", ""))
    expected_adapter = {
        "openpi_array": "openpi", "http_array": "http", "http_symbolic": "http",
        "diagnostic_rule_expert": "rule_expert",
    }.get(name)
    if expected_adapter is not None and expected_adapter != profile.adapter:
        raise ValueError("policy_interface transport conflicts with the policy profile adapter")
    legacy_reports = frozenset(interface.get("supported_reports", []))
    if legacy_reports and legacy_reports != profile.reports:
        raise ValueError("policy_interface reports conflict with the policy profile")
    legacy_transports = frozenset(interface.get("oracle_transports", []))
    if legacy_transports and legacy_transports != profile.oracle_transports:
        raise ValueError("policy_interface Oracle transports conflict with the policy profile")


def load_policy_profile(path: str | Path) -> PolicyProfile:
    return PolicyProfile.from_dict(json.loads(Path(path).read_text()))


def load_intervention_spec(path: str | Path) -> InterventionSpec:
    return InterventionSpec.from_dict(json.loads(Path(path).read_text()))


def resolve_runtime_contract(evaluation: Mapping[str, Any], *, rgb_channels: tuple[str, ...],
                             policy_profile_path: str | Path | None = None,
                             intervention_path: str | Path | None = None) -> tuple[PolicyProfile, InterventionSpec]:
    interface = evaluation.get("policy_interface", {})
    profile = (load_policy_profile(policy_profile_path) if policy_profile_path else
               PolicyProfile.from_dict(evaluation["policy_profile"]) if "policy_profile" in evaluation else
               PolicyProfile.legacy(interface, rgb_channels))
    intervention = (load_intervention_spec(intervention_path) if intervention_path else
                    InterventionSpec.from_dict(evaluation["intervention"]) if "intervention" in evaluation else
                    InterventionSpec.legacy(str(evaluation.get("oracle_profile", "none")),
                                            str(evaluation.get("oracle_transport", "structured"))))
    check = check_capabilities(profile, intervention)
    if not check.compatible and not (
            profile.strict_handshake and check.errors == ("server metadata lacks a labference capability handshake",)):
        raise ValueError("; ".join(check.errors))
    return profile, intervention
