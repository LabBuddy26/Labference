"""Simulator-independent driver contract used by external runtimes."""

from __future__ import annotations

import re
from typing import Any, Mapping, Protocol, runtime_checkable

from eai4s_eval.oracle_runtime import PolicyProfile


_PUBLIC_OBSERVATION_FIELDS = frozenset({
    "joint_positions", "task_rule", "camera_calibration", "authorized_history",
    "visible_slot_ids", "visible_weight_labels_g", "display_reading_g",
    "stable_indicator", "history", "calibration", "opportunity_id", "evidence_step",
})
_PUBLIC_TASK_FIELDS = frozenset({
    "name", "suite", "seed", "task_version", "difficulty", "instruction", "task_rule",
    "episode_idx", "pair_member",
})
_PRIVATE_KEY = re.compile(
    r"(?:evaluator|ground_truth|hidden|correct_|answer|success_label|legal_actions|"
    r"target_(?:world_)?(?:pose|position|frame)|sample_mass)", re.IGNORECASE)


def _reject_private(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _PRIVATE_KEY.search(str(key)):
                raise ValueError(f"private evaluator field at external policy boundary: {key}")
            _reject_private(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_private(nested)


def public_observation(raw: Mapping[str, Any], profile: PolicyProfile) -> dict[str, Any]:
    """Project an untrusted driver observation onto the declared policy boundary."""
    if not isinstance(raw, Mapping):
        raise TypeError("driver observation must be a mapping")
    images = raw.get("images")
    depth = raw.get("depth", {})
    if not isinstance(images, Mapping) or set(images) != set(profile.sensors.rgb_channels):
        raise ValueError("driver RGB channels differ from the policy profile")
    if not isinstance(depth, Mapping) or set(depth) != set(profile.sensors.depth_channels):
        raise ValueError("driver depth channels differ from the policy profile")
    result: dict[str, Any] = {
        "images": {name: images[name] for name in profile.sensors.rgb_channels},
        "depth": {name: depth[name] for name in profile.sensors.depth_channels},
    }
    for name in _PUBLIC_OBSERVATION_FIELDS:
        if name in raw:
            result[name] = raw[name]
    if "joint_positions" not in result or "task_rule" not in result:
        raise ValueError("driver observation needs joint_positions and task_rule")
    _reject_private({key: value for key, value in result.items() if key not in {"images", "depth"}})
    return result


def public_task(raw: Mapping[str, Any]) -> dict[str, Any]:
    result = {name: raw[name] for name in _PUBLIC_TASK_FIELDS if name in raw}
    if "name" not in result:
        raise ValueError("public task needs a name")
    _reject_private(result)
    return result


def public_metadata(raw: Mapping[str, Any], profile: PolicyProfile) -> dict[str, Any]:
    allowed = {key: raw[key] for key in (
        "benchmark", "task_version", "max_steps", "physics_hz", "simulator_version") if key in raw}
    cameras = raw.get("cameras", {})
    if not isinstance(cameras, Mapping):
        raise ValueError("driver camera metadata must be a mapping")
    channels = profile.sensors.rgb_channels + profile.sensors.depth_channels
    if set(cameras) != set(channels):
        raise ValueError("driver camera metadata differs from the policy profile")
    allowed["cameras"] = {name: cameras[name] for name in channels}
    _reject_private(allowed)
    return allowed


@runtime_checkable
class OnlineEpisodeDriver(Protocol):
    """One owner of simulator stepping, regardless of the external orchestrator."""

    def metadata(self) -> Mapping[str, Any]: ...
    def tasks(self) -> list[dict[str, Any]]: ...
    def reset(self, task: Mapping[str, Any], seed: int) -> Mapping[str, Any]: ...
    def observation(self) -> Mapping[str, Any]: ...
    def step(self, action: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def wait(self, physics_steps: int) -> Mapping[str, Any]: ...
    def done(self) -> bool: ...
    def result(self) -> Mapping[str, Any]: ...
    def close(self) -> None: ...


class DriverOracleValidator:
    """Delegate Oracle scope checks to the task-owned online driver.

    The transport adapter must not infer task truth.  Drivers that expose an
    intervention therefore implement both validation methods against their
    versioned task contract and current evaluator state.
    """

    def __init__(self, driver: OnlineEpisodeDriver):
        self.driver = driver
        missing = [name for name in ("validate_oracle_assist", "validate_oracle_replacement")
                   if not callable(getattr(driver, name, None))]
        if missing:
            raise TypeError(f"active Oracle driver is missing validation methods: {missing}")

    def validate_assist(self, profile: str, context: Mapping[str, Any],
                        observation: Mapping[str, Any]) -> None:
        self.driver.validate_oracle_assist(profile, context, observation)

    def validate_replacement(self, profile: str, site: str, original: Any,
                             replacement: Any,
                             runtime_context: Mapping[str, Any]) -> None:
        self.driver.validate_oracle_replacement(
            profile, site, original, replacement, runtime_context)


def validate_driver(driver: OnlineEpisodeDriver) -> None:
    missing = [name for name in ("metadata", "tasks", "reset", "observation", "step",
                                  "wait", "done", "result", "close")
               if not callable(getattr(driver, name, None))]
    if missing:
        raise TypeError(f"online episode driver is missing methods: {missing}")
