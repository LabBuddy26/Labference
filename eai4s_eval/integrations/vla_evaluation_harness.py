"""AllenAI vla-evaluation-harness adapter around a Labference driver."""

from __future__ import annotations

import importlib
from typing import Any, Mapping

from eai4s_eval.oracle_runtime import (
    InterventionController, load_intervention_spec, load_policy_profile,
)

from .base import (DriverOracleValidator, OnlineEpisodeDriver, public_metadata,
                   public_observation, public_task, validate_driver)

try:
    from vla_eval.benchmarks.base import StepBenchmark, StepResult, repeat_last_hold
    from vla_eval.specs import DimSpec, LANGUAGE, RAW, STATE_JOINT
except ImportError:  # Keep Labference importable without the optional harness.
    class StepBenchmark:  # type: ignore[no-redef]
        pass

    class StepResult:  # type: ignore[no-redef]
        pass

    DimSpec = None  # type: ignore[assignment]
    LANGUAGE = RAW = STATE_JOINT = None
    repeat_last_hold = None  # type: ignore[assignment]


def _resolve(path: str) -> Any:
    if ":" not in path:
        raise ValueError("import paths must use module:attribute syntax")
    module, name = path.split(":", 1)
    return getattr(importlib.import_module(module), name)


class LabferenceVLAHarnessBenchmark(StepBenchmark):
    """Expose one Labference online driver through the harness StepBenchmark API."""

    render_backends = frozenset({"gpu"})

    def __init__(self, *, driver_factory: str, policy_profile: str,
                 intervention_spec: str, driver_kwargs: Mapping[str, Any] | None = None,
                 oracle_provider: str | None = None,
                 oracle_validator_factory: str | None = None):
        if DimSpec is None:
            raise ImportError("install vla-evaluation-harness to use this adapter")
        super().__init__()
        self.profile = load_policy_profile(policy_profile)
        self.intervention = load_intervention_spec(intervention_spec)
        if self.profile.adapter != "vla_evaluation_harness":
            raise ValueError("policy profile is not a vla-evaluation-harness profile")
        if self.intervention.mode == "replace":
            raise ValueError("the action-only harness adapter has no module replacement hook")
        factory = _resolve(driver_factory)
        self.driver: OnlineEpisodeDriver = factory(**dict(driver_kwargs or {}))
        validate_driver(self.driver)
        self.oracle_provider = _resolve(oracle_provider) if oracle_provider else None
        validator_factory = (_resolve(oracle_validator_factory)
                             if oracle_validator_factory else None)
        validator = (validator_factory(self.driver) if validator_factory else
                     DriverOracleValidator(self.driver) if self.intervention.profile != "none" else None)
        self.interventions = InterventionController(self.profile, self.intervention, validator)
        self._raw_observation: dict[str, Any] = {}
        self._observation_index = 0

    def get_tasks(self) -> list[dict[str, Any]]:
        return [public_task(task) for task in self.driver.tasks()]

    def reset(self, task: dict[str, Any]) -> Any:
        self._observation_index = 0
        seed = int(task.get("seed", task.get("episode_idx", 101)))
        self._raw_observation = public_observation(self.driver.reset(task, seed), self.profile)
        return self._raw_observation

    def step(self, action: dict[str, Any]) -> Any:
        values = action.get("actions")
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], (int, float)):
            values = [values]
        chunk = self.profile.action.validate_chunk(values)
        observation: Mapping[str, Any] = self._raw_observation
        for waypoint in chunk:
            observation = public_observation(
                self.driver.step({"actions": waypoint}), self.profile)
            if self.driver.done():
                break
        self._raw_observation = dict(observation)
        result = dict(self.driver.result()) if self.driver.done() else {}
        return StepResult(obs=self._raw_observation, reward=float(result.get("reward", 0.0)),
                          done=self.driver.done(), info=result)

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        raw = dict(raw_obs)
        images = raw.get("images", {})
        depth = raw.get("depth", {})
        if not set(self.profile.sensors.rgb_channels) <= set(images):
            raise ValueError("driver observation is missing declared RGB channels")
        if not set(self.profile.sensors.depth_channels) <= set(depth):
            raise ValueError("driver observation is missing declared depth channels")
        observation = {
            "images": {name: images[name] for name in self.profile.sensors.rgb_channels},
            "state": raw["joint_positions"],
            "task_description": str(raw["task_rule"]),
        }
        if self.profile.sensors.depth_channels:
            observation["depth"] = {
                name: depth[name] for name in self.profile.sensors.depth_channels}
        if self.intervention.mode == "assist":
            if self.oracle_provider is None:
                raise RuntimeError("harness assist condition requires an Oracle provider")
            opportunity_id = str(raw.get("opportunity_id", f"observation-{self._observation_index}"))
            evidence_step = int(raw.get("evidence_step", self._observation_index))
            context = self.oracle_provider(self.intervention.profile, raw, task)
            assisted = self.interventions.assist(
                observation, context, opportunity_id=opportunity_id, evidence_step=evidence_step)
            if self.intervention.transport != "text":
                raise ValueError("the current harness model boundary supports text assistance only")
            assisted["task_description"] += "\n" + assisted.pop("oracle_prompt")
            observation = assisted
        self._observation_index += 1
        return observation

    def get_step_result(self, step_result: Any) -> dict[str, Any]:
        result = dict(step_result.info or self.driver.result())
        result.setdefault("success", bool(result.get("physical_completion", False)))
        return result

    def get_action_spec(self) -> dict[str, Any]:
        action = self.profile.action
        return {"actions": DimSpec("actions", action.dimensions,
                                   f"{action.mode}_{action.representation}:{action.frame}:{action.units}")}

    def get_observation_spec(self) -> dict[str, Any]:
        return {"images": RAW, "state": STATE_JOINT, "task_description": LANGUAGE,
                **({"depth": RAW} if self.profile.sensors.depth_channels else {})}

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean", "online_legal_next_action_rate": "mean",
                "physical_completion": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        return {**public_metadata(self.driver.metadata(), self.profile),
                "labference_policy_profile": self.profile.public(),
                "intervention": self.intervention.__dict__,
                "step_owner": "vla_evaluation_harness"}

    def get_hold_action(self, last_action: dict[str, Any] | None) -> dict[str, Any]:
        if self.profile.action.mode != "absolute":
            raise NotImplementedError("only absolute joint targets have a declared safe hold")
        return repeat_last_hold(last_action, self.profile.action.dimensions)

    def render(self) -> Any:
        observation = self._raw_observation or dict(self.driver.observation())
        return observation.get("images", {}).get("front")

    def cleanup(self) -> None:
        self.driver.close()
