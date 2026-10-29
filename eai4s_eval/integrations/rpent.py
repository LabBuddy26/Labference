"""Thin RPent-compatible environment and toolkit bridge.

RPent owns the agent loop.  This bridge owns no policy and exposes only the
authorized online driver, so evaluator-only state remains outside the toolkit.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from eai4s_eval.oracle_runtime import (
    InterventionController, InterventionSpec, PolicyProfile, PolicySession,
)

from .base import (OnlineEpisodeDriver, public_metadata, public_observation,
                   public_task, validate_driver)


def build_rpent_tool_specs() -> list[dict[str, Any]]:
    """Return the fixed primitive vocabulary exposed to an RPent planner."""
    return [
        {"name": "observe", "description": "Read the latest authorized task observation.",
         "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": "recognize_state", "description": "Submit the agent's structured state report.",
         "input_schema": {"type": "object", "properties": {"report": {"type": "object"}},
                          "required": ["report"], "additionalProperties": False}},
        {"name": "commit_operation", "description": "Commit the next symbolic operation before execution.",
         "input_schema": {"type": "object", "properties": {"operation": {"type": "object"}},
                          "required": ["operation"], "additionalProperties": False}},
        {"name": "localize_target", "description": "Submit a target frame from authorized observations.",
         "input_schema": {"type": "object", "properties": {"geometry": {"type": "object"}},
                          "required": ["geometry"], "additionalProperties": False}},
        {"name": "vla_act", "description": "Execute a bounded action chunk from the frozen VLA primitive.",
         "input_schema": {"type": "object", "properties": {"actions": {"type": "array"}},
                          "required": ["actions"], "additionalProperties": False}},
        {"name": "move_to", "description": "Execute declared non-contact motion for a selected target.",
         "input_schema": {"type": "object", "properties": {"actions": {"type": "array"}},
                          "required": ["actions"], "additionalProperties": False}},
        {"name": "execute_operation", "description": "Execute the already committed symbolic operation.",
         "input_schema": {"type": "object", "properties": {"operation": {"type": "object"}},
                          "required": ["operation"], "additionalProperties": False}},
        {"name": "wait", "description": "Advance physics while holding the current robot target.",
         "input_schema": {"type": "object", "properties": {"physics_steps": {"type": "integer", "minimum": 1}},
                          "required": ["physics_steps"], "additionalProperties": False}},
        {"name": "finish", "description": "Ask the environment to evaluate the current terminal state.",
         "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    ]


class LabferenceEnvAdapter:
    """Methods mirror RPent's BaseEnvFacade RPC surface."""

    def __init__(self, driver: OnlineEpisodeDriver, profile: PolicyProfile,
                 policy_session: PolicySession | None = None):
        validate_driver(driver)
        if profile.system_type != "agent" or profile.adapter != "rpent":
            raise ValueError("RPent environment requires an rpent agent profile")
        self.driver = driver
        self.profile = profile
        self.policy_session = policy_session
        if profile.strict_handshake and (policy_session is None or
                                         policy_session.profile.hash != profile.hash or
                                         not policy_session.capability_check.verified_from_server):
            raise ValueError("RPent agent requires a verified policy capability session")
        self.last_obs: dict[str, Any] | None = None
        self.episode_id: str | None = None
        self.sequence = 0
        self.last_reset_receipt: dict[str, Any] | None = None

    def get_env_meta(self) -> dict[str, Any]:
        metadata = public_metadata(self.driver.metadata(), self.profile)
        metadata["labference"] = {
            "schema_version": self.profile.schema_version,
            "rgb_channels": list(self.profile.sensors.rgb_channels),
            "depth_channels": list(self.profile.sensors.depth_channels),
            "action": asdict(self.profile.action),
            "oracle_transports": sorted(self.profile.oracle_transports),
            "replacement_sites": sorted(self.profile.replacement_sites),
            "max_prompt_bytes": self.profile.max_prompt_bytes,
            "reset": asdict(self.profile.reset),
            "agent_tools": list(self.profile.agent_tools),
            "step_owner": "rpent_env_adapter",
        }
        return metadata

    def reset(self, task: Mapping[str, Any], seed: int) -> dict[str, Any]:
        self.sequence = 0
        self.episode_id = f"{task.get('name', 'task')}:{seed}"
        if self.policy_session is not None:
            self.last_reset_receipt = self.policy_session.reset(seed)
        self.last_obs = public_observation(self.driver.reset(task, seed), self.profile)
        return self._packet(self.last_obs)

    def step(self, action: Mapping[str, Any]) -> dict[str, Any]:
        self.last_obs = public_observation(self.driver.step(action), self.profile)
        return self._packet(self.last_obs)

    def chunk_step(self, actions: list[list[float]], **_: Any) -> dict[str, Any]:
        validated = self.profile.action.validate_chunk(actions)
        observation: Mapping[str, Any] = self.last_obs or {}
        for waypoint in validated:
            observation = public_observation(
                self.driver.step({"actions": waypoint}), self.profile)
            if self.driver.done():
                break
        self.last_obs = dict(observation)
        return self._packet(self.last_obs)

    def render_camera(self, camera_name: str, **_: Any) -> Any:
        authorized = set(self.profile.sensors.rgb_channels + self.profile.sensors.depth_channels)
        if camera_name not in authorized:
            raise KeyError(f"camera is not authorized: {camera_name}")
        observation = self.last_obs or public_observation(self.driver.observation(), self.profile)
        if camera_name in observation.get("images", {}):
            return observation["images"][camera_name]
        if camera_name in observation.get("depth", {}):
            return observation["depth"][camera_name]
        raise KeyError(f"camera is not authorized: {camera_name}")

    def get_camera_meta(self, camera_name: str, **_: Any) -> dict[str, Any]:
        metadata = public_metadata(self.driver.metadata(), self.profile).get("cameras", {})
        if camera_name not in metadata:
            raise KeyError(f"unknown camera metadata: {camera_name}")
        return dict(metadata[camera_name])

    def get_task_language(self) -> str:
        observation = self.last_obs or public_observation(self.driver.observation(), self.profile)
        return str(observation["task_rule"])

    def get_tasks(self) -> list[dict[str, Any]]:
        return [public_task(task) for task in self.driver.tasks()]

    def get_result(self) -> dict[str, Any]:
        result = dict(self.driver.result())
        if not self.driver.done():
            return {"done": False}
        allowed = {key: result[key] for key in (
            "success", "reward", "physical_completion", "termination_reason") if key in result}
        return {"done": True, **allowed}

    def _packet(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        return {"episode_id": self.episode_id, "sequence": self.sequence,
                "policy_session_id": (None if self.last_reset_receipt is None else
                                      self.last_reset_receipt.get("session_id")),
                "observation": dict(observation), "done": self.driver.done()}

    def policy_observation(self) -> dict[str, Any]:
        self.last_obs = public_observation(self.driver.observation(), self.profile)
        return dict(self.last_obs)


class RPentToolkitBridge:
    """Fixed tool handlers with intervention hooks at accessible interfaces."""

    def __init__(self, env: LabferenceEnvAdapter, interventions: InterventionController,
                 *, oracle_provider: Any | None = None,
                 operation_executor: Any | None = None,
                 skill_executor: Any | None = None):
        self.env = env
        self.interventions = interventions
        self.oracle_provider = oracle_provider
        self.operation_executor = operation_executor
        self.skill_executor = skill_executor
        self.opportunity_id = "unopened"
        self.evidence_step = -1
        self.committed_operation: dict[str, Any] | None = None
        self.call_counts: dict[str, int] = {spec["name"]: 0 for spec in build_rpent_tool_specs()}

    def _count(self, name: str) -> None:
        self.call_counts[name] += 1

    def _oracle(self, profile: str, *args: Any) -> Any:
        if self.oracle_provider is None:
            raise RuntimeError(f"{profile} intervention requires an Oracle provider")
        value = self.oracle_provider(profile, *args)
        if isinstance(value, Mapping) and "replacement_output" in value:
            return value["replacement_output"]
        return value

    def set_opportunity(self, opportunity_id: str, evidence_step: int) -> None:
        if not opportunity_id or evidence_step < 0:
            raise ValueError("invalid decision opportunity")
        self.opportunity_id = opportunity_id
        self.evidence_step = evidence_step

    def observe(self) -> dict[str, Any]:
        self._count("observe")
        observation = self.env.policy_observation()
        if self.interventions.spec.mode == "assist":
            context = self._oracle(self.interventions.spec.profile)
            observation = self.interventions.assist(
                observation, context, opportunity_id=self.opportunity_id,
                evidence_step=self.evidence_step)
        return observation

    def recognize_state(self, report: Mapping[str, Any]) -> dict[str, Any]:
        self._count("recognize_state")
        selected: Any = dict(report)
        if self.interventions.spec.site == "semantic_report":
            selected = self.interventions.replace(
                selected, self._oracle("semantic"), site="semantic_report",
                opportunity_id=self.opportunity_id, evidence_step=self.evidence_step,
                runtime_context={"policy_observation": self.env.policy_observation()})
        return {"semantic_report": dict(selected)}

    def commit_operation(self, operation: Mapping[str, Any]) -> dict[str, Any]:
        self._count("commit_operation")
        selected: Any = dict(operation)
        if self.interventions.spec.site == "committed_action":
            selected = self.interventions.replace(
                selected, self._oracle("decision"), site="committed_action",
                opportunity_id=self.opportunity_id, evidence_step=self.evidence_step,
                runtime_context={"policy_observation": self.env.policy_observation()})
        self.committed_operation = dict(selected)
        return {"committed_operation": self.committed_operation}

    def localize_target(self, geometry: Mapping[str, Any]) -> dict[str, Any]:
        self._count("localize_target")
        selected: Any = dict(geometry)
        if self.interventions.spec.site == "geometry_report":
            selected = self.interventions.replace(
                selected, self._oracle("geometry", self.committed_operation),
                site="geometry_report", opportunity_id=self.opportunity_id,
                evidence_step=self.evidence_step,
                runtime_context={"committed_operation": self.committed_operation,
                                 "policy_observation": self.env.policy_observation()})
        return {"geometry": selected}

    def vla_act(self, actions: list[list[float]]) -> dict[str, Any]:
        self._count("vla_act")
        packet = self.env.chunk_step(actions)
        return {"ok": True, **packet}

    def move_to(self, actions: list[list[float]]) -> dict[str, Any]:
        self._count("move_to")
        packet = self.env.chunk_step(actions)
        return {"ok": True, **packet}

    def execute_operation(self, operation: Mapping[str, Any]) -> dict[str, Any]:
        """Execute a selected operation without allowing Skill Oracle to change its identity."""
        self._count("execute_operation")
        selected = dict(operation)
        if self.committed_operation is None:
            raise ValueError("an operation must be committed before execution")
        if selected != self.committed_operation:
            raise ValueError("execution operation differs from the committed operation")
        executor = self.operation_executor
        if self.interventions.spec.site == "executor":
            executor = self.skill_executor
            self.interventions.replace(
                {"executor": "configured_operation_executor"},
                {"executor": "validated_reference_executor"}, site="executor",
                opportunity_id=self.opportunity_id, evidence_step=self.evidence_step,
                runtime_context={"committed_operation": selected})
        if executor is None:
            raise RuntimeError("the selected operation has no registered executor")
        result = executor(selected)
        if not isinstance(result, Mapping):
            raise RuntimeError("operation executor must return a result mapping")
        return {"operation": selected, "executor": (
            "skill_reference" if self.interventions.spec.site == "executor" else "configured"),
                "result": dict(result)}

    def wait(self, physics_steps: int) -> dict[str, Any]:
        self._count("wait")
        if type(physics_steps) is not int or physics_steps <= 0:
            raise ValueError("physics_steps must be a positive integer")
        observation = public_observation(
            self.env.driver.wait(physics_steps), self.env.profile)
        self.env.last_obs = observation
        return self.env._packet(observation)

    def finish(self) -> dict[str, Any]:
        self._count("finish")
        return self.env.get_result()

    def metrics(self) -> dict[str, Any]:
        return {
            "agent_tool_calls": sum(self.call_counts.values()),
            "tool_call_counts": dict(self.call_counts),
            "intervention_records": self.interventions.public_records(),
        }
