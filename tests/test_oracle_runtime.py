"""Shared VLA/agent Oracle protocol tests without simulator dependencies."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from eai4s_eval.integrations.base import DriverOracleValidator, OnlineEpisodeDriver
from eai4s_eval.integrations.rpent import (
    LabferenceEnvAdapter, RPentToolkitBridge, build_rpent_tool_specs,
)
from eai4s_eval.integrations.openpi_receipt import LabferenceOpenPIWrapper
from eai4s_eval.oracle_runtime import (
    ActionSpec, InterventionController, InterventionSpec, PolicyProfile, PolicySession,
    ResetSpec, SensorSpec, content_digest, load_policy_profile,
    compare_matched_conditions, condition_id, plan_matched_conditions,
    ranking_eligibility,
)
from eai4s_eval.centrifuge_v2.policy import OpenPIArrayPolicy, PolicyServiceError
from eai4s_eval.oracle_runtime.contracts import check_capabilities
from eai4s_eval.oracle_runtime.contracts import validate_connection_binding
from eai4s_eval.oracle_runtime.comparison import MATCH_FIELDS


ROOT = Path(__file__).resolve().parents[1] / "eai4s_eval/oracle_runtime"


class TestOracleValidator:
    __test__ = False

    def validate_assist(self, profile, context, observation):
        if profile == "semantic" and "correct_slot" in json.dumps(context):
            raise ValueError("semantic answer leakage")

    def validate_replacement(self, profile, site, original, replacement, runtime_context):
        if profile == "decision" and replacement != {"type": "wait"}:
            raise ValueError("illegal decision replacement")
        if profile == "skill" and runtime_context.get("committed_operation") is None:
            raise ValueError("skill replacement needs a committed operation")


def native_profile() -> PolicyProfile:
    return PolicyProfile(
        "native", "native_vla", "openpi", ActionSpec.legacy("array"),
        SensorSpec(("front", "wrist")), oracle_transports=frozenset({"text"}),
        reset=ResetSpec("stateless"), strict_handshake=False)


def test_openpi_stateful_reset_requires_a_verifiable_server_extension():
    class NoReceiptClient:
        def reset(self):
            return None

    adapter = object.__new__(OpenPIArrayPolicy)
    adapter.client = NoReceiptClient()
    with pytest.raises(PolicyServiceError, match="reset_with_receipt"):
        adapter._reset_client(101)


def test_openpi_stateful_reset_forwards_wrapper_receipt():
    expected = {"status": "ok", "seed": 101, "session_id": "episode-101",
                "action_buffer_cleared": True, "agent_context_cleared": True,
                "episode_memory_cleared": True}

    class ReceiptClient:
        def reset_with_receipt(self, seed):
            assert seed == 101
            return expected

    adapter = object.__new__(OpenPIArrayPolicy)
    adapter.client = ReceiptClient()
    assert adapter._reset_client(101) == expected


def agent_profile() -> PolicyProfile:
    return load_policy_profile(ROOT / "profiles/rpent_agent_rgbd.json")


def handshake(profile: PolicyProfile) -> dict:
    return {"labference": {
        "rgb_channels": list(profile.sensors.rgb_channels),
        "depth_channels": list(profile.sensors.depth_channels),
        "action": asdict(profile.action),
        "oracle_transports": sorted(profile.oracle_transports),
        "replacement_sites": sorted(profile.replacement_sites),
        "max_prompt_bytes": profile.max_prompt_bytes,
        "reset": asdict(profile.reset),
        "agent_tools": list(profile.agent_tools),
    }}


def verified_agent_session(profile: PolicyProfile) -> PolicySession:
    counter = {"value": 0}

    def reset(seed):
        counter["value"] += 1
        return {"status": "ok", "seed": seed, "session_id": f"session-{counter['value']}",
                "action_buffer_cleared": True, "agent_context_cleared": True,
                "episode_memory_cleared": True}

    return PolicySession(profile, InterventionSpec(), server_metadata=handshake(profile),
                         reset_callback=reset)


def test_action_spec_rejects_wrong_dimension_and_nonfinite_values():
    action = ActionSpec.legacy("array")
    assert action.validate_chunk([[0.0] * 8]) == [[0.0] * 8]
    assert action.hash == ActionSpec.legacy("array").hash
    assert action.hash != ActionSpec("joint_position", 7, "absolute", "arm", "radians", "none", 20).hash
    with pytest.raises(ValueError):
        action.validate_chunk([[0.0] * 7])
    with pytest.raises(ValueError):
        action.validate_chunk([[float("nan")] * 8])


def test_assist_and_replace_are_distinct_contracts():
    assist = InterventionSpec("semantic", "assist", "policy_input", "text")
    replace = InterventionSpec("decision", "replace", "committed_action", "none")
    assert assist.mode == "assist" and replace.mode == "replace"
    with pytest.raises(ValueError):
        InterventionSpec("decision", "replace", "policy_input", "none")
    with pytest.raises(ValueError):
        InterventionSpec("skill", "assist", "policy_input", "text")


def test_native_action_only_profile_rejects_module_replacement():
    result = check_capabilities(
        native_profile(), InterventionSpec("decision", "replace", "committed_action", "none"))
    assert not result.compatible
    assert "action-only" in ";".join(result.errors)


def test_legacy_connection_cannot_override_profile_capabilities():
    profile = native_profile()
    validate_connection_binding(profile, {
        "name": "openpi_array", "action_space": "array", "oracle_transports": ["text"]})
    with pytest.raises(ValueError, match="action_space"):
        validate_connection_binding(profile, {"name": "openpi_array", "action_space": "symbolic"})
    with pytest.raises(ValueError, match="transport"):
        validate_connection_binding(profile, {"name": "http_array", "action_space": "array"})


def test_strict_profile_requires_matching_server_handshake():
    profile = agent_profile()
    intervention = InterventionSpec("semantic", "assist", "policy_input", "tool")
    assert not check_capabilities(profile, intervention).compatible
    assert check_capabilities(profile, intervention, handshake(profile)).compatible


def test_agent_restart_requires_verified_session_and_fresh_reset_receipt():
    profile = agent_profile()
    with pytest.raises(ValueError, match="verified policy capability session"):
        LabferenceEnvAdapter(FakeDriver(), profile)

    def stale_reset(seed):
        return {"status": "ok", "seed": seed, "session_id": "same-session",
                "action_buffer_cleared": True, "agent_context_cleared": True,
                "episode_memory_cleared": True}

    session = PolicySession(profile, InterventionSpec(), server_metadata=handshake(profile),
                            reset_callback=stale_reset)
    env = LabferenceEnvAdapter(FakeDriver(), profile, session)
    env.reset({"name": "task"}, 1)
    with pytest.raises(RuntimeError, match="reused"):
        env.reset({"name": "task"}, 2)


def test_input_receipt_covers_every_declared_camera_and_oracle():
    profile = agent_profile()
    intervention = InterventionSpec("semantic", "assist", "policy_input", "structured")
    session = PolicySession(profile, intervention, server_metadata=handshake(profile))
    request = {"images": {"front": "a", "wrist": "b"},
               "depth": {"front_depth": "c", "wrist_depth": "d"},
               "oracle_context": {"profile": "semantic", "facts": {"reading": 10.0}}}
    digest = content_digest(request)
    prompt_digest = content_digest("prompt")
    audit = session.audit_request(
        request, adapter_boundary="fake",
        receipt={"request_digest": digest, "prompt_digest": prompt_digest},
        serialized_prompt="prompt", adapter_details={"input_mapping": {"front": "camera_1"}})
    assert audit.receipt_status == "verified" and audit.oracle_digest
    assert audit.adapter_details["input_mapping"]["front"] == "camera_1"
    with pytest.raises(ValueError):
        session.audit_request({**request, "images": {"front": "a"}}, adapter_boundary="fake")


class FakeDriver:
    def __init__(self):
        self.value = 0

    def metadata(self):
        return {"cameras": {name: {"shape": [4, 4, 3]} for name in
                            ("front", "wrist", "front_depth", "wrist_depth")}}

    def tasks(self):
        return [{"name": "task", "seed": 1}]

    def reset(self, task, seed):
        self.value = 0
        return self.observation()

    def observation(self):
        return {"images": {"front": "a", "wrist": "b"},
                "depth": {"front_depth": "c", "wrist_depth": "d"},
                "joint_positions": [0.0] * 9, "task_rule": "test"}

    def step(self, action):
        self.value += 1
        return self.observation()

    def wait(self, physics_steps):
        self.value += physics_steps
        return self.observation()

    def done(self):
        return self.value >= 2

    def result(self):
        return {"success": self.done()}

    def close(self):
        pass

    def validate_oracle_assist(self, profile, context, observation):
        TestOracleValidator().validate_assist(profile, context, observation)

    def validate_oracle_replacement(self, profile, site, original, replacement,
                                    runtime_context):
        TestOracleValidator().validate_replacement(
            profile, site, original, replacement, runtime_context)


def test_rpent_bridge_has_one_step_owner_and_explicit_replacement_hook():
    profile = agent_profile()
    env = LabferenceEnvAdapter(FakeDriver(), profile, verified_agent_session(profile))
    assert env.get_env_meta()["labference"]["step_owner"] == "rpent_env_adapter"
    env.reset({"name": "task"}, 1)
    controller = InterventionController(
        profile, InterventionSpec("decision", "replace", "committed_action", "none"),
        TestOracleValidator())
    toolkit = RPentToolkitBridge(env, controller,
                                 oracle_provider=lambda profile, *_: {"type": "wait"})
    toolkit.set_opportunity("op-1", 4)
    result = toolkit.commit_operation({"type": "stop"})
    assert result["committed_operation"] == {"type": "wait"}
    assert controller.public_records()[0]["site"] == "committed_action"


def test_active_intervention_requires_validator_and_registered_opportunity():
    profile = agent_profile()
    spec = InterventionSpec("semantic", "assist", "policy_input", "structured")
    with pytest.raises(ValueError, match="task-owned"):
        InterventionController(profile, spec)
    controller = InterventionController(profile, spec, TestOracleValidator())
    with pytest.raises(ValueError, match="registered decision opportunity"):
        controller.assist({}, {"facts": {}}, opportunity_id="unopened", evidence_step=-1)
    first = controller.assist({}, {"facts": {}}, opportunity_id="op-1", evidence_step=1)
    second = controller.assist({}, {"facts": {}}, opportunity_id="op-1", evidence_step=1)
    assert first == second and len(controller.public_records()) == 1


def test_external_oracle_validation_is_owned_by_driver():
    validator = DriverOracleValidator(FakeDriver())
    validator.validate_assist("semantic", {"facts": {}}, {"task_rule": "test"})
    with pytest.raises(ValueError, match="answer leakage"):
        validator.validate_assist(
            "semantic", {"correct_slot": "A1"}, {"task_rule": "test"})

    class MissingValidatorDriver(FakeDriver):
        validate_oracle_assist = None

    with pytest.raises(TypeError, match="validation methods"):
        DriverOracleValidator(MissingValidatorDriver())


def test_ranking_requires_versioned_non_gt_policy_and_excludes_skill_ceiling():
    profile = native_profile()
    eligible, revision, reasons = ranking_eligibility(
        profile, InterventionSpec(), {"policy_revision": "openpi-abc123"},
        uses_evaluator_state=False)
    assert eligible and revision == "openpi-abc123" and not reasons
    assert not ranking_eligibility(
        profile, InterventionSpec(), {}, uses_evaluator_state=False)[0]
    assert not ranking_eligibility(
        profile, InterventionSpec(), {"policy_revision": "rev"},
        uses_evaluator_state=True)[0]
    assert not ranking_eligibility(
        profile, InterventionSpec("skill", "replace", "executor", "none"),
        {"policy_revision": "rev"}, uses_evaluator_state=False)[0]
    from_handshake = ranking_eligibility(
        profile, InterventionSpec(), {}, uses_evaluator_state=False,
        server_metadata={"labference": {"policy_revision": "server-checkpoint"}})
    assert from_handshake[:2] == (True, "server-checkpoint")


def test_rpent_wait_reapplies_observation_projection_and_finish_is_terminal_only():
    class WaitLeakDriver(FakeDriver):
        def wait(self, physics_steps):
            value = super().wait(physics_steps)
            value["evaluator_state"] = {"correct_slot": "A1"}
            return value

    profile = agent_profile()
    driver = WaitLeakDriver()
    env = LabferenceEnvAdapter(driver, profile, verified_agent_session(profile))
    env.reset({"name": "task"}, 1)
    controller = InterventionController(profile, InterventionSpec())
    toolkit = RPentToolkitBridge(env, controller)
    packet = toolkit.wait(1)
    assert "evaluator_state" not in packet["observation"]
    assert toolkit.finish() == {"done": False}


def test_external_driver_cannot_expose_private_fields_or_undeclared_camera():
    class LeakyDriver(FakeDriver):
        def observation(self):
            value = super().observation()
            value["evaluator_state"] = {"correct_slot": "A1"}
            value["images"]["secret_gt_camera"] = "answer"
            return value

        def reset(self, task, seed):
            return self.observation()

    profile = agent_profile()
    env = LabferenceEnvAdapter(LeakyDriver(), profile, verified_agent_session(profile))
    with pytest.raises(ValueError, match="RGB channels differ"):
        env.reset({"name": "task", "correct_slot": "A1"}, 1)


def test_rpent_bridge_exposes_all_replace_sites_and_skill_preserves_operation():
    profile = agent_profile()
    env = LabferenceEnvAdapter(FakeDriver(), profile, verified_agent_session(profile))
    env.reset({"name": "task"}, 1)
    names = {spec["name"] for spec in build_rpent_tool_specs()}
    assert {"recognize_state", "commit_operation", "localize_target", "execute_operation"} <= names

    semantic = InterventionController(
        profile, InterventionSpec("semantic", "replace", "semantic_report", "none"),
        TestOracleValidator())
    bridge = RPentToolkitBridge(
        env, semantic, oracle_provider=lambda *_: {"occupied": ["A1"]})
    bridge.set_opportunity("op-semantic", 2)
    assert bridge.recognize_state({"occupied": []})["semantic_report"] == {"occupied": ["A1"]}

    executed = []
    skill = InterventionController(
        profile, InterventionSpec("skill", "replace", "executor", "none"),
        TestOracleValidator())
    bridge = RPentToolkitBridge(
        env, skill, oracle_provider=lambda *_: {},
        operation_executor=lambda operation: {"ok": False},
        skill_executor=lambda operation: executed.append(operation) or {"ok": True})
    bridge.set_opportunity("op-skill", 3)
    operation = {"type": "add", "weight_g": 50}
    bridge.commit_operation(operation)
    result = bridge.execute_operation(operation)
    assert executed == [operation] and result["result"]["ok"] is True
    assert bridge.metrics()["tool_call_counts"]["execute_operation"] == 1
    assert skill.public_records()[0]["site"] == "executor"


def test_rpent_bridge_rejects_executor_target_correction():
    profile = agent_profile()
    env = LabferenceEnvAdapter(FakeDriver(), profile, verified_agent_session(profile))
    env.reset({"name": "task"}, 1)
    controller = InterventionController(profile, InterventionSpec())
    bridge = RPentToolkitBridge(env, controller, operation_executor=lambda operation: {"ok": True})
    bridge.commit_operation({"type": "wait"})
    with pytest.raises(ValueError, match="differs from the committed"):
        bridge.execute_operation({"type": "stop"})


def test_openpi_wrapper_echoes_exact_request_digest_and_capabilities():
    profile = native_profile()

    class Policy:
        def infer(self, observation):
            return {"actions": [[0.0] * 8]}

    wrapper = LabferenceOpenPIWrapper(Policy(), profile, policy_revision="openpi-test-rev")
    prompt = "place the tube"
    observation = {"labference_request_digest": "abc",
                   "labference_prompt_digest": content_digest(prompt),
                   "prompt": prompt, "camera_1_rgb": "front", "camera_2_rgb": "wrist"}
    observation["labference_model_input_digest"] = content_digest(observation)
    result = wrapper.infer(observation)
    assert result["_labference_receipt"]["request_digest"] == "abc"
    assert wrapper.metadata()["labference"]["rgb_channels"] == ["front", "wrist"]
    assert wrapper.metadata()["labference"]["max_prompt_bytes"] == profile.max_prompt_bytes
    assert wrapper.metadata()["labference"]["episode_memory"] == "none"
    assert wrapper.metadata()["labference"]["history_protocol"] == "current_frame_only"
    assert wrapper.metadata()["labference"]["policy_revision"] == "openpi-test-rev"
    with pytest.raises(ValueError, match="checkpoint identifier"):
        LabferenceOpenPIWrapper(Policy(), profile, policy_revision=" ")
    with pytest.raises(ValueError, match="prompt differs"):
        wrapper.infer({"labference_request_digest": "abc",
                       "labference_prompt_digest": "wrong", "prompt": prompt,
                       "camera_1_rgb": "front", "camera_2_rgb": "wrist"})


def test_openpi_wrapper_forwards_auditable_stateful_reset():
    profile = PolicyProfile(
        "memory", "native_vla", "openpi", ActionSpec.legacy("array"),
        SensorSpec(("front", "wrist")), reset=ResetSpec("rpc"),
        episode_memory="model_state", history_protocol="online_once",
        strict_handshake=True)

    class Policy:
        def infer(self, observation):
            return {"actions": [[0.0] * 8]}

        def reset_episode(self, seed):
            return {"status": "ok", "seed": seed,
                    "session_id": f"episode-{seed}",
                    "action_buffer_cleared": True,
                    "agent_context_cleared": True,
                    "episode_memory_cleared": True}

    wrapper = LabferenceOpenPIWrapper(Policy(), profile, policy_revision="memory-v1")
    receipt = wrapper.reset_with_receipt(101)
    assert receipt["session_id"] == "episode-101"
    assert wrapper.metadata()["labference"]["episode_memory"] == "model_state"
    assert wrapper.metadata()["labference"]["history_protocol"] == "online_once"
    assert check_capabilities(
        profile, InterventionSpec(), wrapper.metadata()).compatible is True


def test_all_shipped_profile_and_intervention_files_validate():
    for path in (ROOT / "profiles").glob("*.json"):
        load_policy_profile(path)
    for path in (ROOT / "intervention_specs").glob("*.json"):
        InterventionSpec.from_dict(json.loads(path.read_text()))


def test_first_online_matrix_is_fixed_and_references_registered_files():
    plan = json.loads((ROOT / "experiment_plans/first_online_matrix.json").read_text())
    assert plan["schema_version"] == "labference/online-experiment-plan-v1"
    assert plan["seeds"] == [101, 102, 103, 104, 105]
    assert len(plan["tasks"]) == 4 and len({row["id"] for row in plan["tasks"]}) == 4
    for task in plan["tasks"]:
        assert (ROOT.parents[1] / task["config"]).is_file()
    for system in plan["systems"]:
        assert system["conditions"][0] == "base"
        for profile_path in system["profiles"].values():
            assert (ROOT.parents[1] / profile_path).is_file()

    centrifuge_rgb = load_policy_profile(ROOT / "profiles/rpent_agent_rgb.json")
    centrifuge_rgbd = load_policy_profile(ROOT / "profiles/rpent_agent_rgbd.json")
    assert centrifuge_rgb.sensors.rgb_channels == centrifuge_rgbd.sensors.rgb_channels
    assert not centrifuge_rgb.sensors.depth_channels
    assert centrifuge_rgbd.sensors.depth_channels


def test_matched_comparison_keeps_missing_pairs_in_denominator_and_modes_separate():
    semantic = InterventionSpec("semantic", "assist", "policy_input", "text")
    decision = InterventionSpec("decision", "replace", "committed_action", "none")
    plan = plan_matched_conditions(
        task_version="task/v1", contract_hash="hash", instance_ids=["i1", "i2"],
        policy_profile_id="policy", policy_profile_hash="profile-hash",
        policy_revision="rev", sensor_condition="rgb",
        action_spec_hash="action", toolset_hash="tools", reset_protocol_hash="reset",
        memory_revision="memory", budget_id="budget", interventions=[semantic, decision],
        instance_metadata={
            "i1": {"scene_layout_hash": "layout-1", "initial_state_hash": "state-1",
                   "scene_seed": 1, "mass_seed": 11, "pair_member": "a"},
            "i2": {"scene_layout_hash": "layout-2", "initial_state_hash": "state-2",
                   "scene_seed": 2, "mass_seed": 12, "pair_member": "b"},
        })
    shared = {"task_version": "task/v1", "contract_hash": "hash", "policy_profile_id": "policy",
              "policy_profile_hash": "profile-hash", "policy_revision": "rev",
              "sensor_condition": "rgb", "action_spec_hash": "action",
              "toolset_hash": "tools", "reset_protocol_hash": "reset",
              "memory_revision": "memory", "budget_id": "budget", "evaluation_valid": True,
              "scene_layout_hash": "layout-1", "initial_state_hash": "state-1",
              "scene_seed": 1, "mass_seed": 11, "pair_member": "a"}
    runs = [
        {**shared, "instance_id": "i1", "condition_id": "base",
         "online_legal_next_action_rate": 0.0, "physical_completion": 0.0},
        {**shared, "instance_id": "i1", "condition_id": condition_id(semantic),
         "condition_supported": True, "online_legal_next_action_rate": 1.0, "physical_completion": 0.0},
        {**shared, "instance_id": "i1", "condition_id": condition_id(decision),
         "condition_supported": True, "online_legal_next_action_rate": 1.0, "physical_completion": 1.0},
    ]
    result = compare_matched_conditions(plan, runs)
    assert result["planned_pairs"] == 4 and result["matched_pairs"] == 2
    assert result["conditions"][condition_id(semantic)]["coverage"] == 0.5
    assert result["conditions"][condition_id(semantic)]["paired_legal_choice_gain"] is None


def test_matched_comparison_rejects_different_physical_initial_state():
    semantic = InterventionSpec("semantic", "assist", "policy_input", "text")
    plan = plan_matched_conditions(
        task_version="task/v1", contract_hash="hash", instance_ids=["i1"],
        policy_profile_id="policy", policy_profile_hash="profile", policy_revision="rev",
        sensor_condition="rgb", action_spec_hash="action", toolset_hash="tools",
        reset_protocol_hash="reset", memory_revision="memory", budget_id="budget",
        interventions=[semantic], instance_metadata={"i1": {
            "scene_layout_hash": "layout", "initial_state_hash": "state-a",
            "scene_seed": 1, "mass_seed": 2, "pair_member": "a"}})
    common = {key: plan["pairs"][0][key] for key in MATCH_FIELDS if key != "initial_state_hash"}
    base = {**common, "instance_id": "i1", "condition_id": "base",
            "initial_state_hash": "state-a", "evaluation_valid": True}
    support = {**common, "instance_id": "i1", "condition_id": condition_id(semantic),
               "initial_state_hash": "state-b", "evaluation_valid": True,
               "condition_supported": True}
    with pytest.raises(ValueError, match="initial_state_hash"):
        compare_matched_conditions(plan, [base, support])
