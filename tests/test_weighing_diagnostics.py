"""Input-boundary and scoped Oracle tests without Isaac Sim."""

import pytest

from eai4s_eval.weighing_v1.core import Contract, Instance, State, WEIGHTS
from eai4s_eval.weighing_v1.diagnostics import (
    PolicyCapabilities, VerifiedSkillExecutor, check_policy_request, deliver_input,
    evaluate_policy_checkpoint, oracle_context, policy_observation, score_checkpoint,
)
from eai4s_eval.weighing_v1.loadcell import format_grams, low_pass, segment_state


def fixture_state():
    contract = Contract("easy")
    instance = Instance("easy", 1, 1, 1, 160, 150, (10,), (), "pair")
    state = State(displayed_g=150.0, sensor_valid=True, sample_supported=True,
                  reading_stable=True, sample_untouched=True, released=True)
    return contract, instance, state


def fixture_observation(instance):
    return policy_observation({"front": "image", "wrist": "image"}, [0.0] * 9,
                              instance.public()["rule"], list(WEIGHTS), 150.0, True)


def fixture_geometry():
    frame = {"position_world_m": [0.0, 0.0, 1.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    return {"frame": "world", "units": "m", **{
        group: {str(weight): frame for weight in WEIGHTS}
        for group in ("grasp_frames", "pan_frames", "return_frames")}}


def test_loadcell_format_and_filter_are_physical_readout_helpers():
    assert format_grams(100.04) == "100.0"
    assert format_grams(100.06) == "100.1"
    assert segment_state("100.0")["d0_b"]
    assert not segment_state("100.0")["d0_g"]
    assert segment_state("----")["d2_g"]
    value = low_pass(0.0, 100.0, 1 / 120, .25)
    assert 0.0 < value < 100.0


def test_private_mass_and_answer_do_not_cross_policy_boundary():
    _, instance, _ = fixture_state()
    view = fixture_observation(instance)
    check_policy_request(view)
    with pytest.raises(ValueError):
        check_policy_request(dict(view, evaluator_state={"sample_mass_g": 150}))
    with pytest.raises(ValueError):
        check_policy_request(dict(view, history=[{"correct_weight": 10}]))
    with pytest.raises(ValueError):
        check_policy_request(dict(view, task_rule="Add 10 g to reach the target"))


def test_semantic_decision_geometry_oracles_remain_scoped():
    contract, instance, state = fixture_state()
    view = fixture_observation(instance)
    facts = {"visible_weight_locations_g": {"10": "table"}}
    geometry = fixture_geometry()
    semantic = oracle_context("semantic", contract, instance, state, view, facts, geometry)
    assert semantic["facts"]["visible_weight_locations_g"] == {"10": "table"}
    assert "sample_mass_g" not in str(semantic)
    decision = oracle_context("decision", contract, instance, state, view, facts, geometry)
    assert decision["advice"] == {"type": "add", "weight_g": 10}
    assert "position_world_m" not in str(decision)
    candidate = oracle_context("geometry", contract, instance, state, view, facts, geometry)
    assert set(candidate["candidate_geometry"]["grasp_frames"]) == {str(v) for v in WEIGHTS}
    assert "advice" not in candidate


def test_visual_base_input_and_semantic_readout_are_separate():
    contract, instance, state = fixture_state()
    visual = policy_observation({"front": "image", "wrist": "image", "display": "readout_image"},
                                [0.0] * 9, instance.public()["rule"], list(WEIGHTS), None, None)
    assert visual["display_reading_g"] is None and visual["stable_indicator"] is None
    support = oracle_context("semantic", contract, instance, state, visual,
                             {"display_reading_g": 150.0, "stable_indicator": True,
                              "visible_weight_locations_g": {}}, fixture_geometry())
    delivered = deliver_input(visual, support, "structured", {"structured"})
    assert delivered["oracle_context"]["facts"]["display_reading_g"] == 150.0
    assert "sample_mass_g" not in str(delivered)


def test_declared_transport_no_fallback_or_double_delivery():
    contract, instance, state = fixture_state()
    view = fixture_observation(instance)
    context = oracle_context("decision", contract, instance, state, view, {}, fixture_geometry())
    structured = deliver_input(view, context, "structured", {"structured"})
    assert "oracle_context" in structured and "oracle_prompt" not in structured
    text = deliver_input(view, context, "text", {"text"})
    assert text["oracle_prompt"].startswith("<ORACLE>")
    with pytest.raises(ValueError):
        deliver_input(view, context, "text", {"structured"})
    with pytest.raises(ValueError):
        check_policy_request(dict(structured, oracle_prompt=text["oracle_prompt"]))


def test_online_decision_checkpoint_scores_correct_legality():
    contract, instance, state = fixture_state()
    view = fixture_observation(instance)
    caps = PolicyCapabilities(frozenset({"decision"}), frozenset({"structured"}), "symbolic", True)
    record = evaluate_policy_checkpoint(
        contract, instance, state, view, {}, fixture_geometry(),
        lambda request: {"committed_action": {"type": "add", "weight_g": 10}}, caps,
        instance_id="easy-101", layer="decision", reset_policy=lambda: None)
    assert record["status"] == "scored" and record["scores"]["decision"] == "pass"
    wrong = score_checkpoint({"instance_id": "easy-101", "layer": "decision", "supported": True,
                              "committed_action": {"type": "add", "weight_g": 20},
                              "legal_actions": [{"type": "add", "weight_g": 10}]})
    assert wrong["decision"] == "fail"
    unsupported = evaluate_policy_checkpoint(
        contract, instance, state, view, {}, fixture_geometry(), lambda request: {},
        PolicyCapabilities(frozenset(), frozenset({"structured"}), "symbolic", True),
        instance_id="easy-101", layer="semantic", reset_policy=lambda: None)
    assert unsupported["status"] == "unsupported"


def test_skill_executor_cannot_change_selected_weight():
    contract, instance, state = fixture_state()
    view = fixture_observation(instance)
    executor = VerifiedSkillExecutor(
        "validated-seed-101", frozenset({"add"}),
        lambda operation: {"executed_action": {"type": "add", "weight_g": 20}},
        lambda operation: True, verified=True)
    record = evaluate_policy_checkpoint(
        contract, instance, state, view, {}, fixture_geometry(), lambda request: {},
        PolicyCapabilities(frozenset(), frozenset({"structured"}), "symbolic", True),
        instance_id="easy-101", layer="execution", profile="skill",
        selected_action={"type": "add", "weight_g": 10},
        reset_policy=lambda: None, skill_executor=executor)
    assert record["status"] == "invalid_executor_result"


def test_verified_skill_executor_scores_independent_physical_result():
    contract, instance, state = fixture_state()
    view = fixture_observation(instance)
    selected = {"type": "add", "weight_g": 10}
    executor = VerifiedSkillExecutor(
        "validated-seed-101", frozenset({"add"}),
        lambda operation: {"executed_action": operation},
        lambda operation: operation == selected, verified=True)
    record = evaluate_policy_checkpoint(
        contract, instance, state, view, {}, fixture_geometry(), lambda request: {},
        PolicyCapabilities(frozenset({"execution"}), frozenset({"structured"}), "symbolic", True),
        instance_id="easy-101", layer="execution", profile="skill",
        selected_action=selected, reset_policy=lambda: None, skill_executor=executor)
    assert record["reference_executor_success"] is True
    assert record["scores"]["execution"] == "pass"


def test_openpi_adapter_delivers_display_image_without_hidden_numeric_readout():
    import base64
    from io import BytesIO
    from PIL import Image
    from eai4s_eval.weighing_v1.policy import OpenPIArrayPolicy

    buffer = BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    encoded = {"base64": base64.b64encode(buffer.getvalue()).decode("ascii")}
    _, instance, _ = fixture_state()
    view = policy_observation({"front": encoded, "wrist": encoded, "display": encoded, "weights": encoded},
                              [0.0] * 9, instance.public()["rule"], list(WEIGHTS), None, None)

    class Client:
        observation = None

        def infer(self, observation):
            self.observation = observation
            return {"action": [[0.0] * 8]}

    adapter = object.__new__(OpenPIArrayPolicy)
    adapter.client = Client()
    from dataclasses import replace
    from eai4s_eval.oracle_runtime import PolicyProfile
    profile = PolicyProfile.legacy(
        {"name": "openpi_array", "action_space": "array", "oracle_transports": ["text"]},
        ("front", "wrist", "display", "weights"))
    adapter.profile = replace(profile, input_mapping={
        "front": "camera_1_rgb", "wrist": "camera_2_rgb",
        "display": "camera_3_rgb", "weights": "camera_4_rgb"})
    adapter.session = None
    assert adapter.infer_full(view)["action_chunk"] == [[0.0] * 8]
    assert adapter.client.observation["camera_3_rgb"].shape == (4, 4, 3)
    assert adapter.client.observation["camera_4_rgb"].shape == (4, 4, 3)
    assert '"display_reading_g":null' in adapter.client.observation["prompt"]
