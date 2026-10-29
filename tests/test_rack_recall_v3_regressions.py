"""Scoring and aggregation regressions for rack-placement-recall/v3."""

from __future__ import annotations

import json

import numpy as np
import pytest

from eai4s_eval.rack_recall_v3.cli import _summarize_autonomous_rollouts, main
from eai4s_eval.rack_recall_v3.core import (
    SCORING_VERSION,
    TARGET_TUBE,
    Contract,
    Measurement,
    RackPose,
    SLOTS,
    State,
    TubeObservation,
    advance_state,
    generate_coverage_manifest,
    generate_instance,
)
from eai4s_eval.rack_recall_v3.online import (
    BehaviorSelectionTracker,
    apply_runtime_validity,
    compare_pairs,
    measurement_to_dict,
    score_trace,
)
from eai4s_eval.rack_recall_v3.runner import (
    _bounded_cartesian_target, _preserve_tube_rotation, _rack_transport_errors,
    _reference_arm_step, _retract_clearance_reached, _tube_grasp_target,
    _upright_downward_rotation)
from eai4s_eval.rack_recall_v3.scene import (
    OCCLUDER_CLEARANCE_ABOVE_RACK_M, RECOVERY_MARKER_BORDER_M,
    RECOVERY_MARKER_HEIGHT_M, TRANSPORT_SUPPORT_DYNAMIC_FRICTION,
    TRANSPORT_SUPPORT_STATIC_FRICTION,
)


def test_recovery_marker_is_thin_and_non_supporting_by_contract():
    contract = Contract("easy", "single")
    assert RECOVERY_MARKER_HEIGHT_M < .002
    assert RECOVERY_MARKER_BORDER_M < contract.tube_cap_radius_m
    report = __import__(
        "eai4s_eval.rack_recall_v3.scene", fromlist=["validate_scene_geometry"]
    ).validate_scene_geometry(contract, generate_instance(contract, 101))
    assert report["recovery_marker"]["collision_enabled"] is False
    assert report["recovery_marker"]["table_contact_required"] is True


def test_rack_transport_uses_high_friction_support_and_covers_caps():
    contract = Contract("hard", "same_cap")
    report = __import__(
        "eai4s_eval.rack_recall_v3.scene", fromlist=["validate_scene_geometry"]
    ).validate_scene_geometry(contract, generate_instance(contract, 101))
    support = report["rack_transport_support"]
    assert support["static_friction"] == pytest.approx(
        TRANSPORT_SUPPORT_STATIC_FRICTION)
    assert support["dynamic_friction"] == pytest.approx(
        TRANSPORT_SUPPORT_DYNAMIC_FRICTION)
    assert support["friction_combine_mode"] == "max"
    seated_cap_above_rack = (
        contract.rack_support_top_height_m + contract.tube_length_m -
        contract.rack_height_m)
    assert OCCLUDER_CLEARANCE_ABOVE_RACK_M > seated_cap_above_rack


def test_rack_transport_error_is_measured_in_current_rack_frame():
    contract = Contract("hard", "same_cap")
    pose = RackPose((.2, .1, contract.table_surface_z_m), 90.0)
    mouth = contract.slot_world_center("A2", pose)
    sample = TubeObservation(
        TARGET_TUBE, (mouth[0] + .001, mouth[1] - .002, mouth[2]),
        (0.0, 0.0, -1.0), 0.0, True, True)
    errors = _rack_transport_errors(
        contract, pose, {TARGET_TUBE: sample}, {"A2": TARGET_TUBE})
    assert errors[TARGET_TUBE] == pytest.approx(np.hypot(.001, .002))


def test_recovery_requires_table_support_and_does_not_call_descent_a_drop():
    contract = Contract("easy", "single")
    instance = generate_instance(contract, 101)
    state = State.initial(instance)
    state.phase = "recovery"
    state.target_status = "held"
    center = (*instance.recovery_center_xy_m,
              contract.table_surface_z_m + contract.tube_length_m / 2.0)

    def measurement(step, *, landed, above=True):
        tube = TubeObservation(
            TARGET_TUBE, center, (0.0, 0.0, -1.0), 0.0,
            True, False, in_recovery_region=landed,
            position_delta_m=0.0, above_recovery_region=above,
            recovery_table_supported=landed)
        return Measurement(step, instance.rack_initial_pose, "recovery", False,
                           {TARGET_TUBE: tube}, True, True)

    events = advance_state(contract, instance, state, measurement(0, landed=False))
    assert state.target_status == "held"
    assert not any(row.get("event") == "target_dropped_during_recovery" for row in events)
    assert state.recovery_stable_count == 0

    for step in range(1, contract.final_stable_steps + 1):
        advance_state(contract, instance, state, measurement(step, landed=True))
    assert state.phase == "complete"
    assert state.target_status == "recovered"


def _pair(contract, pair_id="p1"):
    return {"pair_id": pair_id, "task_version": contract.task_version,
            "contract_hash": contract.hash, "scoring_version": SCORING_VERSION,
            "scoring_track": "explicit_slot_report",
            "history_protocol": "online_once", "oracle_profile": "none",
            "members": [{"member": "a"}, {"member": "b"}]}


def _pair_row(contract, pair_id, member, *, success=True, valid=True):
    return {"pair_id": pair_id, "member": member,
            "task_version": contract.task_version, "contract_hash": contract.hash,
            "scoring_version": SCORING_VERSION,
            "scoring_track": "explicit_slot_report",
            "history_protocol": "online_once", "oracle_profile": "none",
            "evaluation_valid": valid, "benchmark_success": success}


def _checkpoint_measurement(contract, pose, step, phase, occluded, delivered):
    mouth = contract.slot_world_center("A1", pose)
    center = (mouth[0], mouth[1],
              mouth[2] - contract.rack_bottom_support_depth_m +
              contract.tube_length_m / 2)
    tube = TubeObservation(TARGET_TUBE, center, (0.0, 0.0, -1.0), 0.0,
                           True, True, False, False, False, 0.0)
    return Measurement(step, pose, phase, occluded, {TARGET_TUBE: tube},
                       True, delivered)


def _checkpoint_trace(contract, instance, *, report_slot="A1", choice_slot=None):
    pose = instance.rack_initial_pose
    trace = [
        {"type": "policy_capabilities", "step": -1, "reports": ["recall"],
         "scoring_track": "explicit_slot_report", "history_protocol": "online_once",
         "episode_memory": "none", "ranking_eligible": False},
        {"type": "checkpoint_initialization", "step": -1, "slot": "A1"},
        {"type": "checkpoint_validation", "step": -1, "slot": "A1",
         "verified": True, "consecutive_stable_samples": contract.stable_steps},
        {"type": "measurement", "step": 0,
         "measurement": measurement_to_dict(
             _checkpoint_measurement(contract, pose, 0, "occluded", True, False))},
        {"type": "evidence_delivery", "step": 1, "sim_time_s": 1 / 120,
         "frame_ids": {"front": "front-step-1", "wrist": "wrist-step-1"},
         "history_protocol": "online_once"},
        {"type": "measurement", "step": 1,
         "measurement": measurement_to_dict(
             _checkpoint_measurement(contract, pose, 1, "recall", False, True))},
        {"type": "opportunity_open", "opportunity_id": "opportunity-0000",
         "step": 1, "evidence_step": 1, "interface": "reported",
         "legal_actions": [{"type": "retrieve", "slot": "A1"}]},
        {"type": "recall_report", "step": 1,
         "opportunity_id": "opportunity-0000", "slot": report_slot},
    ]
    if choice_slot is not None:
        trace.append({"type": "choice", "step": 1, "evidence_step": 1,
                      "opportunity_id": "opportunity-0000",
                      "source": "explicit_commitment",
                      "action": {"type": "retrieve", "slot": choice_slot}})
    return trace


def test_pair_comparison_keeps_missing_members_in_denominator():
    contract = Contract("hard", "same_cap")
    plan = [_pair(contract, "p1"), _pair(contract, "p2")]
    rows = [_pair_row(contract, "p1", member) for member in ("a", "b")]
    result = compare_pairs(plan, rows)
    assert result["complete_pair_count"] == 1
    assert result["state_pair_accuracy"] == 0.5
    assert result["missing_pair_ids"] == ["p2"]


def test_pair_comparison_uses_primary_slot_reasoning_result():
    contract = Contract("hard", "same_cap")
    plan = [_pair(contract, "reasoning")]
    rows = [_pair_row(contract, "reasoning", member)
            for member in ("a", "b")]
    result = compare_pairs(plan, rows)
    assert result["state_pair_accuracy"] == 1.0


def test_correct_checkpoint_slot_is_primary_success_without_physical_completion():
    contract = Contract("easy", "single")
    instance = generate_instance(contract, 101, reserved_slot="A1")
    pose = instance.rack_initial_pose
    occluded = _checkpoint_measurement(contract, pose, 0, "occluded", True, False)
    recall = _checkpoint_measurement(contract, pose, 1, "recall", False, True)
    trace = [
        {"type": "policy_capabilities", "step": -1, "reports": ["recall"],
         "scoring_track": "explicit_slot_report", "history_protocol": "online_once",
         "episode_memory": "none", "ranking_eligible": False},
        {"type": "checkpoint_initialization", "step": -1, "slot": "A1"},
        {"type": "checkpoint_validation", "step": -1, "slot": "A1",
         "verified": True, "consecutive_stable_samples": contract.stable_steps},
        {"type": "measurement", "step": 0,
         "measurement": measurement_to_dict(occluded)},
        {"type": "evidence_delivery", "step": 1, "sim_time_s": 1 / 120,
         "frame_ids": {"front": "front-step-1", "wrist": "wrist-step-1"},
         "history_protocol": "online_once"},
        {"type": "measurement", "step": 1,
         "measurement": measurement_to_dict(recall)},
        {"type": "opportunity_open", "opportunity_id": "opportunity-0000",
         "step": 1, "evidence_step": 1, "interface": "reported",
         "legal_actions": [{"type": "retrieve", "slot": "A1"}]},
        {"type": "recall_report", "step": 1,
         "opportunity_id": "opportunity-0000", "slot": "A1"},
    ]
    result = score_trace(contract, instance, trace)
    assert result["slot_reasoning_success"] is True
    assert result["benchmark_success"] is True
    assert result["evaluation_track"] == "verified_state_checkpoint"
    assert result["checkpoint_state_verified"] is True
    assert result["placement_skill_gate_applied"] is False
    assert result["placement_execution_applicable"] is False
    assert result["physical_completion"] is False
    assert result["valid_success"] is False
    assert result["ranking_success"] is False


def test_coverage_summary_uses_reasoning_not_placement_skill(tmp_path):
    contract = Contract("easy", "single")
    plan = generate_coverage_manifest(contract, (101,))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    member = plan["members"][0]
    run = tmp_path / member["member_id"]
    run.mkdir()
    (run / "run.json").write_text(json.dumps({
        "pair_id": member["member_id"], "runtime_status": "finished"}))
    (run / "scores.json").write_text(json.dumps({
        "task_version": contract.task_version,
        "contract_hash": contract.hash,
        "scoring_version": SCORING_VERSION,
        "condition_variant": contract.condition_variant,
        "history_protocol": "online_once",
        "evaluation_track": "verified_state_checkpoint",
        "evaluation_valid": True,
        "reached_selection_opportunity": True,
        "selection_identified": True,
        "benchmark_success": True,
        "reasoning_valid_success": True,
        "slot_reasoning_success": True,
        "valid_success": False,
    }))
    output = tmp_path / "summary.json"
    main(["summarize-coverage", "--plan", str(plan_path),
          "--runs", str(run), "--output", str(output)])
    summary = json.loads(output.read_text())
    assert summary["reasoning_passed_count"] == 1
    assert summary["physical_completion_passed_count"] == 0
    assert summary["slot_reasoning_rate_fixed_denominator"] == 1 / 12
    assert summary["physical_completion_rate_fixed_denominator"] == 0.0


def test_coverage_summary_counts_missing_runs_as_failures(tmp_path):
    contract = Contract("easy", "single")
    plan = generate_coverage_manifest(contract, (101,))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / "summary.json"
    main(["summarize-coverage", "--plan", str(plan_path), "--output", str(output)])
    summary = json.loads(output.read_text())
    assert summary["fixed_denominator"] == 12
    assert summary["coverage_rate"] == 0.0
    assert summary["slot_reasoning_rate_fixed_denominator"] == 0.0
    assert summary["physical_completion_rate_fixed_denominator"] == 0.0
    assert len(summary["missing_member_ids"]) == 12
    assert all(summary["per_slot"][slot]["planned"] == 1 for slot in SLOTS)


def test_autonomous_summary_preserves_unreached_and_unidentified_denominators():
    base = {
        "task_version": "rack-placement-recall/v3",
        "scoring_version": SCORING_VERSION,
        "scoring_track": "native_behavior_selection",
        "history_protocol": "online_once",
        "oracle_profile": "none",
        "policy_interface": "openpi_array",
        "evaluation_track": "autonomous_rollout",
        "evaluation_valid": True,
        "physical_completion": False,
    }
    rows = [
        {**base, "reached_selection_opportunity": False,
         "selection_identified": False, "slot_reasoning_success": False},
        {**base, "reached_selection_opportunity": True,
         "selection_identified": False, "slot_reasoning_success": None},
        {**base, "reached_selection_opportunity": True,
         "selection_identified": True, "slot_reasoning_success": False},
        {**base, "reached_selection_opportunity": True,
         "selection_identified": True, "slot_reasoning_success": True,
         "physical_completion": True},
    ]
    summary = _summarize_autonomous_rollouts(rows)
    assert summary["opportunity_reach_rate"] == .75
    assert summary["identification_rate_given_reached"] == 2 / 3
    assert summary["selection_accuracy_given_identified"] == .5
    assert summary["correct_selection_rate_all_runs"] == .25
    assert summary["result_class_counts"] == {
        "invalid": 0, "unreached": 1, "unidentified": 1,
        "wrong": 1, "correct": 1,
    }


def test_autonomous_summary_rejects_checkpoint_results():
    with pytest.raises(ValueError, match="autonomous_rollout"):
        _summarize_autonomous_rollouts([{
            "evaluation_track": "verified_state_checkpoint",
        }])


def test_incomplete_runtime_is_invalid_but_keeps_diagnostic_choice():
    result = apply_runtime_validity({
        "evaluation_valid": True,
        "selection_outcome": "correct",
        "slot_reasoning_success": True,
        "benchmark_success": True,
        "reasoning_valid_success": True,
        "ranking_success": True,
        "valid_success": False,
    }, "error")
    assert result["runtime_complete"] is False
    assert result["evaluation_valid"] is False
    assert result["selection_outcome"] == "invalid"
    assert result["benchmark_success"] is False
    assert result["slot_reasoning_success"] is True


def test_hover_separates_translation_from_near_hole_orientation_correction():
    assert _preserve_tube_rotation("hover", .100, 8.0, 4.0) is True
    assert _preserve_tube_rotation("hover", .010, 3.0, 4.0) is True
    assert _preserve_tube_rotation("hover", .010, 5.0, 4.0) is False
    assert _preserve_tube_rotation("recover", .100, 8.0, 4.0) is True
    assert _preserve_tube_rotation("upper_guide", .001, 1.0, 4.0) is True


def test_retract_completion_requires_vertical_and_lateral_clearance():
    target = np.asarray([.224, .156, 1.142])
    assert _retract_clearance_reached([.2265, .1545, 1.1330], target) is True
    assert _retract_clearance_reached([.2265, .1545, 1.1310], target) is False
    assert _retract_clearance_reached([.2310, .1560, 1.1420], target) is False
    assert _retract_clearance_reached([float("nan"), .1560, 1.1420], target) is False


def test_upright_target_preserves_heading_and_points_tube_axis_down():
    source = np.asarray([
        [0.0, -1.0, 0.0],
        [-0.98, 0.0, 0.20],
        [-0.20, 0.0, -0.98],
    ])
    target = _upright_downward_rotation(source)
    assert target[:, 2] == pytest.approx([0.0, 0.0, 1.0])
    assert target[:, 0] == pytest.approx([0.0, -1.0, 0.0])
    assert target.T @ target == pytest.approx(np.eye(3))
    assert np.linalg.det(target) == pytest.approx(1.0)


def test_tube_grasp_target_tracks_tilted_cap_in_xy():
    angle = np.radians(10.0)
    rotation = np.asarray([
        [np.cos(angle), 0.0, -np.sin(angle)],
        [0.0, 1.0, 0.0],
        [np.sin(angle), 0.0, np.cos(angle)],
    ])
    center = np.asarray([0.2, 0.1, 0.8])
    target = _tube_grasp_target(center, rotation, 0.069, 0.0065)
    downward_axis = rotation @ np.asarray([0.0, 0.0, -1.0])
    cap_grasp_point = center - downward_axis * 0.069

    assert target[:2] == pytest.approx(cap_grasp_point[:2])
    assert target[2] == pytest.approx(cap_grasp_point[2] + 0.0065)
    assert abs(target[0] - center[0]) > 0.01


def test_bounded_cartesian_target_limits_each_lift_step():
    current = np.asarray([0.1, 0.2, 0.8])
    target = np.asarray([0.1, 0.2, 0.92])
    stepped = _bounded_cartesian_target(current, target, 0.0004)

    assert np.linalg.norm(stepped - current) == pytest.approx(0.0004)
    assert stepped[:2] == pytest.approx(current[:2])
    assert stepped[2] > current[2]
    assert _bounded_cartesian_target(target, target, 0.0004) == pytest.approx(target)


def test_recovery_transport_uses_conservative_joint_step():
    assert _reference_arm_step({"type": "place_recovery"}) == pytest.approx(.018)
    assert _reference_arm_step({"type": "retrieve", "slot": "B2"}) == pytest.approx(.030)


def test_rmpflow_hover_is_limited_to_free_space():
    from eai4s_eval.rack_recall_v3.runner import _use_rmpflow_hover

    assert _use_rmpflow_hover("hover", .024)
    assert not _use_rmpflow_hover("hover", .015)
    assert not _use_rmpflow_hover("upper_guide", .024)


def test_invalid_pair_never_receives_consistency_credit():
    contract = Contract("hard", "same_cap")
    result = compare_pairs(
        [_pair(contract)],
        [_pair_row(contract, "p1", "a"),
         _pair_row(contract, "p1", "b", valid=False)])
    assert result["invalid_pair_count"] == 1
    assert result["state_pair_accuracy"] == 0.0


def test_pair_rejects_mixed_history_protocol():
    contract = Contract("hard", "same_cap")
    rows = [_pair_row(contract, "p1", member) for member in ("a", "b")]
    rows[1]["history_protocol"] = "current_frame_only"
    with pytest.raises(ValueError, match="mixes task contracts"):
        compare_pairs([_pair(contract)], rows)


def test_behavior_selection_requires_unique_dwell_and_descent_and_freezes():
    contract = Contract("easy", "single")
    pose = RackPose((.2, .1, contract.table_surface_z_m), 0.0)
    tracker = BehaviorSelectionTracker(contract)
    mouth = contract.slot_world_center("B2", pose)
    tube_top = mouth[2] - contract.rack_bottom_support_depth_m + contract.tube_length_m
    start = (mouth[0], mouth[1], tube_top + .030)
    assert tracker.update(start, pose, 0) is None
    dwell = int(contract.behavior_selection_dwell_s * contract.physics_hz)
    assert tracker.update((start[0], start[1], start[2] - .006), pose, dwell) == {
        "type": "retrieve", "slot": "B2"}
    other = contract.slot_world_center("C4", pose)
    assert tracker.update((other[0], other[1], tube_top + .020), pose, dwell + 10) is None


def test_checkpoint_without_positive_validation_is_invalid():
    contract = Contract("easy", "single")
    instance = generate_instance(contract, 101, reserved_slot="A1")
    measurement = _checkpoint_measurement(
        contract, instance.rack_initial_pose, 0, "recall", False, True)
    trace = [
        {"type": "policy_capabilities", "step": -1, "reports": ["recall"],
         "scoring_track": "explicit_slot_report", "history_protocol": "online_once",
         "episode_memory": "none", "ranking_eligible": False},
        {"type": "checkpoint_initialization", "step": -1, "slot": "A1"},
        {"type": "measurement", "step": 0,
         "measurement": measurement_to_dict(measurement)},
    ]
    result = score_trace(contract, instance, trace)
    assert result["evaluation_valid"] is False
    assert result["checkpoint_state_verified"] is False


def test_declared_report_track_is_not_overwritten_by_correct_action():
    contract = Contract("easy", "single")
    instance = generate_instance(contract, 101, reserved_slot="A1")
    result = score_trace(
        contract, instance,
        _checkpoint_trace(contract, instance, report_slot="B2", choice_slot="A1"))
    assert result["explicit_slot_report_correct"] is False
    assert result["symbolic_slot_selection_correct"] is True
    assert result["slot_reasoning_success"] is False
    assert result["selection_outcome"] == "wrong"


def test_response_before_evidence_delivery_invalidates_run():
    contract = Contract("easy", "single")
    instance = generate_instance(contract, 101, reserved_slot="A1")
    trace = _checkpoint_trace(contract, instance)
    trace.insert(4, {"type": "recall_report", "step": 0,
                     "opportunity_id": "opportunity-0000", "slot": "A1"})
    result = score_trace(contract, instance, trace)
    assert result["evaluation_valid"] is False
    assert result["selection_outcome"] == "invalid"
