"""Acceptance tests for rack-placement-recall/v3."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eai4s_eval.oracle_runtime import PolicyProfile

from eai4s_eval.rack_recall_v3.cli import load_config, main
from eai4s_eval.rack_recall_v3.core import (
    CAP_COLORS,
    DISTRACTOR_TUBES,
    SLOTS,
    TARGET_TUBE,
    Contract,
    Measurement,
    RackPose,
    State,
    TubeObservation,
    advance_state,
    evaluate_tube,
    generate_coverage_manifest,
    generate_instance,
)
from eai4s_eval.rack_recall_v3.diagnostics import (
    check_policy_request,
    oracle_context,
    policy_observation,
)
from eai4s_eval.rack_recall_v3.scene import validate_scene_geometry


ROOT = Path("eai4s_eval/rack_recall_v3")
CONFIGS = (
    "easy.json",
    "hard_same_cap.json",
    "hard_distinct_cap.json",
    "native_easy.json",
    "native_hard_same_cap.json",
    "native_hard_distinct_cap.json",
)


def _inserted(contract: Contract, pose: RackPose, tube: str, slot: str,
              *, fingers: bool = False) -> TubeObservation:
    mouth = contract.slot_world_center(slot, pose)
    center = (mouth[0], mouth[1],
              mouth[2] - contract.rack_bottom_support_depth_m +
              contract.tube_length_m / 2)
    return TubeObservation(tube, center, (0.0, 0.0, -1.0), 0.0,
                           not fingers, True, fingers, fingers, False, 0.0)


@pytest.mark.parametrize("name", CONFIGS)
def test_all_v2_configs_validate(name):
    source, contract = load_config(ROOT / "configs" / name)
    assert contract.task_version == "rack-placement-recall/v3"
    assert source["scene"]["scene_revision"] == "rack-recall-step-3x4-v3.1"


def test_step_geometry_matches_frozen_measurements():
    contract = Contract("easy")
    assert contract.row_pitch_m == pytest.approx(0.050)
    assert contract.column_pitch_m == pytest.approx(0.065)
    assert contract.socket_radius_m == pytest.approx(0.021)
    assert contract.tube_scale == pytest.approx(1.20)
    assert contract.tube_cap_scale == pytest.approx(1.25)
    assert 2 * contract.tube_radius_m == pytest.approx(0.033864)
    assert 2 * contract.tube_cap_radius_m == pytest.approx(0.043125)
    assert contract.tube_length_m == pytest.approx(0.14040)
    assert contract.rack_bottom_support_depth_m == pytest.approx(.100)
    assert contract.rack_height_m == pytest.approx(.180)
    assert contract.rack_mouth_height_m == pytest.approx(.170)
    assert contract.rack_lower_guide_height_m == pytest.approx(.100)
    assert contract.rack_support_top_height_m == pytest.approx(.070)
    assert contract.rack_support_thickness_m == pytest.approx(.008)
    assert contract.socket_depth_m == pytest.approx(.070)
    assert contract.tube_cap_radius_m > contract.socket_radius_m
    assert contract.tcp_to_finger_pad_z_m == pytest.approx(.0065)
    assert contract.radial_tolerance_m <= contract.socket_radius_m - contract.tube_radius_m
    centers = {slot: contract.slot_local_center(slot) for slot in SLOTS}
    assert len(set(centers.values())) == 12
    assert centers["A2"][0] - centers["A1"][0] == pytest.approx(0.065)
    assert centers["B1"][1] - centers["A1"][1] == pytest.approx(0.050)
    assert all(center[2] == pytest.approx(0.170) for center in centers.values())


def test_asset_manifest_preserves_sources_parts_and_locator_policy():
    manifest = json.loads((ROOT / "assets/asset_manifest.json").read_text())
    assert manifest["assets"]["rack"]["source_sha256"] == (
        "59d2281d664ff9848e1f5d86d1ba99205c1cae81e779c5bbce35b975b19f874e")
    assert manifest["assets"]["tube"]["source_sha256"] == (
        "4c8cd308c92cd824ca5cd07b8b984924f26fc50c72e6a821404fe25f16d27978")
    assert manifest["assets"]["rack"]["calibration_locator_count"] == 12
    assert manifest["assets"]["rack"]["calibration_locators_rendered"] is False
    assert manifest["assets"]["rack"]["calibration_locators_collidable"] is False
    roles = manifest["assets"]["tube"]["part_roles"]
    assert "CutExtrude1" in roles["cap"] and "CutExtrude3" in roles["body"]
    assert (ROOT / manifest["assets"]["rack"]["source"]).is_file()
    assert (ROOT / manifest["assets"]["tube"]["output"]).is_file()


def test_same_cap_and_distinct_cap_are_balanced_without_unique_target_shortcut():
    same = generate_instance(Contract("hard", "same_cap"), 101)
    distinct = generate_instance(Contract("hard", "distinct_cap"), 101,
                                 layout_seed=same.layout_seed,
                                 motion_seed=same.motion_seed)
    assert len(set(dict(same.cap_colors).values())) == 1
    distinct_colors = dict(distinct.cap_colors)
    assert set(distinct_colors) == {TARGET_TUBE, *DISTRACTOR_TUBES}
    assert set(distinct_colors.values()) == set(CAP_COLORS)
    assert len(set(distinct_colors.values())) == 4
    assert same.distractor_slots == distinct.distractor_slots
    assert same.rack_initial_pose == distinct.rack_initial_pose
    assert same.rack_final_pose == distinct.rack_final_pose


def test_color_seed_is_independent_of_layout_motion_and_policy():
    contract = Contract("hard", "distinct_cap")
    first = generate_instance(contract, 101)
    recolored = generate_instance(contract, 999, layout_seed=first.layout_seed,
                                  motion_seed=first.motion_seed,
                                  policy_seed=first.policy_seed,
                                  color_seed=first.color_seed + 7)
    assert first.distractor_slots == recolored.distractor_slots
    assert first.rack_initial_pose == recolored.rack_initial_pose
    assert first.rack_final_pose == recolored.rack_final_pose
    assert first.cap_colors != recolored.cap_colors


def test_checkpoint_slot_is_reserved_before_distractor_sampling():
    contract = Contract("hard", "same_cap")
    for slot in SLOTS:
        instance = generate_instance(contract, 103, reserved_slot=slot)
        assert slot not in instance.distractor_slots


def test_public_task_description_does_not_reveal_target_color_or_slot():
    instance = generate_instance(Contract("hard", "distinct_cap"), 104)
    serialized = json.dumps(instance.public())
    assert "target_cap_color" not in serialized
    assert "actual_placement_slot" not in serialized
    assert instance.cap_color(TARGET_TUBE) not in instance.public()["rule"]


def test_semantic_oracle_exposes_visible_colors_without_target_identity():
    contract = Contract("hard", "distinct_cap")
    instance = generate_instance(contract, 105)
    state = State.initial(instance)
    context = oracle_context("semantic", contract, state,
                             instance.rack_initial_pose, instance=instance)
    facts = context["facts"]
    assert set(facts["visible_cap_color_by_slot"]) == set(instance.distractor_slots)
    assert TARGET_TUBE not in json.dumps(facts)


def test_policy_history_declares_delivery_mode_and_rejects_private_state():
    request = policy_observation(
        {"front": {"frame": 1}, "wrist": {"frame": 1}}, [0.0] * 9,
        "retrieve the previously placed tube", [], evidence_step=10,
        observation_id="policy-000001", sim_time_s=1.0,
        frame_ids={"front": "front-step-10", "wrist": "wrist-step-10"},
        appearance_condition="distinct_cap", history_delivery_mode="online_once")
    assert request["history_delivery_mode"] == "online_once"
    check_policy_request(request)
    replayed = dict(request)
    replayed["authorized_history"] = [{"images": {"front": "old-frame"}}]
    with pytest.raises(ValueError, match="cannot replay"):
        check_policy_request(replayed)
    request["actual_placement_slot"] = "A1"
    with pytest.raises(ValueError):
        check_policy_request(request)


def test_all_twelve_slots_use_the_same_physical_insertion_predicate():
    contract = Contract("easy")
    pose = RackPose((0.2, 0.1, contract.table_surface_z_m), 90.0)
    for slot in SLOTS:
        result = evaluate_tube(contract, _inserted(contract, pose, TARGET_TUBE, slot), pose)
        assert result["actual_slot"] == slot
        assert result["stable_sample"] is True


def test_old_cap_supported_pose_is_not_a_seated_tube():
    contract = Contract("easy")
    pose = RackPose((0.2, 0.1, contract.table_surface_z_m), 0.0)
    mouth = contract.slot_world_center("B2", pose)
    center = (mouth[0], mouth[1],
              mouth[2] - (contract.tube_length_m / 2 -
                           contract.tube_cap_height_m))
    observation = TubeObservation(
        TARGET_TUBE, center, (0.0, 0.0, -1.0), 0.0,
        True, True, False, False, False, 0.0)
    result = evaluate_tube(contract, observation, pose)
    assert result["geometry_pass"] is False
    assert result["bottom_support_error_m"] > contract.bottom_support_tolerance_m


def test_actual_stable_slot_not_planned_slot_advances_state():
    contract = Contract("easy")
    instance = generate_instance(contract, 106)
    state = State.initial(instance)
    state.target_status = "held"
    tubes = {TARGET_TUBE: _inserted(contract, instance.rack_initial_pose,
                                    TARGET_TUBE, "C4")}
    all_events = []
    for step in range(contract.stable_steps):
        all_events.extend(advance_state(contract, instance, state, Measurement(
            step, instance.rack_initial_pose, "placement", False, tubes, False, False)))
    assert state.actual_placement_slot == "C4"
    assert state.phase == "transition"
    assert any(row.get("actual_slot") == "C4" for row in all_events)


@pytest.mark.parametrize("condition,difficulty", [
    ("single", "easy"), ("same_cap", "hard"), ("distinct_cap", "hard")])
def test_coverage_manifest_has_fixed_twelve_by_five_denominator(condition, difficulty):
    plan = generate_coverage_manifest(Contract(difficulty, condition))
    assert plan["fixed_denominator"] == 60
    assert plan["scoring_version"] == "labference/rack-recall-memory-selection-v2"
    assert {row["checkpoint_slot"] for row in plan["members"]} == set(SLOTS)
    assert all(sum(row["checkpoint_slot"] == slot for row in plan["members"]) == 5
               for slot in SLOTS)
    assert {row["initial_yaw_deg"] for row in plan["members"]} == {0.0, 90.0, 180.0, 270.0}
    assert {row["target_cap_color"] for row in plan["members"]} == set(CAP_COLORS)


def test_combined_coverage_cli_builds_180_members(tmp_path):
    output = tmp_path / "coverage.json"
    main(["plan-coverage", "--configs",
          str(ROOT / "configs/easy.json"),
          str(ROOT / "configs/hard_same_cap.json"),
          str(ROOT / "configs/hard_distinct_cap.json"),
          "--output", str(output)])
    plan = json.loads(output.read_text())
    assert plan["fixed_denominator"] == 180
    assert len({row["member_id"] for row in plan["members"]}) == 180
    assert all(not Path(row["config"]).is_absolute() for row in plan["members"])


def test_scene_validation_reports_cad_migration_and_claim_boundary():
    contract = Contract("hard", "distinct_cap")
    instance = generate_instance(contract, 107)
    report = validate_scene_geometry(contract, instance)
    assert report["locator_count"] == 12
    assert report["locator_rendered"] is False
    assert report["locator_collision_enabled"] is False
    assert report["appearance_claim"] == "appearance_cue_assisted_retrieval"
    assert report["tube_scale"] == pytest.approx(1.20)
    assert report["tube_cap_scale"] == pytest.approx(1.25)
    assert report["socket_clearance_m"] == pytest.approx(.004068)
    assert report["rack_collision_shape"] == (
        "twelve_open_two_plate_annular_guides_with_passive_funnels_and_visible_support")
    assert report["passive_guide"]["slot_count"] == 12
    assert report["passive_guide"]["height_m"] == pytest.approx(0.016)
    assert report["passive_guide"]["throat_radial_clearance_m"] == pytest.approx(0.002)
    assert report["passive_guide"]["answer_conditioned"] is False
    assert report["passive_guide"]["adjacent_openings_connected"] is False
    assert report["recovery_marker"] == {
        "shape": "square_border",
        "size_m": pytest.approx(.160),
        "collision_enabled": False,
        "table_contact_required": True,
        "release_tip_clearance_m": pytest.approx(.005),
    }
    assert report["tube_collision_shape"] == "tapered_body_plus_faceted_cap_and_top_lip"
    assert report["grasp_geometry"]["cap_top_lip_overhang_m"] > 0
    assert report["grasp_geometry"]["cap_static_friction"] > 2.0
    assert report["grasp_geometry"]["cap_grip_facet_count"] == 16
    assert report["rack_bottom_support"] == "visible_250x170x8mm_collision_plate"
    assert report["rack_visual_normalization"]["translate_m"] == pytest.approx(
        [-.090, -.130, .165])
    assert report["rack_plate_heights_m"] == {
        "overall_top": pytest.approx(.180),
        "upper_mouth": pytest.approx(.170),
        "lower_guide_surface": pytest.approx(.100),
        "support_top": pytest.approx(.070),
    }
    assert math.isclose(report["minimum_slot_pitch_m"], 0.05)
    contact = report["grasp_contact_parameters"]
    assert contact["friction_combine_mode"] == "max"
    assert contact["torsional_patch_radius_m"] == pytest.approx(.020)
    assert contact["min_torsional_patch_radius_m"] == pytest.approx(.005)
    insertion = report["insertion_contact_parameters"]
    assert insertion["tube_static_friction"] == pytest.approx(.12)
    assert insertion["tube_dynamic_friction"] == pytest.approx(.08)
    assert insertion["tube_friction_combine_mode"] == "min"


def test_invalid_difficulty_condition_combinations_are_rejected():
    with pytest.raises(ValueError):
        Contract("easy", "same_cap")
    with pytest.raises(ValueError):
        Contract("hard", "single")
    with pytest.raises(ValueError):
        Contract("hard", "visual_memory")


def test_memory_profile_declares_online_once_and_real_reset():
    profile_path = Path(
        "eai4s_eval/oracle_runtime/profiles/openpi_rack_recall_memory_rgb.json")
    profile = PolicyProfile.from_dict(json.loads(profile_path.read_text()))
    assert profile.episode_memory == "model_state"
    assert profile.history_protocol == "online_once"
    assert profile.reset.mode == "rpc"
    with pytest.raises(ValueError, match="requires rpc or restart"):
        PolicyProfile(
            name="invalid-memory", system_type="native_vla", adapter="openpi",
            action=profile.action, sensors=profile.sensors,
            episode_memory="model_state", history_protocol="online_once")


def test_tube_cap_remains_outside_the_guide_aperture():
    with pytest.raises(ValueError, match="wider than the rack opening"):
        Contract("easy", tube_cap_radius_m=.0207)


def test_insertion_mode_is_explicit_and_validated():
    assert Contract("easy").insertion_mode == "guided_gravity_release"
    with pytest.raises(ValueError, match="unknown insertion mode"):
        Contract("easy", insertion_mode="teleport")
