"""Simulator-independent acceptance tests for the feedback-weighing contract."""

from dataclasses import asdict
import math

import pytest

from eai4s_eval.weighing_v1.cli import load_config
from eai4s_eval.weighing_v1.core import (Contract, Instance, Measurement, State, advance,
                                          balanced_instances, generate_instance, legal_actions,
                                          paired_instances)
from eai4s_eval.weighing_v1.online import (OpportunityBook, compare_oracle_pairs,
                                           compare_pairs, execution_subgoal_confirmed,
                                           score_trace)
from eai4s_eval.weighing_v1.scene import generate_scene_layout


def _measurement(step, weights=(), grams=150.0, **changes):
    data = dict(step=step, sensor_valid=True, measured_g=grams, displayed_g=grams,
                supported_weights_g=tuple(weights), sample_supported=True,
                sample_untouched=True, released=True, no_external_support=True,
                no_hand_press=True, max_linear_speed_m_s=0.0,
                max_angular_speed_rad_s=0.0)
    data.update(changes)
    return Measurement(**data)


def _settled(contract, instance, weights=(), grams=150.0):
    state = State()
    for step in range(contract.stable_steps + 2):
        advance(contract, instance, state, _measurement(step, weights, grams))
    return state


def test_configs_and_balanced_state_pairs():
    for difficulty in ("easy", "hard"):
        _, contract = load_config(f"eai4s_eval/weighing_v1/configs/{difficulty}.json")
        _, native = load_config(f"eai4s_eval/weighing_v1/configs/native_{difficulty}.json")
        assert native.hash == contract.hash
        for seed in range(101, 106):
            instance = generate_instance(contract, seed)
            assert instance.target_g == instance.sample_mass_g + sum(instance.desired_weights_g)
            left, right = paired_instances(contract, seed)
            assert left.target_g == right.target_g
            assert left.scene_seed == right.scene_seed
            assert left.desired_weights_g != right.desired_weights_g
            assert left.sample_mass_g != right.sample_mass_g


def test_scene_randomization_is_reproducible_and_independent_of_task_answer():
    for difficulty in ("easy", "hard"):
        contract = Contract(difficulty)
        first = generate_instance(contract, 101, mass_seed=101, scene_seed=20101)
        different_layout = generate_instance(contract, 101, mass_seed=101, scene_seed=20102)
        assert (first.target_g, first.sample_mass_g, first.desired_weights_g,
                first.initial_weights_g) == (
                    different_layout.target_g, different_layout.sample_mass_g,
                    different_layout.desired_weights_g, different_layout.initial_weights_g)
        assert generate_scene_layout(first.scene_seed).hash != generate_scene_layout(
            different_layout.scene_seed).hash
        left, right = paired_instances(contract, 101)
        assert generate_scene_layout(left.scene_seed) == generate_scene_layout(right.scene_seed)
    hashes = set()
    for seed in range(101, 201):
        layout = generate_scene_layout(seed)
        assert layout == generate_scene_layout(seed)
        assert len(layout.weight_spawn_xy) == 4
        assert "target" not in str(layout.public()) and "desired" not in str(layout.public())
        for index, position in enumerate(layout.weight_spawn_xy):
            assert 0.068 <= position[0] <= 0.317
            assert -0.195 <= position[1] <= -0.185
            assert math.dist(position, layout.sample_spawn_xy) >= 0.046
            assert all(math.dist(position, other) >= 0.040
                       for other in layout.weight_spawn_xy[:index])
        hashes.add(layout.hash)
    assert len(hashes) == 100


def test_balanced_instance_plan_and_hard_pairs_cover_all_seed_shapes():
    easy = balanced_instances(Contract("easy"))
    hard = balanced_instances(Contract("hard"))
    assert len(easy) == 12 and len(hard) == 30
    assert len({(item.target_g, item.desired_weights_g) for item in easy}) == 12
    assert len({(item.target_g, item.desired_weights_g) for item in hard}) == 30
    assert easy == balanced_instances(Contract("easy"))
    assert len(balanced_instances(Contract("easy"), repetitions=2)) == 24
    with pytest.raises(ValueError, match="repetitions"):
        balanced_instances(Contract("easy"), repetitions=0)
    for seed in range(1000):
        left, right = paired_instances(Contract("hard"), seed)
        assert left.scene_seed == right.scene_seed
        assert left.initial_weights_g == right.initial_weights_g
        assert left.target_g == right.target_g
        assert left.desired_weights_g != right.desired_weights_g
        assert left.sample_mass_g != right.sample_mass_g


def test_pair_comparison_rejects_changed_authored_layout():
    layout_hash = generate_scene_layout(10101).hash
    plan = [{"pair_id": "easy-101", "condition": "none", "scene_layout_hash": layout_hash,
             "members": [{"member": "a", "seed": 101, "scene_seed": 10101, "mass_seed": 101},
                         {"member": "b", "seed": 102, "scene_seed": 10101, "mass_seed": 102}]}]
    rows = [{"pair_id": "easy-101", "condition": "none", "member": member,
             "seed": seed, "scene_seed": 10101, "mass_seed": seed,
             "scene_layout_hash": layout_hash, "first_choice_correct": True}
            for member, seed in (("a", 101), ("b", 102))]
    assert compare_pairs(plan, rows)["state_pair_accuracy"] == 1.0
    with pytest.raises(ValueError, match="authored layouts"):
        compare_pairs(plan, [rows[0], {**rows[1], "scene_layout_hash": "different"}])
    with pytest.raises(ValueError, match="seed differs"):
        compare_pairs(plan, [rows[0], {**rows[1], "mass_seed": 999}])


def test_policy_camera_bundle_contains_readout_and_weight_evidence():
    import numpy as np
    from eai4s_eval.weighing_v1.runner import _encoded_images

    class Camera:
        def get_rgba(self):
            return np.zeros((4, 4, 4), dtype=np.uint8)

    cameras = {name: Camera() for name in ("front", "wrist", "display", "weights")}
    images = _encoded_images(cameras)
    assert set(images) == set(cameras)
    assert all(image["mime_type"] == "image/png" and image["base64"]
               for image in images.values())


def test_paired_run_rejects_seed_override_before_launch(tmp_path):
    from eai4s_eval.weighing_v1.runner import run_supervised

    with pytest.raises(ValueError, match="cannot be overridden"):
        run_supervised("eai4s_eval/weighing_v1/configs/easy.json", str(tmp_path),
                       pair_seed=101, scene_seed=9001)


def test_validate_reports_layout_without_private_answer(capsys):
    import json
    from eai4s_eval.weighing_v1.cli import main

    main(["validate", "--config", "eai4s_eval/weighing_v1/configs/easy.json",
          "--seed", "101"])
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] and report["scene_layout_hash"] == generate_scene_layout(10101).hash
    assert "sample_mass_g" not in str(report) and "desired_weights_g" not in str(report)


def test_balanced_plan_drives_paired_and_oracle_schedules(tmp_path, capsys):
    import json
    from eai4s_eval.weighing_v1.cli import main, planned_seeds

    config = "eai4s_eval/weighing_v1/configs/easy.json"
    source, contract = load_config(config)
    instances = tmp_path / "instances.json"
    pairs = tmp_path / "pairs.json"
    oracles = tmp_path / "oracles.json"
    main(["plan-instances", "--config", config, "--output", str(instances)])
    assert len(planned_seeds(instances, contract, source["scene"]["scene_revision"])) == 12
    main(["plan-pairs", "--config", config, "--instance-plan", str(instances),
          "--output", str(pairs)])
    main(["plan-oracle", "--config", config, "--instance-plan", str(instances),
          "--output", str(oracles)])
    capsys.readouterr()
    assert len(json.loads(pairs.read_text())["pairs"]) == 12
    assert len(json.loads(oracles.read_text())["pairs"]) == 36
    corrupted = json.loads(instances.read_text())
    corrupted["runs"][0]["scene_layout_hash"] = "not-the-authored-layout"
    instances.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="differs from generated"):
        planned_seeds(instances, contract, source["scene"]["scene_revision"])
    corrupted["runs"][0]["scene_layout_hash"] = generate_scene_layout(
        corrupted["runs"][0]["scene_seed"]).hash
    corrupted["runs"].pop()
    instances.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="stratum equally"):
        planned_seeds(instances, contract, source["scene"]["scene_revision"])


def test_easy_choose_from_live_reading_and_hard_allow_multiple_orders():
    easy = Contract("easy")
    left, right = paired_instances(easy, 101)
    assert legal_actions(easy, left, _settled(easy, left, grams=left.sample_mass_g)) == [
        {"type": "add", "weight_g": left.desired_weights_g[0]}]
    assert legal_actions(easy, right, _settled(easy, right, grams=right.sample_mass_g)) != [
        {"type": "add", "weight_g": left.desired_weights_g[0]}]
    hard = Contract("hard")
    instance = Instance("hard", 1, 100, 1, 220, 70, (50, 100), (10, 20), "pair")
    state = _settled(hard, instance, weights=(10, 20), grams=100)
    actions = legal_actions(hard, instance, state)
    assert {tuple(sorted(action.items())) for action in actions} == {
        (("type", "remove"), ("weight_g", 10)),
        (("type", "remove"), ("weight_g", 20)),
        (("type", "add"), ("weight_g", 50)),
        (("type", "add"), ("weight_g", 100)),
    }


def test_unstable_partial_load_and_disturbance_reset_stability():
    contract = Contract("easy")
    instance = Instance("easy", 1, 1, 1, 160, 150, (10,), (), "pair")
    state = _settled(contract, instance)
    assert state.reading_stable
    advance(contract, instance, state, _measurement(state.last_step + 1, (10,), 160))
    assert legal_actions(contract, instance, state) == [{"type": "wait"}]
    for step in range(state.last_step + 1, state.last_step + contract.final_steps + 4):
        advance(contract, instance, state, _measurement(step, (10,), 160))
    assert legal_actions(contract, instance, state) == [{"type": "stop"}]
    advance(contract, instance, state, _measurement(state.last_step + 1, (10,), 160,
                                                   no_hand_press=False))
    assert state.final_steps == 0
    assert legal_actions(contract, instance, state) == [{"type": "wait"}]


def test_opportunity_deduplicates_frames_and_retry_opens_once():
    contract = Contract("easy")
    instance = Instance("easy", 1, 1, 1, 160, 150, (10,), (), "pair")
    state = _settled(contract, instance)
    book = OpportunityBook(contract, instance, "native")
    opened = book.update(state, 125, evidence_step=125)
    assert opened and opened["legal_actions"] == [{"type": "add", "weight_g": 10}]
    assert book.update(state, 126, evidence_step=126) is None
    retry = book.retry_after_failure(state, 127, evidence_step=127)
    assert retry["opportunity_id"] != opened["opportunity_id"]
    assert book.update(state, 128, evidence_step=128) is None


def test_score_replays_measurements_not_logged_success_or_legal_claims():
    contract = Contract("easy", physics_hz=4, stable_window_s=1, final_window_s=3)
    instance = Instance("easy", 1, 1, 1, 160, 150, (10,), (), "pair")
    trace = []
    for step in range(5):
        trace.append({"type": "measurement", "step": step,
                      "measurement": asdict(_measurement(step))})
    trace.append({"type": "opportunity_open", "opportunity_id": "a", "step": 4,
                  "evidence_step": 4, "interface": "reported",
                  "legal_actions": [{"type": "add", "weight_g": 10}]})
    trace.append({"type": "decision", "step": 4,
                  "action": {"type": "add", "weight_g": 10}})
    trace.append({"type": "choice", "opportunity_id": "a", "step": 4,
                  "evidence_step": 4, "source": "explicit_commitment",
                  "action": {"type": "add", "weight_g": 10}})
    for step in range(5, 19):
        trace.append({"type": "measurement", "step": step,
                      "measurement": asdict(_measurement(step, (10,), 160))})
    trace.append({"type": "opportunity_open", "opportunity_id": "stop", "step": 18,
                  "evidence_step": 18, "interface": "reported",
                  "legal_actions": [{"type": "stop"}]})
    trace.append({"type": "decision", "step": 18, "action": {"type": "stop"}})
    trace.append({"type": "choice", "opportunity_id": "stop", "step": 18,
                  "evidence_step": 18, "source": "explicit_commitment", "action": {"type": "stop"}})
    scores = score_trace(contract, instance, trace)
    assert scores["evaluation_valid"] and scores["task_completion"] and scores["valid_success"]
    assert scores["online_legal_next_action_rate"] == 1.0
    assert scores["first_ready_choice_correct"] is True
    assert not score_trace(contract, instance, trace[:-3])["valid_success"]
    trace[5]["legal_actions"] = [{"type": "add", "weight_g": 20}]
    invalid = score_trace(contract, instance, trace)
    assert not invalid["evaluation_valid"]


def test_recovery_requires_actual_failed_attempt():
    contract = Contract("easy", physics_hz=4, stable_window_s=1, final_window_s=3)
    instance = Instance("easy", 1, 1, 1, 160, 150, (10,), (), "pair")
    trace = [{"type": "measurement", "step": step,
              "measurement": asdict(_measurement(step, (), 150))} for step in range(5)]
    trace += [{"type": "measurement", "step": step,
               "measurement": asdict(_measurement(step, (10,), 160))} for step in range(5, 19)]
    assert not score_trace(contract, instance, trace)["recovered_completion"]
    trace.append({"type": "operation_result", "step": 3, "status": "dropped"})
    assert not score_trace(contract, instance, trace)["recovered_completion"]
    assert not score_trace(contract, instance, trace)["evaluation_valid"]
    trace.pop()
    trace.extend([{"type": "operation_start", "step": 2,
                   "action": {"type": "add", "weight_g": 10}},
                  {"type": "operation_result", "step": 4, "started_step": 2,
                   "action": {"type": "add", "weight_g": 10}, "status": "failed",
                   "physical_supported_weights_g": []}])
    assert score_trace(contract, instance, trace)["recovered_completion"]


def test_planned_pairs_keep_missing_runs_in_denominator():
    plan = [{"pair_id": "one", "condition": "none"},
            {"pair_id": "two", "condition": "none"}]
    rows = [{"pair_id": "one", "condition": "none", "member": "a",
             "first_choice_correct": True},
            {"pair_id": "one", "condition": "none", "member": "b",
             "first_choice_correct": True}]
    result = compare_pairs(plan, rows)
    assert result["planned_pairs"] == 2 and result["present_pairs"] == 1
    assert result["state_pair_accuracy"] == 0.5


def test_expert_pairs_are_engineering_only():
    result = compare_pairs([{"pair_id": "one", "condition": "none"}], [
        {"pair_id": "one", "condition": "none", "member": "a",
         "first_choice_correct": True, "uses_evaluator_state": True},
        {"pair_id": "one", "condition": "none", "member": "b",
         "first_choice_correct": True, "uses_evaluator_state": True}])
    assert result["state_pair_accuracy"] == 1.0
    assert result["engineering_only"] and not result["ranking_eligible"]


def test_execution_checkpoint_waits_for_released_stable_load():
    contract = Contract("easy")
    instance = Instance("easy", 1, 1, 1, 160, 150, (10,), (), "pair")
    state = _settled(contract, instance, weights=(10,), grams=160)
    action = {"type": "add", "weight_g": 10}
    assert execution_subgoal_confirmed(action, state, observed_change=True)
    assert not execution_subgoal_confirmed(action, state, observed_change=False)
    state.released = False
    assert not execution_subgoal_confirmed(action, state, observed_change=True)
    state.released = True
    state.reading_stable = False
    assert not execution_subgoal_confirmed(action, state, observed_change=True)
    state.reading_stable = True
    state.supported_weights_g = ()
    assert not execution_subgoal_confirmed(action, state, observed_change=True)


def test_unknown_modes_and_config_mass_not_in_public_instruction():
    with pytest.raises(ValueError):
        Contract("medium")
    instance = generate_instance(Contract("easy"), 101)
    assert "sample_mass_g" not in instance.public()
    assert "desired_weights_g" not in instance.public()


def test_load_cell_overload_resets_stability_without_invalidating_sensor():
    contract = Contract("easy")
    instance = Instance("easy", 1, 1, 1, 160, 150, (10,), (), "pair")
    state = _settled(contract, instance)
    assert state.reading_stable
    advance(contract, instance, state,
            _measurement(state.last_step + 1, grams=999.9, overloaded=True))
    assert state.sensor_valid and not state.reading_stable and state.stable_steps == 0


def test_video_rgb_is_frozen_before_asynchronous_encoding():
    import numpy as np

    from eai4s_eval.weighing_v1.runner import _rgb

    source = np.zeros((2, 3, 4), dtype=np.uint8)
    source[..., :3] = (10, 20, 30)
    frame = _rgb(source)
    source[..., :3] = 255
    assert frame.shape == (2, 3, 3)
    assert frame.flags.c_contiguous
    assert tuple(frame[0, 0]) == (10, 20, 30)


def test_oracle_compare_requires_complete_matched_non_expert_pairs():
    contract = Contract("easy")
    plan = [{"pair_id": "easy-101-semantic", "task_version": contract.task_version,
             "contract_hash": contract.hash, "seed": 101,
             "base_condition": "none", "support_condition": "semantic"},
            {"pair_id": "easy-102-semantic", "task_version": contract.task_version,
             "contract_hash": contract.hash, "seed": 102,
             "base_condition": "none", "support_condition": "semantic"}]
    shared = {"task_version": contract.task_version, "contract_hash": contract.hash,
              "scene_seed": 10101, "mass_seed": 101, "policy_id": "same-policy",
              "sensor_condition": "front+wrist_rgb+visible_scale_display",
              "action_space": "array", "max_steps_budget": 7200,
              "uses_evaluator_state": False, "evaluation_valid": True}
    rows = [{**shared, "seed": 101, "oracle_profile": "none",
             "online_legal_next_action_rate": .4, "valid_success": False},
            {**shared, "seed": 101, "oracle_profile": "semantic",
             "online_legal_next_action_rate": .8, "valid_success": True}]
    incomplete = compare_oracle_pairs(plan, rows)
    assert incomplete["planned_pairs"] == 2 and incomplete["matched_pairs"] == 1
    assert not incomplete["official_gain_available"]
    assert incomplete["paired_choice_gain"] is None
    complete = compare_oracle_pairs(plan[:1], rows)
    assert complete["paired_choice_gain"] == pytest.approx(.4)
    assert complete["paired_completion_gain"] == 1.0
    with pytest.raises(ValueError, match="incomparable"):
        compare_oracle_pairs(plan[:1], [rows[0], {**rows[1], "policy_id": "different"}])
