import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from eai4s_eval.centrifuge_v2.cli import load_contract
from eai4s_eval.centrifuge_v2.core import Contract, State
from eai4s_eval.centrifuge_v2.diagnostics import (
    OnlineEvaluationMonitor, check_policy_request, verify_online_scores,
)
from eai4s_eval.centrifuge_v2.online_v2 import (
    EVALUATION_VERSION, OpportunityBook, native_behavior, online_state_pair_record, score_online_v2,
)
from eai4s_eval.centrifuge_v2.policy import HttpArrayPolicy, PolicyServiceError, validate_action_chunk


def measured_trace():
    frame = {"type": "geometry_frame", "step": -1,
             "slot_mouths_world_m": {"S00": [0, 0, .867], "S06": [.1, 0, .867]},
             "insertion_axis_world": [0, 0, -1]}
    raw = {"position_world_m": [.2, 0, .76], "axis_world": [0, 0, 1],
           "linear_speed_m_s": 0, "position_delta_m": 0,
           "released": True, "supported": False, "gripper_contact": False,
           "both_fingers_contact": False, "nearest_slot": "S06",
           "radial_error_m": .1, "insertion_depth_m": 0,
           "axis_error_deg": 0}
    return [frame, {"type": "measurement", "step": 30,
                    "gripper_width_m": .08,
                    "tubes": {"NMR_01": raw, "NMR_02": dict(raw, position_world_m=[.2, .1, .76])}}]


class OnlineV2Tests(unittest.TestCase):
    def test_native_configs_validate_and_legacy_remains_unchanged(self):
        configs = Path(__file__).resolve().parents[1] / "eai4s_eval/centrifuge_v2/configs"
        for name in ("native_easy.json", "native_hard.json", "native_hard_fixed_subgoal.json",
                     "easy.json", "hard.json"):
            self.assertEqual(load_contract(str(configs / name)).task_version, "centrifuge-opposite/v2")
        self.assertNotIn("evaluation_version", json.loads((configs / "easy.json").read_text())["evaluation"])

    def test_unchanged_state_does_not_open_repeated_opportunities(self):
        contract = Contract("hard")
        state = State("hard")
        book = OpportunityBook(contract, "reported")
        first = book.update(state, 30, evidence_step=30)
        self.assertIsNotNone(first)
        self.assertIsNone(book.update(state, 60, evidence_step=60))
        self.assertIsNotNone(book.record_choice({"type": "pick", "tube": "NMR_01"}, 30,
                                                 "explicit_commitment", 30))
        self.assertIsNone(book.record_choice({"type": "pick", "tube": "NMR_01"}, 60,
                                             "explicit_commitment", 60))
        retry = book.retry_after_failure(state, 90, evidence_step=90)
        self.assertEqual(retry["trigger"], "observable_operation_failure")

    def test_physical_selection_requires_contact_or_socket_entry(self):
        contract = Contract("hard")
        state = State("hard")
        sample = {tube: {"both_fingers_contact": False} for tube in ("NMR_01", "NMR_02")}
        self.assertIsNone(native_behavior(state, sample, contract, .01))
        sample["NMR_01"]["both_fingers_contact"] = True
        self.assertEqual(native_behavior(state, sample, contract, .02),
                         {"type": "pick", "tube": "NMR_01"})
        state.tube_status["NMR_01"] = "held"
        sample["NMR_01"].update(nearest_slot="S04", radial_error_m=.001,
                                insertion_depth_m=.001, axis_error_deg=0)
        self.assertIsNone(native_behavior(state, sample, contract, .02))
        sample["NMR_01"]["insertion_depth_m"] = .01
        self.assertEqual(native_behavior(state, sample, contract, .02),
                         {"type": "insert", "tube": "NMR_01", "slot": "S04"})

    def test_measured_reference_ignores_forged_legal_set_and_duplicate_choice(self):
        contract = Contract("hard")
        trace = measured_trace()
        trace += [{"type": "opportunity_open", "opportunity_id": "opportunity-0000",
                   "step": 30, "evidence_step": 30,
                   "legal_actions": [{"type": "pick", "tube": "NMR_01"}]},
                  {"type": "decision", "step": 30, "action": {"type": "pick", "tube": "NMR_01"}},
                  {"type": "choice", "opportunity_id": "opportunity-0000", "step": 30,
                   "evidence_step": 30, "source": "explicit_commitment",
                   "action": {"type": "pick", "tube": "NMR_01"}},
                  {"type": "choice", "opportunity_id": "opportunity-0000", "step": 30,
                   "evidence_step": 30, "source": "explicit_commitment",
                   "action": {"type": "pick", "tube": "NMR_01"}}]
        scored = score_online_v2(contract, trace, {"decision"})
        self.assertEqual(scored["opportunity_count"], 1)
        self.assertEqual(scored["correct_choice_count"], 1)
        self.assertEqual(scored["online_legal_next_action_rate"], 1)
        stream = StringIO()
        monitor = OnlineEvaluationMonitor(contract, trace, stream, {"decision"},
                                          evaluation_version=EVALUATION_VERSION)
        monitor.snapshot(30, "choice")
        snapshots = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertTrue(verify_online_scores(contract, trace, snapshots, {"decision"},
                                             EVALUATION_VERSION)["matching"])
        trace[2]["legal_actions"] = [{"type": "stop"}]
        self.assertEqual(score_online_v2(contract, trace)["invalid_opportunity_reference_count"], 1)

    def test_native_choice_cannot_be_claimed_without_matching_physics(self):
        contract = Contract("hard")
        trace = measured_trace() + [
            {"type": "opportunity_open", "opportunity_id": "o0", "step": 30,
             "evidence_step": 30, "legal_actions": [{"type": "pick", "tube": "NMR_01"}]},
            {"type": "choice", "opportunity_id": "o0", "step": 30, "evidence_step": 30,
             "source": "physical_interaction", "action": {"type": "pick", "tube": "NMR_01"}},
        ]
        score = score_online_v2(contract, trace)
        self.assertEqual(score["identified_choice_count"], 0)
        self.assertEqual(score["invalid_opportunity_reference_count"], 1)

    def test_pick_choice_uses_pre_lift_state_and_current_contact_frame(self):
        contract = Contract("hard")
        trace = measured_trace()
        lifted = dict(trace[-1]["tubes"]["NMR_01"], position_world_m=[.2, 0, .86],
                      gripper_contact=True, left_finger_contact=True,
                      right_finger_contact=True, both_fingers_contact=True,
                      released=False)
        trace += [
            {"type": "opportunity_open", "opportunity_id": "o0", "step": 30,
             "evidence_step": 30, "legal_actions": [{"type": "pick", "tube": "NMR_01"}]},
            {"type": "measurement", "step": 31, "finger_joints_m": [.01, .01],
             "tubes": {"NMR_01": lifted, "NMR_02": trace[-1]["tubes"]["NMR_02"]}},
            {"type": "choice", "opportunity_id": "o0", "step": 31,
             "evidence_step": 31, "reference_step": 30,
             "source": "physical_interaction", "action": {"type": "pick", "tube": "NMR_01"}},
        ]
        score = score_online_v2(contract, trace)
        self.assertEqual(score["correct_choice_count"], 1)
        self.assertEqual(score["invalid_opportunity_reference_count"], 0)

    def test_native_wait_is_a_temporal_opportunity_not_a_missing_action_choice(self):
        contract = Contract("hard", stable_steps=1, final_stable_steps=3)
        frame = {"type": "geometry_frame", "step": -1,
                 "slot_mouths_world_m": {"S00": [0, 0, .867], "S06": [.1, 0, .867]},
                 "insertion_axis_world": [0, 0, -1]}
        first = {"position_world_m": [0, 0, .84], "axis_world": [0, 0, 1],
                 "linear_speed_m_s": 0, "position_delta_m": 0,
                 "released": True, "supported": True, "gripper_contact": False}
        second = dict(first, position_world_m=[.2, 0, .76], supported=False)
        trace = [frame,
                 {"type": "measurement", "step": 30,
                  "tubes": {"NMR_01": first, "NMR_02": second}},
                 {"type": "measurement", "step": 31,
                  "tubes": {"NMR_01": first,
                            "NMR_02": dict(first, position_world_m=[.1, 0, .84])}},
                 {"type": "opportunity_open", "opportunity_id": "wait0", "step": 31,
                  "evidence_step": 31, "interface": "native",
                  "legal_actions": [{"type": "wait"}]}]
        score = score_online_v2(contract, trace)
        self.assertEqual(score["opportunity_count"], 1)
        self.assertEqual(score["scored_action_opportunity_count"], 0)
        self.assertEqual(score["temporal_opportunity_count"], 1)
        self.assertIsNone(score["online_legal_next_action_rate"])

    def test_optional_native_report_is_scored_separately_from_physical_choice(self):
        contract = Contract("hard")
        trace = measured_trace() + [
            {"type": "opportunity_open", "opportunity_id": "o0", "step": 30,
             "evidence_step": 30, "legal_actions": [{"type": "pick", "tube": "NMR_01"}]},
            {"type": "policy_report", "step": 30, "evidence_step": 30,
             "opportunity_id": "o0", "visible_slots": [],
             "report": {"semantic_report": {},
                        "committed_action": {"type": "pick", "tube": "NMR_01"}}},
        ]
        scores = score_online_v2(contract, trace, {"semantic", "decision"})
        self.assertEqual(scores["diagnostic"]["interface_scores"]["decision"]["accuracy"], 1)
        self.assertIsNone(scores["diagnostic"]["interface_scores"]["semantic"]["accuracy"])
        self.assertEqual(scores["diagnostic"]["interface_scores"]["semantic"]["unavailable"], 1)
        self.assertEqual(scores["identified_choice_count"], 0)
        self.assertEqual(scores["online_legal_next_action_rate"], 0)

    def test_array_policy_rejects_bad_shapes_and_service_response(self):
        self.assertEqual(len(validate_action_chunk([[0.0] * 8])), 1)
        for bad in ([], [[0.0] * 7], [[float("nan")] * 8], [[0.0] * 8] * 41):
            with self.assertRaises(PolicyServiceError):
                validate_action_chunk(bad)
        with patch.object(HttpArrayPolicy, "_post", return_value={"action": [[0.1] * 8]}):
            self.assertEqual(HttpArrayPolicy("http://localhost:8999").infer({})[0][0], .1)

    def test_verified_online_pair_record_reads_first_physical_choice(self):
        contract = Contract("hard", stable_steps=1, final_stable_steps=1)
        frame = {"type": "geometry_frame", "step": -1,
                 "slot_mouths_world_m": {"S00": [0, 0, .867], "S06": [.1, 0, .867]},
                 "insertion_axis_world": [0, 0, -1]}
        first = {"position_world_m": [0, 0, .84], "axis_world": [0, 0, 1],
                 "linear_speed_m_s": 0, "position_delta_m": 0, "released": True,
                 "supported": True, "gripper_contact": False,
                 "both_fingers_contact": False}
        second = dict(first, position_world_m=[.2, 0, .76], supported=False)
        held = dict(second, position_world_m=[.1, 0, .857], released=False,
                    gripper_contact=True, both_fingers_contact=True,
                    nearest_slot="S06", radial_error_m=0,
                    insertion_depth_m=.01, axis_error_deg=0)
        trace = [frame,
                 {"type": "measurement", "step": 30, "gripper_width_m": .04,
                  "tubes": {"NMR_01": first, "NMR_02": second}},
                 {"type": "measurement", "step": 31, "gripper_width_m": .02,
                  "tubes": {"NMR_01": first, "NMR_02": held}},
                 {"type": "opportunity_open", "opportunity_id": "o0", "step": 31,
                  "evidence_step": 31,
                  "legal_actions": [{"type": "insert", "tube": "NMR_02", "slot": "S06"}]},
                 {"type": "choice", "opportunity_id": "o0", "step": 31,
                  "evidence_step": 31, "source": "physical_interaction",
                  "action": {"type": "insert", "tube": "NMR_02", "slot": "S06"}}]
        plan = [{"pair_id": "state-101", "variant": "a", "scene_seed": 101,
                 "first_actual_slot": "S00", "task_version": contract.task_version,
                 "contract_hash": contract.hash,
                 "state_source": "verified_preload_and_reference_pick_plan",
                 "legal_actions": [{"type": "insert", "tube": "NMR_02", "slot": "S06"}]}]
        evaluation = dict(contract.canonical(), evaluation_version=EVALUATION_VERSION,
                          policy_interface={"name": "http_array", "action_space": "array",
                                            "supported_reports": [], "oracle_transports": ["text"],
                                            "resettable": True})
        manifest = {"evaluation": evaluation, "checkpoint_requested_slot": "S00",
                    "checkpoint_verified": True, "runtime_status": "time_limit",
                    "steps_executed": 211, "checkpoint_decision_window_steps": 180,
                    "scene_seed": 101, "decision_seed": 101,
                    "task_version": contract.task_version, "contract_hash": contract.hash,
                    "policy_id": "test", "policy_uses_evaluator_state": False,
                    "sensor_condition": "front+wrist_rgb", "policy_track": "native",
                    "evaluation_version": EVALUATION_VERSION}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.json").write_text(json.dumps(manifest))
            (root / "events.jsonl").write_text("\n".join(map(json.dumps, trace)) + "\n")
            stream = StringIO()
            OnlineEvaluationMonitor(contract, trace, stream, set(),
                                    evaluation_version=EVALUATION_VERSION).snapshot(31, "terminal")
            (root / "online_scores.jsonl").write_text(stream.getvalue())
            result = online_state_pair_record(tmp, plan)
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["committed_action"]["slot"], "S06")

    def test_online_checkpoint_cli_reports_unsupported_without_starting_isaac(self):
        from eai4s_eval.centrifuge_v2 import cli
        config = Path(__file__).resolve().parents[1] / "eai4s_eval/centrifuge_v2/configs/native_easy.json"
        with TemporaryDirectory() as tmp, patch("sys.argv", [
                "centrifuge", "checkpoints", "--online-config", str(config),
                "--checkpoint-slot", "S00", "--output", tmp, "--kind", "execution"]), \
                patch("eai4s_eval.centrifuge_v2.cli.run_supervised") as run, \
                patch("sys.stdout", new_callable=StringIO):
            cli.main()
            run.assert_not_called()
            result = json.loads((Path(tmp) / "checkpoint_result.json").read_text())
        self.assertEqual(result["status"], "unsupported")

    def test_fixed_subgoal_input_is_declared_and_geometry_checked(self):
        request = {"images": {}, "joint_positions": [0] * 9, "task_rule": "insert",
                   "visible_slot_ids": ["S00"],
                   "selected_operation": {"type": "insert", "tube": "NMR_02", "slot": "S06"},
                   "interaction_pose": {"slot": "S06", "frame": "world", "units": "m",
                                        "mouth_world_m": [.1, 0, .867], "axis_world": [0, 0, -1]}}
        with self.assertRaises(ValueError):
            check_policy_request(request)
        check_policy_request(request, allow_fixed_subgoal=True)
        request["interaction_pose"]["mouth_world_m"] = [float("nan"), 0, .867]
        with self.assertRaises(ValueError):
            check_policy_request(request, allow_fixed_subgoal=True)

    def test_supported_execution_checkpoint_forwards_declared_condition(self):
        from eai4s_eval.centrifuge_v2 import cli
        config = Path(__file__).resolve().parents[1] / "eai4s_eval/centrifuge_v2/configs/native_hard_fixed_subgoal.json"
        fake = {"diagnostic": {"interface_scores": {"execution": {"accuracy": 0.0}}}}
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "run.json").write_text(json.dumps({"checkpoint_verified": True}))
            with patch("sys.argv", [
                    "centrifuge", "checkpoints", "--online-config", str(config),
                    "--checkpoint-slot", "S00", "--output", tmp, "--kind", "execution"]), \
                    patch("eai4s_eval.centrifuge_v2.cli.run_supervised", return_value=fake) as run, \
                    patch("sys.stdout", new_callable=StringIO):
                cli.main()
                self.assertEqual(run.call_args.kwargs["checkpoint_kind"], "execution")
                result = json.loads((Path(tmp) / "checkpoint_result.json").read_text())
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["accuracy"], 0.0)

    def test_invalid_online_score_is_not_counted_as_a_scored_rollout(self):
        from eai4s_eval.centrifuge_v2.cli import online_rollout_record
        contract = Contract("hard")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.json").write_text(json.dumps({
                "runtime_status": "time_limit", "difficulty": "hard", "seed": 101,
                "oracle_profile": "none", "task_version": contract.task_version,
                "contract_hash": contract.hash, "initial_slot": None,
                "evaluation_version": EVALUATION_VERSION, "policy_track": "native"}))
            (root / "scores.json").write_text(json.dumps({
                "task_version": contract.task_version, "contract_hash": contract.hash,
                "task_completion": False, "evaluation_valid": False}))
            result = online_rollout_record(tmp, audit_online=False)
        self.assertEqual(result["status"], "invalid_evaluation")
        self.assertNotIn("success", result)

    def test_fixed_execution_requires_delivered_target_and_measured_stability(self):
        contract = Contract("hard", stable_steps=1, final_stable_steps=2)
        frame = {"type": "geometry_frame", "step": -1,
                 "slot_mouths_world_m": {"S00": [0, 0, .867], "S06": [.1, 0, .867]},
                 "insertion_axis_world": [0, 0, -1]}
        first = {"position_world_m": [0, 0, .84], "axis_world": [0, 0, 1],
                 "linear_speed_m_s": 0, "position_delta_m": 0, "released": True,
                 "supported": True, "gripper_contact": False}
        second = dict(first, position_world_m=[.2, 0, .76], supported=False)
        held = dict(second, position_world_m=[.2, 0, .86], released=False,
                    gripper_contact=True)
        inserted = dict(first, position_world_m=[.1, 0, .84])
        action = {"type": "insert", "tube": "NMR_02", "slot": "S06"}
        pose = {"slot": "S06", "frame": "world", "units": "m",
                "mouth_world_m": [.1, 0, .867], "axis_world": [0, 0, -1]}
        trace = [frame,
                 {"type": "measurement", "step": 30, "tubes": {"NMR_01": first, "NMR_02": second}},
                 {"type": "measurement", "step": 31, "tubes": {"NMR_01": first, "NMR_02": held}},
                 {"type": "fixed_subgoal", "step": 31, "evidence_step": 31,
                  "action": action, "interaction_pose": pose}]
        self.assertIsNone(score_online_v2(contract, trace)["diagnostic"]["fixed_subgoal_execution_success"])
        trace.append({"type": "policy_inference", "step": 31,
                      "policy_request": {"selected_operation": action, "interaction_pose": pose}})
        self.assertFalse(score_online_v2(contract, trace)["diagnostic"]["fixed_subgoal_execution_success"])
        for step in (32, 33):
            trace.append({"type": "measurement", "step": step,
                          "tubes": {"NMR_01": first, "NMR_02": inserted}})
        score = score_online_v2(contract, trace)
        self.assertTrue(score["diagnostic"]["fixed_subgoal_execution_success"])
        self.assertEqual(score["diagnostic"]["interface_scores"]["execution"]["accuracy"], 1)
        trace[3]["action"] = {"type": "insert", "tube": "NMR_02", "slot": "S00"}
        self.assertFalse(score_online_v2(contract, trace)["evaluation_valid"])


if __name__ == "__main__":
    unittest.main()
