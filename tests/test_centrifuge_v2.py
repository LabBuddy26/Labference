import unittest
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO

from eai4s_eval.centrifuge_v2.cli import load_contract
from eai4s_eval.centrifuge_v2.runner import camera_rgb, run, scipy_xyzw_to_lula_wxyz
from eai4s_eval.centrifuge_v2.cli import online_rollout_record, run_supervised

from eai4s_eval.centrifuge_v2.core import (
    Contract, Observation, SLOTS, State, advance_state, cartesian_waypoint,
    cartesian_waypoint_reached, choose_accessible_first_slot,
    insertion_clearance_target_z, inspect_joint_targets, joint_waypoint_is_local,
    classify_contacts, evaluate_insertion, gripper_target_for_insertion, legal_actions,
    insertion_matches_commit, joint_limit_avoidance_delta, limit_joint_targets, limit_joint_velocities,
    operation_step_limit,
    slew_lift_height, wrap_joint_targets_near_current,
)
from eai4s_eval.centrifuge_v2.diagnostics import (
    check_policy_request, compare_pairs, compare_state_pairs, deliver_input, normalize_action,
    oracle_context, policy_observation, score_checkpoint, score_trace,
    PolicyCapabilities, evaluate_policy_checkpoint,
    OnlineEvaluationMonitor, plan_online_rollouts, plan_state_pairs,
    summarize_checkpoint_scores, validated_geometry_report, verify_online_scores,
)
from eai4s_eval.centrifuge_v2.policy import HttpSymbolicPolicy, PolicyServiceError


class GeometryTests(unittest.TestCase):
    def test_socket_centers_match_mesh_cross_section(self):
        # Measured from the circular socket cross section in the source USD,
        # also matching the legacy scene's preloaded NMR_03 XY coordinates.
        local = Contract("hard").slot_local_center("S04")
        world = (.05 + .5 * local[0], .41 - .5 * local[1])
        self.assertAlmostEqual(world[0], .0627004, delta=.000005)
        self.assertAlmostEqual(world[1], .36812646, delta=.000005)

    def test_both_versioned_configs_load_with_stable_hash(self):
        root = Path(__file__).resolve().parents[1] / "eai4s_eval/centrifuge_v2/configs"
        easy = load_contract(str(root / "easy.json"))
        hard = load_contract(str(root / "hard.json"))
        self.assertNotEqual(easy.hash, hard.hash)
        self.assertEqual(easy.hash, load_contract(str(root / "easy.json")).hash)
        self.assertNotEqual(easy.hash, Contract("easy", task_version="centrifuge-opposite/v2",
                                               front_camera_focal_length_mm=13).hash)
        self.assertNotEqual(easy.hash, Contract("easy", task_version="centrifuge-opposite/v2",
                                               franka_base_position_m=(-.4, 0, .71)).hash)
        with self.assertRaises(ValueError):
            run(str(root / "easy.json"), "/tmp/should-not-run", max_steps=0)

    def test_opposite_is_involution(self):
        c = Contract("easy")
        for slot in SLOTS:
            self.assertEqual(c.opposite(c.opposite(slot)), slot)
            self.assertNotEqual(c.opposite(slot), slot)

    def test_slow_lift_target_is_bounded(self):
        self.assertAlmostEqual(slew_lift_height(.70, .80), .7005)
        self.assertAlmostEqual(slew_lift_height(.80, .80), .80)
        with self.assertRaises(ValueError):
            slew_lift_height(float("nan"), .80)

    def test_collection_insert_budget_is_extended_without_changing_eval_budget(self):
        self.assertEqual(operation_step_limit("insert", "NMR_01", collection_mode=True), 30000)
        self.assertEqual(operation_step_limit("insert", "NMR_01", collection_mode=False), 14000)
        self.assertEqual(operation_step_limit("pick", "NMR_02", collection_mode=True), 3000)
        self.assertEqual(operation_step_limit("pick", "NMR_01", collection_mode=True), 1300)

    def test_insert_clearance_accounts_for_gripper_to_tube_offset(self):
        self.assertAlmostEqual(
            insertion_clearance_target_z(.867, 1.30, .057), 1.164)
        self.assertAlmostEqual(
            insertion_clearance_target_z(.867, 1.05, .057), 1.107)
        with self.assertRaises(ValueError):
            insertion_clearance_target_z(float("nan"), 1.2, .05)

    def test_free_space_cartesian_waypoints_are_bounded_and_reach_the_goal(self):
        current = (0.0, 0.0, 0.0)
        target = (.3, .4, 0.0)
        waypoint = cartesian_waypoint(current, target, .02)
        self.assertAlmostEqual(sum(value * value for value in waypoint) ** .5, .02)
        final = cartesian_waypoint((.3, .4, 0.0), target, .02)
        self.assertEqual(final, target)
        self.assertTrue(cartesian_waypoint_reached((.105, .178, 1.122),
                                                   (.107, .180, 1.112), .02))
        self.assertFalse(cartesian_waypoint_reached((0.0, 0.0, 0.0),
                                                    (.01, 0.0, 0.0), .003))
        self.assertFalse(cartesian_waypoint_reached((.10, .10, 0), (.13, .10, 0), .02))
        self.assertFalse(cartesian_waypoint_reached((float("nan"), 0, 0), target))
        with self.assertRaises(ValueError):
            cartesian_waypoint(current, target, 0)

    def test_joint_velocity_targets_are_bounded_and_unwind_out_of_band_joints(self):
        current = [0.0] * 6 + [7.0, 0.04, 0.04]
        velocities = [8.0] * 7 + [0.0, 0.0]
        limited = limit_joint_velocities(current, velocities)
        self.assertEqual(limited[:6], [2.0] * 6)
        self.assertEqual(limited[6], -2.0)
        self.assertEqual(limited[7:], [0.0, 0.0])
        near_limit = limit_joint_velocities([0.0] * 6 + [2 * 3.141592653589793],
                                             [0.0] * 6 + [1.5])
        self.assertEqual(near_limit[6], 0.0)
        self.assertTrue(joint_waypoint_is_local([0, .2], [.05, .28]))
        self.assertFalse(joint_waypoint_is_local([0, .2], [2.6, .4]))
        self.assertFalse(joint_waypoint_is_local([0, .2], [0, float("nan")]))

    def test_joint_limit_avoidance_uses_position_task_nullspace(self):
        import numpy as np

        current = [0.0] * 7
        current[6] = 2.8
        limits = [[-2.9, 2.9]] * 7
        jacobian = np.column_stack((np.eye(3), np.zeros((3, 4))))
        correction = np.asarray(joint_limit_avoidance_delta(current, jacobian, limits))
        self.assertLess(correction[6], 0.0)
        np.testing.assert_allclose(jacobian @ correction, np.zeros(3), atol=1e-8)
        with self.assertRaises(ValueError):
            joint_limit_avoidance_delta(current, jacobian, [[0.0, 0.0]] * 7)

    def test_collection_prefers_approach_side_socket_pair(self):
        slots = {
            "S00": (.1737, .2619, .867), "S01": (.1578, .2293, .867),
            "S02": (.1278, .2091, .867), "S03": (.0916, .2065, .867),
            "S04": (.0591, .2224, .867), "S05": (.0388, .2525, .867),
            "S06": (.0363, .2886, .867), "S07": (.0522, .3212, .867),
            "S08": (.0822, .3414, .867), "S09": (.1184, .3440, .867),
            "S10": (.1509, .3281, .867), "S11": (.1712, .2981, .867),
        }
        self.assertEqual(choose_accessible_first_slot(
            slots, [-.4, 0.0], [.2, -.25]), "S05")
        with self.assertRaises(ValueError):
            choose_accessible_first_slot({}, [-.4, 0], [.2, -.25])

    def test_continuous_joint_targets_use_nearest_equivalent_angle(self):
        result = wrap_joint_targets_near_current(
            [2.8, 17.8, -2.8], [4.0, 17.8 + 2 * 3.141592653589793 + .1, -4.0],
            [-2.9, float("-inf"), -2.9], [2.9, float("inf"), 2.9], [0, 1, 2])
        self.assertEqual(result[0], 2.9)
        self.assertAlmostEqual(result[1], 17.9)
        self.assertEqual(result[2], -2.9)
        # The safety limit must still win after an earlier absolute clamp
        # tries to pull an out-of-range continuous joint back in one jump.
        limited = limit_joint_targets([0.0] * 6 + [17.8], [0.0] * 6 + [3.14])
        self.assertAlmostEqual(limited[6], 17.72)
        unwound = limit_joint_targets([-25.7] + [0.0] * 6, [25.0] + [0.0] * 6)
        self.assertAlmostEqual(unwound[0], -25.62)

    def test_measured_grasp_offset_sets_ee_insertion_goal(self):
        mouth = (0.10, 0.20, 0.87)
        target = gripper_target_for_insertion(mouth, (0, 0, -1), .027, (.01, -.02, .045))
        # PlaceController adds its 20 mm vertical offset at the insertion phase.
        for actual, expected in zip(target, (.11, .18, .868)):
            self.assertAlmostEqual(actual, expected)
        with self.assertRaises(ValueError):
            gripper_target_for_insertion(mouth, (0, 0, -1), float("nan"), (0, 0, 0))

    def test_camera_frame_requires_valid_pixels(self):
        import numpy as np
        self.assertIsNone(camera_rgb(np.array([])))
        pixels = camera_rgb(np.ones((2, 3, 4), dtype=np.float32) * .5)
        self.assertEqual(pixels.shape, (2, 3, 3))
        self.assertEqual(int(pixels[0, 0, 0]), 127)

    def test_scipy_xyzw_quaternion_is_converted_for_lula(self):
        import numpy as np

        np.testing.assert_allclose(
            scipy_xyzw_to_lula_wxyz([0, 0, 0, 1]), [1, 0, 0, 0])
        np.testing.assert_allclose(
            scipy_xyzw_to_lula_wxyz([0, 1, 0, 0]), [0, 0, 1, 0])
        with self.assertRaises(ValueError):
            scipy_xyzw_to_lula_wxyz([0, 0, 0, 0])

    def test_run_output_remains_a_directory(self):
        from eai4s_eval.centrifuge_v2 import cli
        with TemporaryDirectory() as tmp:
            with patch("sys.argv", ["centrifuge", "run", "--worker", "--config", "unused.json", "--output", tmp]), \
                 patch("eai4s_eval.centrifuge_v2.runner.run", return_value={"valid_success": False}), \
                 redirect_stdout(StringIO()):
                cli.main()
            self.assertTrue(Path(tmp).is_dir())

    def test_supervisor_records_process_exit_after_kit_close_request(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "run.json").write_text(json.dumps({
                "runtime_status": "finished", "simulation_app_close": {"status": "requested"}}))
            (out / "scores.json").write_text(json.dumps({"valid_success": True}))
            with patch("eai4s_eval.centrifuge_v2.cli.subprocess.Popen") as popen:
                popen.return_value.poll.return_value = 0
                popen.return_value.returncode = 0
                result = run_supervised("config.json", tmp, 101, 100, False, None, None)
            self.assertTrue(result["valid_success"])
            self.assertEqual(json.loads((out / "run.json").read_text())["simulation_app_close"]["status"],
                             "process_exited")

    def test_joint_targets_are_inspected_and_nonfinite_rejected(self):
        self.assertEqual(inspect_joint_targets([0.0, 0.0], [1.0, None]), 1.0)
        self.assertAlmostEqual(inspect_joint_targets([.2, .3], [.7], [1]), .4)
        with self.assertRaises(ValueError):
            inspect_joint_targets([0.0], [float("nan")])

    def test_joint_targets_are_limited_without_moving_unset_joints(self):
        result = limit_joint_targets([0.0] * 9, [1.0, None, .04], [0, 4, 7])
        self.assertEqual(result, [.08, None, .04])
        with self.assertRaises(ValueError):
            limit_joint_targets([0.0], [float("inf")])

    def test_legacy_targets_fit_calibrated_ring(self):
        c = Contract("hard")
        local = c.slot_local_center("S01")
        world = (.05 + .5*local[0], .41 - .5*local[1], .72 - .5*local[2])
        self.assertLess(abs(world[0] + .01812629), .001)
        self.assertLess(abs(world[1] - .31270059), .001)
        self.assertAlmostEqual(.72 - .5*local[2], .867, places=3)

    def test_ambiguous_hover_and_shallow_insert_fail(self):
        c = Contract("hard")
        mouths = {"S01": (0, 0, .83), "S07": (.1, 0, .83)}
        def sample(position, released=True, supported=True):
            obs = Observation("NMR_01", position, (0, 0, 1), 0.0, released, supported)
            return evaluate_insertion(c, obs, mouths, (0, 0, -1))
        self.assertFalse(sample((0, 0, .84))["geometry_pass"])
        self.assertFalse(sample((0, 0, .823))["geometry_pass"])
        self.assertFalse(sample((.008, 0, .80))["geometry_pass"])
        self.assertFalse(sample((0, 0, .80), False)["stable_sample"])
        self.assertFalse(sample((0, 0, .80), True, False)["stable_sample"])
        self.assertEqual(sample((0, 0, .80))["actual_slot"], "S01")

    def test_stability_uses_measured_displacement_when_available(self):
        c = Contract("easy")
        mouth = {"S00": (0, 0, .867)}
        obs = Observation("NMR_01", (0, 0, .840), (0, 0, 1), .0153, True, True, .00004)
        evidence = evaluate_insertion(c, obs, mouth, (0, 0, -1))
        self.assertTrue(evidence["stable_sample"])
        moved = Observation("NMR_01", (0, 0, .840), (0, 0, 1), .0153, True, True, .001)
        self.assertFalse(evaluate_insertion(c, moved, mouth, (0, 0, -1))["stable_sample"])

    def test_hard_uses_actual_not_promised_slot(self):
        c = Contract("hard", stable_steps=2, final_stable_steps=3)
        s = State("hard")
        self.assertEqual(len(legal_actions(c, s)), 1)
        s.tube_status["NMR_01"] = "held"
        self.assertEqual(len(legal_actions(c, s)), 12)
        s.selected_slot["NMR_01"] = "S01"
        ev = {"actual_slot": "S02", "stable_sample": True}
        advance_state(c, s, {"NMR_01": ev})
        self.assertIsNone(s.first_actual_slot)
        events = advance_state(c, s, {"NMR_01": ev})
        self.assertEqual(events, [{"type": "inserted", "tube": "NMR_01", "slot": "S02"}])
        self.assertEqual(s.first_actual_slot, "S02")
        s.tube_status["NMR_02"] = "held"
        self.assertEqual(legal_actions(c, s), [{"type": "insert", "tube": "NMR_02", "slot": "S08"}])
        s.tube_status["NMR_02"] = "table"
        self.assertEqual(legal_actions(c, s), [{"type": "pick", "tube": "NMR_02"}])

    def test_inserted_tube_requires_a_matching_committed_slot(self):
        self.assertTrue(insertion_matches_commit(Contract("easy"), "NMR_01", None, "S02"))
        self.assertFalse(insertion_matches_commit(Contract("easy"), "NMR_02", None, "S08"))
        self.assertFalse(insertion_matches_commit(Contract("hard"), "NMR_01", None, "S02"))
        self.assertFalse(insertion_matches_commit(Contract("hard"), "NMR_02", "S09", "S08"))
        self.assertTrue(insertion_matches_commit(Contract("hard"), "NMR_02", "S08", "S08"))

    def test_easy_changes_opposite_with_preload(self):
        c = Contract("easy")
        s = State("easy", first_actual_slot="S00")
        self.assertEqual(legal_actions(c, s), [{"type": "pick", "tube": "NMR_02"}])
        s.tube_status["NMR_02"] = "held"
        self.assertEqual(legal_actions(c, s)[0]["slot"], "S06")
        s.first_actual_slot = "S04"
        self.assertEqual(legal_actions(c, s)[0]["slot"], "S10")

    def test_lost_first_insertion_clears_stale_target(self):
        c = Contract("hard", stable_steps=1)
        s = State("hard")
        advance_state(c, s, {"NMR_01": {"actual_slot": "S03", "stable_sample": True}})
        self.assertEqual(s.first_actual_slot, "S03")
        events = advance_state(c, s, {"NMR_01": {"actual_slot": None, "stable_sample": False}})
        self.assertEqual(events[0]["type"], "insertion_lost")
        self.assertIsNone(s.first_actual_slot)
        self.assertEqual(legal_actions(c, s), [{"type": "pick", "tube": "NMR_01"}])

    def test_only_rotor_contact_is_socket_support(self):
        table = classify_contacts([{"body0": "/World/NMR_01/geometry", "body1": "/World/table"}])
        self.assertFalse(table["rotor"])
        rotor = classify_contacts([{"body0": "/World/NMR_01/geometry", "body1": "/World/Centrifugal/geometry"}])
        self.assertTrue(rotor["rotor"])
        finger = classify_contacts([{"body0": "/World/NMR_01", "body1": "/World/Franka/panda_leftfinger"}])
        self.assertTrue(finger["gripper"])


class DiagnosticTests(unittest.TestCase):
    def test_transport_and_leak_boundary(self):
        c = Contract("easy")
        s = State("easy", first_actual_slot="S02")
        obs = policy_observation({"front": "frame"}, [0.0], "insert opposite", list(SLOTS))
        self.assertNotIn("first_actual_slot", obs)
        context = oracle_context("semantic", c, s, {"S02": "NMR_01"}, {})
        self.assertNotIn("advice", context)
        request = deliver_input(obs, context, "structured", {"structured"})
        check_policy_request(request)
        with self.assertRaises(ValueError):
            deliver_input(request, context, "structured", {"structured"})
        with self.assertRaises(ValueError):
            deliver_input(obs, context, "text", {"structured"})
        with self.assertRaises(ValueError):
            check_policy_request({"success": True})
        with self.assertRaises(ValueError):
            check_policy_request({"images": {"metadata": {"correct_slot": "S02"}}})
        with self.assertRaises(ValueError):
            check_policy_request(dict(obs, leaked_result="S08"))
        with self.assertRaises(ValueError):
            check_policy_request(dict(obs, oracle_context={"schema_version": "labference/centrifuge-diagnostic-v1",
                                                           "profile": "semantic", "advice": {"type": "stop"}}))
        with self.assertRaises(ValueError):
            check_policy_request(dict(obs, oracle_prompt="<ORACLE>{bad}</ORACLE>"))
        with self.assertRaises(ValueError):
            check_policy_request(dict(obs, authorized_history=["correct_slot: S08"]))
        with self.assertRaises(ValueError):
            check_policy_request(dict(obs, authorized_history=['{"correct_slot":"S08"}']))

    def test_geometry_does_not_choose_slot(self):
        c = Contract("hard")
        s = State("hard")
        frames = {x: {"mouth_world_m": [i, 0, 0]} for i, x in enumerate(SLOTS)}
        result = oracle_context("geometry", c, s, {}, frames, {"type": "insert", "tube": "NMR_01", "slot": "S04"})
        self.assertEqual(len(result["candidate_frames"]), 12)
        self.assertNotIn("advice", result)
        before_decision = oracle_context("geometry", c, s, {}, frames)
        self.assertEqual(len(before_decision["candidate_frames"]), 12)
        self.assertNotIn("advice", before_decision)

    def test_geometry_report_requires_committed_slot_and_world_meters(self):
        report = {"slot": "S04", "mouth_world_m": [.1, .2, .8],
                  "axis_world": [0, 0, -1], "frame": "world", "units": "m"}
        self.assertEqual(validated_geometry_report(report, "S04")[0], (.1, .2, .8))
        for invalid in [dict(report, slot="S05"), dict(report, units="cm"),
                        dict(report, axis_world=[0, 0, 0]),
                        dict(report, mouth_world_m=[float("nan"), 0, 0])]:
            with self.assertRaises(ValueError):
                validated_geometry_report(invalid, "S04")

    def test_http_symbolic_policy_has_explicit_reset_and_decision(self):
        calls = []
        def respond(request, timeout):
            calls.append((json.loads(request.data), timeout))
            response = {"status": "ok"} if len(calls) == 1 else {
                "committed_action": {"type": "pick", "tube": "NMR_02"}}
            return BytesIO(json.dumps(response).encode())
        with patch("eai4s_eval.centrifuge_v2.policy.urlopen", side_effect=respond):
            policy = HttpSymbolicPolicy("http://127.0.0.1:8123/decide", timeout_s=2)
            policy.reset(101)
            result = policy.decide({"task_rule": "pick opposite"})
        self.assertEqual(calls[0][0], {"command": "reset", "seed": 101})
        self.assertEqual(calls[1][0]["command"], "decide")
        self.assertEqual(calls[1][1], 2)
        self.assertEqual(result["committed_action"]["tube"], "NMR_02")
        with patch("eai4s_eval.centrifuge_v2.policy.urlopen", return_value=BytesIO(b"[]")):
            with self.assertRaises(PolicyServiceError):
                policy.decide({})

    def test_checkpoint_interfaces_are_not_conflated(self):
        record = {"instance_id": "a", "supported_reports": ["decision", "execution"],
                  "committed_action": {"type": "insert", "tube": "NMR_02", "slot": "S07"},
                  "legal_actions": [{"type": "insert", "tube": "NMR_02", "slot": "S07"}],
                  "physical_subgoal_confirmed": False}
        score = score_checkpoint(record)
        self.assertEqual(score["decision"], "pass")
        self.assertEqual(score["execution"], "fail")
        self.assertEqual(score["semantic"], "unsupported")
        malformed = score_checkpoint({"instance_id": "b", "supported_reports": ["semantic", "geometry"],
                                      "expected_semantic": None, "semantic_report": None,
                                      "geometry_report": "bad", "reference_geometry": {"slot": "S02"}})
        self.assertEqual(malformed["semantic"], "fail")
        self.assertEqual(malformed["geometry"], "fail")
        wrong_state = score_checkpoint({"instance_id": "state", "supported_reports": ["semantic"],
                                        "expected_semantic": {"S03": "NMR_01"},
                                        "semantic_report": {"S04": "NMR_01"}})
        self.assertEqual(wrong_state["semantic"], "fail")
        offset = score_checkpoint({"instance_id": "geo", "supported_reports": ["geometry"],
                                   "geometry_report": {"slot": "S03", "mouth_world_m": [.03, 0, 0],
                                                       "axis_world": [0, 0, 1]},
                                   "reference_geometry": {"slot": "S03", "mouth_world_m": [0, 0, 0],
                                                          "axis_world": [0, 0, 1]}})
        self.assertEqual(offset["geometry"], "fail")
        not_applicable = score_checkpoint({"instance_id": "pick", "supported_reports": ["geometry"],
                                           "geometry_applicable": False})
        self.assertEqual(not_applicable["geometry"], "not_applicable")
        summary = summarize_checkpoint_scores([score, malformed, wrong_state, offset, not_applicable])
        self.assertEqual(summary["decision"]["denominator"], 1)
        self.assertEqual(summary["decision"]["accuracy"], 1.0)
        self.assertEqual(summary["geometry"]["not_applicable"], 1)

    def test_strict_failure_preserves_recovered_completion(self):
        c = Contract("hard", stable_steps=2, final_stable_steps=3)
        trace = [
            {"type": "decision", "action": {"type": "insert", "tube": "NMR_01", "slot": "S01"}, "legal_actions": [{"type": "insert", "tube": "NMR_01", "slot": "S01"}]},
            {"type": "decision", "action": {"type": "insert", "tube": "NMR_02", "slot": "S06"}, "legal_actions": [{"type": "insert", "tube": "NMR_02", "slot": "S07"}]},
            {"type": "violation", "kind": "wrong_target"},
            {"type": "physical", "event": "inserted", "tube": "NMR_01", "slot": "S01", "precision_pass": True},
            {"type": "physical", "event": "inserted", "tube": "NMR_02", "slot": "S07", "precision_pass": True},
            {"type": "physical", "event": "grasp_lost", "tube": "NMR_02"},
            {"type": "snapshot", "terminal_steps": 3},
        ]
        score = score_trace(c, trace)
        self.assertTrue(score["completion_success"])
        self.assertFalse(score["valid_success"])
        self.assertTrue(score["recovery_completion"])
        self.assertEqual(score["grasp_lost_count"], 1)

    def test_replay_uses_measurements_not_claimed_legal_set_or_terminal_snapshot(self):
        c = Contract("easy", stable_steps=2, final_stable_steps=3)
        def sample(first, second, step):
            return {"type": "measurement", "step": step, "tubes": {
                "NMR_01": {"actual_slot": "S01" if first else None, "stable_sample": first},
                "NMR_02": {"actual_slot": "S07" if second else None, "stable_sample": second},
            }}
        trace = [sample(True, False, step) for step in range(2)]
        trace += [{"type": "decision", "action": {"type": "stop"},
                   "legal_actions": [{"type": "stop"}]}]
        trace += [sample(True, True, step) for step in range(2, 7)]
        trace += [{"type": "snapshot", "terminal_steps": 999}]
        score = score_trace(c, trace)
        self.assertTrue(score["completion_success"])
        self.assertFalse(score["decision_pass"])
        self.assertEqual(score["terminal_steps"], 4)
        self.assertFalse(score["valid_success"])
        self.assertEqual(score["replay_source"], "measurements")

    def test_offline_replay_recomputes_insertion_from_raw_pose(self):
        c = Contract("easy", stable_steps=1, final_stable_steps=1)
        frame = {"type": "geometry_frame", "slot_mouths_world_m": {
            "S00": [0, 0, .867], "S06": [.1, 0, .867]},
            "insertion_axis_world": [0, 0, -1]}
        def tube(position, supported, contact=False):
            return {"position_world_m": position, "axis_world": [0, 0, 1],
                    "linear_speed_m_s": 0, "position_delta_m": 0,
                    "released": not contact, "supported": supported,
                    "gripper_contact": contact,
                    "actual_slot": "S03", "stable_sample": False}
        first = tube([0, 0, .84], True)
        second_table = tube([.2, 0, .76], False)
        second_held = tube([.2, 0, .83], False, True)
        second_inserted = tube([.1, 0, .84], True)
        trace = [frame,
                 {"type": "measurement", "step": 30,
                  "tubes": {"NMR_01": first, "NMR_02": second_table}},
                 {"type": "measurement", "step": 31,
                  "tubes": {"NMR_01": first, "NMR_02": second_held}},
                 {"type": "decision", "action": {"type": "insert", "tube": "NMR_02", "slot": "S06"}},
                 {"type": "measurement", "step": 32,
                  "tubes": {"NMR_01": first, "NMR_02": second_inserted}},
                 {"type": "decision", "action": {"type": "stop"}}]
        result = score_trace(c, trace)
        self.assertTrue(result["physical_replay_verified"])
        self.assertTrue(result["evaluation_valid"])
        unfinished = score_trace(c, trace[:3])
        self.assertTrue(unfinished['evaluation_valid'])
        self.assertFalse(unfinished['valid_success'])
        self.assertTrue(result["valid_success"])
        self.assertTrue(result["grasp_lift_pass"])
        self.assertEqual(result["operation_attempts"]["insert"]["succeeded"], 1)
        self.assertEqual(result["selected_operation_execution_rate"], 1.0)
        self.assertTrue(result["termination_pass"])
        no_commit = score_trace(c, trace[:3] + trace[4:])
        self.assertFalse(no_commit["valid_success"])
        fake_lift = score_trace(c, [frame, trace[1],
            {"type": "physical", "event": "lifted", "tube": "NMR_02"}, trace[3], trace[4]])
        self.assertFalse(fake_lift["decision_pass"])
        leaked = dict(second_inserted, released=False)
        not_released = score_trace(c, trace[:-2] + [
            {"type": "measurement", "step": 32,
             "tubes": {"NMR_01": first, "NMR_02": leaked}}, trace[-1]])
        self.assertFalse(not_released["completion_success"])
        no_stop = score_trace(c, trace[:-1])
        self.assertTrue(no_stop["completion_success"])
        self.assertFalse(no_stop["termination_pass"])
        self.assertFalse(no_stop["valid_success"])
        no_grasp = score_trace(c, [frame, trace[1], trace[3], trace[4], trace[5]])
        self.assertTrue(no_grasp["completion_success"])
        self.assertFalse(no_grasp["grasp_lift_pass"])
        self.assertFalse(no_grasp["valid_success"])

    def test_module_scores_replay_reports_and_ignore_claimed_checkpoint_scores(self):
        c = Contract("easy", stable_steps=1, final_stable_steps=1)
        frame = {"type": "geometry_frame", "slot_mouths_world_m": {
            "S00": [0, 0, .867], "S06": [.1, 0, .867]},
            "insertion_axis_world": [0, 0, -1]}
        first = {"position_world_m": [0, 0, .84], "axis_world": [0, 0, 1],
                 "linear_speed_m_s": 0, "position_delta_m": 0,
                 "released": True, "supported": True, "gripper_contact": False}
        second = dict(first, position_world_m=[.2, 0, .76], supported=False)
        trace = [frame, {"type": "measurement", "step": 30,
                         "tubes": {"NMR_01": first, "NMR_02": second}},
                 {"type": "decision", "step": 30, "action": {"type": "pick", "tube": "NMR_02"},
                  "legal_actions": [{"type": "stop"}],
                  "policy_response": {"semantic_report": {"S00": "NMR_01"}}},
                 {"type": "checkpoint", "scores": {"semantic": "fail", "decision": "fail"}}]
        scored = score_trace(c, trace, {"semantic", "decision", "geometry"})
        self.assertEqual(scored["module_scores"]["semantic"]["accuracy"], 1.0)
        self.assertEqual(scored["module_scores"]["decision"]["accuracy"], 1.0)
        self.assertEqual(scored["module_scores"]["geometry"]["not_applicable"], 1)
        self.assertEqual(scored["module_scores"]["execution"]["denominator"], 0)

    def test_corrupt_physics_readback_is_scored_as_unverified(self):
        c = Contract("easy", stable_steps=1, final_stable_steps=1)
        result = score_trace(c, [
            {"type": "geometry_frame", "slot_mouths_world_m": {"S00": [0, 0, .867]},
             "insertion_axis_world": [0, 0, -1]},
            {"type": "measurement", "step": 30, "tubes": {
                "NMR_01": {"position_world_m": [0, 0, float("nan")],
                           "axis_world": [0, 0, 1], "linear_speed_m_s": 0,
                           "position_delta_m": 0, "released": True,
                           "supported": True, "gripper_contact": False},
                "NMR_02": {"position_world_m": "invalid", "axis_world": [0, 0, 1],
                           "linear_speed_m_s": 0, "position_delta_m": 0,
                           "released": True, "supported": False, "gripper_contact": False}}}])
        self.assertFalse(result["physical_replay_verified"])
        self.assertGreater(result["raw_measurement_error_count"], 0)
        self.assertFalse(result["valid_success"])
        json.dumps(result, allow_nan=False)

    def test_cli_offline_score_matches_direct_replay(self):
        from eai4s_eval.centrifuge_v2 import cli
        root = Path(__file__).resolve().parents[1] / "eai4s_eval/centrifuge_v2/configs/easy.json"
        config = json.loads(root.read_text())
        contract = load_contract(str(root))
        trace = [{"type": "decision", "step": 30, "action": {"type": "stop"},
                  "legal_actions": [{"type": "pick", "tube": "NMR_02"}]}]
        reports = set(config["evaluation"]["policy_interface"]["supported_reports"])
        direct = score_trace(contract, trace, reports)
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "run.json").write_text(json.dumps({"evaluation": config["evaluation"],
                                                       "contract_hash": contract.hash}))
            (out / "events.jsonl").write_text("\n".join(map(json.dumps, trace)) + "\n")
            with patch("sys.argv", ["centrifuge", "score", "--run", str(out / "run.json"),
                                    "--trace", str(out / "events.jsonl")]), redirect_stdout(StringIO()) as stream:
                cli.main()
            self.assertEqual(json.loads(stream.getvalue()), direct)
            self.assertEqual(direct["decision_error_counts"]["premature_stop"], 1)

    def test_online_scores_are_published_and_auditable_by_prefix(self):
        contract = Contract("easy")
        trace = [{"type": "decision", "step": 30, "action": {"type": "stop"},
                  "legal_actions": [{"type": "pick", "tube": "NMR_02"}]}]
        stream = StringIO()
        monitor = OnlineEvaluationMonitor(contract, trace, stream, {"decision"})
        first = monitor.snapshot(30, "decision")
        self.assertFalse(first["decision_pass"])
        trace.append({"type": "violation", "step": 31, "kind": "safety"})
        monitor.snapshot(31, "terminal")
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(monitor.sample_count, 2)
        self.assertTrue(verify_online_scores(contract, trace, rows, {"decision"})["matching"])
        rows[0]["scores"]["decision_pass"] = True
        audit = verify_online_scores(contract, trace, rows, {"decision"})
        self.assertFalse(audit["matching"])
        self.assertEqual(audit["mismatches"][0]["reason"], "score_mismatch")

    def test_online_operation_outcome_remains_pending_until_physical_result(self):
        c = Contract("hard")
        trace = [{"type": "decision", "step": 1, "action": {"type": "pick", "tube": "NMR_01"},
                  "legal_actions": [{"type": "pick", "tube": "NMR_01"}]}]
        in_progress = score_trace(c, trace)
        self.assertEqual(in_progress["operation_attempts"]["pick"]["pending"], 1)
        trace.append({"type": "physical", "event": "lifted", "step": 2, "tube": "NMR_01"})
        observed = score_trace(c, trace)
        self.assertEqual(observed["operation_attempts"]["pick"]["pending_observed_success"], 1)
        trace.append({"type": "snapshot", "step": 3, "terminal_steps": 0})
        completed = score_trace(c, trace)
        self.assertEqual(completed["operation_attempts"]["pick"]["succeeded"], 1)
        self.assertEqual(completed["selected_operation_execution_rate"], 1.0)

    def test_online_run_records_compare_from_saved_artifacts(self):
        contract = Contract("easy")
        plan = plan_online_rollouts(contract, [101])
        self.assertEqual(plan[0]["initial_slot"], "S05")
        with self.assertRaises(ValueError):
            plan_online_rollouts(contract, [101, 101])
        with TemporaryDirectory() as tmp:
            records = []
            for condition, success in (("none", False), ("semantic", True)):
                root = Path(tmp) / condition
                root.mkdir()
                manifest = dict(plan[0], difficulty="easy", seed=101,
                                oracle_profile=condition, runtime_status="finished",
                                policy_id="model-a", policy_uses_evaluator_state=False,
                                sensor_condition="front+wrist_rgb")
                (root / "run.json").write_text(json.dumps(manifest))
                scores = dict(score_trace(contract, []), valid_success=success)
                (root / "scores.json").write_text(json.dumps(scores))
                records.append(online_rollout_record(str(root), audit_online=False))
            registered = {plan[0]["pair_id"]: {key: plan[0][key] for key in
                          ("task_version", "contract_hash", "scene_seed", "decision_seed",
                           "initial_slot", "max_steps_budget")}}
            compared = compare_pairs(records, "semantic", registered)
            self.assertEqual(compared["paired_absolute_gain"], 1.0)
            self.assertEqual(compared["status"], "complete")
            self.assertFalse(compared["ranking_eligible"])
            self.assertIsNone(compared["paired_rollout_decision_accuracy_gain"])
            with self.assertRaises(ValueError):
                compare_pairs([dict(records[0], max_steps_budget=36), records[1]],
                              "semantic", registered)

    def test_online_run_record_rejects_tampered_final_score(self):
        config_path = Path(__file__).resolve().parents[1] / "eai4s_eval/centrifuge_v2/configs/easy.json"
        evaluation = json.loads(config_path.read_text())["evaluation"]
        contract = load_contract(str(config_path))
        trace = [{"type": "decision", "step": 30, "action": {"type": "stop"},
                  "legal_actions": [{"type": "pick", "tube": "NMR_02"}]}]
        stream = StringIO()
        scores = OnlineEvaluationMonitor(contract, trace, stream, {"decision"}).snapshot(30, "terminal")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.json").write_text(json.dumps({
                "evaluation": evaluation, "difficulty": "easy", "seed": 101,
                "task_version": contract.task_version, "contract_hash": contract.hash,
                "max_steps_budget": contract.max_steps,
                "initial_slot": "S05", "oracle_profile": "none", "runtime_status": "time_limit",
                "policy_id": "model-a", "policy_uses_evaluator_state": False}))
            (root / "events.jsonl").write_text(json.dumps(trace[0]) + "\n")
            (root / "online_scores.jsonl").write_text(stream.getvalue())
            (root / "scores.json").write_text(json.dumps(scores))
            self.assertFalse(online_rollout_record(tmp)["success"])
            from eai4s_eval.centrifuge_v2 import cli
            (root / "rollout_plan.json").write_text(json.dumps({
                "pairs": plan_online_rollouts(contract, [101])}))
            with patch("sys.argv", ["centrifuge", "compare-runs", "--plan",
                                    str(root / "rollout_plan.json"), "--condition", "semantic",
                                    "--runs", str(root)]), redirect_stdout(StringIO()) as output:
                cli.main()
            self.assertEqual(json.loads(output.getvalue())["status"], "incomplete")
            (root / "scores.json").write_text(json.dumps(dict(scores, valid_success=True)))
            with self.assertRaises(ValueError):
                online_rollout_record(tmp)

    def test_clean_completion_is_not_called_recovery(self):
        c = Contract("easy", stable_steps=1, final_stable_steps=1)
        trace = [{"type": "decision", "action": {"type": "pick", "tube": "NMR_02"},
                  "legal_actions": [{"type": "pick", "tube": "NMR_02"}]},
                 {"type": "physical", "event": "inserted", "tube": "NMR_01", "slot": "S00", "precision_pass": True},
                 {"type": "physical", "event": "inserted", "tube": "NMR_02", "slot": "S06", "precision_pass": True},
                 {"type": "snapshot", "terminal_steps": 1}]
        score = score_trace(c, trace)
        self.assertTrue(score["completion_success"])
        self.assertFalse(score["recovery_completion"])

    def test_violation_after_completion_is_not_called_recovery(self):
        c = Contract("easy", stable_steps=1, final_stable_steps=1)
        trace = [
            {"type": "decision", "action": {"type": "pick", "tube": "NMR_02"},
             "legal_actions": [{"type": "pick", "tube": "NMR_02"}]},
            {"type": "physical", "event": "inserted", "tube": "NMR_01", "slot": "S00",
             "precision_pass": True},
            {"type": "physical", "event": "inserted", "tube": "NMR_02", "slot": "S06",
             "precision_pass": True},
            {"type": "snapshot", "terminal_steps": 1},
            {"type": "violation", "kind": "operation_timeout"},
        ]
        self.assertFalse(score_trace(c, trace)["recovery_completion"])

    def test_wrong_physical_order_does_not_pass_protocol(self):
        c = Contract("hard", stable_steps=1, final_stable_steps=1)
        trace = [
            {"type": "decision", "action": {"type": "pick", "tube": "NMR_01"},
             "legal_actions": [{"type": "pick", "tube": "NMR_01"}]},
            {"type": "physical", "event": "inserted", "tube": "NMR_02", "slot": "S06", "precision_pass": True},
            {"type": "physical", "event": "inserted", "tube": "NMR_01", "slot": "S00", "precision_pass": True},
            {"type": "snapshot", "terminal_steps": 1},
        ]
        scored = score_trace(c, trace)
        self.assertTrue(scored["completion_success"])
        self.assertFalse(scored["protocol_pass"])
        self.assertFalse(scored["valid_success"])

    def test_premature_stop_is_protocol_error_even_without_reported_violation(self):
        c = Contract("easy")
        result = score_trace(c, [{"type": "decision", "action": {"type": "stop"},
                                  "legal_actions": [{"type": "pick", "tube": "NMR_02"}]}])
        self.assertEqual(result["illegal_decision_count"], 1)
        self.assertFalse(result["protocol_pass"])

    def test_failed_physical_attempt_does_not_relabel_correct_decision_as_protocol_error(self):
        c = Contract("hard")
        score = score_trace(c, [
            {"type": "decision", "action": {"type": "pick", "tube": "NMR_01"},
             "legal_actions": [{"type": "pick", "tube": "NMR_01"}]},
            {"type": "physical", "event": "grasp_attempt_failed", "tube": "NMR_01"},
            {"type": "violation", "kind": "operation_timeout"},
        ])
        self.assertTrue(score["decision_pass"])
        self.assertTrue(score["protocol_pass"])
        self.assertFalse(score["completion_success"])
        self.assertEqual(score["grasp_attempt_failed_count"], 1)
        self.assertEqual(score["operation_timeout_count"], 1)

    def test_pair_denominator_and_duplicate_rejection(self):
        items = [{"pair_id": "a", "condition": "none", "success": False},
                 {"pair_id": "a", "condition": "semantic", "success": True},
                 {"pair_id": "b", "condition": "none", "success": False}]
        score = compare_pairs(items, "semantic")
        self.assertEqual(score["registered_pairs"], 2)
        self.assertEqual(score["complete_pairs"], 1)
        self.assertEqual(score["status"], "incomplete")
        self.assertIsNone(score["paired_absolute_gain"])
        self.assertEqual(score["missing_conditions"]["semantic"], 1)
        complete = compare_pairs(items + [{"pair_id": "b", "condition": "semantic", "success": False}],
                                 "semantic")
        self.assertEqual(complete["paired_absolute_gain"], .5)
        with self.assertRaises(ValueError):
            compare_pairs(items + [items[0]], "semantic")
        blocked = compare_pairs([
            {"pair_id": "x", "condition": "none", "success": False},
            {"pair_id": "x", "condition": "semantic", "status": "unsupported"},
        ], "semantic")
        self.assertEqual(blocked["unsupported_pairs"], 1)
        self.assertEqual(blocked["scored_pairs"], 0)
        self.assertIsNone(blocked["paired_absolute_gain"])
        with self.assertRaises(ValueError):
            compare_pairs([
                {"pair_id": "y", "condition": "none", "success": False, "scene_seed": 1},
                {"pair_id": "y", "condition": "semantic", "success": True, "scene_seed": 2},
            ], "semantic")
        with self.assertRaises(ValueError):
            compare_pairs([{"pair_id": "e", "condition": "none", "success": True,
                            "policy_uses_evaluator_state": True}], "semantic")
        with self.assertRaises(ValueError):
            compare_pairs([{"pair_id": "m", "condition": "none", "success": False,
                            "policy_id": "a"},
                           {"pair_id": "m", "condition": "semantic", "success": True,
                            "policy_id": "b"}], "semantic")

    def test_preregistered_pair_denominator_and_state_match(self):
        c = Contract("hard")
        planned = plan_state_pairs(c, [101])
        registered = {r["oracle_pair_id"]: {k: r[k] for k in
                      ("task_version", "contract_hash", "scene_seed", "decision_seed",
                       "first_actual_slot", "variant")} for r in planned}
        record = dict(registered[planned[0]["oracle_pair_id"]],
                      pair_id=planned[0]["oracle_pair_id"], condition="none", success=False,
                      policy_id="candidate-a", policy_uses_evaluator_state=False)
        compared = compare_pairs([record], "semantic", registered)
        self.assertEqual(compared["registered_pairs"], 2)
        self.assertEqual(compared["missing_pairs"], 2)
        self.assertEqual(compared["status"], "incomplete")
        with self.assertRaises(ValueError):
            compare_pairs([dict(record, first_actual_slot="S11")], "semantic", registered)
        with self.assertRaises(ValueError):
            compare_pairs([dict(record, condition="geometry")], "semantic", registered)
        with self.assertRaises(ValueError):
            compare_pairs([dict(record, policy_uses_evaluator_state=True)], "semantic", registered)

    def test_checkpoint_adapter_scores_report_and_declares_unsupported(self):
        c = Contract("easy")
        state = State("easy", first_actual_slot="S02")
        obs = policy_observation({"front": "frame-1"}, [0.0], "opposite", list(SLOTS))
        caps = PolicyCapabilities(frozenset({"semantic", "decision"}), frozenset({"structured"}))
        def policy(request):
            self.assertEqual(request["oracle_context"]["visible_occupancy"], {"S02": "NMR_01"})
            self.assertNotIn("advice", request["oracle_context"])
            return {"semantic_report": {"S02": "NMR_01"},
                    "committed_action": {"type": "pick", "tube": "NMR_02"}}
        scored = evaluate_policy_checkpoint(c, state, obs, {"S02": "NMR_01"}, {}, policy, caps,
                                             instance_id="easy-02", profile="semantic", reset_policy=lambda: None)
        self.assertEqual(scored["scores"]["semantic"], "pass")
        self.assertEqual(scored["scores"]["decision"], "pass")
        unsupported = evaluate_policy_checkpoint(c, state, obs, {}, {}, policy, caps,
                                                   instance_id="easy-02", profile="semantic", transport="text",
                                                   reset_policy=lambda: None)
        self.assertEqual(unsupported["status"], "unsupported")
        service = evaluate_policy_checkpoint(c, state, obs, {}, {},
                                             lambda _: 1 / 0, caps, instance_id="err", reset_policy=lambda: None)
        self.assertEqual(service["status"], "service_error")
        action_only = PolicyCapabilities(frozenset(), frozenset(), action_space="array")
        separate = evaluate_policy_checkpoint(c, state, obs, {}, {}, lambda _: {"action": [0.0]},
                                               action_only, instance_id="array", reset_policy=lambda: None)
        self.assertEqual(separate["scores"]["decision"], "unsupported")
        self.assertEqual(evaluate_policy_checkpoint(c, state, obs, {}, {}, policy, caps,
                                                     instance_id="missing-reset")["status"], "unsupported")
        from eai4s_eval.centrifuge_v2.diagnostics import VerifiedSkillExecutor
        skill_caps = PolicyCapabilities(frozenset({"execution"}), frozenset({"structured"}))
        chosen = {"type": "pick", "tube": "NMR_02"}
        missing_skill = evaluate_policy_checkpoint(c, state, obs, {}, {}, lambda _: {},
                                                    skill_caps, instance_id="skill", profile="skill",
                                                    selected_action=chosen, reset_policy=lambda: None)
        self.assertEqual(missing_skill["status"], "unsupported")
        executed = evaluate_policy_checkpoint(c, state, obs, {}, {}, lambda _: {},
                                               skill_caps, instance_id="skill", profile="skill",
                                               selected_action=chosen, reset_policy=lambda: None,
                                               skill_executor=VerifiedSkillExecutor(
                                                   "pick-smoke-v1", frozenset({"pick"}),
                                                   lambda action: {"executed_action": action},
                                                   lambda action: action == chosen, verified=True))
        self.assertTrue(executed["reference_executor_success"])
        self.assertEqual(executed["scores"]["execution"], "pass")

    def test_geometry_oracle_candidate_frames_pass_online_input_validation(self):
        c = Contract("hard")
        state = State("hard")
        state.tube_status["NMR_01"] = "held"
        frames = {slot: {"mouth_world_m": [index * .01, 0, .87],
                         "axis_world": [0, 0, -1]} for index, slot in enumerate(SLOTS)}
        chosen = {"type": "insert", "tube": "NMR_01", "slot": "S03"}
        geometry_report = dict(frames["S03"], slot="S03")
        def policy(request):
            self.assertEqual(len(request["oracle_context"]["candidate_frames"]), 12)
            self.assertNotIn("advice", request["oracle_context"])
            return {"committed_action": chosen, "geometry_report": geometry_report}
        result = evaluate_policy_checkpoint(
            c, state, policy_observation({}, [0.0], "place tubes opposite", list(SLOTS)),
            {}, frames, policy, PolicyCapabilities(frozenset({"decision", "geometry"}),
                                                   frozenset({"structured"})),
            instance_id="geometry-03", profile="geometry", selected_action=chosen,
            reference_geometry=geometry_report, reset_policy=lambda: None)
        self.assertEqual(result["scores"]["decision"], "pass")
        self.assertEqual(result["scores"]["geometry"], "pass")

    def test_counterfactual_plan_changes_actual_slot_not_rule(self):
        c = Contract("hard")
        pair = plan_state_pairs(c, [4])
        self.assertEqual([r["variant"] for r in pair], ["a", "b"])
        self.assertEqual(pair[0]["scene_seed"], pair[1]["scene_seed"])
        self.assertEqual(pair[0]["legal_actions"][0]["slot"], "S10")
        self.assertEqual(pair[1]["legal_actions"][0]["slot"], "S11")

    def test_state_pair_consistency_uses_registered_denominator_and_correct_target(self):
        plan = plan_state_pairs(Contract("hard"), [101, 102])
        correct = [dict(item, committed_action=item["legal_actions"][0],
                        policy_id="candidate-a", policy_uses_evaluator_state=False) for item in plan]
        full = compare_state_pairs(correct, plan)
        self.assertEqual(full["registered_pairs"], 2)
        self.assertEqual(full["counterfactual_consistency_score"], 1.0)
        missing = compare_state_pairs(correct[:-1], plan)
        self.assertEqual(missing["status"], "incomplete")
        self.assertIsNone(missing["counterfactual_consistency_score"])
        same_answer = [dict(item, committed_action=correct[0]["committed_action"]) for item in correct]
        wrong = compare_state_pairs(same_answer, plan)
        self.assertEqual(wrong["counterfactual_consistency_score"], 0.0)
        self.assertEqual(wrong["decision_switch_rate"], 0.0)
        with self.assertRaises(ValueError):
            compare_state_pairs(correct + [correct[0]], plan)
        with self.assertRaises(ValueError):
            compare_state_pairs([dict(correct[0], first_actual_slot="S11")], plan)
        with self.assertRaises(ValueError):
            compare_state_pairs([dict(correct[0], policy_uses_evaluator_state=True)], plan)


if __name__ == "__main__":
    unittest.main()
