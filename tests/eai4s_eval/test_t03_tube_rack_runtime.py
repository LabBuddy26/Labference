from __future__ import annotations

import unittest

from eai4s_eval.eval.success import PredicateCall, SuccessSpec, TaskSuccessEvaluator
from eai4s_eval.eval.success.predicates import MissingPredicateState
from eai4s_eval.scenes.registry import available_scene_ids, get_scene_runtime
from eai4s_eval.scenes.tabletop_tube_rack_insert_scene import compute_tube_rack_insert_metrics
from eai4s_eval.utils.assets import tube_rack_insert_asset_paths


def t03_snapshot(
    *,
    object_pos=(0.131, 0.231, 0.155),
    target_slot_pos=(0.131, 0.231, 0.18),
    alignment_cosine=0.99,
    insert_depth_m=0.025,
    hand_pos=(0.35, 0.231, 0.25),
    gripper_pos=(0.04, 0.04),  # Franka: 0.04=open → released
    object_lin_vel=(0.0, 0.0, 0.0),
    object_ang_vel=(0.0, 0.0, 0.0),
    events=None,
):
    return {
        "object_pos": object_pos,
        "initial_object_pos": (0.0, 0.0, 0.0),
        "hand_pos": hand_pos,
        "gripper_pos": gripper_pos,
        "target_slot_pos": target_slot_pos,
        "alignment_cosine": alignment_cosine,
        "insert_depth_m": insert_depth_m,
        "object_lin_vel": object_lin_vel,
        "object_ang_vel": object_ang_vel,
        "forbidden_collision": False,
        "wrong_object_moved": False,
        "wrong_target_reached": False,
        "max_contact_force_n": 0.0,
        "events": events or [],
        "task_state": {
            "insertion_position_error_m": 0.0,
            "alignment_error_deg": 0.0,
            "insertion_depth_m": insert_depth_m,
        },
    }


def strict_t03_spec() -> SuccessSpec:
    return SuccessSpec(
        hold_steps=2,
        goal_all=[
            PredicateCall(
                "near",
                args={"a_key": "object_pos", "b_key": "target_slot_pos", "axes": "xy", "distance_threshold_m": 0.02},
            ),
            PredicateCall("aligned", args={"min_cosine": 0.92}),
            PredicateCall("insert_depth_reached", args={"min_depth_m": 0.025}),
        ],
        sequence_required=True,
        sequence_events=["grasp_tube", "align_to_slot", "insert_tube"],
        terminal_required=True,
        terminal_all=[
            PredicateCall("released", args={"open_threshold": 0.035}),
            PredicateCall("tcp_far", args={"distance_threshold_m": 0.08}),
        ],
        stability_required=True,
        stability_all=[
            PredicateCall("stable_pose", args={"linear_velocity_threshold_mps": 0.02, "angular_velocity_threshold_radps": 0.20})
        ],
        safety_hard_fail=[
            PredicateCall("forbidden_collision"),
            PredicateCall("wrong_object_moved"),
            PredicateCall("wrong_target_reached", args={"expected_target_region_id": "target_slot"}),
            PredicateCall("contact_force_exceeded", args={"max_force_n": 50.0}),
        ],
        quality_metrics=["insertion_position_error_m", "alignment_error_deg", "insertion_depth_m"],
    )


class TubeRackRuntimeTest(unittest.TestCase):
    def test_scene_registry_resolves_t02_and_t03(self) -> None:
        self.assertIn("tabletop_pick_lift_proxy", available_scene_ids())
        self.assertIn("tabletop_tube_rack_insert", available_scene_ids())
        self.assertEqual(get_scene_runtime("tabletop_pick_lift_proxy").target_object_name, "vial")
        self.assertEqual(get_scene_runtime("tabletop_tube_rack_insert").resolve_target_object_name("vial"), "tube")
        self.assertEqual(
            get_scene_runtime("tabletop_tube_rack_insert").distractor_names,
            ("beaker", "vial", "boiling_flask", "Erlenmeyer_flask"),
        )
        with self.assertRaises(ValueError):
            get_scene_runtime("missing_scene")

    def test_t03_asset_resolver_includes_reaction_distractors(self) -> None:
        assets = tube_rack_insert_asset_paths("asset_root", must_exist=False)
        normalized = {key: value.replace("\\", "/") for key, value in assets.__dict__.items()}
        self.assertTrue(normalized["tube"].endswith("assets/test-tube/test_tube.usd"))
        self.assertTrue(normalized["rack"].endswith("assets/tube-rack/tube_rack.usd"))
        self.assertTrue(normalized["beaker"].endswith("assets/beaker/beaker.usd"))
        self.assertTrue(normalized["vial"].endswith("assets/vial/vial_brown/vial_brown.usd"))
        self.assertTrue(normalized["boiling_flask"].endswith("assets/boiling_flask/flask.usd"))
        self.assertTrue(normalized["erlenmeyer_flask"].endswith("assets/Erlenmeyer_flask/flask.usd"))

    def test_t03_geometry_matches_data_collection_thresholds(self) -> None:
        metrics = compute_tube_rack_insert_metrics(
            tube_pos=(0.131, 0.231, 0.155),
            tube_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            rack_pos=(0.1, 0.2, 0.1),
        )
        self.assertAlmostEqual(metrics["target_slot_pos"][0], 0.131)
        self.assertAlmostEqual(metrics["target_slot_pos"][1], 0.231)
        self.assertAlmostEqual(metrics["target_slot_pos"][2], 0.18)
        self.assertAlmostEqual(metrics["slot_xy_error_m"], 0.0)
        self.assertAlmostEqual(metrics["insert_depth_m"], 0.025)
        self.assertAlmostEqual(metrics["alignment_cosine"], 1.0)

    def test_t03_strict_success_requires_sequence_terminal_and_stability(self) -> None:
        evaluator = TaskSuccessEvaluator(strict_t03_spec())
        grasped_only = evaluator.update(
            t03_snapshot(
                object_pos=(0.0, 0.0, 0.1),
                target_slot_pos=(0.131, 0.231, 0.18),
                insert_depth_m=0.0,
                hand_pos=(0.0, 0.0, 0.1),
                gripper_pos=(0.01, 0.01),
                events=["grasp_tube"],
            )
        )
        self.assertFalse(grasped_only.task_success)
        self.assertFalse(grasped_only.goal_pass)

        aligned_not_inserted = evaluator.update(
            t03_snapshot(insert_depth_m=0.0, events=["align_to_slot"])
        )
        self.assertFalse(aligned_not_inserted.task_success)
        self.assertFalse(aligned_not_inserted.goal_pass)

        # 0.01 = closed → gripper never opened, so the release terminal fails.
        inserted_but_not_released = evaluator.update(
            t03_snapshot(gripper_pos=(0.01, 0.01), events=["insert_tube"])
        )
        self.assertFalse(inserted_but_not_released.task_success)
        self.assertFalse(inserted_but_not_released.terminal_pass)

        first_hold = evaluator.update(t03_snapshot())
        self.assertFalse(first_hold.task_success)
        success = evaluator.update(t03_snapshot())
        self.assertTrue(success.task_success)
        self.assertEqual(success.success_hold_counter, 2)
        self.assertEqual(success.quality_metrics["insertion_depth_m"], 0.025)

    def test_t03_missing_insert_depth_reports_clear_error(self) -> None:
        state = t03_snapshot()
        state.pop("insert_depth_m")
        state["task_state"].pop("insertion_depth_m")
        state["task_state"].pop("insert_depth_m", None)
        with self.assertRaises(MissingPredicateState):
            TaskSuccessEvaluator(strict_t03_spec()).update(state)


if __name__ == "__main__":
    unittest.main()
