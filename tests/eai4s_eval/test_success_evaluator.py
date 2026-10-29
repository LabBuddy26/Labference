from __future__ import annotations

import unittest

from eai4s_eval.eval.success import PredicateCall, SuccessSpec, TaskSuccessEvaluator
from eai4s_eval.eval.success.failure_taxonomy import classify_failure
from eai4s_eval.eval.success.predicates import MissingPredicateState


def pick_snapshot(
    *,
    object_pos=(0.0, 0.0, 0.06),
    initial_object_pos=(0.0, 0.0, 0.0),
    hand_pos=(0.0, 0.0, 0.06),
    gripper_pos=(0.02, 0.02),  # Franka: 0.0=closed, 0.04=open
    object_lin_vel=(0.0, 0.0, 0.0),
    object_ang_vel=(0.0, 0.0, 0.0),
    task_state=None,
):
    return {
        "object_pos": object_pos,
        "initial_object_pos": initial_object_pos,
        "hand_pos": hand_pos,
        "gripper_pos": gripper_pos,
        "object_lin_vel": object_lin_vel,
        "object_ang_vel": object_ang_vel,
        "task_state": task_state or {},
    }


class SuccessEvaluatorTest(unittest.TestCase):
    def test_success_requires_sustained_hold(self) -> None:
        spec = SuccessSpec(
            hold_steps=2,
            goal_all=[
                PredicateCall("grasped", args={"distance_threshold_m": 0.15, "grip_threshold": 0.04}),
                PredicateCall("lifted", args={"height_threshold_m": 0.05}),
            ],
        )
        evaluator = TaskSuccessEvaluator(spec)

        first = evaluator.update(pick_snapshot())
        self.assertFalse(first.task_success)
        self.assertEqual(first.success_hold_counter, 1)

        failed = evaluator.update(pick_snapshot(hand_pos=(1.0, 0.0, 0.0)))
        self.assertFalse(failed.task_success)
        self.assertEqual(failed.success_hold_counter, 0)

        evaluator.update(pick_snapshot())
        success = evaluator.update(pick_snapshot())
        self.assertTrue(success.task_success)
        self.assertTrue(success.goal_pass)
        self.assertEqual(success.success_hold_counter, 2)

    def test_hard_failure_blocks_success(self) -> None:
        spec = SuccessSpec(
            hold_steps=1,
            goal_all=[PredicateCall("lifted", args={"height_threshold_m": 0.01})],
            safety_hard_fail=[PredicateCall("object_out_of_workspace", args={"workspace_radius_m": 0.5})],
        )
        result = TaskSuccessEvaluator(spec).update(pick_snapshot(object_pos=(1.0, 0.0, 0.06)))
        self.assertFalse(result.task_success)
        self.assertFalse(result.safety_valid)
        self.assertEqual(result.failure_type, "object_out_of_workspace")

    def test_terminal_release_is_task_configurable(self) -> None:
        spec = SuccessSpec(
            hold_steps=1,
            goal_all=[PredicateCall("lifted", args={"height_threshold_m": 0.01})],
            terminal_required=True,
            terminal_all=[PredicateCall("released", args={"open_threshold": 0.035})],
        )
        evaluator = TaskSuccessEvaluator(spec)
        # 0.04 ≈ open → released; terminal holds once the gripper is open.
        released = evaluator.update(pick_snapshot(gripper_pos=(0.04, 0.04)))
        self.assertTrue(released.task_success)
        self.assertTrue(released.terminal_pass)

        # 0.01 = closed → not released, so terminal fails.
        closed = evaluator.update(pick_snapshot(gripper_pos=(0.01, 0.01)))
        self.assertFalse(closed.task_success)
        self.assertFalse(closed.terminal_pass)

    def test_chemical_predicate_requires_state_source(self) -> None:
        spec = SuccessSpec(
            hold_steps=1,
            goal_all=[PredicateCall("volume_transferred", args={"target_ml": 2.0, "tolerance_ml": 0.1})],
        )
        evaluator = TaskSuccessEvaluator(spec)
        with self.assertRaises(MissingPredicateState):
            evaluator.update(pick_snapshot())

        ok = evaluator.update(pick_snapshot(task_state={"volume_transferred_ml": 2.05}))
        self.assertTrue(ok.task_success)

    def test_required_sequence_must_complete_in_order(self) -> None:
        spec = SuccessSpec(
            hold_steps=1,
            goal_all=[PredicateCall("lifted", args={"height_threshold_m": 0.01})],
            sequence_required=True,
            sequence_events=["attach_tip", "aspirate", "dispense", "retract"],
        )
        evaluator = TaskSuccessEvaluator(spec)

        first = evaluator.update({**pick_snapshot(), "events": ["attach_tip", "aspirate"]})
        self.assertFalse(first.task_success)
        self.assertFalse(first.sequence_pass)
        self.assertEqual(first.sequence_stage, "dispense")

        success = evaluator.update({**pick_snapshot(), "events": ["dispense", "retract"]})
        self.assertTrue(success.task_success)
        self.assertTrue(success.sequence_pass)
        self.assertEqual(success.sequence_stage, "completed")

    def test_sequence_order_violation_blocks_success_when_required(self) -> None:
        spec = SuccessSpec(
            hold_steps=1,
            goal_all=[PredicateCall("lifted", args={"height_threshold_m": 0.01})],
            sequence_required=True,
            sequence_events=["attach_tip", "aspirate", "dispense"],
        )
        result = TaskSuccessEvaluator(spec).update({**pick_snapshot(), "events": ["dispense"]})
        self.assertFalse(result.task_success)
        self.assertFalse(result.sequence_pass)
        self.assertEqual(result.sequence_stage, "invalid:dispense")

    def test_optional_sequence_is_diagnostic_only(self) -> None:
        spec = SuccessSpec(
            hold_steps=1,
            goal_all=[PredicateCall("lifted", args={"height_threshold_m": 0.01})],
            sequence_required=False,
            sequence_events=["attach_tip", "aspirate"],
        )
        result = TaskSuccessEvaluator(spec).update(pick_snapshot())
        self.assertTrue(result.task_success)
        self.assertFalse(result.sequence_pass)

    def test_hard_failure_is_latched(self) -> None:
        spec = SuccessSpec(
            hold_steps=1,
            goal_all=[PredicateCall("lifted", args={"height_threshold_m": 0.01})],
            safety_hard_fail=[PredicateCall("object_out_of_workspace", args={"workspace_radius_m": 0.5})],
        )
        evaluator = TaskSuccessEvaluator(spec)
        failed = evaluator.update(pick_snapshot(object_pos=(1.0, 0.0, 0.06)))
        self.assertEqual(failed.latched_violations, ["object_out_of_workspace"])

        recovered = evaluator.update(pick_snapshot(object_pos=(0.0, 0.0, 0.06)))
        self.assertFalse(recovered.task_success)
        self.assertFalse(recovered.safety_valid)
        self.assertEqual(recovered.latched_violations, ["object_out_of_workspace"])

    def test_quality_metrics_are_extracted_when_configured(self) -> None:
        spec = SuccessSpec(
            hold_steps=1,
            goal_all=[PredicateCall("lifted", args={"height_threshold_m": 0.01})],
            quality_metrics=["volume_error_ml", "spill_ml"],
        )
        result = TaskSuccessEvaluator(spec).update(
            pick_snapshot(task_state={"volume_error_ml": 0.02, "spill_ml": 0.0})
        )
        self.assertTrue(result.task_success)
        self.assertEqual(result.quality_metrics, {"volume_error_ml": 0.02, "spill_ml": 0.0})

    def test_failure_predicates_for_basic_task_taxonomy(self) -> None:
        spec = SuccessSpec(
            hold_steps=1,
            goal_all=[PredicateCall("lifted", args={"height_threshold_m": 0.01})],
            safety_hard_fail=[
                PredicateCall("object_dropped", args={"min_height_m": 0.02}),
                PredicateCall("wrong_target_reached", args={"expected_target_region_id": "slot_a"}),
                PredicateCall("wrong_object_selected", args={"expected_object_id": "tube"}),
                PredicateCall("contact_force_exceeded", args={"max_force_n": 10.0}),
                PredicateCall("object_tipped_over", args={"min_upright_cosine": 0.8}),
            ],
        )
        base_state = {
            "target_region_id": "slot_a",
            "selected_object_id": "tube",
            "max_contact_force_n": 0.0,
            "object_upright_cosine": 1.0,
        }

        dropped = TaskSuccessEvaluator(spec).update({**pick_snapshot(object_pos=(0.0, 0.0, 0.01)), **base_state})
        self.assertEqual(dropped.failure_type, "object_dropped")

        wrong_target = TaskSuccessEvaluator(spec).update({**pick_snapshot(), **base_state, "target_region_id": "slot_b"})
        self.assertEqual(wrong_target.failure_type, "wrong_target_reached")

        wrong_object = TaskSuccessEvaluator(spec).update({**pick_snapshot(), **base_state, "selected_object_id": "beaker"})
        self.assertEqual(wrong_object.failure_type, "wrong_object_selected")

        force = TaskSuccessEvaluator(spec).update({**pick_snapshot(), **base_state, "max_contact_force_n": 12.0})
        self.assertEqual(force.failure_type, "contact_force_exceeded")

        tipped = TaskSuccessEvaluator(spec).update(
            {
                **pick_snapshot(),
                **base_state,
                "object_upright_cosine": 0.5,
            }
        )
        self.assertEqual(tipped.failure_type, "object_tipped_over")

    def test_failure_taxonomy_classifies_common_failures(self) -> None:
        timeout = classify_failure(success=False, termination_reason="timeout", failure_type="timeout")
        self.assertEqual(timeout.failure_category, "timeout")
        self.assertEqual(timeout.failure_code, "timeout")

        collision = classify_failure(success=False, termination_reason="catastrophic_failure", failure_type="forbidden_collision")
        self.assertEqual(collision.failure_category, "collision")

        drop = classify_failure(success=False, termination_reason="catastrophic_failure", failure_type="object_out_of_workspace")
        self.assertEqual(drop.failure_category, "drop")

        wrong_place = classify_failure(success=False, termination_reason="catastrophic_failure", failure_type="wrong_target_reached")
        self.assertEqual(wrong_place.failure_category, "wrong_place")

        policy = classify_failure(success=False, termination_reason="policy_request_failed", failure_type="policy_request_failed")
        self.assertEqual(policy.failure_category, "policy_error")

        spec = SuccessSpec(
            hold_steps=1,
            goal_all=[PredicateCall("lifted", args={"height_threshold_m": 0.01})],
            sequence_required=True,
            sequence_events=["a", "b"],
        )
        sequence_result = TaskSuccessEvaluator(spec).update({**pick_snapshot(), "events": ["b"]})
        sequence_error = classify_failure(
            success=False,
            termination_reason="timeout",
            failure_type=None,
            success_result=sequence_result,
        )
        self.assertEqual(sequence_error.failure_category, "sequence_error")


if __name__ == "__main__":
    unittest.main()
