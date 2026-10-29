from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eai4s_eval.cli.run_benchmark_eval import build_parser
from eai4s_eval.eval.success import PredicateCall, SuccessSpec, TaskSuccessEvaluator
from eai4s_eval.eval.success.progress import EvaluationProgressReporter, build_progress_payload


class EvaluationProgressTest(unittest.TestCase):
    def test_cli_defaults_to_one_second_and_both_outputs(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.progress_interval_sec, 1.0)
        self.assertEqual(args.progress_log_format, "both")

    def test_numeric_predicate_reports_remaining_distance(self) -> None:
        spec = SuccessSpec(
            hold_steps=2,
            goal_all=[PredicateCall("near", args={"distance_threshold_m": 0.05})],
        )
        result = TaskSuccessEvaluator(spec).update(
            {
                "object_pos": [0.08, 0.0, 0.0],
                "target_pos": [0.0, 0.0, 0.0],
            }
        )
        payload = build_progress_payload(
            result,
            task_id="place",
            episode_id="place_seed1",
            episode_index=0,
            split="sim_dev",
            seed=1,
            action_step=3,
            sim_step=18,
            policy_calls=1,
        )
        self.assertEqual(payload["stage_id"], "goal:near")
        condition = payload["criteria"][0]["conditions"][0]
        self.assertAlmostEqual(condition["actual"], 0.08)
        self.assertAlmostEqual(condition["threshold"], 0.05)
        self.assertAlmostEqual(condition["remaining"], 0.03)

    def test_task_state_predicate_uses_runtime_geometry_diagnostics(self) -> None:
        spec = SuccessSpec(
            hold_steps=1,
            goal_all=[
                PredicateCall(
                    "task_state_true",
                    args={"id": "sample_in_target_box", "key": "task_state.sample_in_target_box"},
                )
            ],
        )
        result = TaskSuccessEvaluator(spec).update(
            {
                "task_state": {
                    "sample_in_target_box": False,
                    "predicate_diagnostics": {
                        "sample_in_target_box": {
                            "criteria": [
                                {"name": "xy_error", "actual": 0.024, "operator": "<=", "threshold": 0.02, "unit": "m"},
                                {"name": "z_error", "actual": 0.01, "operator": "<=", "threshold": 0.05, "unit": "m"},
                            ]
                        }
                    },
                }
            }
        )
        payload = build_progress_payload(
            result,
            task_id="sample",
            episode_id="sample_seed1",
            episode_index=0,
            split="sim_dev",
            seed=1,
            action_step=1,
            sim_step=6,
            policy_calls=1,
        )
        conditions = payload["criteria"][0]["conditions"]
        self.assertAlmostEqual(conditions[0]["remaining"], 0.004)
        self.assertEqual(conditions[1]["remaining"], 0.0)

    def test_reporter_is_periodic_but_emits_predicate_changes_immediately(self) -> None:
        spec = SuccessSpec(
            hold_steps=3,
            goal_all=[PredicateCall("lifted", args={"height_threshold_m": 0.10})],
        )
        evaluator = TaskSuccessEvaluator(spec)
        failed = evaluator.update(
            {"object_pos": [0, 0, 0.05], "initial_object_pos": [0, 0, 0.0]}
        )
        passed = evaluator.update(
            {"object_pos": [0, 0, 0.11], "initial_object_pos": [0, 0, 0.0]}
        )
        with tempfile.TemporaryDirectory() as tmp:
            reporter = EvaluationProgressReporter(
                run_dir=tmp,
                run_id="run",
                task_id="pick",
                spec=spec,
                predicate_hz=60,
                interval_sec=1.0,
                output_format="jsonl",
            )
            reporter.start_episode(
                episode_id="pick_seed1", episode_index=0, split="sim_dev", seed=1
            )
            reporter.observe(failed, action_step=1, sim_step=1, policy_calls=1)
            reporter.observe(failed, action_step=1, sim_step=2, policy_calls=1)
            reporter.observe(passed, action_step=1, sim_step=3, policy_calls=1)
            reporter.observe(passed, action_step=10, sim_step=63, policy_calls=2)
            reporter.close()

            rows = [
                json.loads(line)
                for line in (Path(tmp) / "logs" / "progress_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
        self.assertEqual([row["event_type"] for row in rows], [
            "criteria", "stage_change", "stage_change", "periodic"
        ])
        self.assertEqual(rows[2]["stage_id"], "hold")


if __name__ == "__main__":
    unittest.main()
