from __future__ import annotations

import unittest

from eai4s_eval.cli.run_pick_lift_eval import build_parser
from eai4s_eval.server.app import _aggregate_leaderboard_rows, _legacy_eval_group_id


class ServerLeaderboardTest(unittest.TestCase):
    def test_cli_defaults_use_named_model_and_auto_checkpoint(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.model_name, "eai4s_model")
        self.assertIsNone(args.checkpoint)

    def test_legacy_split_suffix_maps_to_base_group(self) -> None:
        self.assertEqual(_legacy_eval_group_id("eval_20260702_100000_model_sim_clean"), "eval_20260702_100000_model")
        self.assertEqual(_legacy_eval_group_id("eval_20260702_100000_model_sim_randomized"), "eval_20260702_100000_model")
        self.assertEqual(_legacy_eval_group_id("eval_20260702_100000_model"), "eval_20260702_100000_model")

    def test_multi_split_runs_merge_into_one_leaderboard_row(self) -> None:
        base = {
            "eval_group_id": "eval_20260702_100000_model",
            "model_name": "model",
            "checkpoint": "eval_20260702_100000",
            "task_ids": ["T02"],
            "task_names": ["Franka pick/lift proxy"],
            "task_families": ["pick_place"],
            "primary_task_id": "T02",
            "primary_task_name": "Franka pick/lift proxy",
            "primary_task_family": "pick_place",
            "success_logic": "robust_semantic_v0",
            "stage_completion_mean": 1.0,
            "inference_latency_p95_ms": 80.0,
            "failure_category_distribution": {"timeout": 2},
            "failure_code_distribution": {},
            "created_at": "2026-07-02T10:00:00+08:00",
            "task_summary": [
                {
                    "task_id": "T02",
                    "task_name": "Franka pick/lift proxy",
                    "task_family": "pick_place",
                    "leaderboard_group": "basic_pick_place",
                    "split": "sim_dev",
                    "success_rate": 0.8,
                    "safe_success_rate": 0.8,
                    "num_trials": 10,
                    "num_success": 8,
                    "failure_category_distribution": {"timeout": 2},
                }
            ],
            "family_summary": [
                {
                    "task_family": "pick_place",
                    "task_ids": ["T02"],
                    "leaderboard_group": "basic_pick_place",
                    "split": "sim_dev",
                    "success_rate": 0.8,
                    "safe_success_rate": 0.8,
                    "num_trials": 10,
                    "num_success": 8,
                    "failure_category_distribution": {"timeout": 2},
                }
            ],
        }
        runs = [
            {**base, "run_id": "eval_20260702_100000_model", "split": "sim_dev", "track": "sim_dev", "num_trials": 10, "num_success": 8, "success_rate": 0.8},
            {
                **base,
                "run_id": "eval_20260702_100000_model_sim_clean",
                "split": "sim_clean",
                "track": "sim_clean",
                "num_trials": 10,
                "num_success": 9,
                "success_rate": 0.9,
                "task_summary": [{**base["task_summary"][0], "split": "sim_clean", "success_rate": 0.9, "num_success": 9, "failure_category_distribution": {"drop": 1}}],
                "family_summary": [{**base["family_summary"][0], "split": "sim_clean", "success_rate": 0.9, "num_success": 9, "failure_category_distribution": {"drop": 1}}],
                "failure_category_distribution": {"drop": 1},
            },
            {
                **base,
                "run_id": "eval_20260702_100000_model_sim_randomized",
                "split": "sim_randomized",
                "track": "sim_randomized",
                "num_trials": 10,
                "num_success": 7,
                "success_rate": 0.7,
                "task_summary": [{**base["task_summary"][0], "split": "sim_randomized", "success_rate": 0.7, "num_success": 7, "failure_category_distribution": {"collision": 3}}],
                "family_summary": [{**base["family_summary"][0], "split": "sim_randomized", "success_rate": 0.7, "num_success": 7, "failure_category_distribution": {"collision": 3}}],
                "failure_category_distribution": {"collision": 3},
            },
        ]

        rows = _aggregate_leaderboard_rows(runs)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["available_splits"], ["sim_dev", "sim_clean", "sim_randomized"])
        self.assertEqual(row["run_ids_by_split"]["sim_clean"], "eval_20260702_100000_model_sim_clean")
        self.assertEqual(row["num_trials"], 30)
        self.assertEqual(row["num_success"], 24)
        self.assertAlmostEqual(row["task_success_rate"], 0.8)
        self.assertAlmostEqual(row["overall"]["sim_randomized_macro_success_rate"], 0.7)
        self.assertIn("pick_place", row["by_family"])
        self.assertTrue(row["mini_leaderboards"]["by_task"])
        self.assertTrue(row["mini_leaderboards"]["by_family"])
        self.assertEqual(row["failure_category_distribution"], {"collision": 3, "drop": 1, "timeout": 2})


if __name__ == "__main__":
    unittest.main()
