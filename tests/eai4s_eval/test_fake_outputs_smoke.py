from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from fake_outputs import EVAL_GROUP_ID, make_fake_eval_outputs

from eai4s_eval.server.app import (
    _aggregate_leaderboard_rows,
    _apply_eval_task_defaults,
    _query_runs,
    _scan_runs,
    _scan_suite_runs,
    create_app,
)


TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "eai4s_eval" / "server" / "templates" / "index.html"


class FakeOutputsSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.outputs_root = Path(self._temporary_directory.name) / "eval_runs"
        self.fake = make_fake_eval_outputs(self.outputs_root)

    def test_scan_runs_reads_fake_outputs(self) -> None:
        runs = _scan_runs(self.outputs_root.resolve())
        self.assertEqual(len(runs), 3)
        self.assertEqual({run["eval_group_id"] for run in runs}, {EVAL_GROUP_ID})
        self.assertEqual({run["split"] for run in runs}, {"sim_dev", "sim_clean", "sim_randomized"})
        self.assertTrue(all(run["task_summary"] for run in runs))
        self.assertTrue(all(run["family_summary"] for run in runs))
        suite_rows = _scan_suite_runs(self.outputs_root.resolve())
        self.assertEqual(len(suite_rows), 1)
        self.assertEqual(suite_rows[0]["row_type"], "suite")
        self.assertEqual(suite_rows[0]["evaluated_task_count"], 1)
        self.assertEqual(suite_rows[0]["suite_task_count"], 6)

    def test_leaderboard_aggregates_fake_three_split_group(self) -> None:
        rows = _aggregate_leaderboard_rows(_scan_runs(self.outputs_root.resolve()))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["eval_group_id"], EVAL_GROUP_ID)
        self.assertEqual(row["available_splits"], ["sim_dev", "sim_clean", "sim_randomized"])
        self.assertEqual(row["num_trials"], 6)
        self.assertEqual(row["num_success"], 3)
        self.assertAlmostEqual(row["task_success_rate"], 0.5)
        self.assertEqual(row["failure_category_distribution"], {"collision": 1, "drop": 1, "timeout": 1})
        self.assertIn("pick_place", row["by_family"])
        self.assertTrue(row["mini_leaderboards"]["by_task"])
        self.assertTrue(row["mini_leaderboards"]["by_family"])

    def test_flask_api_serves_fake_outputs(self) -> None:
        try:
            app = create_app(outputs_root=self.outputs_root.resolve())
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
            return

        client = app.test_client()
        health = client.get("/api/health")
        self.assertEqual(health.status_code, 200)

        runs = client.get("/api/runs")
        self.assertEqual(runs.status_code, 200)
        self.assertEqual(len(runs.get_json()), 3)

        runs_page = client.get("/api/runs/query?limit=2&split=sim_clean")
        self.assertEqual(runs_page.status_code, 200)
        page_payload = runs_page.get_json()
        self.assertEqual(len(page_payload["rows"]), 1)
        self.assertEqual(page_payload["pagination"]["total"], 1)

        catalog = client.get("/api/evaluation_catalog")
        self.assertEqual(catalog.status_code, 200)
        suites = {item["suite_id"]: item for item in catalog.get_json()["suites"]}
        self.assertEqual(len(suites["eai4s_benchmark_longhorizon_reasoning_v1"]["tasks"]), 24)

        leaderboard = client.get("/api/leaderboard")
        self.assertEqual(leaderboard.status_code, 200)
        payload = leaderboard.get_json()
        self.assertEqual(payload["total_runs"], 1)
        self.assertEqual(payload["leaderboard_mode"], "suite")
        row = payload["rows"][0]
        self.assertEqual(row["row_type"], "suite")
        self.assertEqual(row["evaluated_task_count"], 1)
        self.assertEqual(row["suite_task_count"], 6)
        self.assertAlmostEqual(row["coverage_rate"], 1 / 6)
        self.assertAlmostEqual(row["covered_task_macro_success_rate"], 0.5)
        self.assertEqual(row["failure_category_distribution"], {"collision": 1, "drop": 1, "timeout": 1})
        task_scores = {item["task_id"]: item for item in row["task_score_table"]}
        self.assertEqual(task_scores["T02_franka_pick_lift_proxy"]["score_status"], "evaluated")
        self.assertIsNone(task_scores["T03_tube_insert_rack"]["task_success_rate"])
        self.assertEqual(task_scores["T03_tube_insert_rack"]["score_status"], "not_run")
        self.assertEqual(task_scores["T04_scan_label_place_area"]["score_status"], "not_implemented")

        scores = client.get("/api/task_scores")
        self.assertEqual(scores.status_code, 200)
        score_payload = scores.get_json()
        self.assertEqual(len(score_payload["tasks"]), 6)
        self.assertTrue(score_payload["families"])

        randomized_run_id = self.fake["run_ids"]["sim_randomized"]
        episodes = client.get(f"/api/runs/{randomized_run_id}/episodes")
        self.assertEqual(episodes.status_code, 200)
        episode_rows = episodes.get_json()
        self.assertEqual(len(episode_rows), 2)
        self.assertEqual({row["failure_category"] for row in episode_rows}, {"drop", "collision"})

    def test_frontend_template_exposes_basic_tasks_and_failure_hooks(self) -> None:
        html = TEMPLATE_PATH.read_text(encoding="utf-8")
        for token in (
            'data-tab="overview"',
            'data-tab="long"',
            'data-tab="reasoning"',
            'data-tab="atomic"',
            'data-tab="runs"',
            'data-tab="datasets"',
            'id="benchmark-overview-body"',
            'id="atomic-body"',
            "renderBenchmarkOverview",
            "renderBenchmarkTracks",
            "renderAtomicLeaderboard",
            "renderRuns",
            "renderDatasetStatistics",
            "runsQueryPath",
            "applyRunFilters",
            'id="benchmark-overview-radars"',
            "drawRadarChart",
            "drawAllRadars",
            "data-radar",
            "capabilities",
            'id="ev-eval-target"',
            "T03 tube insert into rack",
            "taskId: 't03'",
            "syncEvalTargetDefaults",
            "ev-task-suite",
            "expandedAllRunId",
            "renderRunInlineDetail",
            "failure_category",
            "failure_code",
            "Failure Category",
            "Failure Code",
        ):
            self.assertIn(token, html)
        # The original EAI4S leaderboard, basic-task mini leaderboards and the
        # eval queue were intentionally removed from the dashboard.
        for removed_token in (
            'data-tab="leaderboard"',
            'data-tab="basic"',
            'data-tab="jobs"',
            'data-tab="fine"',
            "#tab-fine",
            'id="tab-basic"',
            'id="tab-jobs"',
            'id="mini-family-filter"',
            'id="mini-level-filter"',
            'id="mini-tbody"',
            "aggregateMiniRows",
            "renderMiniLeaderboards",
            "renderJobs",
            "data-leaderboard-detail",
        ):
            self.assertNotIn(removed_token, html)

    def test_query_runs_sorts_filters_and_pages(self) -> None:
        page = _query_runs(_scan_runs(self.outputs_root.resolve()), limit=2, split="sim_clean")
        self.assertEqual(page["schema_version"], "eai4s.runs_page/v1")
        self.assertEqual(page["pagination"]["total"], 1)
        self.assertEqual(page["rows"][0]["split"], "sim_clean")
        self.assertIn("sim_randomized", page["available_filters"]["splits"])

    def test_server_applies_t03_eval_defaults(self) -> None:
        body = _apply_eval_task_defaults({"task_id": "t03"})
        self.assertTrue(body["task_spec"].endswith("T03_tube_insert_rack_draft.yaml"))
        self.assertEqual(body["instruction"], "place the glass tube in the rack")
        self.assertEqual(body["target_object"], "tube")
        self.assertEqual(body["max_steps"], 300)

    def test_server_preserves_explicit_eval_overrides(self) -> None:
        body = _apply_eval_task_defaults(
            {
                "task_id": "t03",
                "instruction": "custom",
                "target_object": "custom_tube",
                "max_steps": 17,
            }
        )
        self.assertEqual(body["instruction"], "custom")
        self.assertEqual(body["target_object"], "custom_tube")
        self.assertEqual(body["max_steps"], 17)


if __name__ == "__main__":
    unittest.main()
