from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fake_outputs import make_fake_eval_outputs

from eai4s_eval.benchmark_catalog import (
    ATOMIC_ACTIONS,
    benchmark_track_payload,
    build_benchmark_dashboard,
    load_benchmark_catalog,
    score_task_episode,
    validate_benchmark_catalog,
)
from eai4s_eval.server.app import _scan_runs, create_app


class LaboratoryBenchmarkTest(unittest.TestCase):
    def test_catalog_contains_two_tracks_24_tasks_and_11_actions(self) -> None:
        catalog = load_benchmark_catalog()
        validate_benchmark_catalog(catalog)
        self.assertEqual(len(catalog["tracks"]), 2)
        self.assertEqual(len(catalog["tasks"]), 24)
        self.assertEqual(len(catalog["atomic_actions"]), 11)
        self.assertEqual(
            [item["action_id"] for item in catalog["atomic_actions"]],
            [item[0] for item in ATOMIC_ACTIONS],
        )
        self.assertEqual(sum(item["status"] == "implemented" for item in catalog["tasks"]), 24)
        self.assertEqual(next(item for item in catalog["tasks"] if item["task_id"] == "L07")["status"], "implemented")
        self.assertNotIn("fine_manipulation", {item["track_id"] for item in catalog["tracks"]})

    def test_fake_pick_results_populate_atomic_but_not_formal_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "eval_runs"
            make_fake_eval_outputs(root)
            dashboard = build_benchmark_dashboard(_scan_runs(root.resolve()))
        self.assertEqual(len(dashboard["models"]), 1)
        model = dashboard["models"][0]
        self.assertIsNone(model["overall_success_rate"])
        self.assertEqual(model["task_coverage"], {"evaluated": 0, "total": 24, "rate": 0.0})
        by_action = {item["action_id"]: item for item in model["atomic_actions"]}
        self.assertEqual(by_action["grasp"]["attempts"], 6)
        self.assertEqual(by_action["lift"]["attempts"], 6)
        self.assertAlmostEqual(by_action["grasp"]["success_rate"], 0.5)
        self.assertAlmostEqual(model["atomic_success_rate"], 0.5)
        self.assertEqual(model["atomic_coverage"]["evaluated"], 2)

    def test_track_payload_keeps_all_planned_tasks_visible(self) -> None:
        payload = benchmark_track_payload(build_benchmark_dashboard([]), "long_horizon")
        self.assertEqual(payload["track"]["task_count"], 12)
        with self.assertRaises(KeyError):
            benchmark_track_payload(build_benchmark_dashboard([]), "unknown")

    def test_100_point_score_requires_all_metrics_and_hard_failure_is_zero(self) -> None:
        task = next(item for item in load_benchmark_catalog()["tasks"] if item["task_id"] == "L01")
        missing = score_task_episode(task, {"success": True, "safety_valid": True, "quality_metrics": {}})
        self.assertIsNone(missing["score_100"])
        self.assertEqual(missing["status"], "metrics_unavailable")
        metrics = {item["metric"]: 80.0 for item in task["score_components"] if item["metric"] != "success"}
        scored = score_task_episode(task, {"success": True, "safety_valid": True, "quality_metrics": metrics})
        self.assertEqual(scored["status"], "evaluated")
        self.assertAlmostEqual(scored["score_100"], 88.0)
        hard = score_task_episode(task, {"success": True, "hard_failure": True, "quality_metrics": metrics})
        self.assertEqual(hard["score_100"], 0.0)
        self.assertEqual(hard["status"], "hard_failure")

    def test_benchmark_api_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "eval_runs"
            make_fake_eval_outputs(root)
            try:
                app = create_app(outputs_root=root.resolve())
            except ModuleNotFoundError as exc:
                self.skipTest(str(exc))
                return
            client = app.test_client()
            catalog = client.get("/api/benchmark/catalog")
            self.assertEqual(catalog.status_code, 200)
            self.assertEqual(len(catalog.get_json()["tasks"]), 24)
            leaderboard = client.get("/api/benchmark/leaderboard")
            self.assertEqual(leaderboard.status_code, 200)
            model = leaderboard.get_json()["models"][0]
            self.assertEqual(model["atomic_coverage"]["evaluated"], 2)
            self.assertEqual(client.get("/api/benchmark/tracks/fine_manipulation").status_code, 404)
            self.assertEqual(client.get("/api/benchmark/tracks/unknown").status_code, 404)
            self.assertEqual(client.get("/api/benchmark/atomic-actions").status_code, 200)
            self.assertEqual(client.get(f"/api/benchmark/models/{model['model_id']}").status_code, 200)


if __name__ == "__main__":
    unittest.main()
