from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eai4s_eval.benchmark_catalog import benchmark_track_payload, build_benchmark_dashboard, load_benchmark_catalog
from eai4s_eval.cli.generate_fake_benchmark import MODEL_PROFILES, generate_fake_benchmark
from eai4s_eval.server.app import _scan_runs

TRACK_ORDER = ("long_horizon", "reasoning_in_action")


class FakeBenchmarkDashboardTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.outputs_root = Path(self._temporary_directory.name) / "eval_runs"
        self.catalog = load_benchmark_catalog()
        self.run_ids = generate_fake_benchmark(self.outputs_root, self.catalog, episodes_per_task=3, seed=1)

    def test_generator_writes_one_run_per_model_task(self) -> None:
        self.assertEqual(len(self.run_ids), 3 * len(self.catalog["tasks"]))
        self.assertEqual(len(_scan_runs(self.outputs_root.resolve())), 72)

    def test_generator_is_deterministic(self) -> None:
        second_root = Path(self._temporary_directory.name) / "eval_runs_2"
        second = generate_fake_benchmark(second_root, self.catalog, episodes_per_task=3, seed=1)
        self.assertEqual(second, self.run_ids)
        first_run_dir = self.outputs_root / self.run_ids[0]
        second_run_dir = second_root / second[0]
        self.assertEqual(
            (first_run_dir / "episode_metrics.jsonl").read_text(encoding="utf-8"),
            (second_run_dir / "episode_metrics.jsonl").read_text(encoding="utf-8"),
        )

    def test_dashboard_builds_three_models_over_24_tasks(self) -> None:
        dashboard = build_benchmark_dashboard(_scan_runs(self.outputs_root.resolve()), self.catalog)
        self.assertEqual(dashboard["task_count"], 24)
        self.assertEqual(dashboard["track_count"], 2)
        self.assertEqual(len(dashboard["models"]), len(MODEL_PROFILES))
        model_ids = [model["model_id"] for model in dashboard["models"]]
        self.assertEqual(len(model_ids), len(set(model_ids)))

    def test_dashboard_models_are_ranked_by_success(self) -> None:
        dashboard = build_benchmark_dashboard(_scan_runs(self.outputs_root.resolve()), self.catalog)
        rates = [model["overall_success_rate"] for model in dashboard["models"]]
        self.assertEqual(rates, sorted(rates, reverse=True))
        self.assertEqual(dashboard["models"][0]["model_name"], "openpi_prod")
        self.assertEqual(dashboard["models"][-1]["model_name"], "scripted_baseline")

    def test_each_track_covers_twelve_tasks(self) -> None:
        dashboard = build_benchmark_dashboard(_scan_runs(self.outputs_root.resolve()), self.catalog)
        for model in dashboard["models"]:
            self.assertEqual(len(model["tracks"]), 2)
            for track in model["tracks"]:
                self.assertEqual(track["task_count"], 12)
                self.assertEqual(track["evaluated_task_count"], 12)
                self.assertEqual(track["score_coverage_rate"], 1.0)
                self.assertIsNotNone(track["score_100"])
            self.assertEqual(len(model["tasks"]), 24)

    def test_track_payload_exposes_twelve_task_scores(self) -> None:
        dashboard = build_benchmark_dashboard(_scan_runs(self.outputs_root.resolve()), self.catalog)
        for track_id in TRACK_ORDER:
            payload = benchmark_track_payload(dashboard, track_id)
            self.assertEqual(payload["track"]["track_id"], track_id)
            for model in payload["models"]:
                tasks = model["tasks"]
                self.assertEqual(len(tasks), 12)
                self.assertTrue(all(item["task_id"].startswith(payload["track"]["prefix"]) for item in tasks))
                evaluated = [item for item in tasks if item["score_status"] == "evaluated"]
                self.assertEqual(len(evaluated), 12)
                self.assertTrue(all(isinstance(item["score_100"], (int, float)) for item in evaluated))
                ids = [item["task_id"] for item in tasks]
                self.assertEqual(ids, sorted(ids))

    def test_models_expose_capabilities(self) -> None:
        dashboard = build_benchmark_dashboard(_scan_runs(self.outputs_root.resolve()), self.catalog)
        capability_keys = {"overall_success", "partial_success", "efficiency", "safety", "atomic", "task_coverage", "score_coverage"}
        for model in dashboard["models"]:
            self.assertEqual(set(model["capabilities"].keys()), capability_keys)
            for key, value in model["capabilities"].items():
                self.assertIsNotNone(value)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 100.0)


if __name__ == "__main__":
    unittest.main()
