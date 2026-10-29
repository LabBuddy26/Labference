from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from eai4s_eval.eval.aggregator import RunAggregator
from eai4s_eval.eval.runner import parse_seeds
from eai4s_eval.eval.success.failure_taxonomy import FAILURE_CATEGORIES
from eai4s_eval.eval.task_spec import apply_success_cli_overrides_to_spec, success_spec_from_task_spec, task_metadata_from_task_spec


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "eai4s_eval" / "configs"


class TaskSpecSplitAggregatorTest(unittest.TestCase):
    def test_robust_task_spec_parse_and_cli_override(self) -> None:
        task_spec = {
            "success_hold_steps": 10,
            "control_hz": 10,
            "quality_metrics": ["volume_error_ml"],
            "success": {
                "logic": "robust_semantic_v0",
                "hold_steps": 10,
                "hold_time_sec": 2.0,
                "goal": {"all": [{"predicate": "lifted", "args": {"height_threshold_m": 0.05}}]},
                "sequence": {"required": True, "events": ["a", "b"]},
                "terminal": {"required": False, "all": []},
                "stability": {"required": False, "all": []},
                "safety": {"hard_fail": [{"predicate": "object_out_of_workspace"}]},
            },
        }
        spec = success_spec_from_task_spec(task_spec)
        self.assertEqual(spec.hold_steps, 10)
        self.assertEqual(spec.hold_time_sec, 2.0)
        self.assertEqual(spec.goal_all[0].predicate, "lifted")
        self.assertTrue(spec.sequence_required)
        self.assertEqual(spec.sequence_events, ["a", "b"])
        self.assertEqual(spec.quality_metrics, ["volume_error_ml"])

        class Args:
            success_hold_steps = 3
            success_height_threshold = 0.08
            success_distance_threshold = None
            success_grip_threshold = None
            success_require_stable = None
            success_require_gripper_closed = None

        overridden = apply_success_cli_overrides_to_spec(spec, Args())
        self.assertEqual(overridden.hold_steps, 3)
        self.assertEqual(overridden.goal_all[0].args["height_threshold_m"], 0.08)

    def test_hold_time_sec_converts_to_steps_when_hold_steps_absent(self) -> None:
        task_spec = {
            "control_hz": 20,
            "success": {
                "logic": "robust_semantic_v0",
                "hold_time_sec": 0.25,
                "goal": {"all": [{"predicate": "lifted"}]},
            },
        }
        spec = success_spec_from_task_spec(task_spec)
        self.assertEqual(spec.hold_steps, 5)
        self.assertEqual(spec.hold_time_sec, 0.25)

    def test_predicate_frequency_is_independent_from_policy_frequency(self) -> None:
        task_spec = {
            "control_hz": 10,
            "predicate_hz": 60,
            "success": {
                "logic": "robust_semantic_v0",
                "hold_time_sec": 1.0,
                "goal": {"all": [{"predicate": "lifted"}]},
            },
        }
        spec = success_spec_from_task_spec(task_spec)
        self.assertEqual(spec.hold_steps, 60)

    def test_task_metadata_defaults_and_configured_fields(self) -> None:
        default_meta = task_metadata_from_task_spec({})
        self.assertEqual(default_meta["task_family"], "basic")
        self.assertEqual(default_meta["task_level"], "base")
        self.assertEqual(default_meta["leaderboard_group"], "basic")

        configured = task_metadata_from_task_spec(
            {
                "task_family": "insertion",
                "task_level": "base",
                "task_tags": ["tube", "rack"],
                "leaderboard_group": "basic_insertion",
                "failure_modes": [{"code": "wrong_target_reached", "category": "wrong_place"}],
            }
        )
        self.assertEqual(configured["task_family"], "insertion")
        self.assertEqual(configured["task_tags"], ["tube", "rack"])
        self.assertEqual(configured["leaderboard_group"], "basic_insertion")
        self.assertEqual(configured["failure_modes"][0]["code"], "wrong_target_reached")

    def test_split_manifests_auto_resolve(self) -> None:
        for split in ("sim_dev", "sim_clean", "sim_randomized"):
            seeds = parse_seeds(None, None, split)
            self.assertGreaterEqual(len(seeds), 50)
            self.assertTrue(all(isinstance(seed, int) for seed in seeds))

    def test_seed_manifest_files_have_valid_unique_seeds(self) -> None:
        for split in ("sim_dev", "sim_clean", "sim_randomized"):
            path = CONFIG_ROOT / f"seed_manifest_{split}.json"
            self.assertTrue(path.exists(), str(path))
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            seeds: list[int] = []
            for task in data.get("tasks", []):
                for item in task.get("seeds", []):
                    if item.get("valid_scene", True):
                        seeds.append(int(item["seed"]))
            self.assertGreaterEqual(len(seeds), 50)
            self.assertEqual(len(seeds), len(set(seeds)))

    def test_task_spec_yaml_files_parse_and_use_known_failure_categories(self) -> None:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
            return

        task_dir = CONFIG_ROOT / "tasks"
        yaml_paths = sorted(task_dir.glob("*.yaml"))
        self.assertTrue(yaml_paths)
        for path in yaml_paths:
            task_spec = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertIsInstance(task_spec, dict, path.name)
            spec = success_spec_from_task_spec(task_spec)
            metadata = task_metadata_from_task_spec(task_spec)
            self.assertTrue(spec.logic)
            self.assertTrue(metadata["task_family"])
            self.assertTrue(metadata["leaderboard_group"])
            for mode in metadata["failure_modes"]:
                category = mode.get("category")
                if category is not None:
                    self.assertIn(category, FAILURE_CATEGORIES, f"{path.name}: {mode}")

    def test_aggregator_groups_by_success_logic(self) -> None:
        rows = [
            {
                "run_id": "run",
                "episode_id": "ep1",
                "task_id": "T",
                "task_name": "Task",
                "task_family": "pick_place",
                "task_level": "base",
                "task_tags": ["pick"],
                "leaderboard_group": "basic_pick_place",
                "split": "sim_dev",
                "success_logic": "robust_semantic_v0",
                "valid_scene": True,
                "success": True,
                "stage_completion": 1.0,
                "termination_reason": "success",
                "failure_type": None,
                "failure_category": "none",
                "failure_code": "none",
                "failure_source": "none",
                "failure_severity": "none",
                "goal_pass": True,
                "terminal_pass": True,
                "stability_pass": True,
                "safety_valid": True,
                "sequence_pass": True,
                "quality_metrics": {"volume_error_ml": 0.1, "spill_ml": 0.0},
                "predicate_values": {"lifted": {"passed": True}},
            },
            {
                "run_id": "run",
                "episode_id": "ep2",
                "task_id": "T",
                "task_name": "Task",
                "task_family": "pick_place",
                "task_level": "base",
                "task_tags": ["pick"],
                "leaderboard_group": "basic_pick_place",
                "split": "sim_dev",
                "success_logic": "alternate_logic",
                "valid_scene": True,
                "success": False,
                "stage_completion": 0.0,
                "termination_reason": "timeout",
                "failure_type": "timeout",
                "failure_category": "timeout",
                "failure_code": "timeout",
                "failure_source": "runner",
                "failure_severity": "soft",
                "goal_pass": False,
                "terminal_pass": True,
                "stability_pass": True,
                "safety_valid": True,
                "sequence_pass": False,
                "quality_metrics": {"volume_error_ml": 0.3, "spill_ml": 0.2},
                "predicate_values": {"lifted": {"passed": False}},
            },
        ]
        task_spec = {
            "task_id": "T",
            "task_name": "Task",
            "task_family": "pick_place",
            "task_level": "base",
            "task_tags": ["pick"],
            "leaderboard_group": "basic_pick_place",
            "success": {"logic": "robust_semantic_v0"},
            "failure_modes": [{"code": "timeout", "category": "timeout"}],
        }
        manifest = {"run_id": "run", "model_name": "model", "benchmark_version": "test", "split": "sim_dev"}
        with tempfile.TemporaryDirectory() as tmp:
            result = RunAggregator(tmp).write_all(episode_rows=rows, run_manifest=manifest, task_spec=task_spec)
            output_dir = Path(tmp)
            for name in (
                "episode_metrics.jsonl",
                "task_summary.json",
                "family_summary.json",
                "summary.json",
                "evaluation_summary.log",
                "leaderboard_row.json",
                "static_api/episodes.json",
                "static_api/model_detail.json",
                "static_api/family_summary.json",
            ):
                self.assertTrue((output_dir / name).exists(), name)
            self.assertEqual(result["summary"]["task_summary_count"], 2)
            self.assertEqual(result["summary"]["family_summary_count"], 2)
            self.assertEqual(result["leaderboard_row"]["overall"]["sim_dev_macro_success_rate"], 0.5)
            self.assertEqual(result["summary"]["sequence_pass_rate"], 0.5)
            self.assertEqual(result["summary"]["failure_category_distribution"], {"timeout": 1})
            self.assertEqual(result["leaderboard_row"]["by_family"]["pick_place"]["failure_category_distribution"], {"timeout": 1})
            self.assertTrue(result["leaderboard_row"]["mini_leaderboards"]["by_task"])
            self.assertTrue(result["leaderboard_row"]["mini_leaderboards"]["by_family"])
            self.assertAlmostEqual(result["summary"]["quality_metric_means"]["volume_error_ml"], 0.2)
            self.assertAlmostEqual(result["summary"]["quality_metric_means"]["spill_ml"], 0.1)
            summary_text = (output_dir / "evaluation_summary.log").read_text(encoding="utf-8")
            self.assertIn("Overall success rate: 1/2 (50.00%)", summary_text)
            self.assertIn("T (Task): 1/2 (50.00%)", summary_text)


if __name__ == "__main__":
    unittest.main()
