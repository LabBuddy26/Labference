from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from fake_outputs import EVAL_GROUP_ID, make_fake_eval_outputs

from eai4s_eval.cli.run_task_suite_eval import build_parser
from eai4s_eval.eval.task_suite import DEFAULT_SUITE_ID, TaskSuiteAggregator, load_task_catalog
from eai4s_eval.server.app import _scan_suite_runs


class TaskSuiteTest(unittest.TestCase):
    def test_catalog_loads_t02_to_t07_in_order(self) -> None:
        catalog = load_task_catalog(DEFAULT_SUITE_ID)
        task_ids = [task["task_id"] for task in catalog["tasks"]]
        self.assertEqual(
            task_ids,
            [
                "T02_franka_pick_lift_proxy",
                "T03_tube_insert_rack",
                "T04_scan_label_place_area",
                "T05_label_color_sort_vials",
                "T06_move_consumable_box_area",
                "T07_liquid_pour_beaker_to_beaker",
            ],
        )
        statuses = {task["task_id"]: task["status"] for task in catalog["tasks"]}
        self.assertEqual(statuses["T02_franka_pick_lift_proxy"], "implemented")
        self.assertEqual(statuses["T03_tube_insert_rack"], "implemented")

    def test_suite_parser_defaults_to_default_suite(self) -> None:
        args, unknown = build_parser().parse_known_args([])
        self.assertEqual(args.task_suite, DEFAULT_SUITE_ID)
        self.assertFalse(args.suite_only)
        self.assertEqual(unknown, [])

    def test_full_suite_marks_openarm_skipped_and_keeps_twenty_one_eligible(self) -> None:
        catalog = load_task_catalog("eai4s_benchmark_full_v1")
        self.assertEqual(catalog["suite_task_count"], 22)
        skipped = [item for item in catalog["tasks"] if item["status"] == "skipped"]
        self.assertEqual([item["task_id"] for item in skipped], ["level1_pick_openarm"])
        self.assertEqual(skipped[0]["skip_reason"], "openarm_profile_unavailable")
        self.assertEqual(sum(bool(item["eligible"]) for item in catalog["tasks"]), 21)
        ids = {item["task_id"] for item in catalog["tasks"]}
        self.assertIn("level2_L01_sampletransport", ids)
        self.assertIn("level4_L09_DeviceOperation", ids)
        for new_id in (
            "level2_L08_NMRInjection",
            "level2_L07_centrifuge_load",
            "level2_L10_cohort_analysis",
            "level2_L02_sampleorganition",
            "level2_L06_wights",
            "level4_L11_ConditionModule",
            "level4_L04_LiquidMixing",
            "level4_L12_clearreset",
            "level2_R02_wights",
            "level3_R10_rackplace",
        ):
            self.assertIn(new_id, ids)

    def test_suite_aggregator_keeps_missing_tasks_null(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        fake = make_fake_eval_outputs(tmp)
        rows = _scan_suite_runs(tmp)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row["coverage_rate"], 1 / 6)
        self.assertAlmostEqual(row["covered_task_macro_success_rate"], 0.5)
        task_scores = {item["task_id"]: item for item in row["task_score_table"]}
        self.assertEqual(task_scores["T02_franka_pick_lift_proxy"]["score_status"], "evaluated")
        self.assertIsNone(task_scores["T03_tube_insert_rack"]["task_success_rate"])
        self.assertEqual(task_scores["T03_tube_insert_rack"]["score_status"], "not_run")
        self.assertIsNone(task_scores["T04_scan_label_place_area"]["task_success_rate"])
        self.assertEqual(task_scores["T04_scan_label_place_area"]["score_status"], "not_implemented")

        # Re-running the aggregator from existing single-task outputs should
        # produce the same covered macro score.
        out = tmp / "manual_suite"
        result = TaskSuiteAggregator(out).write_all(
            task_catalog=load_task_catalog(DEFAULT_SUITE_ID),
            outputs_root=tmp,
            eval_group_id=EVAL_GROUP_ID,
            model_name="smoke_model",
            checkpoint="eval_20990101_000000",
            robot="franka_panda",
        )
        self.assertEqual(fake["eval_group_id"], EVAL_GROUP_ID)
        self.assertAlmostEqual(result["suite_summary"]["covered_task_macro_success_rate"], 0.5)
        summary_log = out / "evaluation_summary.log"
        self.assertTrue(summary_log.exists())
        summary_text = summary_log.read_text(encoding="utf-8")
        self.assertIn("Overall episode success rate", summary_text)
        self.assertIn("T02_franka_pick_lift_proxy", summary_text)


if __name__ == "__main__":
    unittest.main()
