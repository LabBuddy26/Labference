from __future__ import annotations

import unittest
import sys
import tempfile
from pathlib import Path

from eai4s_eval.cli.run_benchmark_suite_eval import _run_task_process, build_parser
from eai4s_eval.eval.model_identity import resolve_model_name, validate_checkpoint_name
from eai4s_eval.eval.task_suite import default_official_suite_id


class OfficialEvaluationIdentityTest(unittest.TestCase):
    def test_registry_drives_suite_cli_default(self) -> None:
        args = build_parser().parse_args(["--checkpoint", "gr00t_n15_v1"])
        self.assertEqual(args.task_suite, "eai4s_benchmark_longhorizon_reasoning_v1")
        self.assertEqual(args.task_suite, default_official_suite_id())

    def test_suite_checkpoint_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])

    def test_provider_defaults_and_mismatch_guard(self) -> None:
        self.assertEqual(resolve_model_name("openpi"), "OpenPI")
        self.assertEqual(resolve_model_name("gr00t"), "GR00T N1.5")
        with self.assertRaisesRegex(ValueError, "OpenPI model name"):
            resolve_model_name("gr00t", "openpi-checkpoint")
        with self.assertRaisesRegex(ValueError, "GR00T model name"):
            resolve_model_name("openpi", "GR00T N1.5")

    def test_checkpoint_is_stable_and_path_free(self) -> None:
        self.assertEqual(validate_checkpoint_name("gr00t_v1-a1b2"), "gr00t_v1-a1b2")
        with self.assertRaises(ValueError):
            validate_checkpoint_name("/models/gr00t", required=True)
        with self.assertRaises(ValueError):
            validate_checkpoint_name(None, required=True)

    def test_task_failure_persists_combined_output_and_exception_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = _run_task_process(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('stdout marker'); print('stderr marker', file=sys.stderr); raise SystemExit(7)",
                    "--api-key",
                    "secret-token",
                ],
                cwd=root,
                log_path=root / "task_logs" / "R10_sim_clean.log",
                identity={"task_id": "R10", "split": "sim_clean", "run_id": "suite_R10_sim_clean"},
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["returncode"], 7)
            self.assertNotIn("secret-token", result["command"])
            log = (root / "task_logs" / "R10_sim_clean.log").read_text(encoding="utf-8")
            self.assertIn("stdout marker", log)
            self.assertIn("stderr marker", log)
            self.assertNotIn("secret-token", log)

    def test_task_launch_failure_records_traceback_without_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = _run_task_process(
                [str(root / "missing-python")],
                cwd=root,
                log_path=root / "task_logs" / "launch.log",
                identity={"task_id": "R10", "split": "sim_clean", "run_id": "run"},
            )
            self.assertEqual(result["status"], "launch_failed")
            self.assertEqual(result["returncode"], 127)
            self.assertEqual(result["exception_type"], "FileNotFoundError")
            self.assertIn("Traceback", result["traceback"])


if __name__ == "__main__":
    unittest.main()
