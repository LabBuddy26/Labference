from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from eai4s_eval.server.job_queue import PersistentEvalQueue


class PersistentEvalQueueTest(unittest.TestCase):
    def _wait(self, work: PersistentEvalQueue, job_id: str) -> dict:
        deadline = time.time() + 10
        while time.time() < deadline:
            job = work.get(job_id)
            if job and job.get("status") in {"completed", "failed", "interrupted"}:
                return job
            time.sleep(0.05)
        self.fail(f"job did not finish: {job_id}")

    def test_fifo_worker_streams_logs_and_runs_one_process_at_a_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = PersistentEvalQueue(Path(tmp) / "jobs")
            first = queue.enqueue(
                run_id="run_a", command=[sys.executable, "-c", "import time; print('first', flush=True); time.sleep(.2)"], cwd=tmp
            )
            second = queue.enqueue(
                run_id="run_b", command=[sys.executable, "-c", "print('second', flush=True)"], cwd=tmp
            )
            a = self._wait(queue, first["job_id"])
            b = self._wait(queue, second["job_id"])
            self.assertEqual(a["status"], "completed")
            self.assertEqual(b["status"], "completed")
            self.assertLessEqual(a["started_at"], b["started_at"])
            self.assertIn("first", a["log_tail"])
            self.assertIn("second", b["log_tail"])
            queue.stop()

    def test_restart_marks_running_interrupted_without_repeating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            root.mkdir()
            (root / "job1.json").write_text(
                json.dumps({"job_id": "job1", "run_id": "r", "status": "running", "created_at": "2025-01-01T00:00:00Z"}),
                encoding="utf-8",
            )
            queue = PersistentEvalQueue(root, autostart=False)
            recovered = queue.get("job1")
            self.assertEqual(recovered["status"], "interrupted")
            self.assertIn("not repeated", recovered["error"])


if __name__ == "__main__":
    unittest.main()
