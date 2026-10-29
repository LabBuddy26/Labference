from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from eai4s_eval.eval.benchmark_runner import BenchmarkTaskEvalRunner
from eai4s_eval.eval.replay_backend import ReplayHDF5Backend
from eai4s_eval.eval.replay_backend import ReplayStep
from eai4s_eval.eval.runner import load_task_spec
from eai4s_eval.policy import JointActionChunk
from eai4s_eval.server.app import create_app


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]


class _FakePolicy:
    def reset(self):
        return None

    def predict(self, state):
        self.last_state = state
        return JointActionChunk(np.zeros((2, 8), dtype=np.float32), 1.25, {"fake": True})

    def close(self):
        return None


class BenchmarkReplayE2ETest(unittest.TestCase):
    def test_replay_hdf5_generates_outputs_and_frontend_api_reads_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "episode.h5"
            frames = 60
            snapshot = {
                "object_pos": [0.0, 0.0, 0.91], "initial_object_pos": [0.0, 0.0, 0.80],
                "hand_pos": [0.0, 0.0, 0.91], "gripper_pos": [0.02, 0.02],
                "object_lin_vel": [0.0, 0.0, 0.0], "object_ang_vel": [0.0, 0.0, 0.0],
                "events": [], "task_state": {},
            }
            strings = h5py.string_dtype(encoding="utf-8")
            with h5py.File(replay, "w") as target:
                target.create_dataset("camera_1_rgb", data=np.zeros((frames, 3, 4, 5), dtype=np.uint8))
                target.create_dataset("camera_3_rgb", data=np.zeros((frames, 3, 4, 5), dtype=np.uint8))
                target.create_dataset("agent_pose", data=np.zeros((frames, 8), dtype=np.float32))
                target.create_dataset("actions", data=np.zeros((frames, 8), dtype=np.float32))
                target.create_dataset("language_instruction", data="Pick up the beaker", dtype=strings)
                target.create_dataset("snapshot_json", data=[json.dumps(snapshot)] * frames, dtype=strings)

            output_root = root / "eval_runs"
            task_path = BENCHMARK_ROOT / "eai4s_eval" / "configs" / "tasks" / "B01_level1_pick.yaml"
            task_spec = load_task_spec(str(task_path), "franka", "", 800, None)
            args = SimpleNamespace(
                checkpoint="fake_ckpt", run_id="replay_pick", eval_group_id=None, output_root=str(output_root),
                split="sim_dev", num_episodes=1, seeds="123", seed_manifest=None, model_name="fake_openpi",
                max_steps=800, action_horizon=2, save_video=False, run_type="single_task", suite_id=None,
                checkpoint_hash="abc", dataset_version="replay-v1",
            )
            backend = ReplayHDF5Backend(replay, policy=_FakePolicy())
            result = BenchmarkTaskEvalRunner(
                args=args, backend=backend, task_spec=task_spec,
                policy_metadata={"profile_id": "fake_joint8", "server_metadata": {"action_dim": 8}},
            ).run()
            backend.close()
            run_dir = output_root / "replay_pick"
            self.assertTrue((run_dir / "run_manifest.json").exists())
            self.assertTrue((run_dir / "episode_metrics.csv").exists())
            self.assertTrue((run_dir / "evaluation_summary.log").exists())
            self.assertEqual(result["summary"]["num_success"], 1)
            row = json.loads((run_dir / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(row["success"])
            self.assertFalse(row["legacy_success"])
            self.assertTrue(row["success_disagreement"])
            self.assertEqual(row["policy_calls"], frames)

            app = create_app(outputs_root=output_root)
            client = app.test_client()
            self.assertEqual(client.get("/api/runs").status_code, 200)
            self.assertEqual(client.get("/api/runs/replay_pick/episodes").get_json()[0]["official_success"], True)

    def test_timeout_is_an_official_failure_not_success(self):
        class TimeoutBackend:
            def __init__(self):
                self.done = False

            def reset_episode(self, *, seed, split):
                self.done = False
                return {"seed": seed, "split": split, "object_position": [0, 0, 0.8]}

            def step(self):
                self.done = True
                return ReplayStep(
                    state={"camera_display": {}},
                    snapshot={
                        "object_pos": [0, 0, 0.8], "initial_object_pos": [0, 0, 0.8],
                        "hand_pos": [1, 1, 1], "gripper_pos": [0.04, 0.04],
                        "object_lin_vel": [0, 0, 0], "object_ang_vel": [0, 0, 0],
                    },
                    legacy_done=True, legacy_success=False, termination_reason="timeout",
                )

            def actual_state(self):
                return {"ee_pos": [0, 0, 0], "ee_quat": [0, 0, 0, 1], "joint_pos": [0] * 8, "gripper_pos": [0.04, 0.04]}

            def enter_terminal_hold(self):
                return None

            def policy_stats(self):
                return {"policy_calls": 0, "latencies_ms": [], "last_action_chunk": []}

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            task_path = BENCHMARK_ROOT / "eai4s_eval" / "configs" / "tasks" / "B01_level1_pick.yaml"
            task_spec = load_task_spec(str(task_path), "franka", "", 800, None)
            args = SimpleNamespace(
                checkpoint="timeout", run_id="timeout_run", eval_group_id=None, output_root=tmp,
                split="sim_dev", num_episodes=1, seeds="999", seed_manifest=None, model_name="fake",
                max_steps=800, action_horizon=2, save_video=False, run_type="single_task", suite_id=None,
                checkpoint_hash=None, dataset_version=None,
            )
            BenchmarkTaskEvalRunner(
                args=args, backend=TimeoutBackend(), task_spec=task_spec,
                policy_metadata={"profile_id": "fake", "server_metadata": {}},
            ).run()
            row = json.loads((Path(tmp) / "timeout_run" / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse(row["success"])
            self.assertEqual(row["failure_category"], "timeout")
            self.assertEqual(row["termination_reason"], "timeout")


if __name__ == "__main__":
    unittest.main()
