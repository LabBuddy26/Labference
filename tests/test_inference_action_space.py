import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from controllers.inference_engines.base_inference_engine import BaseInferenceEngine
    from controllers.inference_engines.replay_inference_engine import ReplayInferenceEngine
else:
    BaseInferenceEngine = object
    ReplayInferenceEngine = None


class _TrajectoryController:
    def __init__(self):
        self.ee_chunks = []
        self.joint_chunks = []

    def is_trajectory_complete(self):
        return True

    def generate_ee_trajectory(self, chunk):
        self.ee_chunks.append(np.asarray(chunk))

    def generate_trajectory(self, chunk):
        self.joint_chunks.append(np.asarray(chunk))

    def get_next_action(self):
        return "action"

    def reset(self):
        pass


class _InferenceEngine(BaseInferenceEngine):
    def _get_n_obs_steps(self):
        return 1

    def _init_inference_engine(self):
        pass

    def _predict_action(self, obs_dict, language_instruction=""):
        return np.arange(8, dtype=np.float32).reshape(1, 8)


def _cfg(**infer_overrides):
    infer = {
        "obs_names": {},
        "n_obs_steps": 1,
        **infer_overrides,
    }
    return SimpleNamespace(infer=SimpleNamespace(**infer))


def _state():
    return {
        "camera_data": {},
        "joint_positions": np.zeros(9, dtype=np.float32),
    }


@unittest.skipUnless(torch is not None, "torch is required for inference-engine tests")
class InferenceActionSpaceTest(unittest.TestCase):
    def test_default_inference_executes_ee_actions(self):
        trajectory = _TrajectoryController()
        engine = _InferenceEngine(_cfg(), trajectory)
        self.assertEqual(engine.step_inference(_state()), "action")
        self.assertEqual(len(trajectory.ee_chunks), 1)
        self.assertEqual(len(trajectory.joint_chunks), 0)

    def test_joint_action_space_remains_an_explicit_compatibility_mode(self):
        trajectory = _TrajectoryController()
        engine = _InferenceEngine(_cfg(action_space="joint"), trajectory)
        engine.step_inference(_state())
        self.assertEqual(len(trajectory.ee_chunks), 0)
        self.assertEqual(len(trajectory.joint_chunks), 1)

    def test_replay_defaults_to_actions_ee(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            expected = np.arange(16, dtype=np.float32).reshape(2, 8)
            with h5py.File(path, "w") as episode:
                episode.create_dataset("actions", data=np.zeros((2, 8), dtype=np.float32))
                episode.create_dataset("actions_ee", data=expected)
            engine = ReplayInferenceEngine(
                _cfg(replay_data_path=str(path), episode_index=0),
                _TrajectoryController(),
            )
            self.assertEqual(engine.action_dataset, "actions_ee")
            np.testing.assert_array_equal(engine.actions, expected)
            engine.close()


if __name__ == "__main__":
    unittest.main()
