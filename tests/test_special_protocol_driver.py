import json
from pathlib import Path

import h5py
import numpy as np

from utils.special_protocol_driver import _publish


def episode(root: Path, success=True, mismatch=False):
    data = root / "dataset"
    (data / "meta").mkdir(parents=True)
    (data / "meta/episode.jsonl").write_text(json.dumps({
        "episode_index": 0, "length": 2,
        "task_properties": {"physical_success": success}}) + "\n")
    with h5py.File(data / "episode_0000.h5", "w") as h5:
        h5["task_properties"] = json.dumps({"physical_success": success})
        for key in ("actions", "actions_ee", "action_gripper", "camera_1_rgb", "camera_3_rgb"):
            h5[key] = np.zeros((1 if mismatch and key == "camera_3_rgb" else 2, 8))


def test_publish_requires_physical_success(tmp_path):
    episode(tmp_path / "input", success=False)
    assert not _publish(tmp_path / "input", tmp_path / "out", 0)
    assert not (tmp_path / "out/dataset").exists()


def test_publish_rejects_misaligned_streams(tmp_path):
    episode(tmp_path / "input", mismatch=True)
    assert not _publish(tmp_path / "input", tmp_path / "out", 0)


def test_publish_preserves_data_and_assigns_unique_index(tmp_path):
    episode(tmp_path / "input")
    assert _publish(tmp_path / "input", tmp_path / "out", 3)
    assert (tmp_path / "out/dataset/episode_0003.h5").is_file()
    row = json.loads((tmp_path / "out/dataset/meta/episode.jsonl").read_text())
    assert row["episode_index"] == 3
    assert row["task_properties"]["physical_success"] is True
