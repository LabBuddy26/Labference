import h5py
import numpy as np

from data_collectors.release_episode_recorder import ReleaseEpisodeRecorder


def test_release_recorder_writes_current_aligned_schema(tmp_path):
    recorder = ReleaseEpisodeRecorder(
        str(tmp_path), camera_names=("camera_1_rgb", "camera_3_rgb"),
        source_hz=120, sample_hz=20, max_episodes=1,
        task_name="test task",
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    joints = np.arange(9, dtype=np.float32)
    tcp = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]
    for step in range(0, 18, 6):
        recorder.record(
            cameras={"front": image, "wrist": image}, joints=joints,
            tcp_pose_xyzw=tcp, command=np.r_[joints[:7], 0.04],
            sim_step=step, subtask="reference_pick",
        )
    recorder.finish(joints=joints, tcp_pose_xyzw=tcp)
    recorder.close()
    with h5py.File(tmp_path / "dataset" / "episode_0000.h5", "r") as episode:
        assert episode["agent_pose"].shape == (3, 8)
        assert episode["actions"].shape == (3, 8)
        assert episode["actions_ee"].shape == (3, 8)
        assert episode["action_gripper"].shape == (3,)
        assert episode["action_gripper"][:].tolist() == [1, 1, 1]
        # xyzw input is stored as current wxyz output.
        np.testing.assert_allclose(episode["actions_ee"][0, 3:7], [0.9, 0.1, 0.2, 0.3])


def test_release_recorder_rejects_unconfigured_camera(tmp_path):
    recorder = ReleaseEpisodeRecorder(
        str(tmp_path), camera_names=("camera_1_rgb", "camera_3_rgb"),
        source_hz=60, sample_hz=10, task_name="test task",
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    try:
        recorder.record(
            cameras={"front": image}, joints=np.zeros(9),
            tcp_pose_xyzw=np.r_[np.zeros(3), [0, 0, 0, 1]],
            sim_step=0, subtask="pick",
        )
    except ValueError as exc:
        assert "camera streams mismatch" in str(exc)
    else:
        raise AssertionError("missing camera stream was accepted")
    recorder.close()
