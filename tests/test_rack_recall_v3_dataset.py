"""Training-data protocol tests for rack-placement-recall/v3."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from eai4s_eval.rack_recall_v3.cli import main
from eai4s_eval.rack_recall_v3.dataset import (
    ACTION_NAMES,
    EpisodeDatasetWriter,
    generate_collection_plan,
    validate_dataset,
    validate_hdf5_episode,
)


ROOT = Path("eai4s_eval/rack_recall_v3")
CONFIGS = (
    (str(ROOT / "configs/easy.json"), "single"),
    (str(ROOT / "configs/hard_same_cap.json"), "same_cap"),
    (str(ROOT / "configs/hard_distinct_cap.json"), "distinct_cap"),
)


def test_collection_plan_is_balanced_and_pairs_appearance_conditions():
    plan = generate_collection_plan(CONFIGS, split="train")
    assert plan["fixed_denominator"] == 144
    assert plan["physics_hz"] == 120
    assert plan["control_hz"] == 20
    assert min(row["seed"] for row in plan["units"]) >= 10_000
    indexed = {(row["condition"], row["collection_slot"], row["initial_yaw_deg"]): row
               for row in plan["units"]}
    for slot in tuple(f"{row}{column}" for row in "ABC" for column in range(1, 5)):
        for yaw in (0.0, 90.0, 180.0, 270.0):
            same = indexed[("same_cap", slot, yaw)]
            distinct = indexed[("distinct_cap", slot, yaw)]
            assert same["layout_seed"] == distinct["layout_seed"]
            assert same["motion_seed"] == distinct["motion_seed"]
            assert same["color_seed"] == distinct["color_seed"]
            assert same["policy_seed"] == distinct["policy_seed"]


def test_training_and_validation_collection_seeds_are_disjoint():
    train = generate_collection_plan(CONFIGS, split="train")
    validation = generate_collection_plan(CONFIGS, split="validation")
    assert {row["seed"] for row in train["units"]}.isdisjoint(
        row["seed"] for row in validation["units"])


def test_collection_plan_cli_writes_portable_144_unit_manifest(tmp_path):
    output = tmp_path / "collection.json"
    main(["plan-collection", "--configs", *(config for config, _ in CONFIGS),
          "--output", str(output)])
    plan = json.loads(output.read_text())
    assert plan["fixed_denominator"] == 144
    assert all(not Path(row["config"]).is_absolute() for row in plan["units"])
    assert all(len(row["contract_hash"]) == 64 for row in plan["units"])


def test_streaming_hdf5_episode_has_synchronized_public_data_and_private_diagnostics(tmp_path):
    path = tmp_path / "episode.h5"
    writer = EpisodeDatasetWriter(
        path, {"instruction": "place, remember, and retrieve", "seed": 10000})
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[100:120, 200:240] = (10, 80, 220)
    for index in range(2):
        writer.add_physics_step(
            step=index, joints=np.zeros(9), command_index=max(index - 1, -1),
            force_axial_n=0.1, force_lateral_n=0.2, phase="pick")
        writer.add_control_frame(
            step=index * 6, front=image, wrist=image, joint_state=np.zeros(9),
            tcp_pose_xyzw=np.asarray([0, 0, 0, 0, 0, 0, 1]),
            action=np.zeros(8), evaluator_state={"actual_placement_slot": "A1"},
            tube_evidence={"target": {"rack_axial_force_n": 0.1}},
            controller_status={"phase": "pick"})
    writer.set_terminal_observation(
        front=image, wrist=image, joint_state=np.zeros(9),
        tcp_pose_xyzw=np.asarray([0, 0, 0, 0, 0, 0, 1]),
        evaluator_state={"phase": "complete"})
    writer.finalize(scores={"valid_success": True, "clean_completion": True},
                    runtime_status="finished", accepted_for_training=True)

    report = validate_hdf5_episode(path)
    assert report["valid"] is True
    assert report["frame_count"] == 2
    assert report["accepted_for_training"] is True
    with h5py.File(path, "r") as episode:
        assert tuple(json.loads(episode.attrs["action_names_json"])) == ACTION_NAMES
        assert episode.attrs["action_units"] == "radians+meter"
        assert "actual_placement_slot" not in episode["observation"]
        private_state = episode["diagnostics/evaluator_state_json"].asstr()[0]
        assert "actual_placement_slot" in private_state
        assert episode["terminal_observation/front_jpeg"].shape == (1,)

    dataset_report = validate_dataset(path)
    assert dataset_report["valid"] is True
    assert dataset_report["episode_count"] == 1
    assert dataset_report["accepted_episode_count"] == 1


def test_dataset_validator_accepts_hdf5_suffix(tmp_path):
    path = tmp_path / "episode.hdf5"
    writer = EpisodeDatasetWriter(path, {"instruction": "test"})
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    writer.add_control_frame(
        step=6, front=image, wrist=image, joint_state=np.zeros(9),
        tcp_pose_xyzw=np.zeros(7), action=np.zeros(8), evaluator_state={},
        tube_evidence={}, controller_status={})
    writer.set_terminal_observation(
        front=image, wrist=image, joint_state=np.zeros(9),
        tcp_pose_xyzw=np.zeros(7), evaluator_state={})
    writer.finalize(scores={}, runtime_status="finished", accepted_for_training=True)

    report = validate_dataset(path)
    assert report["valid"] is True
    assert report["episode_count"] == 1
    assert report["accepted_episode_count"] == 1


def test_incomplete_partial_file_blocks_dataset_validation(tmp_path):
    writer = EpisodeDatasetWriter(
        tmp_path / "episode.h5", {"instruction": "unfinished"})
    writer.abort("interrupted")
    report = validate_dataset(tmp_path)
    assert report["valid"] is False
    assert len(report["partial_files"]) == 1
