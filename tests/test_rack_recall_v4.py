import json
from pathlib import Path

import numpy as np
import pytest

from eai4s_eval.rack_recall_v4.cli import load_config
from eai4s_eval.rack_recall_v4.core import (
    Contract,
    Measurement,
    RackPose,
    State,
    TASK_VERSION,
    advance_state,
    generate_coverage_manifest,
    generate_instance,
)
from eai4s_eval.rack_recall_v4.dataset import EpisodeDatasetWriter, validate_hdf5_episode
from eai4s_eval.rack_recall_v4.online import compare_pairs
from eai4s_eval.rack_recall_v4.scene import REBUILD_REVISION, SCENE_REVISION, SceneHandles


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "eai4s_eval/rack_recall_v4/configs"


@pytest.mark.parametrize(
    "name,mode",
    [
        ("easy.json", "none"),
        ("hard_same_cap.json", "occluded_rebuild"),
        ("hard_distinct_cap.json", "occluded_rebuild"),
        ("hard_same_cap_physical_transport.json", "physical_transport"),
    ],
)
def test_v4_configs_declare_versioned_environment_mode(name, mode):
    source, contract = load_config(CONFIGS / name)
    assert contract.task_version == TASK_VERSION
    assert contract.environment_mode == mode
    assert contract.rebuild_revision == REBUILD_REVISION
    assert source["scene"]["scene_revision"] == SCENE_REVISION
    assert source["scene"]["environment_mode"] == mode


def test_environment_mode_validation_is_explicit():
    with pytest.raises(ValueError, match="environment_mode"):
        Contract(difficulty="easy", environment_mode="occluded_rebuild")
    with pytest.raises(ValueError, match="environment_mode"):
        Contract(difficulty="hard", condition_variant="same_cap", environment_mode="none")


def test_absent_rebuild_measurement_does_not_advance_task_state():
    contract = Contract(
        difficulty="hard", condition_variant="same_cap",
        environment_mode="occluded_rebuild")
    instance = generate_instance(contract, 101)
    state = State.initial(instance)
    state.phase = "transition"
    before = state.evaluator()
    measurement = Measurement(
        step=10,
        rack_pose=instance.rack_initial_pose,
        environment_phase="rebuild_absent",
        occluder_visible=True,
        tubes={},
        rack_motion_complete=False,
        evidence_delivered=False,
        state_valid=False,
        tube_state="absent",
    )
    assert advance_state(contract, instance, state, measurement) == []
    assert state.evaluator() == before


def test_rack_local_pose_round_trip_preserves_offset_and_orientation():
    from scipy.spatial.transform import Rotation

    initial = RackPose((0.196, 0.1637, 0.707), 180.0)
    final = RackPose((0.1545, 0.1776, 0.707), 0.0)
    initial_q = Rotation.from_euler("z", initial.yaw_deg, degrees=True).as_quat()
    final_q = Rotation.from_euler("z", final.yaw_deg, degrees=True).as_quat()
    initial_world = SceneHandles._matrix(
        initial.position_m, (initial_q[3], *initial_q[:3]))
    final_world = SceneHandles._matrix(final.position_m, (final_q[3], *final_q[:3]))
    tube_q = Rotation.from_euler("xyz", (0.2, -0.1, 179.9), degrees=True).as_quat()
    tube_world = SceneHandles._matrix(
        (0.2935, 0.2137, 0.84745), (tube_q[3], *tube_q[:3]))
    local = np.linalg.inv(initial_world) @ tube_world
    rebuilt = final_world @ local
    recovered = np.linalg.inv(final_world) @ rebuilt
    assert np.max(np.abs(recovered - local)) < 1e-12


def test_world_target_adapter_converts_scipy_xyzw_to_lula_wxyz():
    from eai4s_eval.rack_recall_v4.runner import _WorldTargetMotionAdapter

    class FakeMotion:
        def __init__(self):
            self.rmp_flow = self
            self.call = None

        def get_end_effector_pose(self, joints):
            return np.asarray([1.0, 2.0, 3.0]), None

        def forward(self, **kwargs):
            self.call = kwargs
            return "action"

    class FakeFranka:
        def get_joint_positions(self):
            return np.zeros(7)

        def get_gripper_position(self):
            return np.asarray([1.1, 2.0, 3.0])

    motion = FakeMotion()
    adapter = _WorldTargetMotionAdapter(motion, FakeFranka())
    action = adapter.forward(
        [2.0, 2.0, 3.0], target_end_effector_orientation=[0.0, 0.0, 0.0, 1.0])

    assert action == "action"
    np.testing.assert_allclose(
        motion.call["target_end_effector_orientation"], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        motion.call["target_end_effector_position"], [1.9, 2.0, 3.0])


def test_dataset_masks_environment_control_frames(tmp_path):
    path = tmp_path / "episode.h5"
    writer = EpisodeDatasetWriter(path, metadata={"instruction": "test"})
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    writer.add_physics_step(
        step=1, joints=np.zeros(9), command_index=-1,
        force_axial_n=0.0, force_lateral_n=0.0,
        phase="rebuild_absent", environment_state_valid=False,
        policy_control_enabled=False)
    writer.add_control_frame(
        step=1, front=image, wrist=image, joint_state=np.zeros(9),
        tcp_pose_xyzw=np.zeros(7), action=np.zeros(8), evaluator_state={},
        tube_evidence={}, controller_status={}, environment_phase="rebuild_absent",
        environment_state_valid=False, policy_control_enabled=False)
    writer.set_terminal_observation(
        front=image, wrist=image, joint_state=np.zeros(9),
        tcp_pose_xyzw=np.zeros(7), evaluator_state={})
    writer.finalize(scores={}, runtime_status="finished", accepted_for_training=False)
    report = validate_hdf5_episode(path)
    assert report["valid"], report


def test_v4_runtime_contains_rebuild_and_sync_guards():
    runner = (ROOT / "eai4s_eval/rack_recall_v4/runner.py").read_text()
    scene = (ROOT / "eai4s_eval/rack_recall_v4/scene.py").read_text()
    for token in (
        "tube_snapshot_saved", "tubes_deleted", "rack_motion_complete",
        "tubes_rebuilt", "rebuild_validated", "tube_rebuild_pose_error",
        "environment_state_valid", "policy_control_enabled",
    ):
        assert token in runner
    assert "world.reset()" not in scene
    assert 'settings.set_bool("/physics/updateToUsd", True)' in runner
    assert "diagnostic_top" in scene
    assert "GUIDE_HANDOFF_ALIGNMENT_LIMIT_M = .0023" in runner
    assert "TRANSPORT_ALIGNMENT_LIMIT_M = .0010" in runner


def test_hover_precision_mode_latches_and_prevents_controller_boundary_chatter():
    from eai4s_eval.rack_recall_v4.runner import (
        GUIDED_RELEASE_HANDOFF_EEF_TOLERANCE_M,
        HOVER_PRECISION_HOLD_BIAS_M,
        _precision_hover_target_xy,
        _rmpflow_hover_xy_correction,
        _update_hover_precision_mode,
        _use_rmpflow_hover,
    )

    # Keep obstacle-aware RMPflow active through the long transfer. A 9 mm
    # lead compensates for its near-field stop band; on first entry to the
    # 5 mm precision region, hold a latched target compensating for the
    # measured end-effector-to-tube offset. Release still requires the
    # independent 2.3 mm physical alignment threshold.
    assert _use_rmpflow_hover("hover", .024, False)
    assert _use_rmpflow_hover("hover", .007, False)
    assert not _update_hover_precision_mode(False, "hover", .007)
    assert _update_hover_precision_mode(False, "hover", .0045)
    precision = _update_hover_precision_mode(False, "hover", .002)
    assert precision
    assert _use_rmpflow_hover("hover", .030, precision)
    assert _use_rmpflow_hover("hover", .002, precision)
    assert _use_rmpflow_hover("recover", .030, precision)
    assert _update_hover_precision_mode(precision, "hover", .030)
    assert _use_rmpflow_hover("upper_guide", .030, precision)
    assert GUIDED_RELEASE_HANDOFF_EEF_TOLERANCE_M == pytest.approx(.006)
    correction = _rmpflow_hover_xy_correction((.010, 0.0), (0.0, 0.0), .010)
    assert correction == pytest.approx((.019, 0.0))
    precision_correction = _rmpflow_hover_xy_correction(
        (.010, 0.0), (0.008, 0.0), .002, precision_mode=True)
    assert precision_correction == pytest.approx((.002, 0.0))
    precision_target = _precision_hover_target_xy(
        (.010, 0.0), (.008, 0.0), HOVER_PRECISION_HOLD_BIAS_M)
    assert precision_target == pytest.approx((.014, 0.0))
    correction = _rmpflow_hover_xy_correction((.020, 0.0), (0.0, 0.0), .020)
    assert correction == pytest.approx((.020, 0.0))


def test_guided_release_descends_to_upper_guide_before_releasing():
    runner = (ROOT / "eai4s_eval/rack_recall_v4/runner.py").read_text()
    assert 'self._set_insertion_phase("upper_guide", depth)' in runner
    assert '"upper_guide": .060 if guided_release else .012' in runner
    assert 'depth >= .060 and radial <= alignment_radial' in runner
    assert '(self.insertion_phase == "upper_guide" and depth >= -.010)' in runner
    assert 'self.insertion_phase_step > 20 * self.control_hz' in runner
    assert 'if self.insertion_phase == "recover":' in runner
    assert 'xy_correction = np.zeros(2, dtype=float)' in runner
    assert 'self.contract.insertion_retract_m + .015' in runner
    assert 'self.insertion_phase == "recover" and target_error <= .006' in runner
    assert 'depth <= -.020' in runner
    assert 'self.insertion_precision_mode = False' in runner
    assert 'self.insertion_precision_target_xy = None' in runner


def test_coverage_protocol_freezes_environment_implementation():
    contract = Contract(
        difficulty="hard", condition_variant="same_cap",
        environment_mode="occluded_rebuild")
    member = generate_coverage_manifest(contract, (101,))["members"][0]
    assert member["environment_mode"] == "occluded_rebuild"
    assert member["rebuild_revision"] == REBUILD_REVISION


def test_pair_comparison_rejects_environment_mode_mismatch():
    contract = Contract(
        difficulty="hard", condition_variant="same_cap",
        environment_mode="occluded_rebuild")
    plan = [{
        "pair_id": "p1",
        "task_version": contract.task_version,
        "contract_hash": contract.hash,
        "scoring_version": "labference/rack-recall-memory-selection-v4",
        "scoring_track": "behavior",
        "history_protocol": "online_once",
        "oracle_profile": "none",
        "environment_mode": "occluded_rebuild",
        "rebuild_revision": REBUILD_REVISION,
        "members": [{"member": "a"}, {"member": "b"}],
    }]
    base = {
        "pair_id": "p1",
        "task_version": contract.task_version,
        "contract_hash": contract.hash,
        "scoring_version": "labference/rack-recall-memory-selection-v4",
        "scoring_track": "behavior",
        "history_protocol": "online_once",
        "oracle_profile": "none",
        "environment_mode": "occluded_rebuild",
        "rebuild_revision": REBUILD_REVISION,
        "evaluation_valid": True,
        "benchmark_success": True,
    }
    rows = [{**base, "member": "a"},
            {**base, "member": "b", "environment_mode": "physical_transport"}]
    with pytest.raises(ValueError, match="mixes task contracts"):
        compare_pairs(plan, rows)
