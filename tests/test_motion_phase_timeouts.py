from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from utils.rotation_utils import (
    isaac_quaternion_from_euler_deg,
)


class _Action:
    def __init__(self, joint_positions=None):
        self.joint_positions = joint_positions


class _IsaacBaseController:
    def __init__(self, name=None):
        self.name = name

    def reset(self):
        pass


class _Cspace:
    def __init__(self):
        self.targets = []
        self.reset_calls = 0

    def forward(self, *, target_end_effector_position, target_end_effector_orientation):
        self.targets.append(np.asarray(target_end_effector_position).copy())
        return _Action([0.0] * 9)

    def reset(self):
        self.reset_calls += 1


def _load_atomic_module(filename: str, module_name: str):
    replacements = {
        "isaacsim": types.ModuleType("isaacsim"),
        "isaacsim.core": types.ModuleType("isaacsim.core"),
        "isaacsim.core.api": types.ModuleType("isaacsim.core.api"),
        "isaacsim.core.api.controllers": types.ModuleType("isaacsim.core.api.controllers"),
        "isaacsim.core.utils": types.ModuleType("isaacsim.core.utils"),
        "isaacsim.core.utils.stage": types.ModuleType("isaacsim.core.utils.stage"),
        "isaacsim.core.utils.types": types.ModuleType("isaacsim.core.utils.types"),
        "isaacsim.core.utils.rotations": types.ModuleType("isaacsim.core.utils.rotations"),
        "isaacsim.robot": types.ModuleType("isaacsim.robot"),
        "isaacsim.robot.manipulators": types.ModuleType("isaacsim.robot.manipulators"),
        "isaacsim.robot.manipulators.grippers": types.ModuleType("isaacsim.robot.manipulators.grippers"),
        "isaacsim.robot.manipulators.grippers.gripper": types.ModuleType(
            "isaacsim.robot.manipulators.grippers.gripper"
        ),
    }
    replacements["isaacsim.core.api.controllers"].BaseController = _IsaacBaseController
    replacements["isaacsim.core.utils.stage"].get_stage_units = lambda: 1.0
    replacements["isaacsim.core.utils.stage"].get_current_stage = lambda: None
    replacements["isaacsim.core.utils.types"].ArticulationAction = _Action
    replacements["isaacsim.core.utils.rotations"].euler_angles_to_quat = lambda value: value
    replacements["isaacsim.robot.manipulators.grippers.gripper"].Gripper = object

    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        path = Path(__file__).resolve().parents[1] / "controllers" / "atomic_actions" / filename
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


PickRController = _load_atomic_module("pickr_controller.py", "_test_pickr_controller").PickRController
PickController = _load_atomic_module("pick_controller.py", "_test_pick_controller").PickController
PlaceController = _load_atomic_module("place_controller.py", "_test_place_controller").PlaceController
PressZController = _load_atomic_module("pressz_controller.py", "_test_pressz_controller").PressZController
_tube_insert_module = _load_atomic_module(
    "tube_insert_controller.py",
    "_test_tube_insert_controller",
)
TubeInsertController = _tube_insert_module.TubeInsertController
TubeInsertPhase = _tube_insert_module.TubeInsertPhase


class _Gripper:
    def forward(self, action):
        return _Action([0.04, 0.04])


class MotionPhaseTimeoutTest(unittest.TestCase):
    def test_euler_conversion_uses_isaac_wxyz_order(self):
        np.testing.assert_allclose(
            isaac_quaternion_from_euler_deg((0.0, 180.0, 0.0)),
            np.array([0.0, 0.0, 1.0, 0.0]),
            atol=1e-7,
        )

    def test_pick_motion_timeout_fails_instead_of_advancing(self):
        controller = PickRController(
            name="pick",
            cspace_controller=_Cspace(),
            events_dt=[1.0] * 7,
        )
        joints = np.zeros(9)
        controller.forward(
            np.array([0.3, 0.0, 0.7]), joints, "weight_10g", np.array([0.04] * 3),
            None, np.array([0.0, 0.0, 0.0]),
        )
        controller.forward(
            np.array([0.3, 0.0, 0.7]), joints, "weight_10g", np.array([0.04] * 3),
            None, np.array([0.0, 0.0, 0.0]),
        )
        self.assertTrue(controller.is_done())
        self.assertTrue(controller.has_failed())
        self.assertIn("event 0 timed out", controller.last_failure_reason)
        self.assertIn("xyz_error", controller.last_failure_reason)

    def test_strict_pick_controller_timeout_is_terminal(self):
        controller = PickController(
            name="strict-pick",
            cspace_controller=_Cspace(),
            events_dt=[1.0] * 7,
            require_pose_reached=True,
            max_motion_phase_steps=1,
        )
        joints = np.zeros(9)
        position = np.array([0.3, 0.0, 0.7])
        size = np.array([0.03, 0.03, 0.05])
        controller.forward(position.copy(), joints, "beaker", size, None, np.zeros(3))
        controller.forward(position.copy(), joints, "beaker", size, None, np.zeros(3))
        self.assertTrue(controller.is_done())
        self.assertTrue(controller.has_failed())
        self.assertIn("pick motion event 0 timed out", controller.last_failure_reason)
        self.assertIn("xyz_error", controller.last_failure_reason)

    def test_pick_phase_only_reset_preserves_shared_cspace(self):
        cspace = _Cspace()
        controller = PickController(name="pick", cspace_controller=cspace)
        controller._event = 5
        controller._failed = True
        controller.reset_phase_state(initial_event=1)
        self.assertEqual(cspace.reset_calls, 0)
        self.assertEqual(controller._event, 1)
        self.assertFalse(controller.has_failed())

    def test_pick_event_zero_can_reuse_previous_motion_target(self):
        cspace = _Cspace()
        controller = PickController(name="pick", cspace_controller=cspace)
        target = np.array([0.15, 0.26, 1.08])
        controller.reset_phase_state(initial_pregrasp_target=target)
        controller.forward(
            np.array([0.20, 0.30, 0.80]), np.zeros(9), "beaker",
            np.array([0.05, 0.05, 0.10]), None, np.zeros(3),
        )
        controller.forward(
            np.array([0.20, 0.30, 0.80]), np.zeros(9), "beaker",
            np.array([0.05, 0.05, 0.10]), None, np.zeros(3),
        )
        np.testing.assert_allclose(cspace.targets[-1], target)

    def test_pick_reached_target_advances_without_timeout(self):
        controller = PickRController(
            name="pick",
            cspace_controller=_Cspace(),
            events_dt=[1.0] * 7,
        )
        joints = np.zeros(9)
        position = np.array([0.3, 0.0, 0.7])
        size = np.array([0.04, 0.04, 0.04])
        controller.forward(position, joints, "weight_10g", size, None, position)
        event_zero_target = np.array([0.2, 0.0, 0.86])
        controller.forward(position, joints, "weight_10g", size, None, event_zero_target)
        self.assertFalse(controller.has_failed())
        self.assertEqual(controller._event, 1)

    def test_pick_explicit_z_offset_overrides_object_default(self):
        cspace = _Cspace()
        controller = PickRController(
            name="pick",
            cspace_controller=cspace,
            events_dt=[1.0] * 7,
        )
        joints = np.zeros(9)
        position = np.array([0.3, 0.0, 0.7])
        size = np.array([0.045, 0.045, 0.075])
        controller.forward(position, joints, "weight_50g", size, None, np.zeros(3))
        controller._event = 2
        controller.forward(
            position,
            joints,
            "weight_50g",
            size,
            None,
            np.zeros(3),
            pick_z_offset=0.010,
        )
        np.testing.assert_allclose(cspace.targets[-1], np.array([0.3, 0.0, 0.71]))

    def test_pickr_configurable_grasp_tolerance_advances_event_two(self):
        controller = PickRController(
            name="pick",
            cspace_controller=_Cspace(),
            events_dt=[1.0] * 7,
            grasp_position_threshold=0.012,
        )
        controller._start = False
        controller._event = 2
        target = np.array([0.18, 0.02, 0.79])
        actual = target + np.array([0.008, 0.010, 0.006])
        controller.forward(
            target.copy(), np.zeros(9), "beaker_05", np.array([0.08] * 3),
            None, actual,
            pick_z_offset=0.0,
        )
        self.assertEqual(controller._event, 3)
        self.assertFalse(controller.has_failed())

    def test_pickr_z_delta_is_relative_to_default_object_offset(self):
        cspace = _Cspace()
        controller = PickRController(name="pick", cspace_controller=cspace)
        controller._start = False
        controller._event = 2
        position = np.array([0.18, 0.02, 0.75])
        size = np.array([0.08, 0.08, 0.10])
        controller.forward(
            position, np.zeros(9), "beaker_05", size, None, np.zeros(3),
            pick_z_offset_delta=-0.01,
        )
        # Default fallback is 0.4 * object height; lower it by 1 cm.
        np.testing.assert_allclose(cspace.targets[-1][2], 0.78)

    def test_pick_second_approach_uses_configured_x_offset(self):
        cspace = _Cspace()
        controller = PickController(
            name="pick",
            cspace_controller=cspace,
            events_dt=[1.0] * 7,
        )
        controller._event = 1
        controller._start = False
        controller.forward(
            picking_position=np.array([0.20, 0.25, 0.80]),
            current_joint_positions=np.zeros(9),
            object_name="beaker",
            object_size=np.array([0.05, 0.05, 0.10]),
            gripper_control=None,
            gripper_position=np.zeros(3),
            pre_offset_x=0.03,
        )
        np.testing.assert_allclose(cspace.targets[-1][0], 0.17)

    def test_place_motion_timeout_reports_target_and_actual(self):
        controller = PlaceController(
            name="place",
            cspace_controller=_Cspace(),
            gripper=None,
            events_dt=[1.0] * 6,
        )
        joints = np.zeros(9)
        target = np.array([0.1, 0.2, 0.7])
        controller.forward(target, joints, None, gripper_position=np.zeros(3))
        controller.forward(target, joints, None, gripper_position=np.zeros(3))
        self.assertTrue(controller.is_done())
        self.assertTrue(controller.has_failed())
        self.assertIn("place motion event 0 timed out", controller.last_failure_reason)
        self.assertIn("target=", controller.last_failure_reason)
        self.assertIn("actual=", controller.last_failure_reason)

    def test_press_motion_timeout_reports_target_and_actual(self):
        controller = PressZController(
            name="press",
            cspace_controller=_Cspace(),
            events_dt=[1.0] * 4,
        )
        joints = np.zeros(9)
        target = np.array([0.1, 0.2, 0.7])
        controller.forward(target, joints, None, gripper_position=np.zeros(3))
        controller.forward(target, joints, None, gripper_position=np.zeros(3))
        self.assertTrue(controller.is_done())
        self.assertTrue(controller.has_failed())
        self.assertIn("press motion event 0 timed out", controller.last_failure_reason)
        self.assertIn("target=", controller.last_failure_reason)

    def test_tube_approach_timeout_reports_socket_and_xyz_error(self):
        cfg = SimpleNamespace(
            approach_height=0.12,
            alignment_height=0.035,
            xy_alignment_tolerance=0.0025,
            insertion_tracking_tolerance=0.004,
            insertion_step=0.00035,
            pre_release_hold_frames=30,
            post_release_settle_frames=45,
            retreat_height=0.12,
            phase_timeout_steps={"approach": 2},
        )
        controller = TubeInsertController(
            name="tube-insert",
            cspace_controller=_Cspace(),
            gripper=_Gripper(),
            insertion_cfg=cfg,
        )
        kwargs = dict(
            socket_position=np.array([0.28, -0.32, 0.837]),
            inserted_tube_center=np.array([0.28, -0.32, 0.852]),
            insertion_axis=np.array([0.0, 0.0, 1.0]),
            current_joint_positions=np.zeros(9),
            gripper_position=np.zeros(3),
            object_position=np.array([0.1, 0.0, 0.8]),
            end_effector_orientation=np.array([0.0, 1.0, 0.0, 0.0]),
        )
        controller.forward(**kwargs)
        controller.forward(**kwargs)
        self.assertTrue(controller.is_done())
        self.assertTrue(controller.has_failed())
        self.assertIn("phase approach timed out", controller.last_failure_reason)
        self.assertIn("socket=", controller.last_failure_reason)
        self.assertIn("tube_center=", controller.last_failure_reason)
        self.assertIn("rmp_target=", controller.last_failure_reason)
        self.assertIn("actual_end_effector=", controller.last_failure_reason)
        self.assertIn("xyz_error=", controller.last_failure_reason)

    def test_tube_drop_is_reported_before_alignment_timeout(self):
        cfg = SimpleNamespace(
            approach_height=0.12,
            alignment_height=0.035,
            xy_alignment_tolerance=0.0025,
            fine_alignment_z_tolerance=0.008,
            insertion_tracking_tolerance=0.004,
            insertion_step=0.00035,
            pre_release_hold_frames=30,
            post_release_settle_frames=45,
            retreat_height=0.12,
            grasp_tracking_tolerance=0.025,
            phase_timeout_steps={"approach": 100},
        )
        controller = TubeInsertController(
            name="tube-insert",
            cspace_controller=_Cspace(),
            gripper=_Gripper(),
            insertion_cfg=cfg,
        )
        kwargs = dict(
            socket_position=np.array([0.28, -0.32, 0.837]),
            inserted_tube_center=np.array([0.28, -0.32, 0.852]),
            insertion_axis=np.array([0.0, 0.0, 1.0]),
            current_joint_positions=np.zeros(9),
            gripper_position=np.zeros(3),
            object_position=np.array([0.1, 0.0, 0.8]),
            end_effector_orientation=np.array([0.0, 1.0, 0.0, 0.0]),
        )
        controller.forward(**kwargs)
        controller.forward(**{**kwargs, "object_position": np.array([0.2, 0.0, 0.65])})
        self.assertTrue(controller.has_failed())
        self.assertIn("tube lost from gripper", controller.last_failure_reason)
        self.assertIn("expected_tube_center=", controller.last_failure_reason)

    def test_tube_insert_progress_separates_lateral_and_axial_tolerance(self):
        cfg = SimpleNamespace(
            insertion_tracking_tolerance=0.004,
            insertion_lateral_tolerance=0.003,
            insertion_axial_tolerance=0.007,
        )
        controller = TubeInsertController(
            name="tube-insert",
            cspace_controller=_Cspace(),
            gripper=_Gripper(),
            insertion_cfg=cfg,
        )
        controller._insertion_axis = np.array([0.0, 0.0, 1.0])
        actual = np.zeros(3)
        # Matches the measured stalled case: 2.1 mm lateral, 6.0 mm axial.
        self.assertTrue(
            controller._insertion_target_reached(
                np.array([0.001753, 0.001179, 0.005979]),
                actual,
            )
        )
        self.assertFalse(
            controller._insertion_target_reached(np.array([0.0031, 0.0, 0.001]), actual)
        )
        self.assertFalse(
            controller._insertion_target_reached(np.array([0.001, 0.0, 0.0071]), actual)
        )

    def test_tube_insert_object_gate_rejects_socket_edge_displacement(self):
        cfg = SimpleNamespace(
            object_lateral_tolerance=0.0015,
            object_axial_tolerance=0.006,
        )
        controller = TubeInsertController(
            name="tube-insert",
            cspace_controller=_Cspace(),
            gripper=_Gripper(),
            insertion_cfg=cfg,
        )
        controller._insertion_axis = np.array([0.0, 0.0, 1.0])
        target = np.array([0.28, -0.32, 0.90])
        self.assertTrue(
            controller._object_target_reached(
                target, target + np.array([0.0010, 0.0, 0.0050])
            )
        )
        self.assertFalse(
            controller._object_target_reached(
                target, target + np.array([0.0016, 0.0, 0.0])
            )
        )

    def test_fine_alignment_feedback_cancels_measured_lateral_residual(self):
        cfg = SimpleNamespace(fine_alignment_max_correction=0.004)
        controller = TubeInsertController(
            name="tube-insert",
            cspace_controller=_Cspace(),
            gripper=_Gripper(),
            insertion_cfg=cfg,
        )
        controller._insertion_axis = np.array([0.0, 0.0, 1.0])
        target = np.array([0.281554, -0.318786, 0.931970])
        measured = np.array([0.279958, -0.320775, 0.925607])
        correction = controller._lateral_object_feedback(target, measured)
        np.testing.assert_allclose(correction, [0.001596, 0.001989, 0.0])

        # A large displacement is bounded so a bad measurement cannot create
        # an aggressive lateral command.
        correction = controller._lateral_object_feedback(
            target, measured - np.array([0.020, 0.0, 0.0])
        )
        self.assertAlmostEqual(float(np.linalg.norm(correction)), 0.004)

    def test_fine_alignment_bias_is_retained_and_bounded(self):
        cfg = SimpleNamespace(
            fine_alignment_max_correction=0.004,
            fine_alignment_feedback_gain=0.2,
        )
        controller = TubeInsertController(
            name="tube-insert",
            cspace_controller=_Cspace(),
            gripper=_Gripper(),
            insertion_cfg=cfg,
        )
        controller._insertion_axis = np.array([0.0, 0.0, 1.0])
        target = np.array([0.281554, -0.318786, 0.931970])
        measured = np.array([0.279958, -0.320775, 0.925607])
        first_bias = controller._update_lateral_command_bias(target, measured)
        self.assertGreater(float(np.linalg.norm(first_bias)), 0.0)

        for _ in range(100):
            retained_bias = controller._update_lateral_command_bias(target, measured)
        self.assertAlmostEqual(float(np.linalg.norm(retained_bias)), 0.004)
        np.testing.assert_allclose(controller._lateral_command_bias, retained_bias)

    def test_guarded_insert_lifts_before_horizontal_approach(self):
        cfg = SimpleNamespace(
            lift_before_approach=True,
            approach_height=0.12,
            alignment_height=0.08,
            xy_alignment_tolerance=0.003,
            insertion_tracking_tolerance=0.004,
            phase_timeout_steps={"lift_clearance": 100},
        )
        cspace = _Cspace()
        controller = TubeInsertController(
            name="tube-insert",
            cspace_controller=cspace,
            gripper=_Gripper(),
            insertion_cfg=cfg,
        )
        start_object = np.array([0.18, 0.15, 0.80])
        start_hand = np.array([0.18, 0.15, 0.87])
        kwargs = dict(
            socket_position=np.array([0.30, -0.15, 0.74]),
            inserted_tube_center=np.array([0.30, -0.15, 0.78]),
            insertion_axis=np.array([0.0, 0.0, 1.0]),
            current_joint_positions=np.zeros(9),
            gripper_position=start_hand,
            object_position=start_object,
            end_effector_orientation=np.array([0.0, 1.0, 0.0, 0.0]),
        )
        controller.forward(**kwargs)
        # The first target retains source XY; it must not move diagonally to
        # the rack while establishing clearance.
        np.testing.assert_allclose(cspace.targets[-1][:2], start_hand[:2])
        self.assertEqual(controller._phase, TubeInsertPhase.LIFT_CLEARANCE)

    def test_weight_gripper_targets_close_inside_scaled_widths(self):
        controller = PickRController(
            name="weight-pick",
            cspace_controller=_Cspace(),
        )
        self.assertEqual(controller.get_gripper_distance("weight_20g"), 0.010)
        self.assertEqual(controller.get_gripper_distance("weight_50g"), 0.020)


if __name__ == "__main__":
    unittest.main()
