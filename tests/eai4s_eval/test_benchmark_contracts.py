from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import yaml
from types import SimpleNamespace

from eai4s_eval.eval.benchmark_runner import load_benchmark_seeds
from eai4s_eval.eval.runner import load_task_spec
from eai4s_eval.eval.success import TaskSuccessEvaluator
from eai4s_eval.eval.task_spec import success_spec_from_task_spec
from eai4s_eval.policy import OpenPIJointPolicyAdapter, OpenPIProfile, PolicyClientError, PolicyOutputInvalid


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = BENCHMARK_ROOT / "eai4s_eval"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from evaluation.config import BenchmarkConfigError, chw_to_hwc_uint8, split_interval, validate_camera_contract
from evaluation.runtime import BenchmarkRuntime
from utils.camera_utils import _rgb_hwc


class _FakeOpenPI:
    def __init__(self, action=None, metadata=None):
        self.action = np.zeros((2, 8), dtype=np.float32) if action is None else action
        self.metadata = metadata or {
            "action_dim": 8,
            "observation_keys": ["observation/front_image", "observation/wrist_image", "observation/state", "prompt"],
        }
        self.last_observation = None

    def get_server_metadata(self):
        return self.metadata

    def infer(self, observation):
        self.last_observation = observation
        return {"actions": self.action, "model_latency_ms": 4.5}


class BenchmarkContractTest(unittest.TestCase):
    def test_all_collection_configs_have_valid_inference_cameras(self):
        for path in sorted((BENCHMARK_ROOT / "config").glob("*.yaml")):
            with self.subTest(path=path.name):
                cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
                if "defaults" in cfg and "cameras" not in cfg:
                    # Hydra defaults-only alias (e.g. level1_pick_infer.yaml)
                    # that resolves to a canonical config at compose time.
                    continue
                self.assertTrue(validate_camera_contract(cfg))

    def test_camera_validation_reports_missing_stream(self):
        cfg = {"cameras": [{"name": "camera_1", "image_type": "rgb"}], "infer": {"obs_names": {"camera_3_rgb": "x"}}}
        with self.assertRaisesRegex(BenchmarkConfigError, "camera_3_rgb"):
            validate_camera_contract(cfg)

    def test_chw_to_hwc_is_not_dimension_truncation(self):
        image = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
        converted = chw_to_hwc_uint8(image)
        self.assertEqual(converted.shape, (4, 5, 3))
        np.testing.assert_array_equal(converted[..., 2], image[2])

    def test_camera_rgb_normalizes_batched_and_channel_first_frames(self):
        hwc = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        np.testing.assert_array_equal(_rgb_hwc(hwc), hwc)
        np.testing.assert_array_equal(_rgb_hwc(hwc[None, ...]), hwc)
        np.testing.assert_array_equal(_rgb_hwc(np.moveaxis(hwc, -1, 0)), hwc)
        with self.assertRaisesRegex(ValueError, r"got \(0,\)"):
            _rgb_hwc(np.asarray(()))

    def test_r11_move_success_uses_cartesian_hand_position(self):
        spec = load_task_spec(
            str(PACKAGE_ROOT / "configs" / "tasks" / "B27_level2_R11_plateposition.yaml"),
            "franka", "", 500, None,
        )
        success_spec = success_spec_from_task_spec(spec)
        runtime = self._runtime_shell("move", SimpleNamespace(controller_type="move"))
        snapshot = runtime.build_snapshot({
            "object_position": [0.0, 0.0, 0.8],
            "target_position": [0.1, 0.2, 0.9],
            "object_quaternion": [0.0, 0.0, 0.0, 1.0],
            "joint_positions": [0.0] * 7 + [0.04, 0.04],
            "gripper_position": [0.101, 0.199, 0.901],
        })
        evaluator = TaskSuccessEvaluator(success_spec)
        result = None
        for _ in range(success_spec.hold_steps):
            result = evaluator.update(snapshot)
        assert result is not None
        self.assertTrue(result.task_success)

    def test_r01_scale_success_uses_cartesian_hand_position(self):
        spec = load_task_spec(
            str(PACKAGE_ROOT / "configs" / "tasks" / "B28_level3_R01_scaledigital.yaml"),
            "franka", "", 500, None,
        )
        success_spec = success_spec_from_task_spec(spec)
        runtime = self._runtime_shell("press", SimpleNamespace(controller_type="press"))
        snapshot = runtime.build_snapshot({
            "object_position": [0.1, 0.2, 0.9],
            "target_position": [0.1, 0.2, 0.9],
            "object_quaternion": [0.0, 0.0, 0.0, 1.0],
            "joint_positions": [0.0] * 7 + [0.04, 0.04],
            "gripper_position": [0.101, 0.199, 0.901],
        })
        evaluator = TaskSuccessEvaluator(success_spec)
        result = None
        for _ in range(success_spec.hold_steps):
            result = evaluator.update(snapshot)
        assert result is not None
        self.assertTrue(result.task_success)

    def test_openpi_maps_chw_images_state_prompt_and_validates_8d(self):
        fake = _FakeOpenPI()
        adapter = OpenPIJointPolicyAdapter(host="127.0.0.1", port=8765, profile=OpenPIProfile(), client=fake)
        state = {
            "camera_data": {
                "camera_1_rgb": np.zeros((3, 6, 8), dtype=np.uint8),
                "camera_3_rgb": np.zeros((3, 6, 8), dtype=np.uint8),
            },
            "agent_pose": np.arange(8, dtype=np.float32),
            "language_instruction": "pick the beaker",
        }
        chunk = adapter.predict(state)
        self.assertEqual(chunk.joint_positions.shape, (2, 8))
        self.assertEqual(fake.last_observation["observation/front_image"].shape, (6, 8, 3))
        self.assertEqual(fake.last_observation["prompt"], "pick the beaker")
        np.testing.assert_array_equal(fake.last_observation["observation/state"], np.arange(8, dtype=np.float32))

    def test_openpi_rejects_bad_metadata_shape_and_nonfinite_actions(self):
        with self.assertRaises(PolicyClientError):
            OpenPIJointPolicyAdapter(
                host="127.0.0.1", port=8765, profile=OpenPIProfile(),
                client=_FakeOpenPI(metadata={"action_dim": 7}),
            )
        adapter = OpenPIJointPolicyAdapter(
            host="127.0.0.1", port=8765, profile=OpenPIProfile(),
            client=_FakeOpenPI(action=np.full((1, 8), np.nan, dtype=np.float32)),
        )
        with self.assertRaisesRegex(PolicyOutputInvalid, "NaN"):
            adapter.predict({
                "camera_data": {"camera_1_rgb": np.zeros((3, 2, 2)), "camera_3_rgb": np.zeros((3, 2, 2))},
                "agent_pose": np.zeros(8), "language_instruction": "x",
            })

    def test_split_ranges_and_fixed_seed_manifest_are_reproducible(self):
        clean_low, clean_high = split_interval((0.2, 0.4), "sim_clean")
        self.assertAlmostEqual(clean_low, 0.3)
        self.assertAlmostEqual(clean_high, 0.3)
        low, high = split_interval((0.2, 0.4), "sim_randomized")
        self.assertAlmostEqual(low, 0.175)
        self.assertAlmostEqual(high, 0.425)
        first = load_benchmark_seeds(task_id="level1_pick", split="sim_dev", count=100)
        second = load_benchmark_seeds(task_id="level1_pick", split="sim_dev", count=100)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(
            load_benchmark_seeds(task_id="level2_L01_sampletransport", split="sim_dev", count=2),
            [430000, 430001],
        )
        self.assertEqual(
            load_benchmark_seeds(task_id="level4_L09_DeviceOperation", split="sim_dev", count=2),
            [440000, 440001],
        )
        for task_id, base in (
            ("level2_L08_NMRInjection", 610000),
            ("level2_L07_centrifuge_load", 620000),
            ("level2_L10_cohort_analysis", 630000),
            ("level2_L02_sampleorganition", 640000),
            ("level2_L06_wights", 650000),
            ("level4_L11_ConditionModule", 660000),
            ("level4_L04_LiquidMixing", 670000),
            ("level4_L12_clearreset", 680000),
        ):
            with self.subTest(task=task_id):
                self.assertEqual(
                    load_benchmark_seeds(task_id=task_id, split="sim_dev", count=2),
                    [base, base + 1],
                )
        self.assertEqual(
            load_benchmark_seeds(task_id="level4_L12_clearreset", split="sim_randomized", count=2),
            [780000, 780001],
        )
        # R-series tasks merged from main keep their own seed ranges per split.
        self.assertEqual(
            load_benchmark_seeds(task_id="level2_R02_wights", split="sim_dev", count=2),
            [830000, 830001],
        )
        self.assertEqual(
            load_benchmark_seeds(task_id="level3_R10_rackplace", split="sim_dev", count=2),
            [840000, 840001],
        )
        self.assertEqual(
            load_benchmark_seeds(task_id="level2_R02_wights", split="sim_randomized", count=2),
            [850000, 850001],
        )
        self.assertEqual(
            load_benchmark_seeds(task_id="level3_R10_rackplace", split="sim_randomized", count=2),
            [860000, 860001],
        )
        self.assertEqual(
            load_benchmark_seeds(task_id="level2_R08_NMRSample", split="sim_clean", count=2),
            [875000, 875001],
        )

    def test_every_published_task_has_a_seed_pool_for_every_split(self):
        manifest_path = PACKAGE_ROOT / "configs" / "benchmark_seed_manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8-sig"))
        published_task_ids = {
            str(spec["task_id"])
            for path in (PACKAGE_ROOT / "configs" / "tasks").glob("B*.yaml")
            if (spec := yaml.safe_load(path.read_text(encoding="utf-8-sig")))
            and spec.get("task_id")
        }
        for split in ("sim_clean", "sim_dev", "sim_randomized"):
            with self.subTest(split=split):
                self.assertEqual(
                    published_task_ids.difference(manifest["splits"][split]),
                    set(),
                )

    def test_sample_transport_snapshot_requires_all_three_slots(self):
        cfg = SimpleNamespace(
            controller_type="sample_transport",
            sample_transport_success={"place_xy_tolerance_m": 0.02, "place_z_tolerance_m": 0.05},
        )
        runtime = self._runtime_shell("sample_transport", cfg)
        base = {
            "object_position": [0.30, 0.00, 0.76],
            "target_position": [0.20, 0.30, 0.70],
            "object_quaternion": [0, 0, 0, 1],
            "joint_positions": [0.0] * 7 + [0.04, 0.04],
            "gripper_position": [0.0, 0.0, 0.8],
            "vial_positions": [[0.30, 0.00, 0.76], [0.32, 0.10, 0.76], [0.34, 0.20, 0.76]],
            "target_slot_positions": [[0.15, 0.30, 0.70], [0.20, 0.30, 0.70], [0.25, 0.30, 0.70]],
        }
        runtime.build_snapshot(base)
        placed = dict(base, vial_positions=[[0.15, 0.30, 0.76], [0.20, 0.30, 0.76], [0.25, 0.30, 0.76]])
        snapshot = runtime.build_snapshot(placed)
        self.assertTrue(snapshot["task_state"]["sample_transport_complete"])
        one_wrong = dict(base, vial_positions=[[0.15, 0.30, 0.76], [0.24, 0.30, 0.76], [0.25, 0.30, 0.76]])
        snapshot = runtime.build_snapshot(one_wrong)
        self.assertFalse(snapshot["task_state"]["sample_transport_complete"])

    def test_device_operation_snapshot_requires_every_subgoal(self):
        cfg = SimpleNamespace(
            controller_type="device_operate",
            device_operation_success={
                "door_handle_displacement_m": 0.13,
                "place_xy_tolerance_m": 0.20,
                "place_z_tolerance_m": 0.10,
                "button_press_axis": 0,
                "button_press_displacement_m": 0.005,
            },
        )
        runtime = self._runtime_shell("device_operate", cfg)
        base = {
            "object_position": [0.15, 0.23, 0.84],
            "target_position": [0.55, 0.13, 0.98],
            "object_quaternion": [0, 0, 0, 1],
            "joint_positions": [0.0] * 9,
            "gripper_position": [0.0, 0.0, 1.0],
            "door_handle_position": [0.50, 0.20, 1.0],
            "button_position": [0.45, 0.20, 0.77],
            "initial_button_position": [0.45, 0.20, 0.77],
            "beaker_position": [0.15, 0.23, 0.84],
            "beaker3_position": [0.15, 0.35, 0.84],
            "device_interior_position": [0.52, 0.04, 1.11],
            "beaker3_target_position": [0.42, 0.10, 1.12],
        }
        runtime.build_snapshot(base)
        success = dict(
            base,
            door_handle_position=[0.64, 0.20, 1.0],
            button_position=[0.456, 0.20, 0.77],
            beaker_position=[0.52, 0.04, 1.11],
            beaker3_position=[0.42, 0.10, 1.12],
        )
        snapshot = runtime.build_snapshot(success)
        self.assertTrue(snapshot["task_state"]["device_operation_complete"])
        not_pressed = dict(success, button_position=[0.45, 0.20, 0.77])
        snapshot = runtime.build_snapshot(not_pressed)
        self.assertFalse(snapshot["task_state"]["device_operation_complete"])

    def test_nmr_injection_snapshot_requires_placed_released_stable(self):
        cfg = SimpleNamespace(
            controller_type="nmr_injection",
            nmr_injection_success={"place_xy_tolerance_m": 0.02, "place_z_tolerance_m": 0.05},
        )
        runtime = self._runtime_shell("nmr_injection", cfg)
        base = {
            "object_position": [0.30, -0.10, 0.76],
            "target_position": [0.15, 0.00, 0.76],
            "object_quaternion": [0, 0, 0, 1],
            "joint_positions": [0.0] * 7 + [0.04, 0.04],
            "gripper_position": [0.0, 0.0, 0.8],
        }
        runtime.build_snapshot(base)  # capture episode-start object position
        moved = dict(base, object_position=[0.15, 0.00, 0.76])
        snapshot = runtime.build_snapshot(moved)
        self.assertTrue(snapshot["task_state"]["nmr_placed"])
        self.assertTrue(snapshot["task_state"]["gripper_released"])
        self.assertFalse(snapshot["task_state"]["object_stable"])
        settled = runtime.build_snapshot(dict(moved, object_position=[0.15, 0.00, 0.76]))
        self.assertTrue(settled["task_state"]["object_stable"])

    def test_sample_organition_snapshot_resolves_seed_chosen_box(self):
        cfg = SimpleNamespace(
            controller_type="sample_organition",
            sample_organization_success={"place_xy_tolerance_m": 0.02, "place_z_tolerance_m": 0.05},
        )
        runtime = self._runtime_shell("sample_organition", cfg)
        base = {
            "object_position": [0.30, -0.10, 0.76],
            "target_position": [0.15, 0.05, 0.76],
            "target_a_position": [0.15, 0.05, 0.76],
            "target_b_position": [0.15, 0.25, 0.76],
            "target_box_name": "SamplePlaceBoxA",
            "object_quaternion": [0, 0, 0, 1],
            "joint_positions": [0.0] * 7 + [0.04, 0.04],
            "gripper_position": [0.0, 0.0, 0.8],
        }
        runtime.build_snapshot(base)
        in_a = dict(base, object_position=[0.15, 0.05, 0.76])
        snapshot = runtime.build_snapshot(in_a)
        self.assertTrue(snapshot["task_state"]["sample_in_box_a"])
        self.assertFalse(snapshot["task_state"]["sample_in_box_b"])
        self.assertTrue(snapshot["task_state"]["sample_in_target_box"])
        self.assertEqual(snapshot["task_state"]["target_box_name"], "SamplePlaceBoxA")
        in_b = dict(base, object_position=[0.15, 0.25, 0.76])
        snapshot = runtime.build_snapshot(in_b)
        self.assertTrue(snapshot["task_state"]["sample_in_box_b"])
        self.assertTrue(snapshot["task_state"]["sample_in_target_box"])

    def test_centrifuge_load_snapshot_requires_both_nmr_slots(self):
        cfg = SimpleNamespace(
            controller_type="centrifuge_load",
            centrifuge_success={"place_xy_tolerance_m": 0.02, "place_z_tolerance_m": 0.05},
        )
        runtime = self._runtime_shell("centrifuge_load", cfg)
        base = {
            "object_position": [0.25, -0.25, 0.76],
            "target_position": [0.05, 0.57, 0.76],
            "nmr_02_position": [0.38, -0.15, 0.76],
            "target_02_position": [0.02, 0.63, 0.76],
            "object_quaternion": [0, 0, 0, 1],
            "joint_positions": [0.0] * 9,
            "gripper_position": [0.0, 0.0, 0.8],
        }
        runtime.build_snapshot(base)
        partial = dict(base, object_position=[0.05, 0.57, 0.76])
        snapshot = runtime.build_snapshot(partial)
        self.assertTrue(snapshot["task_state"]["nmr_01_placed"])
        self.assertFalse(snapshot["task_state"]["nmr_02_placed"])
        self.assertFalse(snapshot["task_state"]["centrifuge_loaded"])
        complete = dict(
            partial,
            nmr_02_position=[0.02, 0.63, 0.76],
        )
        snapshot = runtime.build_snapshot(complete)
        self.assertTrue(snapshot["task_state"]["centrifuge_loaded"])

    def test_cohort_analysis_snapshot_requires_three_vials(self):
        cfg = SimpleNamespace(
            controller_type="cohort_analysis",
            cohort_success={"place_xy_tolerance_m": 0.02, "place_z_tolerance_m": 0.05},
        )
        runtime = self._runtime_shell("cohort_analysis", cfg)
        base = {
            "object_position": [0.38, -0.15, 0.76],
            "target_position": [0.15, 0.30, 0.76],
            "vial_01_position": [0.38, 0.00, 0.76],
            "target_01_position": [0.20, 0.30, 0.76],
            "vial_02_position": [0.38, 0.15, 0.76],
            "target_02_position": [0.25, 0.30, 0.76],
            "object_quaternion": [0, 0, 0, 1],
            "joint_positions": [0.0] * 9,
            "gripper_position": [0.0, 0.0, 0.8],
        }
        runtime.build_snapshot(base)
        complete = dict(
            base,
            object_position=[0.15, 0.30, 0.76],
            vial_01_position=[0.20, 0.30, 0.76],
            vial_02_position=[0.25, 0.30, 0.76],
        )
        snapshot = runtime.build_snapshot(complete)
        self.assertTrue(snapshot["task_state"]["cohort_vial_0_placed"])
        self.assertTrue(snapshot["task_state"]["cohort_vial_1_placed"])
        self.assertTrue(snapshot["task_state"]["cohort_vial_2_placed"])
        self.assertTrue(snapshot["task_state"]["cohort_analysis_complete"])

    def test_scale_weights_snapshot_requires_three_pads(self):
        cfg = SimpleNamespace(
            controller_type="scale_weights",
            scale_weights_success={"place_xy_tolerance_m": 0.02, "place_z_tolerance_m": 0.05},
        )
        runtime = self._runtime_shell("scale_weights", cfg)
        base = {
            "object_position": [0.25, -0.15, 0.76],
            "target_position": [0.25, 0.20, 0.76],
            "weight_01_position": [0.20, -0.10, 0.76],
            "target_01_position": [0.30, 0.20, 0.76],
            "weight_02_position": [0.15, 0.00, 0.76],
            "target_02_position": [0.35, 0.20, 0.76],
            "object_quaternion": [0, 0, 0, 1],
            "joint_positions": [0.0] * 9,
            "gripper_position": [0.0, 0.0, 0.8],
        }
        runtime.build_snapshot(base)
        complete = dict(
            base,
            object_position=[0.25, 0.20, 0.76],
            weight_01_position=[0.30, 0.20, 0.76],
            weight_02_position=[0.35, 0.20, 0.76],
        )
        snapshot = runtime.build_snapshot(complete)
        self.assertTrue(snapshot["task_state"]["weight_0_placed"])
        self.assertTrue(snapshot["task_state"]["weight_1_placed"])
        self.assertTrue(snapshot["task_state"]["weight_2_placed"])
        self.assertTrue(snapshot["task_state"]["weights_on_scale"])

    def test_clear_reset_snapshot_requires_vial_and_tube(self):
        cfg = SimpleNamespace(
            controller_type="clear_reset",
            clear_reset_success={"place_xy_tolerance_m": 0.02, "place_z_tolerance_m": 0.05},
        )
        runtime = self._runtime_shell("clear_reset", cfg)
        base = {
            "object_position": [0.23, 0.15, 0.76],
            "target_position": [0.25, 0.30, 0.76],
            "object1_position": [0.18, 0.15, 0.71],
            "target1_position": [0.32, -0.10, 0.80],
            "object_quaternion": [0, 0, 0, 1],
            "joint_positions": [0.0] * 9,
            "gripper_position": [0.0, 0.0, 0.8],
        }
        runtime.build_snapshot(base)
        # The tube center returns to its episode-start height (~0.71) once it
        # rests in the rack; the z-check compares against that initial height.
        complete = dict(
            base,
            object_position=[0.25, 0.30, 0.76],
            object1_position=[0.32, -0.10, 0.71],
        )
        snapshot = runtime.build_snapshot(complete)
        self.assertTrue(snapshot["task_state"]["vial_placed"])
        self.assertTrue(snapshot["task_state"]["tube_placed"])
        self.assertTrue(snapshot["task_state"]["clear_reset_complete"])

    def test_condition_module_snapshot_tracks_press_and_return(self):
        cfg = SimpleNamespace(
            controller_type="ConditionModule",
            condition_module_success={
                "place_xy_tolerance_m": 0.02,
                "place_z_tolerance_m": 0.05,
                "button_press_axis": 2,
                "button_press_displacement_m": 0.005,
            },
        )
        runtime = self._runtime_shell("ConditionModule", cfg)
        base = {
            "object_position": [0.16, -0.30, 0.72],
            "target_position": [0.30, 0.31, 0.81],
            "beaker_position": [0.16, -0.30, 0.72],
            "beaker_initial_position": [0.16, -0.30, 0.72],
            "button_position": [0.30, 0.31, 0.71],
            "initial_button_position": [0.30, 0.31, 0.71],
            "object_quaternion": [0, 0, 0, 1],
            "joint_positions": [0.0] * 9,
            "gripper_position": [0.0, 0.0, 0.8],
        }
        runtime.build_snapshot(base)
        on_device = dict(
            base,
            beaker_position=[0.30, 0.31, 0.81],
            button_position=[0.30, 0.31, 0.716],
        )
        snapshot = runtime.build_snapshot(on_device)
        self.assertTrue(snapshot["task_state"]["beaker_on_device"])
        self.assertTrue(snapshot["task_state"]["button_pressed"])
        self.assertFalse(snapshot["task_state"]["beaker_returned"])
        self.assertFalse(snapshot["task_state"]["condition_module_complete"])
        returned = dict(on_device, beaker_position=[0.16, -0.30, 0.72])
        snapshot = runtime.build_snapshot(returned)
        self.assertTrue(snapshot["task_state"]["beaker_returned"])
        self.assertTrue(snapshot["task_state"]["condition_module_complete"])

    def test_liquid_mixing_snapshot_requires_pour_then_return_per_beaker(self):
        cfg = SimpleNamespace(
            controller_type="LiquidMixing",
            liquid_mixing_success={
                "min_tilt_degrees": 45,
                "near_tolerance_m": 0.15,
                "return_rotation_tolerance_degrees": 30,
                "place_xy_tolerance_m": 0.02,
                "place_z_tolerance_m": 0.05,
            },
        )
        runtime = self._runtime_shell("LiquidMixing", cfg)
        base = {
            "object_position": [0.16, -0.30, 0.72],
            "target_position": [0.21, 0.31, 0.71],
            "pour_target_position": [0.20, 0.31, 0.76],
            "beaker_0_position": [0.16, -0.30, 0.72],
            "initial_beaker_0_position": [0.16, -0.30, 0.72],
            "beaker_0_quaternion": [0, 0, 0, 1],
            "initial_beaker_0_quaternion": [0, 0, 0, 1],
            "beaker_1_position": [0.17, -0.10, 0.72],
            "initial_beaker_1_position": [0.17, -0.10, 0.72],
            "beaker_1_quaternion": [0, 0, 0, 1],
            "initial_beaker_1_quaternion": [0, 0, 0, 1],
            "beaker_2_position": [0.17, 0.035, 0.72],
            "initial_beaker_2_position": [0.17, 0.035, 0.72],
            "beaker_2_quaternion": [0, 0, 0, 1],
            "initial_beaker_2_quaternion": [0, 0, 0, 1],
            "object_quaternion": [0, 0, 0, 1],
            "joint_positions": [0.0] * 9,
            "gripper_position": [0.0, 0.0, 0.8],
        }
        tilted = [0.5, 0.0, 0.0, 0.866]  # 60 deg about x
        runtime.build_snapshot(base)
        pour_0 = dict(
            base,
            beaker_0_position=[0.20, 0.30, 0.80],
            beaker_0_quaternion=tilted,
        )
        snapshot = runtime.build_snapshot(pour_0)
        self.assertTrue(snapshot["task_state"]["beaker_0_poured"])
        self.assertFalse(snapshot["task_state"]["beaker_0_returned"])
        self.assertFalse(snapshot["task_state"]["liquid_mixing_complete"])
        pour_1 = dict(
            pour_0,
            beaker_0_position=[0.16, -0.30, 0.72],
            beaker_0_quaternion=[0, 0, 0, 1],
            beaker_1_position=[0.20, 0.30, 0.80],
            beaker_1_quaternion=tilted,
        )
        snapshot = runtime.build_snapshot(pour_1)
        self.assertTrue(snapshot["task_state"]["beaker_0_returned"])
        self.assertTrue(snapshot["task_state"]["beaker_1_poured"])
        pour_2 = dict(
            pour_1,
            beaker_1_position=[0.17, -0.10, 0.72],
            beaker_1_quaternion=[0, 0, 0, 1],
            beaker_2_position=[0.20, 0.30, 0.80],
            beaker_2_quaternion=tilted,
        )
        snapshot = runtime.build_snapshot(pour_2)
        self.assertTrue(snapshot["task_state"]["beaker_1_returned"])
        self.assertTrue(snapshot["task_state"]["beaker_2_poured"])
        complete = dict(
            pour_2,
            beaker_2_position=[0.17, 0.035, 0.72],
            beaker_2_quaternion=[0, 0, 0, 1],
        )
        snapshot = runtime.build_snapshot(complete)
        self.assertTrue(snapshot["task_state"]["beaker_2_returned"])
        self.assertTrue(snapshot["task_state"]["liquid_mixing_complete"])

    def test_core_truth_specs_succeed_only_after_hold_windows(self):
        task_dir = PACKAGE_ROOT / "configs" / "tasks"
        cases = [
            (
                "B01_level1_pick.yaml", 60,
                {"object_pos": [0, 0, 0.91], "initial_object_pos": [0, 0, 0.80], "hand_pos": [0, 0, 0.91], "gripper_pos": [0.02, 0.02]},
            ),
            (
                "B02_level1_place.yaml", 30,
                {"object_pos": [0.01, 0.01, 0.80], "target_pos": [0, 0, 0.80], "initial_object_pos": [0, 0, 0.80], "gripper_pos": [0.04, 0.04], "object_lin_vel": [0, 0, 0], "object_ang_vel": [0, 0, 0]},
            ),
        ]
        for filename, hold, snapshot in cases:
            with self.subTest(task=filename):
                spec = load_task_spec(str(task_dir / filename), "franka", "", 3000, None)
                evaluator = TaskSuccessEvaluator(success_spec_from_task_spec(spec))
                for _ in range(hold - 1):
                    result = evaluator.update(snapshot)
                    self.assertFalse(result.task_success)
                self.assertTrue(evaluator.update(snapshot).task_success)

        pour = load_task_spec(str(task_dir / "B03_level1_pour.yaml"), "franka", "", 3000, None)
        evaluator = TaskSuccessEvaluator(success_spec_from_task_spec(pour))
        base = {
            "object_pos": [0, 0, 0.90], "target_pos": [0.01, 0.01, 0.80], "initial_object_pos": [0, 0, 0.80],
            "task_state": {"pour_returned": True, "rotation_deg": 0.0, "xy_distance_m": 0.01},
        }
        for event in ("near_target", "tilt_source", "return_upright"):
            snap = dict(base, events=[event])
            evaluator.update(snap)
        for _ in range(58):
            result = evaluator.update(dict(base, events=[]))
        self.assertTrue(evaluator.update(dict(base, events=[])).task_success)

    def test_labbench_task_specs_use_runtime_truth_and_hold_windows(self):
        task_dir = PACKAGE_ROOT / "configs" / "tasks"
        cases = {
            "B11_level2_sampletransport.yaml": {
                "task_state": {
                    "sample_slot_0_filled": True,
                    "sample_slot_1_filled": True,
                    "sample_slot_2_filled": True,
                },
                "gripper_pos": [0.04, 0.04],
            },
            "B12_level4_device_operation.yaml": {
                "task_state": {
                    "device_door_opened": True,
                    "device_beaker_inside": True,
                    "device_beaker3_placed": True,
                    "button_pressed": True,
                },
            },
        }
        for filename, snapshot in cases.items():
            with self.subTest(task=filename):
                task = load_task_spec(str(task_dir / filename), "franka", "", 500, None)
                evaluator = TaskSuccessEvaluator(success_spec_from_task_spec(task))
                for _ in range(29):
                    self.assertFalse(evaluator.update(snapshot).task_success)
                self.assertTrue(evaluator.update(snapshot).task_success)

    def test_stir_snapshot_declares_pending_trajectory_truth_as_false(self):
        runtime = object.__new__(BenchmarkRuntime)
        runtime.cfg = SimpleNamespace(controller_type="stir")
        runtime.controller = SimpleNamespace()
        runtime.world = SimpleNamespace(get_physics_dt=lambda: 1.0 / 60.0)
        runtime._initial_object_pos = np.asarray([0.0, 0.0, 0.8])
        runtime._initial_object_quat = np.asarray([0.0, 0.0, 0.0, 1.0])
        runtime._previous_object_pos = runtime._initial_object_pos.copy()
        runtime._step_index = 0
        runtime._event_history = []
        state = {
            "object_position": [0.0, 0.0, 0.8],
            "target_position": [0.1, 0.1, 0.8],
            "object_quaternion": [0.0, 0.0, 0.0, 1.0],
            "joint_positions": [0.0] * 8,
            "gripper_position": [0.0, 0.0, 0.0],
        }
        snapshot = runtime.build_snapshot(state)
        self.assertIn("stir_trajectory_valid", snapshot["task_state"])
        self.assertFalse(snapshot["task_state"]["stir_trajectory_valid"])

    @staticmethod
    def _runtime_shell(controller_type, cfg=None):
        runtime = object.__new__(BenchmarkRuntime)
        runtime.cfg = cfg or SimpleNamespace(controller_type=controller_type)
        runtime.controller = SimpleNamespace()
        runtime.world = SimpleNamespace(get_physics_dt=lambda: 1.0 / 60.0)
        runtime._initial_object_pos = np.asarray([0.0, 0.0, 0.8])
        runtime._initial_object_quat = np.asarray([0.0, 0.0, 0.0, 1.0])
        runtime._previous_object_pos = runtime._initial_object_pos.copy()
        runtime._step_index = 0
        runtime._event_history = []
        return runtime

    def test_door_snapshot_exposes_release_and_open_truth(self):
        cfg = SimpleNamespace(
            controller_type="open",
            open_door={"minimum_handle_displacement": 0.10, "minimum_release_distance": 0.05, "open_stable_steps": 2},
        )
        runtime = self._runtime_shell("open", cfg)
        base = {"target_position": [0, 0, 0], "joint_positions": [0.0] * 8, "object_quaternion": [0, 0, 0, 1]}
        first = dict(base, object_position=[0.0, 0.0, 0.8], gripper_position=[0.0, 0.0, 0.8])
        runtime.build_snapshot(first)
        moved = dict(base, object_position=[0.12, 0.0, 0.8], gripper_position=[0.3, 0.0, 0.8])
        snapshot = runtime.build_snapshot(moved)
        self.assertTrue(snapshot["task_state"]["door_open_threshold_reached"])
        self.assertTrue(snapshot["task_state"]["door_released"])

    def test_stir_snapshot_requires_circular_work_path_and_lift(self):
        cfg = SimpleNamespace(
            controller_type="stir",
            stir_success={
                "min_radius_m": 0.004, "max_radius_m": 0.02,
                "min_work_height_m": 0.81, "max_work_height_m": 0.90,
                "min_samples": 20, "min_angular_sweep_degrees": 300,
                "min_path_length_m": 0.03, "max_sample_jump_m": 0.03,
                "final_lift_height_m": 0.93,
            },
        )
        runtime = self._runtime_shell("stir", cfg)
        for index in range(24):
            angle = 2 * np.pi * index / 23.0
            pos = [0.01 * np.cos(angle), 0.01 * np.sin(angle), 0.84]
            runtime.build_snapshot({
                "object_position": pos, "glass_rod_position": pos,
                "target_position": [0.0, 0.0, 0.8],
                "joint_positions": [0.0] * 8, "gripper_position": pos,
            })
        snapshot = runtime.build_snapshot({
            "object_position": [0.0, 0.0, 0.95], "glass_rod_position": [0.0, 0.0, 0.95],
            "target_position": [0.0, 0.0, 0.8], "joint_positions": [0.0] * 8,
            "gripper_position": [0.0, 0.0, 0.95],
        })
        self.assertTrue(snapshot["task_state"]["stir_trajectory_valid"])

    def test_tube_snapshot_exposes_insertion_geometry(self):
        cfg = SimpleNamespace(
            controller_type="pickplace",
            tube_insertion={"success_xy_tolerance": 0.003, "success_z_tolerance": 0.005, "success_axis_tolerance_degrees": 8},
        )
        runtime = self._runtime_shell("pickplace", cfg)
        snapshot = runtime.build_snapshot({
            "object_position": [0.0, 0.0, 0.63], "target_position": [0.0, 0.0, 0.63],
            "tube_socket_position": [0.0, 0.0, 0.60], "tube_inserted_center": [0.0, 0.0, 0.63],
            "object_axis": [0.0, 0.0, -1.0], "tube_axis": [0.0, 0.0, 1.0],
            "joint_positions": [0.0] * 8, "gripper_position": [0.0, 0.0, 0.63],
        })
        self.assertTrue(snapshot["task_state"]["tube_inserted"])
        self.assertTrue(snapshot["task_state"]["tube_aligned"])
        self.assertAlmostEqual(snapshot["insert_depth_m"], 0.03, places=6)

        # The real rack supports the released tube with its geometry centre
        # below the socket plane. Depth must remain positive in that convention.
        below_socket = runtime.build_snapshot({
            "object_position": [0.0, 0.0, 0.811],
            "target_position": [0.0, 0.0, 0.811],
            "tube_socket_position": [0.0, 0.0, 0.838],
            "tube_inserted_center": [0.0, 0.0, 0.811],
            "object_axis": [0.0, 0.0, -1.0],
            "tube_axis": [0.0, 0.0, 1.0],
            "joint_positions": [0.0] * 7 + [0.04, 0.04],
            "gripper_position": [0.0, 0.0, 0.95],
        })
        self.assertTrue(below_socket["task_state"]["tube_inserted"])
        self.assertTrue(below_socket["task_state"]["tube_released"])
        self.assertAlmostEqual(below_socket["insert_depth_m"], 0.027, places=6)


if __name__ == "__main__":
    unittest.main()
