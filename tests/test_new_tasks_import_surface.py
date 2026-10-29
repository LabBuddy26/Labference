"""Import surface for the LabBench B13-B20 tasks merged into this repo.

The task and controller modules import ``isaacsim``, so real imports only
succeed when the Isaac extension system is active (e.g. after creating a
SimulationApp).  To keep this test runnable in any environment:

- static checks verify the factory registrations and module files;
- ``evaluation.runtime`` is imported for real because it intentionally keeps
  Isaac imports inside ``__init__`` (this also exercises the new snapshot
  branches' module-level code paths);
- a guarded real-import test runs the full task/controller imports when
  ``isaacsim.core`` is importable, and skips otherwise.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IS_PUBLIC_RELEASE = (REPO_ROOT / "PUBLIC_RELEASE.json").is_file()


class NewTaskImportSurfaceTest(unittest.TestCase):
    CONTROLLER_TYPES = (
        "nmr_injection",
        "centrifuge_load",
        "cohort_analysis",
        "sample_organition",
        "scale_weights",
        "ConditionModule",
        "LiquidMixing",
        "clear_reset",
        "rack_place",
        "weight_select",
        "press",
        "move",
    )
    TASK_TYPES = (
        "nmrinjection",
        "centrifugeload",
        "cohortanalysis",
        "sampleorganition",
        "scaleweights",
        "ConditionModule",
        "LiquidMixing",
        "clearreset",
        "rackplace",
        "weightselect",
        "nmrsample",
        "nmrrack",
        "temperaturcontrol",
        "devicestatus",
        "centrifugecontrol",
        "scaledigital",
        "plateposition",
    )
    AUDITED_CONTROLLER_MODULES = (
        "sample_transport",
        "device_operate",
        "nmr_injection",
        "centrifuge_load",
        "cohort_analysis",
        "sample_organition",
        "scale_weights",
        "condition_module",
        "liquid_mixing",
        "clear_reset",
    )
    # Module-name → controller-file overrides for the few controllers whose
    # files do not follow the snake_case ``<name>_controller.py`` convention.
    CONTROLLER_FILENAME_OVERRIDES = {
        "liquid_mixing": "LiquidMixing_controller.py",
    }

    def _controller_path(self, module_name: str) -> Path:
        filename = self.CONTROLLER_FILENAME_OVERRIDES.get(
            module_name, f"{module_name}_controller.py"
        )
        return REPO_ROOT / "controllers" / filename

    def test_controller_factory_registers_all_new_types(self) -> None:
        text = (REPO_ROOT / "factories" / "controller_factory.py").read_text(encoding="utf-8")
        for name in self.CONTROLLER_TYPES:
            self.assertIn(f'register_controller("{name}"', text, name)

    def test_task_factory_registers_all_new_types(self) -> None:
        text = (REPO_ROOT / "factories" / "task_factory.py").read_text(encoding="utf-8")
        for name in self.TASK_TYPES:
            self.assertIn(f'register_task("{name}"', text, name)

    def test_new_controller_modules_exist_with_phase_specs(self) -> None:
        # Literal PhaseSpec( call sites per multi-object module.  liquid_mixing
        # is not part of this family: this branch keeps main's TaskPhase-based
        # LiquidMixing implementation (pinned by the platform test below).
        multi_object = {
            "cohort_analysis": 6,
            "scale_weights": 6,
            "clear_reset": 4,
        }
        for module_name, expected_calls in multi_object.items():
            path = REPO_ROOT / "controllers" / f"{module_name}_controller.py"
            self.assertTrue(path.exists(), str(path))
            text = path.read_text(encoding="utf-8")
            specs = re.findall(r"PhaseSpec\(", text)
            self.assertEqual(len(specs), expected_calls, module_name)
        # nmr_injection / sample_organition reuse the single-object machine.
        for module_name in ("nmr_injection", "sample_organition"):
            text = (REPO_ROOT / "controllers" / f"{module_name}_controller.py").read_text(encoding="utf-8")
            self.assertIn("SingleObjectPlaceTaskController", text, module_name)

        # ConditionModule deliberately follows main's dedicated five-stage
        # controller instead of the generic PhaseSpec state machine.
        condition_text = (
            REPO_ROOT / "controllers" / "condition_module_controller.py"
        ).read_text(encoding="utf-8")
        self.assertIn("class TaskPhase(Enum)", condition_text)
        for phase in ("PICKING1", "PLACEING1", "PRESS", "PICKING2", "PLACEING2"):
            self.assertIn(f"TaskPhase.{phase}", condition_text)
        self.assertIn("cache_collection_step(state, action)", condition_text)
        self.assertIn("write_cached_episode(state)", condition_text)

    def test_shared_phase_machine_and_press_module_exist(self) -> None:
        self.assertTrue((REPO_ROOT / "controllers" / "multi_object_task_controller.py").exists())
        self.assertTrue((REPO_ROOT / "controllers" / "single_object_place_controller.py").exists())
        self.assertTrue((REPO_ROOT / "controllers" / "atomic_actions" / "pressz_controller.py").exists())

    def test_reasoning_tasks_follow_merged_collection_contract(self) -> None:
        configs = (
            "level2_R08_NMRSample.yaml",
            "level2_R09_NMRRack.yaml",
            "level3_R03_temperaturepanel.yaml",
            "level3_R05_devicestatus.yaml",
        )
        for filename in configs:
            text = (REPO_ROOT / "config" / filename).read_text(encoding="utf-8")
            self.assertIn("max_episodes: 200", text, filename)
            self.assertIn("sample_hz: 10", text, filename)
            self.assertIn("action_space: ee_pos", text, filename)
            self.assertIn("camera_1_rgb: camera_1_rgb", text, filename)
            self.assertIn("camera_3_rgb: camera_3_rgb", text, filename)
            self.assertNotIn("camera_2_rgb", text, filename)
            self.assertNotIn("name: camera_2", text, filename)
            self.assertIn("local_translation: [-0.20, 0.0, -0.01]", text, filename)

        for filename in (
            "level3_R03_temperaturepanel.yaml",
            "level3_R05_devicestatus.yaml",
        ):
            text = (REPO_ROOT / "config" / filename).read_text(encoding="utf-8")
            self.assertIn("translation: [-0.7, 1.4, 1.4]", text, filename)
            self.assertIn(
                "orientation: [-0.2297, -0.18336, 0.59631, 0.74701]",
                text,
                filename,
            )

        press = (REPO_ROOT / "controllers" / "press_controller.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("cache_collection_step(state, action)", press)
        self.assertIn("write_cached_episode(state)", press)
        if not IS_PUBLIC_RELEASE:
            self.assertTrue(
                (REPO_ROOT / "assets" / "chemistry_lab" / "assets" / "TempInstrument.usd").is_file()
            )

        display_task = (REPO_ROOT / "tasks" / "display_button_task.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("self._texture_generation += 1", display_task)
        self.assertIn('with_suffix(".png.tmp")', display_task)
        self.assertIn("os.replace(temporary_path, texture_path)", display_task)

        task_list = (REPO_ROOT / "collection_tasks.txt").read_text(encoding="utf-8")
        for filename in configs:
            self.assertIn(Path(filename).stem, task_list)

    def test_latest_main_tasks_follow_collection_contract(self) -> None:
        configs = (
            "level2_R11_plateposition.yaml",
            "level3_R01_scaledigital.yaml",
            "level3_R04_centrifugepanel.yaml",
        )
        for filename in configs:
            text = (REPO_ROOT / "config" / filename).read_text(encoding="utf-8")
            self.assertIn("max_episodes: 200", text, filename)
            self.assertIn("sample_hz: 10", text, filename)
            self.assertIn("action_space: ee_pos", text, filename)
            self.assertIn("camera_1_rgb: camera_1_rgb", text, filename)
            self.assertIn("camera_3_rgb: camera_3_rgb", text, filename)
            self.assertNotIn("camera_2_rgb", text, filename)
            self.assertNotIn("name: camera_2", text, filename)
            self.assertIn("local_translation: [-0.20, 0.0, -0.01]", text, filename)

        for filename in ("level3_R01_scaledigital.yaml", "level3_R04_centrifugepanel.yaml"):
            text = (REPO_ROOT / "config" / filename).read_text(encoding="utf-8")
            self.assertIn("translation: [-0.7, 1.4, 1.4]", text, filename)
            self.assertIn("orientation_deg: [0.0, 90.0, 0.0]", text, filename)
            self.assertIn("press_offset_x_m: 0.0", text, filename)
            self.assertIn("hold_requires_reach: false", text, filename)

        move = (REPO_ROOT / "controllers" / "move_controller.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("cache_collection_step(state, action)", move)
        self.assertIn("write_cached_episode(state)", move)
        self.assertIn("xyz_error", move)

        if not IS_PUBLIC_RELEASE:
            for asset in ("RpmInstrument.usd", "ScaleInstrument.usd"):
                self.assertTrue(
                    (REPO_ROOT / "assets" / "chemistry_lab" / "assets" / asset).is_file()
                )
        task_list = (REPO_ROOT / "collection_tasks.txt").read_text(encoding="utf-8")
        for filename in configs:
            self.assertIn(Path(filename).stem, task_list)

    def test_l07_uses_main_single_tube_control_with_aligned_collection(self) -> None:
        text = (
            REPO_ROOT / "controllers" / "centrifuge_load_controller.py"
        ).read_text(encoding="utf-8")
        self.assertIn("class CentrifugeLoadTaskController(BaseController)", text)
        self.assertIn("PickController", text)
        self.assertIn("np.radians([0, 90, 30])", text)
        self.assertIn("np.radians([0, 90, 80])", text)
        self.assertIn("np.radians([0, 90, 30])", text)
        self.assertIn("pre_place_z=self.place_pre_z_m", text)
        self.assertIn("place_offset_z=self.place_offset_z_m", text)
        self.assertIn("gripper_distances=self.grasp_width_m", text)
        self.assertIn("cache_collection_step(state, action)", text)
        self.assertIn("write_cached_episode(state)", text)
        self.assertNotIn("pick_nmr_02", text)
        config = (
            REPO_ROOT / "config" / "level2_L07_centrifuge_load.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("grasp_width_m: 0.001", config)
        self.assertIn("place_descent_tolerance_m: 0.004", config)
        self.assertIn("place_pre_z_m: 0.20", config)
        self.assertIn("place_offset_z_m: 0.070", config)
        self.assertIn("place_xy_compensation_m: [0.0, 0.0]", config)
        self.assertIn("CentrifugalMachine_mesh.usd", config)
        self.assertIn("preloaded_position: [0.0627004, 0.36812646, 0.82909817]", config)
        self.assertIn("target1_position: [0.041, 0.23, 0.8175]", config)

    def test_reasoning_centrifuge_configs_provide_preloaded_objects(self) -> None:
        r06 = (REPO_ROOT / "config" / "level2_R06_centrifuge_load.yaml").read_text(
            encoding="utf-8"
        )
        r07 = (REPO_ROOT / "config" / "level2_R07_centrifuge_load.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn('sub_object1: "/World/NMR_01"', r06)
        self.assertIn('sub_object2: "/World/NMR_02"', r07)
        self.assertIn('prim_path: "/World/NMR_02"', r07)
        self.assertGreaterEqual(r07.count('path: "/World/NMR_01"'), 2)
        self.assertIn("x: [0.25, 0.30]", r07)
        self.assertIn("y: [-0.15, -0.10]", r07)
        self.assertIn("x: [0.25, 0.25]", r07)
        self.assertIn("y: [-0.25, -0.25]", r07)
        self.assertIn("pick_grasp_tolerance_m: 0.005", r07)
        self.assertIn("pick_position_tolerance_m: 0.025", r07)
        self.assertIn("place_position_tolerance_m: 0.020", r07)
        for module in ("centrifugeloadR_task.py", "centrifugeloadR1_task.py"):
            text = (REPO_ROOT / "tasks" / module).read_text(encoding="utf-8")
            self.assertIn("_place_preloaded_object", text)
            self.assertIn("_stabilize_object(obj_path)", text)
            self.assertIn("GetVelocityAttr().Set", text)
            self.assertIn("GetAngularVelocityAttr().Set", text)
        r07_task = (
            REPO_ROOT / "tasks" / "centrifugeloadR1_task.py"
        ).read_text(encoding="utf-8")
        self.assertIn("_set_object_kinematic(self.sub_object1, True)", r07_task)
        self.assertIn("target_body.CreateKinematicEnabledAttr().Set(True)", r07_task)
        r07_controller = (
            REPO_ROOT / "controllers" / "centrifuge_loadR1_controller.py"
        ).read_text(encoding="utf-8")
        self.assertGreaterEqual(
            r07_controller.count("np.radians([0, 90, 30])"), 2
        )
        self.assertGreaterEqual(
            r07_controller.count("gripper_distances=self.grasp_width_m"), 2
        )

    def test_new_collection_tasks_restore_the_stable_robot_start_pose(self) -> None:
        configs = (
            "level2_L03_singlepipetting.yaml",
            "level2_L05_formulation_spotting.yaml",
            "level2_L07_centrifuge_load.yaml",
            "level2_R06_centrifuge_load.yaml",
            "level2_R07_centrifuge_load.yaml",
            "level3_R12_devicecover_status.yaml",
        )
        for filename in configs:
            text = (REPO_ROOT / "config" / filename).read_text(encoding="utf-8")
            self.assertIn("drive_robot: true", text, filename)

    def test_condition_module_uses_main_motion_parameters_with_merged_collection(self) -> None:
        text = (
            REPO_ROOT / "controllers" / "condition_module_controller.py"
        ).read_text(encoding="utf-8")
        self.assertIn("[0.002, 0.002, 0.005, 1, 0.05, 0.01, 1]", text)
        self.assertIn("[0.002, 0.002, 0.005, 0.2, 0.05, 0.01, 0.1]", text)
        self.assertIn("events_dt=[0.004, 0.01, 0.01]", text)
        self.assertIn("np.radians([0, 90, 10])", text)
        self.assertIn("np.radians([0, 90, 30])", text)
        self.assertIn("np.radians([0, 90, 40])", text)
        self.assertIn('state["button_position"]', text)
        self.assertIn("button_press_target_offset_z_m", text)
        self.assertIn("max_button_press_displacement", text)
        self.assertNotIn("ButtonBase", text)
        self.assertIn('failure_reason.startswith("place motion event 4 timed out")', text)
        self.assertIn("and self._check_phase_success(state)", text)
        self.assertIn("cache_collection_step(state, action)", text)
        self.assertIn("write_cached_episode(state)", text)
        config = (
            REPO_ROOT / "config" / "level4_L11_ConditionModule.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'button_path: "/World/heat_device/heat_device/button"', config
        )
        self.assertIn("button_press_target_offset_z_m: 0.002", config)
        self.assertIn("button_press_displacement_m: 0.003", config)
        self.assertNotIn("/World/heat_device/Root/heat_device/button", config)

    def test_weight_select_allows_a_long_strict_pregrasp_timeout(self) -> None:
        config = (REPO_ROOT / "config" / "level2_R02_wights.yaml").read_text(
            encoding="utf-8"
        )
        controller = (
            REPO_ROOT / "controllers" / "weight_select_controller.py"
        ).read_text(encoding="utf-8")
        self.assertIn("pick_event_0_timeout_steps: 1200", config)
        self.assertIn("1.0 / self.pick_event_0_timeout_steps", controller)
        self.assertIn("isaac_quaternion_from_euler_deg", controller)
        self.assertNotIn("R.from_euler", controller)
        self.assertIn('"camera_3_rgb": "camera_3_rgb"', config)
        self.assertNotIn('name: "camera_2"', config)
        self.assertIn("local_translation: [-0.20, 0.0, -0.01]", config)

    def test_liquid_mixing_platform_is_never_loaded_and_controller_independent(self) -> None:
        config = (REPO_ROOT / "config" / "level4_L04_LiquidMixing.yaml").read_text(
            encoding="utf-8"
        )
        # The platform remains commented out in the config (never loaded);
        # no active line may reference it.
        for line in config.splitlines():
            if "plat" in line and not line.lstrip().startswith("#"):
                self.fail(f"un-commented platform reference: {line}")
        for beaker in ("beaker_03", "beaker_04", "beaker_05"):
            self.assertIn(f'prim_path: "/World/{beaker}"', config)
        self.assertIn('task_type: "LiquidMixing"', config)
        self.assertIn('controller_type: "LiquidMixing"', config)

        task = (REPO_ROOT / "tasks" / "LiquidMixing_task.py").read_text(
            encoding="utf-8"
        )
        # The source beaker the task tracks is part of the active scene.
        self.assertIn('self.beaker_path = "/World/beaker_4"', task)

        controller = (
            REPO_ROOT / "controllers" / "LiquidMixing_controller.py"
        ).read_text(encoding="utf-8")
        # This branch keeps main's LiquidMixing implementation: a TaskPhase
        # machine (not a multi-object PhaseSpec controller) that never
        # references the platform prim.
        self.assertIn("TaskPhase", controller)
        self.assertIn("self.cache_collection_step(state, action)", controller)
        self.assertIn("reset_needed = True", controller)
        for line in controller.splitlines():
            if "plat" in line and not line.lstrip().startswith("#"):
                self.fail(f"controller platform reference: {line}")

    def test_audited_controllers_do_not_pass_scipy_xyzw_to_isaac(self) -> None:
        # Scipy's R.from_euler(...).as_quat() yields xyzw, the same convention
        # isaacsim uses.  Passing it to the atomic controllers' orientation
        # kwargs (end_effector_orientation / target_end_effector_orientation /
        # target_orientation) is the accepted pattern in this branch (used by
        # device_operate and LiquidMixing).  What is forbidden is using a
        # scipy xyzw quaternion anywhere else (e.g. direct isaac pose calls).
        allowed_kwargs = {
            "end_effector_orientation",
            "target_end_effector_orientation",
            "target_orientation",
        }
        for module_name in self.AUDITED_CONTROLLER_MODULES:
            text = self._controller_path(module_name).read_text(encoding="utf-8")
            tree = ast.parse(text)
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "as_quat"
            ]
            for call in calls:
                if not any(
                    kw.arg in allowed_kwargs and kw.value is call
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    for kw in node.keywords
                ):
                    self.fail(
                        f"{module_name}: scipy xyzw quat used outside "
                        f"atomic-controller orientation kwargs: {ast.unparse(call)}"
                    )

    def test_device_operation_matches_main_collection_motion(self) -> None:
        text = (
            REPO_ROOT / "controllers" / "device_operate_controller.py"
        ).read_text(encoding="utf-8")
        self.assertIn("target_position[0] -= 0.4", text)
        self.assertIn("target_position[2] += 0.22", text)
        self.assertNotIn("require_pose_reached=True", text)
        self.assertNotIn("forward_multi_segment", text)

        config = (
            REPO_ROOT / "config" / "level4_L09_DeviceOperation.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("max_steps: 4000", config)
        self.assertIn("local_translation: [-0.20, 0.0, -0.01]", config)
        self.assertIn(
            "local_orientation: [0.20083, 0.67799, -0.67799, -0.20083]",
            config,
        )
        self.assertIn("position: [0.15, 0.23, 0.84]", config)
        self.assertIn("position: [0.15, 0.35, 0.84]", config)

        object_utils_text = (
            REPO_ROOT / "utils" / "object_utils.py"
        ).read_text(encoding="utf-8")
        self.assertIn("UsdGeom.XformOp.TypeTranslate", object_utils_text)

    def test_multi_object_pick_lifts_clear_success_threshold(self) -> None:
        modules = (
            "cohort_analysis",
            "scale_weights",
            "clear_reset",
        )
        for module_name in modules:
            path = REPO_ROOT / "controllers" / f"{module_name}_controller.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Name) or node.func.id != "PhaseSpec":
                    continue
                if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
                    continue
                if node.args[1].value != "pick":
                    continue
                lift = 0.30
                for keyword in node.keywords:
                    if keyword.arg == "after_offset_z":
                        lift = float(ast.literal_eval(keyword.value))
                self.assertGreaterEqual(lift, 0.12, f"{module_name}: {ast.unparse(node)}")

    def test_scale_third_pick_completion_diagnostics_are_wired(self) -> None:
        scale = (REPO_ROOT / "controllers" / "scale_weights_controller.py").read_text(
            encoding="utf-8"
        )
        shared = (
            REPO_ROOT / "controllers" / "multi_object_task_controller.py"
        ).read_text(encoding="utf-8")
        for field in (
            "initial_z=",
            "current_z=",
            "lift_height=",
            "left_contact_force_n=",
            "right_contact_force_n=",
            "finger_joint_positions=",
        ):
            self.assertIn(field, scale)
        self.assertIn("self._on_phase_completion(state, spec)", shared)
        self.assertIn('spec.name == "pick_weight_2"', scale)

    def test_clear_reset_tube_approach_stays_inside_workspace(self) -> None:
        path = REPO_ROOT / "controllers" / "clear_reset_controller.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tube_place = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "PhaseSpec":
                continue
            if not node.args or ast.literal_eval(node.args[0]) != "place_tube":
                continue
            tube_place = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in node.keywords
            }
            break
        self.assertIsNotNone(tube_place)
        self.assertLessEqual(float(tube_place["pre_place_z"]), 0.20)

    def test_clear_reset_uses_guarded_insertion_for_both_objects(self) -> None:
        controller_text = (
            REPO_ROOT / "controllers" / "clear_reset_controller.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"place_vial", "insert"', controller_text)
        self.assertIn('"place_tube", "insert"', controller_text)

        shared_text = (
            REPO_ROOT / "controllers" / "multi_object_task_controller.py"
        ).read_text(encoding="utf-8")
        self.assertIn("TubeInsertController", shared_text)
        self.assertIn('if spec.kind == "insert"', shared_text)
        self.assertIn("inserted_center_key", shared_text)
        self.assertIn("socket_key", shared_text)
        self.assertIn("pre_release_offset_z", shared_text)

        task_text = (
            REPO_ROOT / "tasks" / "clearreset_task.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"target1_inserted_center"', task_text)
        self.assertIn('"tube_socket_position"', task_text)
        self.assertIn("tube_bottom_seat_local_position", task_text)
        self.assertIn("_local_point_to_world", task_text)
        self.assertIn("vial_settled_center_local_position", task_text)

        config_text = (
            REPO_ROOT / "config" / "level4_L12_clearreset.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("guarded_insertion:", config_text)
        self.assertIn("object_lateral_tolerance: 0.002", config_text)
        self.assertIn("insertion_lateral_tolerance: 0.005", config_text)
        self.assertIn("vial_socket_local_position: [0.024, 0.020, 0.0]", config_text)
        self.assertIn(
            "tube_socket_local_position: [-0.04224623, 0.280, -0.00624620]",
            config_text,
        )
        self.assertIn("max_steps: 6000", config_text)
        self.assertIn("tube_minimum_insertion_depth_m: 0.015", config_text)

        self.assertIn("insertion_depth", controller_text)
        self.assertIn("tube_bottom_z", controller_text)

    def test_evaluation_runtime_imports_without_isaac(self) -> None:
        from evaluation.runtime import BenchmarkRuntime  # noqa: F401

        self.assertTrue(hasattr(BenchmarkRuntime, "build_snapshot"))

    def test_main_sync_tasks_follow_current_collection_and_eval_contract(self) -> None:
        configs = (
            "level2_L03_singlepipetting.yaml",
            "level2_L05_formulation_spotting.yaml",
            "level2_L07_centrifuge_load.yaml",
            "level2_R06_centrifuge_load.yaml",
            "level2_R07_centrifuge_load.yaml",
            "level3_R12_devicecover_status.yaml",
        )
        for filename in configs:
            text = (REPO_ROOT / "config" / filename).read_text(encoding="utf-8")
            self.assertIn("max_episodes: 200", text, filename)
            self.assertIn("sample_hz: 10", text, filename)
            self.assertIn("action_space:", text, filename)
            self.assertIn("ee_pos", text, filename)
            self.assertIn("camera_1_rgb", text, filename)
            self.assertIn("camera_3_rgb", text, filename)
            self.assertNotIn("camera_2_rgb", text, filename)
            self.assertNotIn('name: "camera_2"', text, filename)
            self.assertIn("local_translation: [-0.20, 0.0, -0.01]", text, filename)
            self.assertIn(
                "local_orientation: [0.20083, 0.67799, -0.67799, -0.20083]",
                text,
                filename,
            )
            self.assertIn("focal_length: 10.0", text, filename)
            self.assertIn("horizontal_aperture: 20.955", text, filename)
            self.assertIn("focus_distance: 0.50", text, filename)
            self.assertIn("clipping_range: [0.05, 10.0]", text, filename)

        for filename in (
            "single_pipetting_controller.py",
            "formula_spotting_controller.py",
            "centrifuge_loadR_controller.py",
            "centrifuge_loadR1_controller.py",
        ):
            text = (REPO_ROOT / "controllers" / filename).read_text(encoding="utf-8")
            self.assertIn("cache_collection_step(state, action)", text, filename)
            self.assertIn("write_cached_episode(state)", text, filename)
            self.assertNotIn("data_collector.cache_step(", text, filename)
            self.assertNotIn("data_collector.write_cached_data(", text, filename)

        suite = (
            REPO_ROOT
            / "eai4s_eval/configs/task_suites/eai4s_benchmark_longhorizon_reasoning_v1.yaml"
        ).read_text(encoding="utf-8")
        self.assertEqual(suite.count("status: implemented"), 24)

    def test_full_imports_when_isaac_extension_system_active(self) -> None:
        try:
            import isaacsim  # noqa: F401
            from isaacsim.core.utils.prims import get_prim_at_path  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("isaacsim.core not importable without an active Isaac session")
        from factories.controller_factory import _controller_registry
        from factories.task_factory import _task_registry

        for name in self.CONTROLLER_TYPES:
            self.assertIn(name, _controller_registry, name)
        for name in self.TASK_TYPES:
            self.assertIn(name, _task_registry, name)


if __name__ == "__main__":
    unittest.main()
