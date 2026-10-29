from pathlib import Path
import unittest

import yaml
from omegaconf import OmegaConf

from eai4s_eval.cli.run_benchmark_eval import build_parser as build_benchmark_eval_parser
from eai4s_eval.configured_runner import _args_from_yaml
from eai4s_eval.policy import load_openpi_profile


ROOT = Path(__file__).resolve().parents[1]


class EvalModeConfigTest(unittest.TestCase):
    def test_every_task_yaml_can_resolve_its_eval_assets(self):
        for path in sorted((ROOT / "config").glob("*.yaml")):
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
                if "defaults" in config and "cameras" not in config:
                    continue  # alias configs only hold `defaults: [<canonical>]`
                self.assertIn(config.get("mode"), {"collect", "eval"})
                evaluation = config.get("evaluation")
                self.assertIsInstance(evaluation, dict)
                self.assertTrue((ROOT / evaluation["task_spec"]).is_file())
                policy = evaluation.get("policy")
                self.assertIsInstance(policy, dict)
                self.assertIn(policy.get("provider"), {"openpi", "gr00t"})
                self.assertTrue((ROOT / policy["profile"]).is_file())
                self.assertIn(policy.get("protocol"), {"websocket", "zmq"})

    def test_eval_arguments_come_from_the_same_task_yaml(self):
        cfg = OmegaConf.load(ROOT / "config" / "level1_pick.yaml")
        args = _args_from_yaml(
            cfg,
            benchmark_root=ROOT,
            backend="numpy",
            headless=True,
            no_video=False,
        )
        self.assertEqual(args.task_spec, str(ROOT / "eai4s_eval/configs/tasks/B01_level1_pick.yaml"))
        self.assertEqual(args.policy_profile, str(ROOT / "eai4s_eval/configs/policies/gr00t_franka_ee_pos_v1.yaml"))
        self.assertEqual(args.output_root, str(ROOT / "outputs/eval_runs"))
        self.assertEqual(args.max_steps, int(cfg.evaluation.max_steps))
        self.assertEqual(args.execution_horizon, int(cfg.evaluation.policy.execution_horizon))
        self.assertEqual(args.policy_provider, "gr00t")
        self.assertEqual(args.policy_backend, "zmq")
        self.assertEqual(args.vla_port, int(cfg.evaluation.policy.port))
        self.assertFalse(args.reanchor_chunks)
        self.assertFalse(args.gripper_reach_gate)
        self.assertFalse(args.chunk_boundary_guard)
        self.assertEqual(args.chunk_boundary_error_threshold, 0.03)
        self.assertEqual(args.max_joint_velocity, 2.0)
        self.assertEqual(args.max_joint_acceleration, 10.0)
        self.assertEqual(args.control_hz, 10)
        self.assertIsNone(args.sim_steps_per_action)
        cfg.task.max_steps = 17
        separated_args = _args_from_yaml(
            cfg,
            benchmark_root=ROOT,
            backend="numpy",
            headless=True,
            no_video=False,
        )
        self.assertEqual(separated_args.max_steps, int(cfg.evaluation.max_steps))
        self.assertTrue(args.save_video)

    def test_gr00t_profile_declares_binary_training_gripper(self):
        profile = load_openpi_profile(
            ROOT / "eai4s_eval/configs/policies/gr00t_franka_joint8_v1.yaml"
        )
        self.assertEqual(profile.observation_mode, "gr00t")
        self.assertEqual(profile.action_dim, 8)
        self.assertEqual(profile.state_arm_dim, 7)
        self.assertEqual(profile.gripper_mode, "binary")
        self.assertEqual(profile.gripper_closed_value, 0.0)
        self.assertEqual(profile.gripper_open_value, 0.04)
        self.assertEqual(profile.control_hz, 10)

    def test_gr00t_ee_pos_profile_declares_ee_action_space(self):
        profile = load_openpi_profile(
            ROOT / "eai4s_eval/configs/policies/gr00t_franka_ee_pos_v1.yaml"
        )
        self.assertEqual(profile.observation_mode, "gr00t")
        self.assertEqual(profile.action_space, "ee_pos")
        self.assertEqual(profile.action_dim, 8)
        self.assertEqual(profile.state_arm_dim, 7)
        self.assertEqual(profile.gripper_mode, "binary")
        self.assertEqual(profile.control_hz, 10)

    def test_openpi_ee_pos_profile_declares_ee_action_space(self):
        profile = load_openpi_profile(
            ROOT / "eai4s_eval/configs/policies/openpi_franka_ee_pos_v1.yaml"
        )
        self.assertEqual(profile.observation_mode, "openpi")
        self.assertEqual(profile.action_space, "ee_pos")
        self.assertEqual(profile.action_dim, 8)
        self.assertEqual(profile.gripper_mode, "binary")
        self.assertEqual(profile.control_hz, 10)

    def test_all_franka_tasks_default_to_ee_actions(self):
        for path in sorted((ROOT / "config").glob("*.yaml")):
            config = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
            robot_type = (config.get("robot") or {}).get("type")
            if not robot_type:
                continue
            with self.subTest(config=path.name):
                infer = config.get("infer") or {}
                profile_path = ROOT / config["evaluation"]["policy"]["profile"]
                profile = load_openpi_profile(profile_path)
                if robot_type == "franka":
                    self.assertEqual(infer.get("action_space"), "ee_pos")
                    self.assertEqual(profile.action_space, "ee_pos")
                else:
                    self.assertEqual(robot_type, "openarm")
                    self.assertEqual(infer.get("action_space"), "joint")

    def test_standalone_eval_defaults_to_openpi_ee_profile(self):
        args = build_benchmark_eval_parser().parse_args([])
        self.assertEqual(
            Path(args.policy_profile).name,
            "openpi_franka_ee_pos_v1.yaml",
        )

    def test_standalone_eval_default_budget_is_episode_steps(self):
        args = build_benchmark_eval_parser().parse_args([])
        self.assertEqual(args.max_steps, 500)
        self.assertIsNone(args.sim_steps_per_action)


if __name__ == "__main__":
    unittest.main()
