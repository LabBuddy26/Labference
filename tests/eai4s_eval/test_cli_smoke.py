from __future__ import annotations

import unittest

from eai4s_eval.cli.run_pick_lift_eval import build_parser
from eai4s_eval.cli.run_tube_rack_insert_eval import build_parser as build_t03_parser


class CliSmokeTest(unittest.TestCase):
    def test_parser_defaults_do_not_conflict_with_app_launcher_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.model_name, "eai4s_model")
        self.assertIsNone(args.checkpoint)
        self.assertEqual(args.robot, "franka_panda")
        self.assertEqual(args.policy_api, "auto")
        self.assertTrue(str(args.task_spec).endswith("T02_franka_pick_lift_proxy.yaml"))

        option_strings = {option for action in parser._actions for option in action.option_strings}
        self.assertNotIn("--device", option_strings)
        self.assertNotIn("--headless", option_strings)
        self.assertNotIn("--enable_cameras", option_strings)

    def test_parser_accepts_smoke_flags(self) -> None:
        args = build_parser().parse_args(
            [
                "--all-splits",
                "--save_video",
                "--policy_api",
                "predict_action",
                "--num_episodes",
                "2",
                "--checkpoint",
                "ckpt_smoke",
            ]
        )
        self.assertTrue(args.all_splits)
        self.assertTrue(args.save_video)
        self.assertEqual(args.policy_api, "predict_action")
        self.assertEqual(args.num_episodes, 2)
        self.assertEqual(args.checkpoint, "ckpt_smoke")

    def test_t03_parser_defaults_to_tube_rack_task(self) -> None:
        args = build_t03_parser().parse_args([])
        self.assertTrue(str(args.task_spec).endswith("T03_tube_insert_rack_draft.yaml"))
        self.assertEqual(args.instruction, "place the glass tube in the rack")
        self.assertEqual(args.target_object, "tube")
        self.assertEqual(args.max_steps, 300)
        self.assertEqual(args.sim_steps_per_action, 6)
        self.assertEqual(args.reaction_interpolation_steps, 1)

    def test_t03_parser_allows_user_overrides(self) -> None:
        args = build_t03_parser().parse_args(["--instruction", "custom tube task", "--max_steps", "42"])
        self.assertEqual(args.instruction, "custom tube task")
        self.assertEqual(args.max_steps, 42)


if __name__ == "__main__":
    unittest.main()
