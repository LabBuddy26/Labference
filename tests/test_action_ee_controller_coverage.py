from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ActionEEControllerCoverageTest(unittest.TestCase):
    def test_franka_controllers_use_the_shared_episode_writer(self):
        controller_root = ROOT / "controllers"
        offenders = []
        for path in sorted(controller_root.glob("*_controller.py")):
            if path.name in {"base_controller.py", "openarm_pick_controller.py"}:
                continue
            source = path.read_text(encoding="utf-8-sig")
            if "cache_collection_step(" not in source:
                continue
            if "data_collector.write_cached_data(" in source:
                offenders.append(path.name)
        self.assertEqual(offenders, [])

    def test_shared_collection_path_always_supplies_franka_ee(self):
        source = (ROOT / "controllers" / "base_controller.py").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn('if ee_pose is None and self._robot_type == "franka"', source)
        self.assertIn("final_ee_pose=self.transform_ee_pose", source)


if __name__ == "__main__":
    unittest.main()
