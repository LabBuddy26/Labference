from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eai4s_eval.public_site import build_release_leaderboard, load_release_protocols
from eai4s_eval.server.app import create_app


class ReleasePublicSiteTest(unittest.TestCase):
    def test_old_rack_version_is_not_ranked_with_v4(self) -> None:
        board = build_release_leaderboard([{
            "protocol_id": "rack_same_cap", "task_version": "rack-placement-recall/v3",
            "ranking_eligible": True, "model_name": "old-model", "checkpoint": "old",
            "scores": {"valid_success": True},
        }])
        self.assertEqual(board["models"], [])
    def test_release_catalog_contains_eight_independent_protocols(self) -> None:
        catalog = load_release_protocols()
        self.assertEqual(len(catalog["protocols"]), 8)
        self.assertEqual(catalog["protocols"][0]["camera_streams"], ["camera_1_rgb", "camera_3_rgb"])
        self.assertIn("actions_ee", catalog["protocols"][0]["action_streams"])
        self.assertIn("display_rgb", catalog["protocols"][2]["camera_streams"])
        self.assertEqual(catalog["protocols"][-1]["task_version"], "device-operation-cycle/v1")

    def test_release_board_keeps_reference_runs_out_of_model_rank(self) -> None:
        runs = [
            {"protocol_id": "centrifuge_easy", "model_name": "Reference executor", "checkpoint": "reference", "ranking_eligible": False, "scores": {"valid_success": True}},
            {"protocol_id": "centrifuge_easy", "model_name": "OpenPI", "checkpoint": "ckpt-1", "ranking_eligible": True, "evaluation_valid": True, "scores": {"valid_success": True, "decision_pass": True}},
        ]
        board = build_release_leaderboard(runs)
        self.assertEqual(board["run_count"], 2)
        self.assertEqual(len(board["models"]), 1)
        self.assertEqual(board["models"][0]["task_coverage"]["evaluated"], 1)
        self.assertEqual(board["models"][0]["overall_score_100"], 100.0)

    def test_release_run_is_visible_in_public_api(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "centrifuge_easy_openpi"
            run.mkdir()
            (run / "run.json").write_text(json.dumps({
                "task_version": "centrifuge-opposite/v2",
                "difficulty": "hard",
                "policy_interface": "openpi_array",
                "policy_id": "openpi",
                "policy_revision": "ckpt-1",
                "ranking_eligible": True,
                "runtime_status": "finished",
            }), encoding="utf-8")
            (run / "scores.json").write_text(json.dumps({
                "evaluation_valid": True,
                "valid_success": True,
            }), encoding="utf-8")
            app = create_app(outputs_root=root, dataset_statistics_path=root / "missing.json")
            client = app.test_client()
            payload = client.get("/api/public/release-leaderboard").get_json()
            self.assertEqual(payload["run_count"], 1)
            self.assertEqual(payload["models"][0]["model_name"], "openpi")
            self.assertEqual(payload["models"][0]["family_coverage"], {"evaluated": 1, "total": 4})
            page = client.get("/leaderboard/").get_data(as_text=True)
            self.assertIn('id="release-leaderboard"', page)
            self.assertNotIn('id="ordinary-leaderboard"', page)
            self.assertEqual(page.count('class="family-result-heading"'), 4)
            self.assertIn("1/4 families", page)
            self.assertEqual(client.get("/tasks/protocol/centrifuge_easy/").status_code, 200)
            self.assertEqual(client.get("/api/public/release-protocols/nope").status_code, 404)

    def test_homepage_shows_four_special_families_but_only_two_verified_main_videos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(outputs_root=Path(directory), dataset_statistics_path=Path(directory) / "missing.json")
            response = app.test_client().get("/")
            self.assertEqual(response.status_code, 200)
            page = response.get_data(as_text=True)
            self.assertEqual(page.count('class="special-family-card"'), 4)
            self.assertIn("Verified successful collection · Main camera · 10 Hz", page)
            self.assertNotIn("labbench-family-assets-v2.webp", page)
            self.assertIn("Centrifuge opposite-slot", page)
            self.assertIn("Feedback weighing", page)
            self.assertIn("Rack placement recall", page)
            self.assertIn("Drying-oven operation cycle", page)
            self.assertEqual(page.count('data-public-status="implemented"'), 2)
            self.assertEqual(page.count('data-public-status="pending"'), 2)
            self.assertNotIn('data-task-card', page)
            self.assertNotIn('data-demo-card', page)
            self.assertIn("weighing-hard-front.mp4", page)
            self.assertIn("device-operation-hard-front.mp4", page)
            self.assertEqual(page.count('<video '), 2)
            self.assertNotIn("rack-recall-hard-front.mp4", page)
            self.assertNotIn("centrifuge-hard-front.mp4", page)


if __name__ == "__main__":
    unittest.main()
