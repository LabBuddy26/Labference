from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eai4s_eval.server.app import create_app


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": "dataset_statistics/v1",
        "generated_at": "2026-07-17T10:30:00+08:00",
        "real_robot_teleoperation": {
            "status": "empty",
            "totals": {"batch_count": 0, "sample_count": 0, "duration_seconds": 0},
            "rows": [],
        },
        "umi": {
            "status": "empty",
            "totals": {"batch_count": 0, "sample_count": 0, "duration_seconds": 0},
            "rows": [],
        },
        "simulation": {
            "status": "empty",
            "totals": {
                "dataset_batch_count": 0,
                "episode_count": 0,
                "frame_count": 0,
                "duration_seconds": 0,
                "scene_count": 0,
            },
            "rows": [],
        },
    }


class DatasetStatisticsApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.outputs_root = self.root / "outputs"
        self.statistics_path = self.root / "dataset_statistics.json"
        try:
            self.app = create_app(
                outputs_root=self.outputs_root,
                dataset_statistics_path=self.statistics_path,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
            return
        self.client = self.app.test_client()

    def _write_json(self, payload: object) -> None:
        self.statistics_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def test_returns_current_payload_and_rereads_file(self) -> None:
        first = _valid_payload()
        self._write_json(first)

        response = self.client.get("/api/dataset_statistics")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), first)
        self.assertIn("no-store", response.headers["Cache-Control"])

        second = _valid_payload()
        second["simulation"] = {
            "status": "active",
            "totals": {
                "dataset_batch_count": 1,
                "episode_count": 20,
                "frame_count": 900,
                "duration_seconds": 30,
                "scene_count": 1,
            },
            "rows": [{
                "simulator": "ISAAC_SIM",
                "robot_type": "FRANKA",
                "task_id": "liquid_pour",
                "task_name": "液体倾倒",
                "dataset_batch_count": 1,
                "episode_count": 20,
                "frame_count": 900,
                "duration_seconds": 30,
                "scene_count": 1,
                "generation_date_start": "2026-07-17",
                "generation_date_end": "2026-07-17",
            }],
        }
        self._write_json(second)

        refreshed = self.client.get("/api/dataset_statistics")
        self.assertEqual(refreshed.status_code, 200)
        self.assertEqual(refreshed.get_json(), second)

    def test_default_statistics_merge_physical_umi_and_simulation(self) -> None:
        app = create_app(outputs_root=self.outputs_root)
        response = app.test_client().get("/api/dataset_statistics")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["real_robot_teleoperation"]["totals"]["sample_count"], 873)
        self.assertEqual(payload["umi"]["totals"]["sample_count"], 743)
        self.assertEqual(payload["simulation"]["totals"]["dataset_batch_count"], 24)
        self.assertEqual(payload["simulation"]["totals"]["episode_count"], 4785)
        self.assertEqual(payload["simulation"]["totals"]["frame_count"], 1177697)

    def test_missing_file_returns_explicit_404(self) -> None:
        response = self.client.get("/api/dataset_statistics")

        self.assertEqual(response.status_code, 404)
        error = response.get_json()["error"]
        self.assertEqual(error["code"], "dataset_statistics_not_found")
        self.assertEqual(Path(error["path"]), self.statistics_path.resolve())

    def test_corrupt_json_returns_explicit_500(self) -> None:
        self.statistics_path.write_text("{broken", encoding="utf-8")

        response = self.client.get("/api/dataset_statistics")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"]["code"], "dataset_statistics_invalid_json")

    def test_incompatible_schema_and_sections_return_422(self) -> None:
        payload = _valid_payload()
        payload["schema_version"] = "dataset_statistics/v0"
        del payload["simulation"]
        self._write_json(payload)

        response = self.client.get("/api/dataset_statistics")

        self.assertEqual(response.status_code, 422)
        error = response.get_json()["error"]
        self.assertEqual(error["code"], "dataset_statistics_schema_incompatible")
        self.assertEqual(error["actual_schema"], "dataset_statistics/v0")
        self.assertEqual(error["missing_or_invalid_sections"], ["simulation"])

    def test_nested_contract_damage_returns_explicit_422(self) -> None:
        payload = _valid_payload()
        payload["real_robot_teleoperation"]["status"] = "active"  # type: ignore[index]
        payload["real_robot_teleoperation"]["rows"] = "not-a-list"  # type: ignore[index]
        self._write_json(payload)

        response = self.client.get("/api/dataset_statistics")

        self.assertEqual(response.status_code, 422)
        error = response.get_json()["error"]
        self.assertEqual(error["code"], "dataset_statistics_schema_incompatible")
        self.assertIn("real_robot_teleoperation.rows", error["invalid_contract_fields"])

    def test_unhashable_nested_values_return_explicit_422(self) -> None:
        payload = _valid_payload()
        payload["real_robot_teleoperation"]["status"] = []  # type: ignore[index]
        self._write_json(payload)

        response = self.client.get("/api/dataset_statistics")

        self.assertEqual(response.status_code, 422)
        error = response.get_json()["error"]
        self.assertIn("real_robot_teleoperation.status", error["invalid_contract_fields"])

        payload = _valid_payload()
        payload["real_robot_teleoperation"] = {
            "status": "active",
            "totals": {"batch_count": 1, "sample_count": 1, "duration_seconds": 1},
            "rows": [
                {
                    "device_model": "UR3",
                    "task_id": "liquid_pour",
                    "task_name": "液体倾倒",
                    "batch_count": 1,
                    "sample_count": 1,
                    "duration_seconds": 1,
                    "collection_date_start": "2026-07-17",
                    "collection_date_end": "2026-07-17",
                    "operator_ids": [{}],
                }
            ],
        }
        self._write_json(payload)

        response = self.client.get("/api/dataset_statistics")

        self.assertEqual(response.status_code, 422)
        error = response.get_json()["error"]
        self.assertIn(
            "real_robot_teleoperation.rows[0].operator_ids",
            error["invalid_contract_fields"],
        )

    def test_statistics_failure_is_isolated_from_existing_endpoints(self) -> None:
        self.statistics_path.write_text("not json", encoding="utf-8")

        self.assertEqual(self.client.get("/api/dataset_statistics").status_code, 500)

        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()["status"], "ok")
        leaderboard = self.client.get("/api/leaderboard")
        self.assertEqual(leaderboard.status_code, 200)
        self.assertEqual(leaderboard.get_json()["rows"], [])


if __name__ == "__main__":
    unittest.main()
