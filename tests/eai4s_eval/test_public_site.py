from __future__ import annotations

import json
import re
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from eai4s_eval.cli.export_public_site import export_public_site
from eai4s_eval.public_site import (
    build_public_leaderboard,
    build_public_snapshot,
    load_asset_licenses,
    load_public_catalogs,
    load_public_site_config,
)
from eai4s_eval.server.app import create_app


class PublicCatalogTest(unittest.TestCase):
    def test_asset_preview_is_hidden_when_audited_source_fingerprint_changes(self) -> None:
        from eai4s_eval.public_site import PUBLIC_IMAGE_AUDIT_PATH
        audit = json.loads(PUBLIC_IMAGE_AUDIT_PATH.read_text())
        audit["objects"]["beaker"]["source_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "image-audit.json"
            manifest.write_text(json.dumps(audit))
            with patch("eai4s_eval.public_site.PUBLIC_IMAGE_AUDIT_PATH", manifest):
                _, catalog = load_public_catalogs()
                beaker = next(item for item in catalog["objects"] if item["object_id"] == "beaker")
                self.assertIsNone(beaker["media"])

    def test_generated_proxy_images_are_hidden_in_all_public_views(self) -> None:
        _, catalog = load_public_catalogs()
        by_id = {item["object_id"]: item for item in catalog["objects"]}
        generated_ids = {"pipette", "pipette-tip-box", "waste-box", "well-plate", "button-panel"}
        with tempfile.TemporaryDirectory() as directory:
            client = create_app(outputs_root=Path(directory)).test_client()
            pages = [client.get("/").get_data(as_text=True),
                     client.get("/objects/").get_data(as_text=True)]
            for object_id in generated_ids:
                self.assertEqual(by_id[object_id]["image_kind"], "generated_concept")
                self.assertIsNone(by_id[object_id]["media"])
                pages.append(client.get(f"/objects/{object_id}/").get_data(as_text=True))
                for page in pages:
                    self.assertNotIn(f"object_images/{object_id}.webp", page)
            self.assertIsNotNone(by_id["beaker"]["media"])

    def test_asset_licence_registry_is_fail_closed(self) -> None:
        registry = load_asset_licenses()
        self.assertEqual(registry["policy"]["software_license"], "MIT")
        defaults = registry["default_record"]
        for override in registry["objects"].values():
            record = {**defaults, **override}
            self.assertEqual(record["review_status"], "pending")
            self.assertEqual(record["spdx_id"], "NOASSERTION")
            self.assertFalse(record["source_redistributable"])
            self.assertFalse(record["derivatives_redistributable"])

    def test_catalogs_are_complete_and_cross_linked(self) -> None:
        catalog, objects = load_public_catalogs()
        self.assertEqual(len(catalog["tasks"]), 24)
        self.assertEqual(
            {task["task_id"] for task in catalog["tasks"]},
            {f"{prefix}{index:02d}" for prefix in "LR" for index in range(1, 13)},
        )
        self.assertEqual(sum(task["availability"] == "available" for task in catalog["tasks"]), 24)
        object_ids = {item["object_id"] for item in objects["objects"]}
        self.assertEqual(len(object_ids), len(objects["objects"]))
        self.assertIn("beaker", object_ids)
        self.assertNotIn("franka", object_ids)
        self.assertNotIn("rubber-stopper", object_ids)
        self.assertIn("pipette", object_ids)
        self.assertIn("pipette-tip-box", object_ids)
        self.assertIn("well-plate", object_ids)
        self.assertIn("waste-box", object_ids)
        self.assertIn("button-panel", object_ids)
        for task in catalog["tasks"]:
            self.assertTrue(task["name_en"])
            self.assertNotIn("name_zh", task)
        self.assertIn("button-panel", {
            object_id for task in catalog["tasks"] for object_id in task["object_ids"]
        })
        demos = [demo for task in catalog["tasks"] for demo in task["media"]["demos"]]
        if load_public_site_config().get("public_media_enabled", True):
            self.assertEqual(len(demos), 18)
            self.assertEqual(
                {task["task_id"] for task in catalog["tasks"] if task["media"]["demos"]},
                {
                    "L01", "L02", "L04", "L06", "L08", "L09", "L10", "L11", "L12",
                    "R01", "R02", "R03", "R04", "R05", "R08", "R09", "R10", "R11",
                },
            )
            self.assertTrue(all(len(demo["views"]) == 2 for demo in demos))
        else:
            self.assertEqual(demos, [])
        static_root = Path(__file__).resolve().parents[2] / "eai4s_eval" / "server" / "static"
        for item in objects["objects"]:
            self.assertNotIn("asset_paths", item)
            self.assertNotIn("asset_distribution", item)
            self.assertNotIn("public_render_rotation_deg", item)
            self.assertNotIn("license", item)
            self.assertNotIn("distribution", item)
            if item["media"]:
                self.assertTrue((static_root / item["media"]).is_file(), item["object_id"])
        self.assertNotIn("license_policy", objects)
        self.assertNotIn("media_licenses", objects)
        public_payload = json.dumps(
            {"catalog": catalog, "objects": objects}, ensure_ascii=False
        )
        self.assertIsNone(re.search(r"[\u4e00-\u9fff]", public_payload))

    def test_public_snapshot_normalizes_dataset_task_names_to_english(self) -> None:
        snapshot = build_public_snapshot(
            [],
            dataset_statistics={
                "simulation": {
                    "rows": [
                        {"task_id": "L01", "task_name": "中文任务名"},
                        {"task_id": "legacy_task", "task_name": "旧任务"},
                    ]
                }
            },
        )
        rows = snapshot["dataset_statistics"]["simulation"]["rows"]
        self.assertEqual(rows[0]["task_name"], "Transport Three Sample Vials")
        self.assertEqual(rows[1]["task_name"], "legacy_task")
        leaderboard_payload = json.dumps(
            snapshot["leaderboards"], ensure_ascii=False
        )
        self.assertIsNone(re.search(r"[\u4e00-\u9fff]", leaderboard_payload))

    def test_empty_registry_keeps_local_runs_out_of_public_board(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "episode_metrics.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": "L01",
                        "success": True,
                        "task_score_100": 95.0,
                        "quality_metrics": {
                            "geometry_quality_score_100": 95,
                            "force_quality_score_100": 90,
                            "efficiency_score_100": 80,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            run = {
                "run_id": "local-debug-run",
                "run_dir": str(run_dir),
                "model_name": "debug-policy",
                "checkpoint": "debug",
                "split": "sim_randomized",
            }
            empty = build_public_leaderboard(
                [run],
                split="sim_randomized",
                registry={"schema_version": "labbench_official_submissions/v1", "submissions": []},
            )
            self.assertEqual(empty["models"], [])

            verified = build_public_leaderboard(
                [run],
                split="sim_randomized",
                registry={
                    "schema_version": "labbench_official_submissions/v1",
                    "submissions": [
                        {
                            "submission_id": "verified-debug-fixture",
                            "status": "verified",
                            "model_name": "debug-policy",
                            "checkpoint": "debug",
                            "run_ids": ["local-debug-run"],
                        }
                    ],
                },
            )
            self.assertEqual(len(verified["models"]), 1)
            self.assertEqual(verified["models"][0]["overall_score_100"], 95.0)

            provider_rejected = build_public_leaderboard(
                [{**run, "policy_provider": "openpi", "suite_id": "suite-v1"}],
                split="sim_randomized",
                registry={
                    "schema_version": "labbench_official_submissions/v1",
                    "submissions": [
                        {
                            "submission_id": "verified-provider-fixture",
                            "status": "verified",
                            "model_name": "debug-policy",
                            "checkpoint": "debug",
                            "policy_provider": "gr00t",
                            "suite_id": "suite-v1",
                            "splits": ["sim_randomized"],
                            "run_ids": ["local-debug-run"],
                        }
                    ],
                },
            )
            self.assertEqual(provider_rejected["models"], [])

            forged = dict(run, model_name="not-the-verified-model")
            rejected = build_public_leaderboard(
                [forged],
                split="sim_randomized",
                registry={
                    "schema_version": "labbench_official_submissions/v1",
                    "submissions": [
                        {
                            "submission_id": "verified-debug-fixture",
                            "status": "verified",
                            "model_name": "debug-policy",
                            "checkpoint": "debug",
                            "run_ids": ["local-debug-run"],
                        }
                    ],
                },
            )
            self.assertEqual(rejected["models"], [])


class PublicSiteRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.outputs = self.root / "runs"
        self.statistics = self.root / "missing-statistics.json"
        self.app = create_app(
            outputs_root=self.outputs,
            dataset_statistics_path=self.statistics,
        )
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_project_resources_and_institution_logo(self) -> None:
        home = self.client.get("/").get_data(as_text=True)
        self.assertIn('data-resource="paper" href="https://arxiv.org/abs/2610.35252"', home)
        self.assertIn('data-resource="code" href="https://github.com/LabBuddy26/Labference"', home)
        self.assertIn('data-resource="homepage" href="https://labbuddy26.github.io/Labference/"', home)
        self.assertIn('data-resource="dataset" href="https://pan.baidu.com/', home)
        self.assertIn("中国科学院自动化研究所", home)
        self.assertIn("中国科学院大学", home)
        self.assertIn("University of Chinese Academy of Sciences logo", home)
        with self.client.get("/static/web_assets/brand/ucas-logo-v2.webp") as response:
            self.assertEqual(response.status_code, 200)
        with self.client.get("/static/web_assets/brand/casia-logo-v2.png") as response:
            self.assertEqual(response.status_code, 200)
        config = load_public_site_config()
        config["links"].update({
            "paper": "https://example.org/labbench-paper",
            "code": "https://example.org/labbench-code",
            "homepage": "https://example.org/labbench-home",
        })
        with patch("eai4s_eval.public_site.load_public_site_config", return_value=config):
            configured = self.client.get("/").get_data(as_text=True)
        for key in ("paper", "code", "homepage"):
            self.assertIn(f'data-resource="{key}" href="{config["links"][key]}"', configured)
            self.assertNotIn(f'data-resource="{key}" aria-disabled="true"', configured)

    def test_huggingface_resources_can_be_enabled_independently(self) -> None:
        home = self.client.get('/').get_data(as_text=True)
        self.assertIn('data-resource="huggingface" aria-disabled="true"', home)
        config = load_public_site_config()
        config['links'].update(huggingface='https://huggingface.co/collections/example/labference',
                               hf_dataset='https://huggingface.co/datasets/example/Labference')
        with patch('eai4s_eval.public_site.load_public_site_config', return_value=config):
            home = self.client.get('/').get_data(as_text=True)
            data = self.client.get('/dataset/').get_data(as_text=True)
        self.assertIn('data-resource="huggingface" href="https://huggingface.co/collections/example/labference"', home)
        self.assertIn('href="https://huggingface.co/datasets/example/Labference"', data)
        self.assertIn('Simulation resources · Coming soon', data)
        self.assertIn('Models · Coming soon', data)

    def test_public_pages_and_console_render(self) -> None:
        expected_markers = {
            "/": "Benchmarking the Reasoning–Action Connection in Robotic Laboratories".encode(),
            "/tasks/": b"4 special workflows + 24 ordinary tasks",
            "/tasks/L01/": b"Transport Three Sample Vials",
            "/objects/": b"The tools and vessels behind every task",
            "/objects/beaker/": b"Laboratory Beaker",
            "/dataset/": b"Simulation data, frame by frame",
            "/leaderboard/": b"Four workflows. One transparent score.",
            "/docs/": b"Build, collect and evaluate with Labference",
            "/console": b"EAI4S-bench",
        }
        for route, marker in expected_markers.items():
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, route)
            self.assertIn(marker, response.data, route)
            english_content = response.get_data(as_text=True)
            for partner in load_public_site_config().get("partners", []):
                english_content = english_content.replace(partner["name"], "")
            self.assertIsNone(re.search(r"[\u4e00-\u9fff]", english_content), route)
        task_detail = self.client.get("/tasks/L01/").get_data(as_text=True)
        media_enabled = self.client.get("/api/public/site").get_json()["site"].get(
            "public_media_enabled", True
        )
        if media_enabled:
            self.assertIn("data-autoplay-video", task_detail)
            self.assertIn("Front camera", task_detail)
        leaderboard_template = (
            Path(__file__).resolve().parents[2]
            / "eai4s_eval/server/templates/public_leaderboard.html"
        ).read_text(encoding="utf-8")
        self.assertIn("data-leaderboard-detail-toggle", leaderboard_template)
        self.assertIn("Public details contain aggregate metrics only", leaderboard_template)
        self.assertNotIn("Atomic-action aggregates", leaderboard_template)
        if media_enabled:
            self.assertIn("Wrist camera", task_detail)
        self.assertIn('class="page-back" href="/tasks/"', task_detail)
        object_detail = self.client.get("/objects/beaker/").get_data(as_text=True)
        self.assertIn('class="page-back" href="/objects/"', object_detail)
        self.assertIn("Metadata only", object_detail)
        self.assertNotIn("Asset licensing", object_detail)
        self.assertNotIn("NOASSERTION", object_detail)
        self.assertEqual(self.client.get("/licenses/").status_code, 404)
        dataset = self.client.get("/dataset/").get_data(as_text=True)
        self.assertIn("CC-BY-4.0", dataset)
        self.assertIn("creativecommons.org/licenses/by/4.0/", dataset)
        self.assertIn("pan.baidu.com/s/1zyJL3zxHqI_j67hUkUF6lw", dataset)
        self.assertNotIn("pwd=", dataset)
        self.assertNotIn("gtc4", dataset)
        self.assertIn("Download the Labference dataset", dataset)
        self.assertNotIn("Extraction code", dataset)
        self.assertEqual(self.client.get("/objects/franka/").status_code, 404)
        self.assertEqual(self.client.get("/objects/pipette/").status_code, 200)
        self.assertEqual(self.client.get("/objects/well-plate/").status_code, 200)
        self.assertEqual(self.client.get("/objects/button-panel/").status_code, 200)
        home = self.client.get("/").get_data(as_text=True)
        if self.client.get("/api/public/site").get_json()["site"].get(
            "public_media_enabled", True
        ):
            self.assertNotIn("web_assets/hero/labbench-family-assets-v2.webp", home)
            self.assertIn("web_assets/brand/eai4s-labbench-mark.webp", home)
        self.assertNotIn('data-featured-track=', home)
        self.assertNotIn('data-demo-card', home)
        self.assertEqual(home.count('data-special-family='), 4)
        for anchor in (
            "abstract", "special-tasks", "dataset", "leaderboard", "objects", "citation"
        ):
            self.assertIn(f'href="#{anchor}"', home)
            self.assertIn(f'id="{anchor}"', home)
        self.assertNotIn('data-filter="all"', home)
        if media_enabled:
            self.assertEqual(home.count('<video '), 2)
        self.assertIn('aria-live="polite"', home)
        self.assertIn('aria-label="Back to top"', home)
        self.assertNotIn(r"howpublished={\url{https://github.com/}}", home)
        if media_enabled:
            for task_id in ("R01", "R02", "R04", "R11"):
                detail = self.client.get(f"/tasks/{task_id}/").get_data(as_text=True)
                self.assertIn("data-autoplay-video", detail)
                self.assertIn("Front camera", detail)
                self.assertIn("Wrist camera", detail)
        self.assertNotIn("data-hover-video", self.client.get("/tasks/L07/").get_data(as_text=True))

    def test_dataset_statistics_map_to_catalog_ids(self) -> None:
        fixture = (
            Path(__file__).resolve().parents[2]
            / "eai4s_eval" / "server" / "static" / "web_assets" / "dataset_stats_20260826.json"
        )
        if not fixture.is_file():
            self.skipTest("published dataset fixture is not included in this release")
        app = create_app(outputs_root=self.outputs, dataset_statistics_path=fixture)
        rows = app.test_client().get("/api/public/site").get_json()["dataset_statistics"]["simulation"]["rows"]
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(row["mapped"] for row in rows))
        self.assertTrue(all(row["fps"] == 10 for row in rows))
        catalog_ids = {row["task_id"] for row in rows if not row["task_id"].startswith("skill:")}
        self.assertEqual(
            catalog_ids,
            {
                "L01", "L02", "L04", "L06", "L08", "L09", "L10", "L11", "L12",
                "R01", "R03", "R04", "R05", "R08", "R09", "R10", "R11",
            },
        )
        skill_ids = {row["task_id"] for row in rows if row["task_id"].startswith("skill:")}
        self.assertEqual(
            skill_ids,
            {"skill:close", "skill:grasp", "skill:open", "skill:place", "skill:pour", "skill:stir"},
        )
        # Catalog rows carry the official English task name instead of the raw
        # lerobot instruction string; skill rows keep their instruction label.
        self.assertIn("Transport Three Sample Vials", {row["task_name"] for row in rows})
        self.assertIn(
            "Close the laboratory device",
            [row["task_name"] for row in rows if row["task_id"] == "skill:close"],
        )

    def test_public_api_contract_and_errors(self) -> None:
        snapshot = self.client.get("/api/public/site").get_json()
        self.assertEqual(snapshot["schema_version"], "labbench_public_snapshot/v1")
        self.assertEqual(snapshot["summary"]["task_count"], 24)
        self.assertEqual(snapshot["summary"]["available_task_count"], 24)
        self.assertEqual(self.client.get("/api/public/tasks/F01").status_code, 404)
        self.assertEqual(self.client.get("/api/public/tasks/L01").status_code, 200)
        self.assertEqual(self.client.get("/api/public/tasks/nope").status_code, 404)
        self.assertEqual(self.client.get("/api/public/objects/beaker").status_code, 200)
        self.assertEqual(self.client.get("/api/public/leaderboard?split=invalid").status_code, 410)

    def test_console_can_be_disabled(self) -> None:
        app = create_app(outputs_root=self.outputs, console_enabled=False)
        client = app.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/console").status_code, 404)
        self.assertEqual(client.get("/api/eval/jobs").status_code, 404)
        self.assertNotIn(b">Console<", client.get("/").data)
        self.assertEqual(
            client.get("/static/web_assets/object_images/beaker.webp").status_code,
            404,
        )
        snapshot = client.get("/api/public/site").get_json()
        self.assertFalse(snapshot["site"]["public_media_enabled"])
        self.assertTrue(all(item["media"] is None for item in snapshot["objects"]["objects"]))
        self.assertTrue(all(not task["media"]["demos"] for task in snapshot["catalog"]["tasks"]))

    def test_static_export_uses_base_url_and_excludes_console(self) -> None:
        output = self.root / "site-dist"
        export_public_site(
            output_dir=output,
            outputs_root=self.outputs,
            dataset_statistics_path=self.statistics,
            base_url="/LabBench",
        )
        manifest = json.loads((output / "export-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["page_count"], 60)
        self.assertFalse(manifest["console_included"])
        self.assertFalse(manifest["public_media_included"])
        self.assertEqual(manifest["media_file_count"], 0)
        self.assertEqual(manifest["media_bytes"], 0)
        self.assertFalse((output / "console").exists())
        home = (output / "index.html").read_text(encoding="utf-8")
        self.assertIn('/LabBench/tasks/', home)
        self.assertIn('/LabBench/static/public-site.css', home)
        self.assertIn('data-resource="homepage" href="https://labbuddy26.github.io/Labference/"', home)
        self.assertIn('class="institution-text">CASIA', home)
        self.assertIn('class="institution-text">UCAS', home)
        self.assertNotIn('src="/LabBench/static/web_assets/brand/casia-logo-v2.png"', home)
        self.assertTrue((output / "tasks" / "L01" / "index.html").is_file())
        self.assertTrue((output / "objects" / "beaker" / "index.html").is_file())
        self.assertFalse((output / "licenses").exists())
        self.assertFalse((output / "static" / "web_assets").exists())
        dataset = (output / "dataset" / "index.html").read_text(encoding="utf-8")
        self.assertIn("CC-BY-4.0", dataset)
        self.assertIn("pan.baidu.com/s/1zyJL3zxHqI_j67hUkUF6lw", dataset)
        self.assertNotIn("pwd=", dataset)
        self.assertNotIn("gtc4", dataset)
        self.assertIn("Download the Labference dataset", dataset)
        self.assertNotIn("Extraction code", dataset)

    def test_static_export_can_include_only_referenced_public_media(self) -> None:
        output = self.root / "site-dist-with-media"
        export_public_site(
            output_dir=output,
            outputs_root=self.outputs,
            dataset_statistics_path=self.statistics,
            base_url="/LabBench",
            include_public_media=True,
        )
        manifest = json.loads((output / "export-manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["public_media_included"])
        self.assertGreater(manifest["media_file_count"], 0)
        self.assertGreater(manifest["media_bytes"], 0)
        media_root = output / "static" / "web_assets"
        self.assertTrue((media_root / "brand" / "eai4s-labbench-mark.webp").is_file())
        self.assertTrue((media_root / "brand" / "casia-logo-v2.png").is_file())
        self.assertTrue((media_root / "brand" / "ucas-logo-v2.webp").is_file())
        self.assertFalse((media_root / "hero" / "labbench-family-assets-v2.webp").exists())
        self.assertFalse((media_root / "hero" / "labbench-family.webp").exists())
        for object_id in ("pipette", "pipette-tip-box", "waste-box", "well-plate", "button-panel"):
            self.assertFalse((media_root / "object_images" / f"{object_id}.webp").exists())
        self.assertTrue((media_root / "object_images" / "beaker.webp").is_file())
        self.assertTrue((media_root / "task_media" / "l01-sample-transport-front.mp4").is_file())
        self.assertFalse((media_root / "brand" / "eai4s-labbench-mark.png").exists())
        self.assertFalse((media_root / "hero" / "labbench-family.png").exists())
        self.assertFalse((media_root / "source_videos").exists())
        self.assertFalse((media_root / "dataset_stats_20260826.json").exists())
        media_files = [path for path in media_root.rglob("*") if path.is_file()]
        self.assertEqual(len(media_files), manifest["media_file_count"])
        self.assertEqual(sum(path.stat().st_size for path in media_files), manifest["media_bytes"])
        self.assertTrue(all(path.suffix.lower() in {".webp", ".mp4", ".jpg", ".jpeg", ".png"} for path in media_files))


if __name__ == "__main__":
    unittest.main()
