from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]


def test_hard_centrifuge_collection_and_eval_are_configured():
    cfg = OmegaConf.load(ROOT / "config/level3_L07_hard_centrifuge_load.yaml")
    assert cfg.task_type == "centrifugeload_H"
    assert cfg.controller_type == "centrifuge_hard_load"
    assert cfg.infer.action_space == "ee_pos"
    assert cfg.collector.sample_hz == 10
    assert cfg.cameras[-1].name == "camera_3"
    assert list(cfg.cameras[-1].local_translation) == [-0.20, 0.0, -0.01]
    assert (ROOT / cfg.evaluation.task_spec).is_file()


def test_easy_centrifuge_uses_new_asset_and_preloaded_tube():
    cfg = OmegaConf.load(ROOT / "config/level2_L07_centrifuge_load.yaml")
    centrifuge = next(item for item in cfg.scene if item.prim_path == "/World/Centrifugal")
    assert centrifuge.usd_path.endswith("CentrifugalMachine_mesh.usd")
    assert list(cfg.preloaded_position) == [0.0627004, 0.36812646, 0.82909817]
    assert [item.path for item in cfg.task.obj_paths] == ["/World/NMR_01"]


def test_hard_controller_uses_aligned_collection_api():
    source = (ROOT / "controllers/centrifuge_hard_load_controller.py").read_text()
    base_source = (ROOT / "controllers/multi_object_task_controller.py").read_text()
    assert "MultiObjectTaskController" in source
    assert "cache_collection_step(state, action)" in base_source
    assert "write_cached_episode(state)" in base_source
    assert "write_cached_data" not in source
