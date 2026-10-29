from __future__ import annotations

from eai4s_eval.scenes.runtime import SceneRuntime
from eai4s_eval.scenes.tabletop_pick_lift_scene import PickLiftSceneRuntime
from eai4s_eval.scenes.tabletop_tube_rack_insert_scene import TubeRackInsertSceneRuntime


_SCENE_RUNTIMES: dict[str, type[SceneRuntime]] = {
    PickLiftSceneRuntime.scene_id: PickLiftSceneRuntime,
    TubeRackInsertSceneRuntime.scene_id: TubeRackInsertSceneRuntime,
}


def available_scene_ids() -> list[str]:
    return sorted(_SCENE_RUNTIMES)


def get_scene_runtime(scene_id: str | None) -> SceneRuntime:
    resolved = scene_id or PickLiftSceneRuntime.scene_id
    try:
        return _SCENE_RUNTIMES[resolved]()
    except KeyError as exc:
        known = ", ".join(available_scene_ids())
        raise ValueError(f"Unknown scene_id '{resolved}'. Available scene runtimes: {known}") from exc
