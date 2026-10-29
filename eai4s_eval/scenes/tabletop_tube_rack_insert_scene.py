from __future__ import annotations

from typing import Any

import numpy as np

from eai4s_eval.scenes.runtime import RuntimeResetResult, SceneRuntime, split_xy_range
from eai4s_eval.scenes.tabletop_pick_lift_scene import _camera_quat_ros
from eai4s_eval.utils.assets import tube_rack_insert_asset_paths


TARGET_SLOT_OFFSET_M = np.array([0.031, 0.031, 0.08], dtype=np.float64)


def compute_tube_rack_insert_metrics(
    *,
    tube_pos: Any,
    tube_quat_wxyz: Any,
    rack_pos: Any,
    slot_offset_m: Any = TARGET_SLOT_OFFSET_M,
) -> dict[str, Any]:
    """Compute T03 geometry metrics from simulator truth state."""
    from eai4s_eval.utils.math import normalize_quat_wxyz

    tube_pos_np = np.asarray(tube_pos, dtype=np.float64)[:3]
    rack_pos_np = np.asarray(rack_pos, dtype=np.float64)[:3]
    offset_np = np.asarray(slot_offset_m, dtype=np.float64)[:3]
    target_slot_pos = rack_pos_np + offset_np
    q = normalize_quat_wxyz(tube_quat_wxyz)
    w, x, y, z = q
    # Rotate local +Z by a wxyz quaternion. This matches the data collection
    # script's quat_rotate(tube_quat, [0, 0, 1]) check.
    tube_z = np.array(
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=np.float64,
    )
    alignment_cosine = float(abs(np.dot(tube_z, np.array([0.0, 0.0, 1.0], dtype=np.float64))))
    slot_xy_error_m = float(np.linalg.norm(tube_pos_np[:2] - target_slot_pos[:2]))
    insert_depth_m = float(max(0.0, target_slot_pos[2] - tube_pos_np[2]))
    above_slot_margin_m = float(tube_pos_np[2] - (target_slot_pos[2] - 0.05))
    return {
        "target_slot_pos": target_slot_pos,
        "alignment_cosine": alignment_cosine,
        "slot_xy_error_m": slot_xy_error_m,
        "insert_depth_m": insert_depth_m,
        "above_slot_margin_m": above_slot_margin_m,
        "tube_z_axis": tube_z,
    }


def build_tabletop_tube_rack_insert_scene_cfg(*, robot_adapter: Any, asset_root: str | None, num_envs: int) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    assets = tube_rack_insert_asset_paths(asset_root, must_exist=True)
    ros_quat = _camera_quat_ros([0.5, 0.5, 0.5, 0.5])
    ros_quat_wrist = _camera_quat_ros([-0.27, -0.65, -0.65, -0.27])

    @configclass
    class TableTopTubeRackInsertSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )
        robot = robot_adapter.build_robot_cfg("{ENV_REGEX_NS}/Robot")
        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        )
        rack = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/rack",
            spawn=sim_utils.UsdFileCfg(
                usd_path=assets.rack,
                scale=(1.2e-3, 1.2e-3, 1e-3),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.001,
                    rest_offset=-0.0008,
                    torsional_patch_radius=0.0,
                    min_torsional_patch_radius=0.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, 0.2, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        tube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/tube",
            spawn=sim_utils.UsdFileCfg(
                usd_path=assets.tube,
                scale=(1e-3, 1e-3, 5e-4),
                visual_material=sim_utils.GlassMdlCfg(glass_color=(0.97, 0.97, 0.97), glass_ior=1.52, thin_walled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=False, kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2, density=None),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.0015,
                    rest_offset=0.0003,
                    torsional_patch_radius=0.02,
                    min_torsional_patch_radius=0.005,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.1, 0.1, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        beaker = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/beaker",
            spawn=sim_utils.UsdFileCfg(
                usd_path=assets.beaker,
                scale=(0.15, 0.15, 0.15),
                visual_material=sim_utils.GlassMdlCfg(glass_color=(0.97, 0.97, 0.97), glass_ior=1.52, thin_walled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=False, kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2, density=None),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.0015,
                    rest_offset=0.0003,
                    torsional_patch_radius=0.02,
                    min_torsional_patch_radius=0.005,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, -0.1, 0.0), rot=(0.7071, 0.7071, 0.0, 0.0)),
        )
        vial = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/vial",
            spawn=sim_utils.UsdFileCfg(
                usd_path=assets.vial,
                scale=(0.1, 0.1, 0.1),
                visual_material=sim_utils.GlassMdlCfg(glass_color=(0.97, 0.97, 0.97), glass_ior=1.52, thin_walled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=False, kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2, density=None),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.0015,
                    rest_offset=0.0003,
                    torsional_patch_radius=0.02,
                    min_torsional_patch_radius=0.005,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.1, 0.1), rot=(0.7071, 0.7071, 0.0, 0.0)),
        )
        boiling_flask = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/boiling_flask",
            spawn=sim_utils.UsdFileCfg(
                usd_path=assets.boiling_flask,
                scale=(0.2, 0.2, 0.2),
                visual_material=sim_utils.GlassMdlCfg(glass_color=(0.97, 0.97, 0.97), glass_ior=1.52, thin_walled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=False, kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2, density=None),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.0015,
                    rest_offset=0.0003,
                    torsional_patch_radius=0.02,
                    min_torsional_patch_radius=0.005,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.3, 0.0), rot=(0.7071, 0.7071, 0.0, 0.0)),
        )
        Erlenmeyer_flask = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Erlenmeyer_flask",
            spawn=sim_utils.UsdFileCfg(
                usd_path=assets.erlenmeyer_flask,
                scale=(0.15, 0.15, 0.15),
                visual_material=sim_utils.GlassMdlCfg(glass_color=(0.98, 0.98, 0.99), glass_ior=1.52, thin_walled=True),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=False, kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2, density=None),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.0015,
                    rest_offset=0.0003,
                    torsional_patch_radius=0.02,
                    min_torsional_patch_radius=0.005,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.5, 0.0), rot=(0.7071, 0.7071, 0.0, 0.0)),
        )
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=(1.5, 0.0, 0.25), rot=ros_quat, convention="ros"),
        )
        camera2 = CameraCfg(
            prim_path=robot_adapter.camera_mounts.get("wrist_camera", "{ENV_REGEX_NS}/Robot/panda_hand/wrist_camera"),
            update_period=0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.03), rot=ros_quat_wrist, convention="ros"),
        )

    return TableTopTubeRackInsertSceneCfg(num_envs=num_envs, env_spacing=2.0)


class TubeRackInsertSceneRuntime(SceneRuntime):
    scene_id = "tabletop_tube_rack_insert"
    target_object_name = "tube"
    distractor_names: tuple[str, ...] = ("beaker", "vial", "boiling_flask", "Erlenmeyer_flask")

    def resolve_target_object_name(self, requested_target: str | None = None) -> str:
        if requested_target in {None, "", "vial"}:
            return self.target_object_name
        return requested_target

    def build_scene_cfg(self, *, robot_adapter: Any, asset_root: str | None, num_envs: int) -> Any:
        return build_tabletop_tube_rack_insert_scene_cfg(
            robot_adapter=robot_adapter,
            asset_root=asset_root,
            num_envs=num_envs,
        )

    def after_setup(self, *, scene: Any, sim: Any, device: str) -> None:
        self.rack = scene["rack"]

    def reset_episode(
        self,
        *,
        scene: Any,
        sim: Any,
        robot: Any,
        adapter: Any,
        target_object: Any,
        original_object_pos: Any,
        original_object_quat: Any,
        seed: int,
        split: str,
        device: str,
        step_sim: Any,
        distractor_objects: dict[str, Any],
        distractor_original_pos: dict[str, Any],
    ) -> RuntimeResetResult:
        import torch
        import numpy as np

        sim.reset()
        adapter.reset_robot(robot)
        rng = np.random.default_rng(seed)
        xy_range = split_xy_range(split)
        noise_xy = rng.uniform(-xy_range, xy_range, size=2)
        new_pos = original_object_pos.clone()
        new_pos[0] += float(noise_xy[0])
        new_pos[1] += float(noise_xy[1])
        if split == "sim_randomized":
            yaw_deg = rng.uniform(-30.0, 30.0)
            yaw_rad = float(np.deg2rad(yaw_deg))
            yaw_quat = np.array([np.cos(yaw_rad / 2.0), 0.0, 0.0, np.sin(yaw_rad / 2.0)], dtype=np.float64)
            from eai4s_eval.utils.math import quat_multiply_wxyz

            orig_quat_np = original_object_quat.detach().cpu().numpy().astype(np.float64)
            new_quat_np = quat_multiply_wxyz(yaw_quat, orig_quat_np)
            new_quat = torch.from_numpy(new_quat_np).float().to(device)
        else:
            new_quat = original_object_quat.clone()
        target_object.write_root_pose_to_sim(torch.cat([new_pos, new_quat], dim=0).unsqueeze(0))
        if hasattr(target_object, "write_root_velocity_to_sim"):
            target_object.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        for _ in range(8):
            step_sim()
        return RuntimeResetResult(
            target_object=target_object,
            original_object_pos=original_object_pos,
            original_object_quat=original_object_quat,
        )

    def build_success_snapshot(
        self,
        *,
        scene: Any,
        target_object: Any,
        initial_object_pos: Any,
        robot: Any,
        resolved_robot: Any,
        step_index: int | None,
        device: str,
    ) -> dict[str, Any]:
        snapshot = super().build_success_snapshot(
            scene=scene,
            target_object=target_object,
            initial_object_pos=initial_object_pos,
            robot=robot,
            resolved_robot=resolved_robot,
            step_index=step_index,
            device=device,
        )
        rack_state = scene["rack"].data.root_state_w[0].detach().cpu().numpy()
        metrics = compute_tube_rack_insert_metrics(
            tube_pos=snapshot["object_pos"],
            tube_quat_wxyz=snapshot["object_quat"],
            rack_pos=rack_state[:3],
        )
        hand_distance = float(np.linalg.norm(np.asarray(snapshot["hand_pos"])[:3] - np.asarray(snapshot["object_pos"])[:3]))
        gripper_mean = float(np.mean(np.asarray(snapshot["gripper_pos"], dtype=np.float64)))
        events = []
        if hand_distance < 0.21 and gripper_mean < 0.035:
            events.append("grasp_tube")
        if metrics["slot_xy_error_m"] < 0.01 and metrics["alignment_cosine"] > 0.92 and metrics["above_slot_margin_m"] > 0.0:
            events.append("align_to_slot")
        if metrics["insert_depth_m"] >= 0.025 and metrics["slot_xy_error_m"] < 0.02:
            events.append("insert_tube")
        snapshot.update(
            {
                "target_slot_pos": metrics["target_slot_pos"],
                "alignment_cosine": metrics["alignment_cosine"],
                "insert_depth_m": metrics["insert_depth_m"],
                "forbidden_collision": False,
                "wrong_object_moved": False,
                "wrong_target_reached": False,
                "max_contact_force_n": 0.0,
                "events": events,
                "task_state": {
                    "target_slot_pos": metrics["target_slot_pos"],
                    "alignment_cosine": metrics["alignment_cosine"],
                    "insert_depth_m": metrics["insert_depth_m"],
                    "slot_xy_error_m": metrics["slot_xy_error_m"],
                    "insertion_position_error_m": metrics["slot_xy_error_m"],
                    "alignment_error_deg": float(np.degrees(np.arccos(min(1.0, max(0.0, metrics["alignment_cosine"]))))),
                    "above_slot_margin_m": metrics["above_slot_margin_m"],
                    "max_contact_force_n": 0.0,
                    "wrong_target_reached": False,
                    "events": events,
                },
            }
        )
        return snapshot

    def build_quality_metrics(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        state = snapshot.get("task_state") or {}
        return {
            "insertion_position_error_m": state.get("slot_xy_error_m"),
            "alignment_error_deg": None if snapshot.get("alignment_cosine") is None else float(np.degrees(np.arccos(min(1.0, max(0.0, snapshot["alignment_cosine"]))))),
            "insertion_depth_m": snapshot.get("insert_depth_m"),
        }
