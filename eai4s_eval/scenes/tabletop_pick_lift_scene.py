from __future__ import annotations

from typing import Any

from eai4s_eval.utils.assets import pick_lift_asset_paths
from eai4s_eval.scenes.runtime import RuntimeResetResult, SceneRuntime, split_xy_range


def _camera_quat_ros(opencv_quat: Any) -> Any:
    """Convert an OpenCV-style quaternion to ROS convention.

    Parameters
    ----------
    opencv_quat:
        A sequence of 4 floats representing a quaternion in OpenCV convention
        (w, x, y, z).  May be a list, tuple, numpy array, or torch tensor.

    Returns
    -------
    torch.Tensor
        A 1-D tensor of 4 floats in ROS convention, detached and cloned so the
        result does not share memory with any upstream autograd graph.
    """
    import torch
    from isaaclab.utils.math import convert_camera_frame_orientation_convention

    if torch.is_tensor(opencv_quat):
        quat = opencv_quat.detach().clone().to(dtype=torch.float32)
    else:
        quat = torch.tensor(opencv_quat, dtype=torch.float32)
    if quat.ndim == 0:
        quat = quat.unsqueeze(0)
    return convert_camera_frame_orientation_convention(
        quat.unsqueeze(0),
        origin="opencv",
        target="ros",
    )[0].detach().clone()


def build_tabletop_pick_lift_scene_cfg(*, robot_adapter: Any, asset_root: str | None, num_envs: int) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    assets = pick_lift_asset_paths(asset_root, must_exist=True)
    ros_quat = _camera_quat_ros([0.5, 0.5, 0.5, 0.5])
    ros_quat_wrist = _camera_quat_ros([-0.27, -0.65, -0.65, -0.27])

    @configclass
    class TableTopPickLiftSceneCfg(InteractiveSceneCfg):
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
        shelf = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/shelf",
            spawn=sim_utils.UsdFileCfg(
                usd_path=assets.shelf,
                scale=(0.4, 0.4, 0.4),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    disable_gravity=False,
                    kinematic_enabled=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.2, density=None),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.0015,
                    rest_offset=0.0003,
                    torsional_patch_radius=0.02,
                    min_torsional_patch_radius=0.005,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, -0.2, 0.1), rot=(0.5, 0.5, 0.5, 0.5)),
        )
        tube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/tube",
            spawn=sim_utils.UsdFileCfg(
                usd_path=assets.tube,
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.1, 0.1), rot=(0.5, 0.5, 0.5, 0.5)),
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.1, 0.0), rot=(0.7071, 0.7071, 0.0, 0.0)),
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

    return TableTopPickLiftSceneCfg(num_envs=num_envs, env_spacing=2.0)


class PickLiftSceneRuntime(SceneRuntime):
    scene_id = "tabletop_pick_lift_proxy"
    target_object_name = "vial"
    distractor_names = ("shelf", "tube", "beaker", "boiling_flask", "Erlenmeyer_flask")

    def build_scene_cfg(self, *, robot_adapter: Any, asset_root: str | None, num_envs: int) -> Any:
        return build_tabletop_pick_lift_scene_cfg(
            robot_adapter=robot_adapter,
            asset_root=asset_root,
            num_envs=num_envs,
        )

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
        noise_xy = rng.uniform(-split_xy_range(split), split_xy_range(split), size=2)
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
        if split == "sim_randomized":
            for name, obj in distractor_objects.items():
                orig_pos = distractor_original_pos[name]
                dist_noise = rng.uniform(-0.03, 0.03, size=2)
                dist_new_pos = orig_pos.clone()
                dist_new_pos[0] += float(dist_noise[0])
                dist_new_pos[1] += float(dist_noise[1])
                dist_quat = obj.data.root_state_w[0, 3:7].clone()
                obj.write_root_pose_to_sim(torch.cat([dist_new_pos, dist_quat], dim=0).unsqueeze(0))
        if hasattr(target_object, "write_root_velocity_to_sim"):
            target_object.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        for _ in range(8):
            step_sim()
        return RuntimeResetResult(
            target_object=target_object,
            original_object_pos=original_object_pos,
            original_object_quat=original_object_quat,
        )
