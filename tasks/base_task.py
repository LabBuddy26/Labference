from abc import ABC, abstractmethod
from typing import Dict, Any
import os
from pathlib import Path
import numpy as np
import carb
from isaacsim.sensors.camera import Camera
from utils.object_utils import ObjectUtils
from isaacsim.core.utils.semantics import add_update_semantics
from utils.camera_utils import process_camera_image
from isaacsim.core.utils.prims import set_prim_visibility
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage


from pxr import UsdPhysics, UsdShade, PhysxSchema, UsdGeom, Gf, Sdf, UsdLux


def _resolve_scene_usd_path(usd_path: str, assets_root_path: str | None) -> str | None:
    """Resolve a scene asset USD path regardless of the process working directory.

    The benchmark eval harness runs from a different CWD than the sim repo (it
    only adds the repo to ``sys.path``), so a bare relative ``usd_path`` such as
    ``assets/chemistry_lab/assets/table.usd`` does not resolve and the whole
    scene silently fails to load (empty table, props at the origin, trivial
    success predicates).  Resolution order:
      1. As given (absolute, or relative to the current working directory).
      2. Relative to this repository's own assets root (``<repo>/assets/...``).
      3. Relative to the Isaac Sim assets root.
    Returns the first existing candidate, or ``None`` if none exist.
    """
    candidates = [
        usd_path,
        str(Path(__file__).resolve().parents[1] / usd_path),
    ]
    if assets_root_path:
        candidates.append(str(Path(assets_root_path) / usd_path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


class BaseTask(ABC):
    """
    Base class for all simulation tasks.
    
    Attributes:
        cfg: Task configuration
        world: Simulation world instance
        cameras: Camera settings
        reset_needed (bool): Flag indicating if task needs reset
        frame_idx (int): Current frame index
        object_utils (ObjectUtils): Utility instance for object operations
    """
    def __init__(self, cfg, world, stage, robot):
        """
        Initialize the task.
        
        Args:
            cfg: Task configuration
            world: Simulation world instance
            cameras: Camera settings
        """
        self.cfg = cfg
        self.world = world
        self.stage = stage
        self.robot = robot
        self.reset_needed = False
        self.frame_idx = 0
        # Outcome of the most recent completed episode (True/False), or None
        # when no episode has completed yet (startup-gate rejections and
        # physics stops never set it).  Read by main.py's collect loop to
        # count consecutive failures; cleared on every reset.
        self.episode_outcome = None
        self.object_utils = ObjectUtils.get_instance()
        self.setup_scene()
        self.setup_cameras()
        self.setup_objects()
        self.setup_materials()
        
        self.current_material_idx = 0
        if len(self.obj_configs) != 0:
            self.episodes_per_obj = int(cfg.max_episodes / len(self.obj_configs))
        else:
            self.episodes_per_obj = 0
        self.current_obj_idx = 0
        self.current_obj_episodes = 0
      
    def reset(self) -> None:
        """
        Reset the task state and simulation world.
        """
        self.world.reset()
        self.reset_needed = False
        self.frame_idx = 0
        self.episode_outcome = None
                
        if self.material_config:
            self.apply_material_to_object(self.material_config.path)
            
    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """
        Execute one step of the task.
        
        Returns:
            Dict[str, Any]: Task step information
        """
        pass

    def get_task_info(self) -> Dict[str, Any]:
        """
        Get task-related information.
        
        Returns:
            Dict[str, Any]: Dictionary containing task information
        """
        return {
            "frame_idx": self.frame_idx,
            "reset_needed": self.reset_needed
        }
        
    def need_reset(self) -> bool:
        """
        Check if task needs to be reset.
        
        Returns:
            bool: True if reset is needed, False otherwise
        """
        return self.reset_needed

    def on_task_complete(self, success: bool) -> None:
        """
        Handle task completion logic.
        Update object and material indices.
        """
        self.update_object_and_material_indices(success)
        self.reset_needed = True
    

    def setup_scene(self) -> None:
        """
        Set up scene for the task.
        """
        self.world.scene.add_default_ground_plane()

        # ground_prim = self.stage.GetPrimAtPath("/World/defaultGroundPlane")
        # if ground_prim.IsValid():
        #     ground_prim.GetAttribute("primvar:displayColor").Set([Gf.Vec3f(0.5, 0.5, 0.5)])
        
        self._add_scene_lights()
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        if hasattr(self.cfg, 'scene'):
            for obj in self.cfg.scene:
                if not self.stage.GetPrimAtPath(obj.prim_path).IsValid():
                    usd_path = _resolve_scene_usd_path(obj.usd_path, assets_root_path)
                    if usd_path is None:
                        carb.log_error(
                            f"Scene asset not found for {obj.prim_path}: {obj.usd_path}"
                        )
                        continue

                    add_reference_to_stage(usd_path=usd_path, prim_path=obj.prim_path)
                    obj_position = obj.get("position", [0.0, 0.0, 0.0])
                    obj_orientation = obj.get("orientation", [1.0, 0.0, 0.0, 0.0])
                    obj_scale = obj.get("scale", [1.0, 1.0, 1.0])
                    is_glass = obj.get("glass", False)

                    self.object_utils.set_object_position(object_path=obj.prim_path, position=obj_position)
                    self.object_utils.set_object_orientation(object_path=obj.prim_path, orientation=obj_orientation)
                    self.object_utils.set_object_scale(object_path=obj.prim_path, scale=obj_scale)
                    if is_glass:
                        self.apply_glass_material(obj.prim_path)
                    if hasattr(obj, "visual_color"):
                        self.apply_visual_color(obj.prim_path, obj.visual_color)

        self.apply_joint_overrides()

    def apply_joint_overrides(self) -> None:
        """Apply task-scoped PhysX joint settings after referenced USDs load."""
        attribute_names = {
            "joint_friction": "physxJoint:jointFriction",
            "angular_restitution": "physxLimit:angular:restitution",
            "angular_stiffness": "physxLimit:angular:stiffness",
            "angular_damping": "physxLimit:angular:damping",
        }
        for override in getattr(self.cfg, "joint_overrides", []) or []:
            joint_path = override.get("prim_path")
            joint_prim = self.stage.GetPrimAtPath(joint_path)
            if not joint_prim.IsValid():
                carb.log_warn(f"Joint override ignored; prim not found at {joint_path}")
                continue

            for config_key, attribute_name in attribute_names.items():
                if config_key not in override:
                    continue
                attribute = joint_prim.GetAttribute(attribute_name)
                if not attribute.IsValid():
                    carb.log_warn(
                        f"Joint override ignored; {attribute_name} not found on {joint_path}"
                    )
                    continue
                attribute.Set(float(override[config_key]))


    def setup_cameras(self) -> None:
        """
        Set up cameras for the task.
        """
        self.cameras = []
        mode = str(self.cfg.get("mode", "collect"))
        if mode == "collect":
            camera_frequency = int(self.cfg.collector.get("sample_hz", 60))
        else:
            evaluation_cfg = self.cfg.get("evaluation") or {}
            policy_cfg = evaluation_cfg.get("policy") or {}
            camera_frequency = int(policy_cfg.get("control_hz", 10))
        if camera_frequency not in {10, 60}:
            raise ValueError(
                f"camera frequency must be 10 or 60 Hz, got {camera_frequency}"
            )
        for cam_cfg in self.cfg.cameras:
            camera_exists = self.stage.GetPrimAtPath(cam_cfg.prim_path).IsValid()
            if camera_exists:
                camera = Camera(
                    prim_path=cam_cfg.prim_path,
                    name=cam_cfg.name,
                    frequency=camera_frequency,
                    resolution=tuple(cam_cfg.resolution)
                )
            else:
                if (
                    hasattr(cam_cfg, 'local_translation')
                    or hasattr(cam_cfg, 'local_orientation')
                    # The main-branch Level4 configs identify the wrist
                    # camera by its panda_hand prim path and intentionally do
                    # not provide world translation/orientation fields.
                    or 'panda_hand' in str(cam_cfg.prim_path)
                ):
                    # A wrist camera is created as a child of its link, then positioned below.
                    camera = Camera(
                        prim_path=cam_cfg.prim_path,
                        name=cam_cfg.name,
                        frequency=camera_frequency,
                        resolution=tuple(cam_cfg.resolution)
                    )
                else:
                    # Preserve the historical world-camera construction and USD-axis conversion.
                    camera = Camera(
                        prim_path=cam_cfg.prim_path,
                        translation=np.asarray(cam_cfg.translation, dtype=np.float64),
                        name=cam_cfg.name,
                        frequency=camera_frequency,
                        resolution=tuple(cam_cfg.resolution)
                    )
                    camera.set_local_pose(
                        orientation=np.asarray(cam_cfg.orientation, dtype=np.float64),
                        camera_axes="usd",
                    )

            local_pose = {}
            if hasattr(cam_cfg, 'local_translation'):
                local_pose['translation'] = np.asarray(cam_cfg.local_translation, dtype=np.float64)
            if hasattr(cam_cfg, 'local_orientation'):
                local_pose['orientation'] = np.asarray(cam_cfg.local_orientation, dtype=np.float64)
            if local_pose:
                camera.set_local_pose(camera_axes="usd", **local_pose)

            if hasattr(cam_cfg, 'focal_length'):
                camera.set_focal_length(float(cam_cfg.focal_length))
            
            if hasattr(cam_cfg, 'clipping_range'):
                camera.set_clipping_range(near_distance=cam_cfg.clipping_range[0], far_distance=cam_cfg.clipping_range[1])
            else:
                camera.set_clipping_range(near_distance=0.1, far_distance=10.0)
            
            if hasattr(cam_cfg, 'horizontal_aperture'):
                camera.set_horizontal_aperture(cam_cfg.horizontal_aperture)
            else:
                camera.set_horizontal_aperture(30)
            
            if hasattr(cam_cfg, 'focus_distance'):
                camera.set_focus_distance(cam_cfg.focus_distance)
            else:
                camera.set_focus_distance(30)
            
            self.cameras.append(camera)
        
        self.world.reset()
        
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            camera.initialize()
            image_types = cam_cfg.image_type.split('+') if '+' in cam_cfg.image_type else [cam_cfg.image_type]
            
            for image_type in image_types:
                if image_type == "depth":
                    camera.add_distance_to_image_plane_to_frame()
                elif image_type == "pointcloud":
                    camera.add_distance_to_image_plane_to_frame()
                    camera.add_pointcloud_to_frame()
                elif image_type == "segmentation":
                    camera.add_instance_segmentation_to_frame()
                    for class_id, class_to_prim in cam_cfg.class_to_prim.items():
                        e_prim = self.stage.GetPrimAtPath(class_to_prim)
                        add_update_semantics(e_prim, class_id)
                elif image_type == "semantic_pointcloud":
                    camera.add_instance_segmentation_to_frame()
                    camera.add_distance_to_image_plane_to_frame()
                    camera.add_pointcloud_to_frame()
                    for class_id, class_to_prim in cam_cfg.class_to_prim.items():
                        e_prim = self.stage.GetPrimAtPath(class_to_prim)
                        add_update_semantics(e_prim, class_id)
        
        
    
    def setup_objects(self) -> None:
        """
        Set up objects in the simulation world.
        """

        self.obj_configs = []
        if hasattr(self.cfg, 'task') and hasattr(self.cfg.task, 'obj_paths'):
            for obj in self.cfg.task.obj_paths:
                if isinstance(obj, str):
                    self.obj_configs.append({
                        'path': obj,
                        'position_range': {
                            'x': [0.24, 0.30],
                            'y': [-0.05, 0.05],
                            'z': [0.85, 0.85]
                        }
                    })
                else:
                    self.obj_configs.append(obj)
                
    def setup_materials(self) -> None:
        """
        Set up materials for the objects.
        """
        self.material_config = None
        self.available_materials = []
        is_infer_mode = hasattr(self.cfg, "mode") and self.cfg.mode == "infer"
        has_infer = hasattr(self.cfg, "infer")
        is_ood = False
        if has_infer and hasattr(self.cfg.infer, "is_test_material"):
            is_ood = bool(self.cfg.infer.is_test_material)
        has_material_paths = hasattr(self.cfg, "task") and hasattr(self.cfg.task, "material_paths") and self.cfg.task.material_paths
        if has_material_paths:
            self.material_config = self.cfg.task.material_paths[0]
            if is_infer_mode and is_ood and hasattr(self.material_config, "test_materials"):
                self.available_materials = getattr(self.material_config, "test_materials", [])
            elif hasattr(self.material_config, "materials"):
                self.available_materials = getattr(self.material_config, "materials", [])

    
    def _add_scene_lights(self):
        stage = self.stage

        dome_light = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
        dome_light.CreateIntensityAttr(1000)
        dome_light.CreateTextureFormatAttr("automatic")

        # rect_light = UsdLux.RectLight.Define(stage, "/World/Lights/RectLight")
        # rect_light.CreateIntensityAttr(800)
        # rect_light.CreateWidthAttr(5.0)
        # rect_light.CreateHeightAttr(5.0)

        # xform = UsdGeom.Xformable(rect_light)
        # xform.AddTranslateOp().Set(Gf.Vec3d(2.0, 3.0, 5.0))
        # xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))


    def add_friction_to_prim(
        self,
        prim_path: str,
        static_friction: float = 1.2,
        dynamic_friction: float = 1.2,
        restitution: float = 0.0,
        material_path: str = "/World/PhysicsMaterials/Friction",
        binding_strength=UsdShade.Tokens.weakerThanDescendants):

        if not self.stage.GetPrimAtPath(material_path).IsValid():
            mat_prim = UsdShade.Material.Define(self.stage, material_path)
            phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim.GetPrim())
            phys_mat.CreateStaticFrictionAttr().Set(static_friction)
            phys_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
            phys_mat.CreateRestitutionAttr().Set(restitution)

            physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim.GetPrim())
            physx_mat.CreateFrictionCombineModeAttr().Set("max")   # average / min / multiply / max
            physx_mat.CreateRestitutionCombineModeAttr().Set("min")
        else:
            mat_prim = self.stage.GetPrimAtPath(material_path)

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Prim not found: {prim_path}")
            return
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)

        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        material = UsdShade.Material(mat_prim)
        binding_api.Bind(material, binding_strength, "physics")


    def apply_glass_material(self, prim_path):
        material_path = f"{prim_path}/Looks/TransparentGlass"
        try:
            import isaaclab.sim as sim_utils

            cfg = sim_utils.GlassMdlCfg(
                glass_color=(0.97, 0.97, 0.97),
                glass_ior=1.52,
                thin_walled=True,
            )
            cfg.func(material_path, cfg)
            material = UsdShade.Material.Get(self.stage, Sdf.Path(material_path))

            prim = self.stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material, 
                    UsdShade.Tokens.strongerThanDescendants
                )
            
            if material and material.GetPrim().IsValid():
                return material
            
            isaaclab_error = "GlassMdlCfg did not create a valid UsdShade.Material"
        except Exception as exc:
            isaaclab_error = f"{type(exc).__name__}: {exc}"

    def apply_visual_color(self, prim_path: str, color) -> None:
        """Bind an opaque, high-visibility material without changing physics."""
        target_prim = self.stage.GetPrimAtPath(prim_path)
        if not target_prim.IsValid():
            carb.log_warn(f"Cannot apply visual color: prim not found at {prim_path}")
            return

        rgb = np.asarray(color, dtype=np.float32)
        if rgb.shape != (3,) or np.any(rgb < 0.0) or np.any(rgb > 1.0):
            raise ValueError(
                f"visual_color for {prim_path} must contain three values in [0, 1], got {color}"
            )

        material_path = f"{prim_path}/Looks/HighVisibilitySurface"
        material = UsdShade.Material.Define(self.stage, Sdf.Path(material_path))
        shader = UsdShade.Shader.Define(
            self.stage, Sdf.Path(f"{material_path}/PreviewSurface")
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*rgb.tolist())
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.38)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # A strong binding overrides the referenced USD's embedded grey material,
        # while collision and physics-material bindings remain unchanged.
        UsdShade.MaterialBindingAPI.Apply(target_prim).Bind(
            material, UsdShade.Tokens.strongerThanDescendants
        )
    
    def get_camera_data(self):
        camera_data = {}
        display_data = {}
        
        for camera, cam_cfg in zip(self.cameras, self.cfg.cameras):
            record, display = process_camera_image(camera, cam_cfg.image_type)
            if record is not None:
                if isinstance(record, dict):
                    for k, v in record.items():
                        camera_data[f"{cam_cfg.name}_{k}"] = v
                else:
                    camera_data[f"{cam_cfg.name}_{cam_cfg.image_type}"] = record
            if display is not None:
                display_data[cam_cfg.name] = display
        return camera_data, display_data
    
    def apply_material_to_object(self, target_path: str, material_idx: int = None) -> None:
        """
        Apply material to the specified object.
        
        Args:
            target_path: Path of the target object
            material_idx: Material index, if None use current material index
        """
        if not self.material_config or not self.available_materials:
            return
            
        if material_idx is None:
            material_idx = self.current_material_idx
            
        target_prim = self.stage.GetPrimAtPath(target_path)
        if target_prim.IsValid():
            material_path = self.available_materials[material_idx]
            mtl_prim = self.stage.GetPrimAtPath(material_path)
            if mtl_prim.IsValid():
                cube_mat_shade = UsdShade.Material(mtl_prim)
                UsdShade.MaterialBindingAPI(target_prim).Bind(
                    cube_mat_shade, 
                    UsdShade.Tokens.strongerThanDescendants
                )
    
    def randomize_object_position(self, obj_path: str, position_range: Dict[str, list]) -> np.ndarray:
        """
        Randomize object position according to position range.
        
        Args:
            obj_path: Object path
            position_range: Position range dictionary containing x, y, z [min, max] values
            
        Returns:
            np.ndarray: Randomly generated position
        """
        position = np.array([
            np.random.uniform(position_range['x'][0], position_range['x'][1]),
            np.random.uniform(position_range['y'][0], position_range['y'][1]),
            np.random.uniform(position_range['z'][0], position_range['z'][1])
        ])
        self.object_utils.set_object_position(object_path=obj_path, position=position)
        
        return position
    
    def place_objects_with_visibility_management(self, current_obj_idx: int, far_distance: float = 10.0) -> str:
        """
        Place objects and manage visibility, move non-current objects to far distance.
        
        Args:
            current_obj_idx: Index of current object
            far_distance: Far distance where non-current objects are placed
            
        Returns:
            str: Path of current object
        """
        for i, obj_config in enumerate(self.obj_configs):
            obj_path = obj_config['path']
            position_range = obj_config['position_range']
            prim = self.stage.GetPrimAtPath(obj_path)
            
            if prim.IsValid():
                if i == current_obj_idx:
                    self.randomize_object_position(obj_path, position_range)
                    set_prim_visibility(prim, True)
                else:
                    # Move non-current objects to far distance
                    angle = 2 * np.pi * i / len(self.obj_configs)
                    far_position = np.array([
                        far_distance * np.cos(angle),
                        far_distance * np.sin(angle),
                        0.1
                    ])
                    self.object_utils.set_object_position(object_path=obj_path, position=far_position)
                    set_prim_visibility(prim, False)
        
        return self.obj_configs[current_obj_idx]['path']
    
    def get_basic_state_info(self, joint_positions: np.ndarray = None, object_path: str = None, 
                           target_path: str = None, additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get basic state information containing common state data for all tasks.
        
        Args:
            joint_positions: Robot joint positions
            object_path: Object path
            target_path: Target object path
            additional_info: Additional state information
            
        Returns:
            Dict[str, Any]: Basic state information dictionary
        """
        
        if joint_positions is None:
            joint_positions = self.robot.get_joint_positions()
            if joint_positions is None:
                return None
    
        camera_data, display_data = self.get_camera_data()
        state = {
            'joint_positions': joint_positions,
            'camera_data': camera_data,
            'camera_display': display_data,
            'done': self.reset_needed,
            'gripper_position': self.robot.get_gripper_position(),
        }
        
        if object_path:
            state.update({
                'object_position': self.object_utils.get_geometry_center(object_path=object_path),
                'object_size': self.object_utils.get_object_size(object_path=object_path),
                'object_path': object_path,
                'object_name': object_path.split("/")[-1]
            })
        
        
        if target_path:
            state.update({
                'target_position': self.object_utils.get_geometry_center(object_path=target_path),
                'target_size': self.object_utils.get_object_size(object_path=target_path),
                'target_path': target_path,
                'target_name': target_path.split("/")[-1]
            })
        
        if additional_info:
            state.update(additional_info)
            
        return state
    
    
    def add_collider_to_prim(self, prim_path: str, sdf_resolution: int = 256) -> bool:
        """Enable GPU-compatible SDF mesh collision on the given prim.

        Particle systems (liquids) and convex hull meshes are incompatible on
        GPU physics; converting the mesh to an SDF collision approximation
        lets particles collide with it.
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return False

        mesh_col_api = UsdPhysics.MeshCollisionAPI(prim)
        mesh_col_api.GetApproximationAttr().Set("sdf")

        sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        sdf_api.CreateSdfResolutionAttr().Set(int(sdf_resolution))

        collision_api = UsdPhysics.CollisionAPI(prim)
        collision_api.GetCollisionEnabledAttr().Set(True)
        return True
    def check_frame_limits(self, max_steps: int = None) -> bool:
        """
        Check frame limits, set reset flag if limit is exceeded.

        Collection and evaluation use different clocks.  ``task.max_steps``
        belongs to the collection task and is measured in simulator frames;
        evaluation limits are enforced by ``BenchmarkTaskEvalRunner`` in
        policy/action steps.  In evaluation mode, do not let a collection
        frame limit terminate an episode early.  A task may opt into an
        explicit simulator-frame safety cap with ``evaluation.sim_max_steps``.
        
        Args:
            max_steps: Maximum steps, if None use value from configuration
            
        Returns:
            bool: Return True if should continue execution, False otherwise
        """
        if self.frame_idx < 5:
            return False

        mode = str(self.cfg.get("mode", "") if hasattr(self.cfg, "get") else getattr(self.cfg, "mode", ""))
        if mode == "eval":
            evaluation_cfg = self.cfg.get("evaluation") if hasattr(self.cfg, "get") else getattr(self.cfg, "evaluation", None)
            sim_max_steps = None
            if evaluation_cfg is not None:
                try:
                    sim_max_steps = evaluation_cfg.get("sim_max_steps")
                except AttributeError:
                    sim_max_steps = getattr(evaluation_cfg, "sim_max_steps", None)
            # No simulator-frame cap by default: the evaluator's action-step
            # budget is authoritative.  This also covers task implementations
            # that pass a hard-coded ``max_steps`` (for example stirring).
            if sim_max_steps in (None, "", 0, "0"):
                return True
            max_steps = int(sim_max_steps)

        elif max_steps is None:
            max_steps = self.cfg.task.max_steps
            
        if self.frame_idx > max_steps:
            reason = (
                f"Task simulator-frame limit exceeded: "
                f"frame_idx={self.frame_idx}, max_steps={max_steps}"
            )
            print(f"[BaseTask] {reason}", flush=True)
            self.on_task_complete(False)
            self.episode_outcome = False
            self.reset_needed = True
            return False

        return True
        
    def update_object_and_material_indices(self, success: bool) -> None:
        """
        Update object and material indices.
        
        Args:
            success: Whether task completed successfully
        """
        if success:
            self.current_obj_episodes += 1
            if self.current_obj_episodes >= self.episodes_per_obj and len(self.obj_configs) > 0:
                self.current_obj_idx = (self.current_obj_idx + 1) % len(self.obj_configs)
                self.current_obj_episodes = 0
            if self.available_materials:
                self.current_material_idx = (self.current_material_idx + 1) % len(self.available_materials)
