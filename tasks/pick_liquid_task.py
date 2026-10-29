import os
import numpy as np
from typing import Any
from .single_object_task import SingleObjectTask
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
import carb
import math
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from omni.physx.scripts import particleUtils
from omni.physx.scripts import physicsUtils


class PickLiquidTask(SingleObjectTask):
    """
    A task class for robotic picking operations.
    Manages object placement, material switching, and task state transitions.
    """
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)

    def setup_scene(self) -> None:

        self._configure_particle_physics_scene()
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        self._add_scene_lights()
        self.world.scene.add_default_ground_plane()
        
        if hasattr(self.cfg, 'scene') and  hasattr(self.cfg.scene, 'base_scene'):
            for obj in self.cfg.scene.base_scene:
                if not self.stage.GetPrimAtPath(obj.prim_path).IsValid():
                    if not os.path.exists(obj.usd_path):
                        usd_path = assets_root_path + obj.usd_path
                    else:
                        usd_path = obj.usd_path
                    
                    add_reference_to_stage(usd_path=usd_path, prim_path=obj.prim_path)
                    obj_position = obj.get("position", [0.0, 0.0, 0.0])
                    obj_orientation = obj.get("orientation", [1.0, 0.0, 0.0, 0.0])
                    obj_scale = obj.get("scale", [1.0, 1.0, 1.0])
                    
                    self.object_utils.set_object_position(object_path=obj.prim_path, position=obj_position)
                    self.object_utils.set_object_orientation(object_path=obj.prim_path, orientation=obj_orientation)
                    self.object_utils.set_object_scale(object_path=obj.prim_path, scale=obj_scale)

        if hasattr(self.cfg, 'scene') and  hasattr(self.cfg.scene, 'liquid_scene'):

            prim_path = self.cfg.scene.liquid_scene.prim_path
            # geometry_cfg = self.cfg.scene.liquid_scene.geometry
            # liquid_cfg = self.cfg.scene.liquid

            # self._author_obj_root(prim_path)
            # self._author_assembly(prim_path, geometry_cfg)

            self.apply_glass_material(prim_path)
            # The beaker is rotated through a large angle during pouring.  Give
            # its SDF collision meshes a high-friction contact material so it
            # remains seated between the Franka fingers while returning upright.
            self.set_high_friction(prim_path)
            self.apply_sdf_collision(prim_path)

            # self.author_physx_water_particles(liquid_cfg, geometry_cfg, [0, 0, 0.1])

    def apply_liquid(self):

        prim_path = self.cfg.scene.liquid_scene.prim_path
        liquid_cfg = self.cfg.scene.liquid
        geometry_cfg = self.cfg.scene.liquid_scene.geometry

        liquid_prim_path = liquid_cfg.prim_path
        if self.stage.GetPrimAtPath(liquid_prim_path).IsValid():
            success = self.stage.RemovePrim(liquid_prim_path)
        center = self.object_utils.get_geometry_center(object_path=prim_path)
        self.author_physx_water_particles(liquid_cfg, geometry_cfg, center)


    def apply_sdf_collision(self, prim_path: str):
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return 
        for p in Usd.PrimRange(prim):
            
            if p.IsA(UsdGeom.Mesh):
                if p.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
                    p.RemoveAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI)
                if p.HasAPI(PhysxSchema.PhysxConvexHullCollisionAPI):
                    p.RemoveAPI(PhysxSchema.PhysxConvexHullCollisionAPI)
                
                collision_api = UsdPhysics.CollisionAPI.Apply(p)
                collision_api.CreateCollisionEnabledAttr().Set(True)
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(p)
                mesh_collision_api.CreateApproximationAttr().Set("sdf")
                sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(p)
                sdf_api.CreateSdfResolutionAttr().Set(256)
        
    def set_high_friction(
            self,
            prim_path,
            static_friction: float = 3.0,
            dynamic_friction: float = 2.5
        ):
        material_path = f"{prim_path}/GripPhysicsMaterial"
        mat_prim = self.stage.GetPrimAtPath(material_path)
        if not mat_prim or not mat_prim.IsValid():
            mat_prim = UsdShade.Material.Define(self.stage, Sdf.Path(material_path)).GetPrim()
        physics_mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
        physics_mat_api.CreateStaticFrictionAttr(static_friction)
        physics_mat_api.CreateDynamicFrictionAttr(dynamic_friction)
        physics_mat_api.CreateRestitutionAttr(0.0)

        physx_mat_api = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
        physx_mat_api.CreateFrictionCombineModeAttr("max")
        physx_mat_api.CreateRestitutionCombineModeAttr("min")

        prim = self.stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            physicsUtils.add_physics_material_to_prim(
                self.stage, prim, Sdf.Path(material_path)
            )

    def _add_scene_lights(self):

        dome = UsdLux.DomeLight.Define(self.stage, Sdf.Path("/World/Light"))
        dome.CreateIntensityAttr(3000.0)
        dome.CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75))

        # A neutral soft source gives OmniGlass a reflection to resolve against the
        # gray SeattleLabTable. It is real lighting, not an authored cup outline.
        key = UsdLux.RectLight.Define(self.stage, Sdf.Path("/World/GlassKeyLight"))
        key.CreateIntensityAttr(1000.0)
        key.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        key.CreateWidthAttr(0.45)
        key.CreateHeightAttr(0.28)


    def _author_obj_root(self, prim_path):
        
        root = UsdGeom.Xform.Define(self.stage, Sdf.Path(prim_path)).GetPrim()
        xformable = UsdGeom.Xformable(root)
        # xformable.ClearXformOpOrder()
        # xformable.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in cfg.object_init_position]))
        # quat = cfg.object_init_rotation
        # xformable.AddOrientOp().Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))))
        body = UsdPhysics.RigidBodyAPI.Apply(root)
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(True)

        mass = UsdPhysics.MassAPI.Apply(root)
        mass.CreateMassAttr(0.18)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, -0.008))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.00025, 0.00025, 0.00016))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        physx_body.CreateLinearDampingAttr(1.2)
        physx_body.CreateAngularDampingAttr(14.0)
        try:
            physx_body.CreateMaxDepenetrationVelocityAttr(0.25)
            physx_body.CreateMaxLinearVelocityAttr(1.0)
        except Exception:
            pass
        try:
            physx_body.CreateMaxAngularVelocityAttr(4.0)
        except Exception:
            pass


    def _author_assembly(self, prim_path, geometry_cfg):

        root = self.stage.GetPrimAtPath(prim_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"Cannot author beaker assembly; root prim does not exist: {prim_path}")

        
        visual_root_path = f"{prim_path}/Visual"
        collision_root_path = f"{prim_path}/CollisionProxies"
        frames_root_path = f"{prim_path}/Frames"

        for path in [visual_root_path, collision_root_path, frames_root_path]:
            UsdGeom.Xform.Define(self.stage, Sdf.Path(path))

        glass_material_path = f"{prim_path}/Looks/TransparentGlass"
        collision_material_path = f"{prim_path}/Looks/GripPhysicsMaterial"

        glass_material_path, _ = self._author_glass_material(glass_material_path)
        
        self._ensure_physics_material(collision_material_path)
        self._author_hollow_beaker_visual(visual_root_path, geometry_cfg, glass_material_path)
        # self._author_contained_liquid_visual(visual_root_path, geometry_cfg, liquid_cfg)
        
    def _author_glass_material(self, material_path):
        # import pdb; pdb.set_trace()
        glass_cfg = {
            "class": "sim_utils.GlassMdlCfg",
            "glass_color": [0.98, 0.98, 0.99],
            "glass_ior": 1.52,
            "thin_walled": True,
        }
        isaaclab_error = None
        try:
            import isaaclab.sim as sim_utils

            cfg = sim_utils.GlassMdlCfg(
                glass_color=(0.98, 0.98, 0.99),
                glass_ior=1.52,
                thin_walled=True,
            )
            cfg.func(material_path, cfg)
            material = UsdShade.Material.Get(self.stage, Sdf.Path(material_path))
            if material and material.GetPrim().IsValid():
                return material, {
                    **glass_cfg,
                    "backend": "isaaclab_glass_mdl_cfg",
                    "material_path": material_path,
                    "mdl_path": str(cfg.mdl_path),
                }
            isaaclab_error = "GlassMdlCfg did not create a valid UsdShade.Material"
        except Exception as exc:
            isaaclab_error = f"{type(exc).__name__}: {exc}"

        material = UsdShade.Material.Define(self.stage, Sdf.Path(material_path))
        shader = UsdShade.Shader.Define(self.stage, Sdf.Path(f"{material_path}/Shader"))
        shader_prim = shader.GetPrim()
        shader_prim.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
        shader_prim.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("OmniGlass.mdl"))
        shader_prim.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("OmniGlass")
        shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.98, 0.98, 0.99))
        shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.52)
        shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        for output_name in ["mdl:surface", "mdl:displacement", "mdl:volume"]:
            material.CreateOutput(output_name, Sdf.ValueTypeNames.Token).ConnectToSource(
                shader.ConnectableAPI(),
                "out",
            )
        return material, {
            **glass_cfg,
            "backend": "authored_omniglass_fallback",
            "material_path": material_path,
            "mdl_path": "OmniGlass.mdl",
            "isaaclab_error": isaaclab_error,
        }
        return material
        
    def _ensure_physics_material(self, material_path: str) -> None:
        from pxr import PhysxSchema

        prim = self.stage.GetPrimAtPath(material_path)
        if not prim or not prim.IsValid():
            prim = UsdShade.Material.Define(self.stage, Sdf.Path(material_path)).GetPrim()
        material = UsdPhysics.MaterialAPI.Apply(prim)
        material.CreateStaticFrictionAttr(2.0)
        material.CreateDynamicFrictionAttr(1.5)
        material.CreateRestitutionAttr(0.0)
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(prim)
        physx_material.CreateFrictionCombineModeAttr("max")
        physx_material.CreateRestitutionCombineModeAttr("min")


    def _author_hollow_beaker_visual(
        self,
        root_path: str,
        geometry,
        material,
    ) -> None:
        segments = max(64, int(geometry.wall_segments))
        inner_radius = float(geometry.outer_radius) - float(geometry.wall_thickness)
        outer_radius = float(geometry.outer_radius)
        # inner_radius = float(inner_radius)
        z_bottom = -float(geometry.height) * 0.5 + float(geometry.bottom_thickness)
        z_top = float(geometry.height) * 0.5

        points = []
        for idx in range(segments):
            angle = 2.0 * math.pi * float(idx) / float(segments)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            points.extend(
                [
                    Gf.Vec3f(cos_a * outer_radius, sin_a * outer_radius, z_bottom),
                    Gf.Vec3f(cos_a * outer_radius, sin_a * outer_radius, z_top),
                    Gf.Vec3f(cos_a * inner_radius, sin_a * inner_radius, z_bottom),
                    Gf.Vec3f(cos_a * inner_radius, sin_a * inner_radius, z_top),
                ]
            )

        counts: list[int] = []
        indices: list[int] = []
        normals: list[Any] = []
        for idx in range(segments):
            nxt = (idx + 1) % segments
            angle_i = 2.0 * math.pi * float(idx) / float(segments)
            angle_j = 2.0 * math.pi * float(nxt) / float(segments)
            outer_i = Gf.Vec3f(math.cos(angle_i), math.sin(angle_i), 0.0)
            outer_j = Gf.Vec3f(math.cos(angle_j), math.sin(angle_j), 0.0)
            inner_i = Gf.Vec3f(-math.cos(angle_i), -math.sin(angle_i), 0.0)
            inner_j = Gf.Vec3f(-math.cos(angle_j), -math.sin(angle_j), 0.0)
            ob_i, ot_i, ib_i, it_i = idx * 4, idx * 4 + 1, idx * 4 + 2, idx * 4 + 3
            ob_j, ot_j, ib_j, it_j = nxt * 4, nxt * 4 + 1, nxt * 4 + 2, nxt * 4 + 3
            # Keep the outer wall, inner wall, and physical glass lip in one mesh.
            # Face-varying normals let the thin lip retain a crisp highlight without
            # adding an opaque outline helper.
            indices.extend([ob_i, ob_j, ot_j, ot_i])
            normals.extend([outer_i, outer_j, outer_j, outer_i])
            indices.extend([ib_j, ib_i, it_i, it_j])
            normals.extend([inner_j, inner_i, inner_i, inner_j])
            indices.extend([ot_i, ot_j, it_j, it_i])
            normals.extend([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
            counts.extend([4, 4, 4])

        mesh = UsdGeom.Mesh.Define(self.stage, Sdf.Path(f"{root_path}/GlassWall"))
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateNormalsAttr(normals)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(False)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


        bottom = UsdGeom.Cylinder.Define(self.stage, Sdf.Path(f"{root_path}/GlassBottom"))
        bottom.CreateAxisAttr("Z")
        bottom.CreateRadiusAttr(float(geometry.outer_radius))
        bottom.CreateHeightAttr(float(geometry.bottom_thickness))
        bottom_xform = UsdGeom.Xformable(bottom.GetPrim())
        bottom_xform.ClearXformOpOrder()
        bottom_xform.AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, -float(geometry.height) * 0.5 + float(geometry.bottom_thickness) * 0.5)
        )
        UsdShade.MaterialBindingAPI.Apply(bottom.GetPrim()).Bind(material)

    def author_beaker_sdf_collision_mesh(
        self,
        root_path: str,
        geometry,
        liquid,
        *,
        mesh_name: str = "SDFLiquidCollider",
        visible: bool = False,
        sdf_resolution: int = 256,
    ) -> dict[str, Any]:
        """Author a HoneyDemo-style hollow beaker SDF collider.

        The segmented box proxies are useful for cheap rigid-body grasping, but
        PhysX particles are much happier with a single SDF collision mesh for the
        container wall and bottom. This mesh represents the glass volume: open at
        the top, with an annular wall and a closed bottom disk.
        """

        root = self.stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"Cannot author SDF beaker collider; root prim does not exist: {root_path}")

        segments = max(32, int(geometry.wall_segments))
        outer_radius = float(geometry.outer_radius)
        inner_radius = float(geometry.outer_radius) - float(geometry.wall_thickness)
        z_bottom = -float(geometry.height) * 0.5
        z_bottom_top = z_bottom + float(geometry.bottom_thickness)
        z_top = float(geometry.height) * 0.5

        points = []
        for idx in range(segments):
            angle = 2.0 * math.pi * float(idx) / float(segments)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            points.extend(
                [
                    Gf.Vec3f(cos_a * outer_radius, sin_a * outer_radius, z_bottom),
                    Gf.Vec3f(cos_a * outer_radius, sin_a * outer_radius, z_bottom_top),
                    Gf.Vec3f(cos_a * outer_radius, sin_a * outer_radius, z_top),
                    Gf.Vec3f(cos_a * inner_radius, sin_a * inner_radius, z_bottom_top),
                    Gf.Vec3f(cos_a * inner_radius, sin_a * inner_radius, z_top),
                ]
            )

        counts: list[int] = []
        indices: list[int] = []

        # Bottom underside and interior support disk. These N-gons seal the bottom
        # volume so particles cannot treat the container as an open tube.
        indices.extend([idx * 5 for idx in reversed(range(segments))])
        counts.append(segments)
        indices.extend([idx * 5 + 3 for idx in range(segments)])
        counts.append(segments)

        for idx in range(segments):
            nxt = (idx + 1) % segments
            ob_i, omt_i, ot_i, ib_i, it_i = idx * 5, idx * 5 + 1, idx * 5 + 2, idx * 5 + 3, idx * 5 + 4
            ob_j, omt_j, ot_j, ib_j, it_j = nxt * 5, nxt * 5 + 1, nxt * 5 + 2, nxt * 5 + 3, nxt * 5 + 4
            # Lower and upper outer wall.
            indices.extend([ob_i, ob_j, omt_j, omt_i])
            counts.append(4)
            indices.extend([omt_i, omt_j, ot_j, ot_i])
            counts.append(4)
            # Bottom top annulus, inner wall, and top rim.
            indices.extend([ib_i, ib_j, omt_j, omt_i])
            counts.append(4)
            indices.extend([ib_j, ib_i, it_i, it_j])
            counts.append(4)
            indices.extend([ot_i, ot_j, it_j, it_i])
            counts.append(4)

        mesh_path = f"{root_path}/CollisionProxies/{mesh_name}"
        mesh = UsdGeom.Mesh.Define(self.stage, Sdf.Path(mesh_path))
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateDoubleSidedAttr(True)
        if not visible:
            UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()
        collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        collision.CreateCollisionEnabledAttr(True)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mesh_collision.CreateApproximationAttr().Set("sdf")
        sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh.GetPrim())
        try:
            sdf_api.CreateSdfResolutionAttr().Set(int(sdf_resolution))
        except Exception:
            pass
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        sdf_contact_offset = (
            float(liquid.collision_proxy_contact_offset)
            if liquid.collision_proxy_contact_offset is not None
            else max(float(liquid.particle_contact_offset) * 1.25, 0.0015)
        )
        sdf_rest_offset = (
            float(liquid.collision_proxy_rest_offset)
            if liquid.collision_proxy_rest_offset is not None
            else min(float(liquid.particle_radius) * 0.25, 0.0005)
        )
        physx_collision.CreateContactOffsetAttr().Set(sdf_contact_offset)
        physx_collision.CreateRestOffsetAttr().Set(sdf_rest_offset)

    def _author_contained_liquid_visual(
        self,
        root_path: str,
        geometry,
        liquid
    ) -> str:
        """Author a low-cost local-frame water surface for particle preview renders.

        The simulated water is still the PhysX particle set. This thin surface only
        makes the fill volume readable in low-particle preview videos where sparse
        points otherwise look like they are detached from the cup.
        """

        inner_radius = float(geometry.outer_radius) - float(geometry.wall_thickness)
        material_path = f"{root_path}/ContainedLiquidMaterial"
        material =  UsdShade.Material.Define(self.stage, Sdf.Path(material_path))
        shader = UsdShade.Shader.Define(self.stage, Sdf.Path(f"{material_path}/PreviewSurface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        
        color = liquid.color
        opacity = float(color[3]) if len(color) > 3 else 0.45
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        )
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(color[0]) * 0.03, float(color[1]) * 0.05, float(color[2]) * 0.08)
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(max(0.58, min(max(opacity, 0.58), 0.78)))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.015)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        margin = max(float(liquid.particle_spacing) * 2.0, float(liquid.particle_radius) * 2.0)
        radius = max(float(inner_radius) - margin, margin)
        usable_height = max(float(geometry.height) - float(geometry.bottom_thickness) - 2.0 * margin, margin)
        fill_height = min(usable_height, usable_height * float(liquid.fill_fraction))
        surface_height = max(float(liquid.particle_spacing) * 2.0, 0.003)
        z_min = -float(geometry.height) * 0.5 + float(geometry.bottom_thickness) + margin
        center_z = z_min + fill_height - surface_height * 0.5

        surface_path = f"{root_path}/ContainedLiquidSurface"
        surface = UsdGeom.Cylinder.Define(self.stage, Sdf.Path(surface_path))
        surface.CreateAxisAttr("Z")
        surface.CreateRadiusAttr(radius)
        surface.CreateHeightAttr(surface_height)
        xformable = UsdGeom.Xformable(surface.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, center_z))
        # xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
        UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(material)
        return surface_path

    def author_physx_water_particles(
        self,
        liquid,
        geometry,
        center,
    ) -> dict[str, Any]:

        positions = self.sample_liquid_particles_in(center, geometry, liquid)
        UsdGeom.Xform.Define(self.stage, Sdf.Path(liquid.prim_path))
        material = self._author_water_material(liquid)

        system_path = Sdf.Path(liquid.particle_system_path)
        spacing = float(liquid.particle_spacing)
        offsets = self.particle_offset_report(liquid)
        fluid_rest_offset = float(offsets["fluid_rest_offset"])
        rest_offset = float(offsets["rest_offset"])
        particle_contact_offset = float(offsets["particle_contact_offset"])
        contact_offset = float(offsets["contact_offset"])
        system_dynamics = self._particle_system_dynamics(liquid)

        system_kwargs: dict[str, Any] = {
            "particle_system_enabled": True,
            "simulation_owner": Sdf.Path("/World/physicsScene"),
            "particle_contact_offset": particle_contact_offset,
            "enable_ccd": True,
            "solver_position_iterations": int(liquid.solver_iterations),
            "max_depenetration_velocity": float(system_dynamics["max_depenetration_velocity"]),
            "max_neighborhood": int(system_dynamics["max_neighborhood"]),
            "max_velocity": float(system_dynamics["max_velocity"]),
            "global_self_collision_enabled": True,
            "non_particle_collision_enabled": True,
        }
        if offsets["offset_mode"] != "physx_autocompute":
            system_kwargs.update(
                {
                    "contact_offset": contact_offset,
                    "rest_offset": rest_offset,
                    "solid_rest_offset": rest_offset,
                    "fluid_rest_offset": fluid_rest_offset,
                }
            )
        system = particleUtils.add_physx_particle_system(self.stage, system_path, **system_kwargs)
        post_processing = self._author_particle_post_processing(system_path, liquid, particleUtils)
        pbd_params = self._pbd_water_material_params(liquid)
        try:
            particleUtils.add_pbd_particle_material(self.stage, Sdf.Path(liquid.material_path), **pbd_params)
        except TypeError:
            particleUtils.add_pbd_particle_material(self.stage, Sdf.Path(liquid.material_path))
        UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(material)
        physicsUtils.add_physics_material_to_prim(self.stage, system.GetPrim(), Sdf.Path(liquid.material_path))

        gf_positions = [Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])) for pos in positions]
        velocities = [Gf.Vec3f(0.0, 0.0, 0.0) for _ in gf_positions]
        widths = [float(liquid.particle_radius) * 2.0 for _ in gf_positions]
        # particle_points = particleUtils.add_physx_particleset_points(
        #     self.stage,
        #     Sdf.Path(liquid.particle_set_path),
        #     gf_positions,
        #     velocities,
        #     widths,
        #     system_path,
        #     self_collision=True,
        #     fluid=True,
        #     particle_group=0,
        #     particle_mass=float(liquid.particle_mass),
        #     density=float(pbd_params["density"]),
        # )
        particle_mass = float(liquid.particle_mass)
        estimated_particle_mass = (
            particle_mass
            if particle_mass > 0.0
            else 8.0 * fluid_rest_offset * fluid_rest_offset * fluid_rest_offset * float(pbd_params["density"])
        )

        particle_points = particleUtils.add_physx_particleset_points(
            self.stage,
            Sdf.Path(liquid.particle_set_path),
            gf_positions,
            velocities,
            widths,
            system_path,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=estimated_particle_mass,
            density=float(pbd_params["density"]),
        )
        UsdShade.MaterialBindingAPI.Apply(particle_points.GetPrim()).Bind(material)
        physicsUtils.add_physics_material_to_prim(self.stage, particle_points.GetPrim(), Sdf.Path(liquid.material_path))

    def sample_liquid_particles_in(
        self,
        center,
        geometry,
        liquid,
    ) -> list[tuple[float, float, float]]:
        
        inner_radius = float(geometry.outer_radius) - float(geometry.wall_thickness)
        # Keep the initial fluid core away from the hidden cup-wall proxies. If too many
        # particles start exactly on the wall contact band, PhysX can spend a very long
        # time resolving the first step and the water looks like it is glued to the glass.
        margin = max(float(liquid.particle_spacing) * 3.0, float(liquid.particle_radius) * 2.25)
        radius = max(inner_radius - margin, margin)
        z_min = float(center[2]) - float(geometry.height) * 0.5 + float(geometry.bottom_thickness) + margin
        fill_height = (float(geometry.height) - float(geometry.bottom_thickness)) * float(liquid.fill_fraction)
        z_max = min(z_min + fill_height, float(center[2]) + float(geometry.height) * 0.5 - margin)
        spacing = float(liquid.particle_spacing)
        positions: list[tuple[float, float, float]] = []
        layer = 0
        z = z_min
        while z <= z_max and len(positions) < int(liquid.max_particles):
            offset = 0.5 * spacing if layer % 2 else 0.0
            x = -radius
            while x <= radius and len(positions) < int(liquid.max_particles):
                y = -radius + offset
                while y <= radius and len(positions) < int(liquid.max_particles):
                    if x * x + y * y <= radius * radius:
                        positions.append((float(center[0]) + x, float(center[1]) + y, z))
                    y += spacing
                x += spacing
            z += spacing
            layer += 1
        return positions


    def _author_water_material(self, liquid) -> Any:
        """Author a render-only water material separate from the PBD material."""

        visual_material_path = f"{liquid.material_path}Visual"
        material = UsdShade.Material.Define(self.stage, Sdf.Path(visual_material_path))
        shader = UsdShade.Shader.Define(self.stage, Sdf.Path(f"{visual_material_path}/PreviewSurface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        color = liquid.color
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        )
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(color[0]) * 0.24, float(color[1]) * 0.32, float(color[2]) * 0.42)
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
            min(max(float(color[3]) if len(color) > 3 else 1.0, 0.90), 1.0)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.08)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material


    def particle_offset_report(self, liquid) -> dict[str, float | str]:
        """Return the effective PhysX particle offsets for diagnostics.

        PhysX autocomputes all offsets from particleContactOffset when they are not
        authored. The formulas mirror the Isaac Sim 5.1 particle documentation.
        """

        particle_contact_offset = float(liquid.particle_contact_offset)
        if liquid.offset_mode == "physx_autocompute":
            contact_offset = particle_contact_offset
            rest_offset = 0.99 * particle_contact_offset
            fluid_rest_offset = 0.99 * 0.6 * particle_contact_offset
            return {
                "offset_mode": "physx_autocompute",
                "particle_contact_offset": particle_contact_offset,
                "contact_offset": contact_offset,
                "rest_offset": rest_offset,
                "solid_rest_offset": rest_offset,
                "fluid_rest_offset": fluid_rest_offset,
                "recommended_particle_spacing": 2.0 * fluid_rest_offset,
            }

        spacing = float(liquid.particle_spacing)
        fluid_rest_offset = spacing * 0.54
        rest_offset = max(spacing * 0.72, fluid_rest_offset)
        return {
            "offset_mode": "explicit_legacy",
            "particle_contact_offset": particle_contact_offset,
            "contact_offset": max(rest_offset + spacing * 0.03, particle_contact_offset),
            "rest_offset": rest_offset,
            "solid_rest_offset": rest_offset,
            "fluid_rest_offset": fluid_rest_offset,
            "recommended_particle_spacing": 2.0 * fluid_rest_offset,
        }

    def _particle_system_dynamics(self, liquid) -> dict[str, float | int]:
        if liquid.material_preset == "physx_water":
            return {
                "max_velocity": 1.0,
                "max_depenetration_velocity": 0.025,
                "max_neighborhood": 96,
            }
        if liquid.material_preset == "container_water_stable":
            return {
                "max_velocity": 0.02,
                "max_depenetration_velocity": 0.006,
                "flow_max_velocity": 0.25,
                "flow_max_depenetration_velocity": 0.012,
                "max_neighborhood": 128,
            }
        return {
            "max_velocity": 0.32,
            "max_depenetration_velocity": 0.025,
            "max_neighborhood": 128,
        }

    def _author_particle_post_processing(
        self,
        system_path: Any,
        liquid,
        particle_utils: Any,
    ) -> dict[str, Any]:
        is_surface = liquid.render_mode == "isosurface" or liquid.enable_isosurface
        if not is_surface:
            return {
                "anisotropy": False,
                "smoothing": False,
                "isosurface": False,
            }

        # Matches Isaac Sim 5.1 ParticlePostProcessingDemo defaults for an
        # isosurface: anisotropy min=1 avoids holes and max=2 limits stretching.
        particle_utils.add_physx_particle_anisotropy(
            self.stage,
            system_path,
            enabled=True,
            scale=5.0,
            min=1.0,
            max=2.0,
        )
        particle_utils.add_physx_particle_smoothing(self.stage, system_path, enabled=True, strength=0.8)
        particle_utils.add_physx_particle_isosurface(
            self.stage,
            system_path,
            enabled=True,
            grid_spacing=float(liquid.particle_spacing),
            surface_distance=float(liquid.particle_spacing) * 0.60,
            grid_filtering_passes="GSRS",
            grid_smoothing_radius=0.00008,
            num_mesh_smoothing_passes=1,
            num_mesh_normal_smoothing_passes=1,
        )
        return {
            "anisotropy": True,
            "anisotropy_scale": 5.0,
            "anisotropy_min": 1.0,
            "anisotropy_max": 2.0,
            "smoothing": True,
            "smoothing_strength": 0.8,
            "isosurface": True,
        }

    def _pbd_water_material_params(self, liquid) -> dict[str, float]:
        if liquid.material_preset == "physx_water":
            # Isaac Sim 5.1's built-in AddPBDMaterialWater values for a meter-scale stage.
            return {
                "friction": 0.1,
                "damping": 0.0,
                "viscosity": 0.0000017,
                "surface_tension": 0.0074,
                "cohesion": 0.01,
                "adhesion": 0.0,
                "particle_adhesion_scale": 1.0,
                "density": 1000.0,
                "drag": 0.0,
                "vorticity_confinement": 0.0,
                "cfl_coefficient": 1.0,
            }
        if liquid.material_preset == "container_water":
            # Low-viscosity water tuned for a millimeter-scale segmented/SDF cup.
            # The cohesion and surface tension keep the PBD neighborhood intact at
            # contacts; viscosity remains over two orders below the old carry
            # preset so the liquid can still pour under gravity.
            return {
                "friction": 0.06,
                "damping": 0.10,
                "viscosity": 0.00005,
                "surface_tension": 0.020,
                "cohesion": 0.014,
                "adhesion": 0.001,
                "particle_adhesion_scale": 0.02,
                "density": 1000.0,
                "drag": 0.05,
                "vorticity_confinement": 0.0,
                "cfl_coefficient": 1.0,
            }
        if liquid.material_preset == "container_water_stable":
            # Small-container water needs more dissipation than the meter-scale
            # PhysX preset, but remains far below the cohesive carry/honey range.
            return {
                "friction": 0.06,
                "damping": 0.35,
                "viscosity": 0.0015,
                "surface_tension": 0.015,
                "cohesion": 0.012,
                "adhesion": 0.0,
                "particle_adhesion_scale": 0.0,
                "density": 1000.0,
                "drag": 0.08,
                "vorticity_confinement": 0.0,
                "cfl_coefficient": 1.0,
            }
        if liquid.material_preset == "water_cohesive":
            return {
                "friction": 0.12,
                "damping": 0.55,
                "viscosity": 0.010,
                "surface_tension": 0.030,
                "cohesion": 0.025,
                "adhesion": 0.003,
                "particle_adhesion_scale": 0.05,
                "density": 1000.0,
                "drag": 0.08,
                "vorticity_confinement": 0.0,
            }
        return {
            "friction": 0.06,
            "damping": 0.42,
            "viscosity": 0.006,
            "surface_tension": 0.020,
            "cohesion": 0.014,
            "adhesion": 0.001,
            "particle_adhesion_scale": 0.02,
            "density": 1000.0,
            "drag": 0.05,
            "vorticity_confinement": 0.0,
        }


    def _configure_particle_physics_scene(self) -> dict[str, Any]:

        scene = UsdPhysics.Scene.Define(self.stage, Sdf.Path("/World/physicsScene"))
        scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
        physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
        physx_scene.CreateSolverTypeAttr().Set("TGS")
        physx_scene.GetTimeStepsPerSecondAttr().Set(60.0)
        physx_scene.CreateGpuFoundLostPairsCapacityAttr().Set(10 * 1024)
        physx_scene.CreateGpuCollisionStackSizeAttr().Set(64 * 1024 * 1024)
        physx_scene.CreateGpuMaxRigidContactCountAttr().Set(1024 * 512)
        physx_scene.CreateGpuMaxRigidPatchCountAttr().Set(1024 * 80)
        physx_scene.CreateGpuMaxParticleContactsAttr().Set(2**20)
        return {
            "physics_scene_path": "/World/physicsScene",
            "gravity_direction": [0.0, 0.0, -1.0],
            "enable_gpu_dynamics": True,
            "broadphase_type": "GPU",
            "solver_type": "TGS",
        }

    def reset(self):
        """
        Reset task state.
        Initialize robot position, apply materials, and place objects.
        """
        super().reset()
        self.apply_liquid()
