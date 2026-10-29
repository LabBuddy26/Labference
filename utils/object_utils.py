from pxr import Usd, UsdGeom, Gf, UsdPhysics
from isaacsim.core.utils.stage import get_stage_units
import numpy as np

class ObjectUtils:
    _instance = None

    @classmethod
    def get_instance(cls, stage: Usd.Stage = None, default_path: str = "/World") -> "ObjectUtils":
        if cls._instance is None:
            if stage is None:
                raise ValueError("Stage must be provided for first instance")
            cls._instance = cls(stage, default_path)
        return cls._instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ObjectUtils, cls).__new__(cls)
        return cls._instance

    def __init__(self, stage: Usd.Stage, default_path: str = "/World"):
        if not hasattr(self, '_initialized'):
            self._stage = stage
            self._default_path = default_path
            self._pick_height_offsets = {
                "rod": 0.04,
                "tube": 0.01,
                "beaker": 0.0,
                "erlenmeyer flask": 0.018,
                "cylinder": 0.0,
                "petri dish": 0.005,
                "pipette": 0.008,
                "microscope slide": 0.002
            }
            self._initialized = True

    def _get_object_path(self, object_name: str = None, object_path: str = None) -> str:
        if object_path:
            return object_path
        if object_name:
            return f"{self._default_path}/{object_name}"
        raise ValueError("Either object_name or object_path must be provided")

    def get_pick_position(self, object_name: str = None, object_path: str = None) -> np.ndarray:
        """Get the object's pick position with height offset."""
        position = self.get_geometry_center(object_name, object_path)
        if position is None:
            return None

        name = object_name or object_path.split('/')[-1]
        for key, offset in self._pick_height_offsets.items():
            if key in name.lower():
                position[2] += offset / get_stage_units()
                return position

        position[2] += 0.02 / get_stage_units()
        return position

    def get_object_size(self, object_name: str = None, object_path: str = None) -> np.ndarray:
        """Get the world-space size of an object."""
        path = self._get_object_path(object_name, object_path)
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        min_point = bbox.GetRange().GetMin()
        max_point = bbox.GetRange().GetMax()
        world_min_point = np.array([min_point[0], min_point[1], min_point[2], 1.0])
        world_max_point = np.array([max_point[0], max_point[1], max_point[2], 1.0])
        transform_matrix = bbox.GetMatrix()
        transform_matrix_np = np.array(transform_matrix)
        world_size = world_max_point @ transform_matrix_np - world_min_point @ transform_matrix_np
        return world_size[:3]

    def get_object_xform_position(self, object_path: str) -> np.ndarray:
        """Get the world-space position from the object's transform."""
        prim = self._stage.GetPrimAtPath(object_path)
        
        if not prim.IsValid():
            print(f"Object at path {object_path} not found.")
            return None

        xformable = UsdGeom.Xformable(prim)
        transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        position = transform.ExtractTranslation()
        return np.array(position)

    def get_local_point_world_position(
        self,
        object_path: str,
        local_point: np.ndarray,
    ) -> np.ndarray:
        """Transform a point from a prim's local frame into world space."""
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            raise ValueError(f"Object at path {object_path} not found")
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        point = transform.Transform(Gf.Vec3d(*np.asarray(local_point, dtype=np.float64)))
        return np.asarray(point, dtype=np.float64)
            
    def get_geometry_center(self, object_name: str = None, object_path: str = None) -> np.ndarray:
        path = self._get_object_path(object_name, object_path)
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        range_3d = bbox.GetRange()
        center = (np.array(range_3d.GetMin()) + np.array(range_3d.GetMax())) / 2.0
        center_hom = np.append(center, 1.0)
        transform = np.array(bbox.GetMatrix())
        world_center = center_hom @ transform
        return world_center[:3]

    
    def get_transform_quat(self, object_path: str, w_first: bool = False) -> np.ndarray:
        """Get the current world-space rotation quaternion of an object.
        
        Args:
            object_path: The USD path to the object.
            w_first: If True, return quaternion in [w, x, y, z] format, else [x, y, z, w].
                    Default is False.
        """
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            print(f"Object at path {object_path} not found.")
            return None

        # Some referenced assets use a non-rigid Xform as their public root
        # and place the actual PhysX body below it.  For example, beaker_2 is
        # rooted at /World/beaker while /World/beaker/Root/beaker_2/mesh owns
        # PhysicsRigidBodyAPI.  The root remains at its authored pose while
        # the descendant receives the simulated rotation.
        pose_prim = prim
        if not pose_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            for descendant in Usd.PrimRange(prim):
                if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
                    pose_prim = descendant
                    break

        # Dynamic PhysX objects update their world transform during the
        # simulation.  Reading only ``xformOp:orient`` returns the authored
        # reset orientation and made a rotating poured cup appear to have a
        # constant 0-degree change.  Extract the rotation from the complete
        # local-to-world matrix so this works for orient, rotateXYZ, and
        # physics-authored transform operations alike.
        transform = UsdGeom.Xformable(pose_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        rotation = transform.ExtractRotationQuat()
        quat_xyzw = np.array(
            [
                rotation.GetImaginary()[0],
                rotation.GetImaginary()[1],
                rotation.GetImaginary()[2],
                rotation.GetReal(),
            ],
            dtype=np.float64,
        )
        return (
            np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                dtype=np.float64,
            )
            if w_first
            else quat_xyzw
        )
        
    def get_revolute_joint_positions(self, joint_path: str) -> np.ndarray:
        joint_prim = self._stage.GetPrimAtPath(joint_path)

        joint_api = UsdPhysics.Joint(joint_prim)
        body1 = joint_api.GetBody1Rel().GetTargets()
        if not body1:
            raise RuntimeError(
                f"Revolute joint {joint_path} has no body1 relationship; "
                "resolve and cache the joint centre before simulation playback"
            )

        body1_prim = self._stage.GetPrimAtPath(body1[0])

        body1_xform = UsdGeom.Xformable(body1_prim)
        body1_world_transform = Gf.Matrix4f(body1_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

        local_pos1 = joint_api.GetLocalPos1Attr().Get() 
        local_rot1 = joint_api.GetLocalRot1Attr().Get()  

        rotation_matrix = Gf.Matrix3f(local_rot1)
        local_transform = Gf.Matrix4f()
        local_transform.SetTranslateOnly(local_pos1)
        local_transform.SetRotateOnly(rotation_matrix)
        joint_position = local_transform * body1_world_transform
        return np.array(joint_position.ExtractTranslation())
    
    def set_object_position(self, object_path: str, position: np.ndarray, local_position: np.ndarray = None, position_offset: np.ndarray = None) -> None:
        """Set the object's position in world or local space."""
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            print(f"Object at path {object_path} not found.")
            return

        xformable = UsdGeom.Xformable(prim)
        xform_ops = xformable.GetOrderedXformOps()
        if local_position is not None and position_offset is not None:
            new_position = Gf.Vec3d(*(local_position + position_offset))
        else:
            new_position = Gf.Vec3d(*position)

        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                    op.Set(Gf.Vec3d(*new_position))
                else:
                    op.Set(Gf.Vec3f(*new_position))
                return
        xformable.AddTranslateOp().Set(new_position)
    
    def set_object_orientation(self, object_path: str, orientation: np.ndarray) -> None:
        """Set the object's position in world or local space."""
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            print(f"Object at path {object_path} not found.")
            return

        xformable = UsdGeom.Xformable(prim)
        xform_ops = xformable.GetOrderedXformOps()
        quat = Gf.Quatf(*orientation)

        if xform_ops:
            for op in xform_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                        quat = Gf.Quatd(*orientation)
                    else:
                        quat = Gf.Quatf(*orientation)
                    op.Set(quat)
                    return
            xformable.AddOrientOp().Set(quat)
        else:
            xformable.AddOrientOp().Set(quat)


    def set_object_scale(self, object_path: str, scale: np.ndarray) -> None:
        """Set the object's scale (uniform or non-uniform)"""
        prim = self._stage.GetPrimAtPath(object_path)
        if not prim.IsValid():
            print(f"Object at path {object_path} not found.")
            return

        xformable = UsdGeom.Xformable(prim)
        xform_ops = xformable.GetOrderedXformOps()

        scale_vec = Gf.Vec3f(*scale)
        # from isaacsim.core.prims import XFormPrim
        # prim = XFormPrim(prim_paths_expr=object_path)
        # prim.set_local_scales(np.array([scale]))

        if xform_ops:
            for op in xform_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    op.Set(scale_vec)
                    return
            xformable.AddScaleOp().Set(scale_vec)
        else:
            xformable.AddScaleOp().Set(scale_vec)
