from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from eai4s_eval.public_site import OBJECT_CATALOG_PATH, REPOSITORY_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render one LabBench USD object on a neutral studio background.")
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--settle-steps", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The public view intentionally strips source paths and hides unaudited
    # previews. A renderer must use the source catalogue, not that public view.
    catalog = json.loads(OBJECT_CATALOG_PATH.read_text(encoding="utf-8"))
    item = next(
        (value for value in catalog["objects"] if value["object_id"] == args.object_id),
        None,
    )
    if item is None:
        raise ValueError(f"unknown object id: {args.object_id}")
    if not item["asset_paths"]:
        raise ValueError(f"object {args.object_id} has no renderable USD asset")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {"headless": True, "width": args.resolution, "height": args.resolution}
    )
    try:
        import omni.usd
        from PIL import Image
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.sensors.camera import Camera

        world = World(stage_units_in_meters=1.0)
        stage = omni.usd.get_context().get_stage()
        asset_path = REPOSITORY_ROOT / item["asset_paths"][0]
        add_reference_to_stage(str(asset_path), "/World/Object")
        # Converted CAD assets often instance their mesh prototypes. Make only
        # this temporary render-stage reference editable for material override.
        while True:
            instances = [value for value in Usd.PrimRange(stage.GetPrimAtPath("/World/Object"))
                         if value.IsInstance()]
            if not instances:
                break
            for value in instances:
                value.SetInstanceable(False)
        # Catalogue previews are static; do not let imported task bodies fall
        # during the camera/material warmup. Source files are not modified.
        for body in Usd.PrimRange(stage.GetPrimAtPath("/World/Object")):
            if body.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(body).CreateRigidBodyEnabledAttr(False)
        world.reset()

        prim = stage.GetPrimAtPath("/World/Object")
        # Catalogue rotations are display-only.  The source USDs and the
        # simulator task transforms must remain unchanged.
        rotation = item.get("public_render_rotation_deg") or [0.0, 0.0, 0.0]
        if any(abs(float(value)) > 1e-6 for value in rotation):
            UsdGeom.Xformable(prim).AddRotateXYZOp(opSuffix="public_catalog").Set(
                Gf.Vec3f(*(float(value) for value in rotation))
            )
        category_colors = {
            "glassware": (0.04, 0.52, 0.55),
            "consumables": (0.36, 0.66, 0.16),
            "fixtures": (0.06, 0.24, 0.31),
            "instruments": (0.16, 0.34, 0.38),
            "tools": (0.84, 0.45, 0.10),
        }
        clay_material = UsdShade.Material.Define(stage, "/World/PublicClayMaterial")
        clay_shader = UsdShade.Shader.Define(stage, "/World/PublicClayMaterial/Surface")
        clay_shader.CreateIdAttr("UsdPreviewSurface")
        clay_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*category_colors.get(str(item.get("category_id")), (0.08, 0.46, 0.48)))
        )
        clay_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.34)
        clay_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.08)
        clay_material.CreateSurfaceOutput().ConnectToSource(
            clay_shader.ConnectableAPI(), "surface"
        )
        for descendant in Usd.PrimRange(prim):
            if descendant.IsA(UsdGeom.Gprim) and not item.get("public_render_preserve_materials", False):
                UsdShade.MaterialBindingAPI.Apply(descendant).Bind(
                    clay_material, UsdShade.Tokens.strongerThanDescendants
                )

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        minimum = np.asarray(box.GetMin(), dtype=np.float64)
        maximum = np.asarray(box.GetMax(), dtype=np.float64)
        center = (minimum + maximum) / 2.0
        radius = max(float(np.linalg.norm(maximum - minimum)) / 2.0, 0.05)

        light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/StudioLight"))
        light.CreateIntensityAttr(950.0)
        light.CreateColorAttr(Gf.Vec3f(0.94, 0.98, 0.96))

        camera_prim = UsdGeom.Camera.Define(stage, Sdf.Path("/World/PublicCamera"))
        # Z is the canonical image-up direction for public catalogue renders.
        view_direction = np.asarray(item.get("public_render_view_direction", (
            [.8, -2., 1.4] if item.get("public_render_preserve_materials") else [1.45, .65, 1.15])), dtype=np.float64)
        view_direction /= np.linalg.norm(view_direction)
        # Leave enough margin for display-only rotations whose projected
        # centre differs slightly from the authored USD origin.
        camera_distance = max(radius * float(item.get("public_render_distance_factor",
                                                     5.5 if item.get("public_render_preserve_materials") else 10.)), .3)
        eye = Gf.Vec3d(*(center + view_direction * camera_distance))
        target = Gf.Vec3d(*center)
        matrix = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
        xformable = UsdGeom.Xformable(camera_prim.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(matrix)
        camera_prim.CreateFocalLengthAttr(42.0)
        camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.001, max(100.0, radius * 20.0)))

        camera = Camera(
            prim_path="/World/PublicCamera",
            name="public_object_camera",
            resolution=(args.resolution, args.resolution),
        )
        camera.initialize()
        for _ in range(max(1, args.settle_steps)):
            world.step(render=True)
        rgba = np.asarray(camera.get_rgba(), dtype=np.uint8)
        if rgba.ndim != 3 or rgba.shape[-1] not in {3, 4}:
            raise RuntimeError(f"unexpected camera image shape: {rgba.shape}")
        rgb = rgba[..., :3]
        channel_range = np.ptp(rgb.astype(np.int16), axis=(0, 1))
        if float(np.max(channel_range)) < 8.0:
            raise RuntimeError(
                f"render for {args.object_id} has insufficient foreground contrast: "
                f"channel_range={channel_range.tolist()}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{args.output.stem}.", suffix=".webp", dir=args.output.parent, delete=False
        ) as handle:
            temporary_output = Path(handle.name)
        try:
            Image.fromarray(rgba).save(
                temporary_output, format="WEBP", quality=88, method=6
            )
            os.replace(temporary_output, args.output)
            args.output.chmod(0o644)
        finally:
            temporary_output.unlink(missing_ok=True)
        provenance = {
            "classification": "usd_render",
            "renderer": "eai4s_eval/cli/render_object_thumbnail.py",
            "source_asset": item["asset_paths"][0],
            "source_sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
            "image_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "image": (str(args.output.resolve().relative_to(REPOSITORY_ROOT))
                      if args.output.resolve().is_relative_to(REPOSITORY_ROOT)
                      else str(args.output.resolve())),
            "display_rotation_deg": rotation,
        }
        args.output.with_suffix(".provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        print(
            f"Rendered {args.object_id} from {asset_path} to {args.output}; "
            f"bounds_min={minimum.tolist()}, bounds_max={maximum.tolist()}, "
            f"camera_distance={camera_distance:.6f}"
        )
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
