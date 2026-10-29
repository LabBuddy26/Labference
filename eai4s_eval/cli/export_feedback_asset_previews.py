"""Export display-only USD snapshots from the real feedback-task authoring code.

Run with env_isaaclab. This does not edit or replace simulator task assets.
"""
from pathlib import Path
import hashlib
import json


def main():
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
        from isaacsim.core.api import World
        from eai4s_eval.weighing_v1 import scene
        repo = Path(__file__).resolve().parents[2]
        output = repo / "eai4s_eval/configs/assets/derived_feedback"
        output.mkdir(parents=True, exist_ok=True)
        world = World(stage_units_in_meters=1.)
        stage = omni.usd.get_context().get_stage()
        scale = UsdGeom.Xform.Define(stage, scene.SCALE_PATH)
        scene._author_materials(stage)
        scene._rigid_box(stage, f"{scene.SCALE_PATH}/Base", (0, 0, .014),
                         (.30, .24, .028), "Body", mass_kg=1.2)
        scene._rigid_box(stage, scene.TRAY_PATH, (0, .018, .044),
                         (.24, .18, .012), "Tray", mass_kg=scene.PAN_MASS_KG)
        scene._author_display(stage, f"{scene.SCALE_PATH}/Display")
        group = UsdGeom.Xform.Define(stage, "/World/WeightSet")
        for index, mass in enumerate((10, 20, 50, 100)):
            scene._author_object(stage, f"/World/WeightSet/weight_{mass}g",
                                 ((index - 1.5) * .055, 0, .02), mass / 1000.,
                                 sample=False, label=str(mass))
        scene._author_object(stage, "/World/FeedbackSample", (0, 0, .022), .075, sample=True)
        source = stage.Flatten()
        for name, path in (("feedback-scale", scene.SCALE_PATH),
                           ("feedback-weight-set", "/World/WeightSet"),
                           ("feedback-sample", "/World/FeedbackSample")):
            target = Usd.Stage.CreateInMemory()
            Sdf.CopySpec(source, path, target.GetRootLayer(), "/Asset")
            if path != scene.SCALE_PATH:
                Sdf.CopySpec(source, scene.SCALE_PATH + "/Looks", target.GetRootLayer(), "/Asset/Looks")
            for prim in target.Traverse():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                for relationship in prim.GetRelationships():
                    targets = relationship.GetTargets()
                    if targets:
                        relationship.SetTargets([value.ReplacePrefix(Sdf.Path(scene.SCALE_PATH + "/Looks"),
                                                                      Sdf.Path("/Asset/Looks")) for value in targets])
                for attribute in prim.GetAttributes():
                    connections = attribute.GetConnections()
                    if connections:
                        attribute.SetConnections([value.ReplacePrefix(Sdf.Path(scene.SCALE_PATH + "/Looks"),
                                                                       Sdf.Path("/Asset/Looks")) for value in connections])
            target.SetDefaultPrim(target.GetPrimAtPath("/Asset"))
            UsdGeom.SetStageUpAxis(target, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(target, 1.)
            destination = output / f"{name}.usda"
            target.Export(str(destination))
            generator = Path(scene.__file__).resolve()
            destination.with_suffix(".source.json").write_text(json.dumps({
                "generator": str(generator.relative_to(repo)),
                "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
                "asset_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "purpose": "Display-only geometry snapshot; rigid-body dynamics disabled for rendering."
            }, indent=2) + "\n")
            print(f"EXPORTED {destination}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
