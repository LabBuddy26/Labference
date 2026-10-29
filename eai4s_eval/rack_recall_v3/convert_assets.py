"""Convert the versioned rack-recall STEP sources into self-contained USD assets."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parent
ASSETS = {
    "rack": (ROOT / "assets/source/rack_3x4.step",
             ROOT / "assets/converted/rack_3x4.usd"),
    "tube": (ROOT / "assets/source/centrifuge_tube.step",
             ROOT / "assets/converted/centrifuge_tube.usd"),
}
OPTIONS = {
    "accurateSurfaceCurvatures": True,
    "accurateTessellation": False,
    "convertCurves": False,
    "convertHidden": False,
    "convertMetadata": True,
    "dedup": True,
    "globalXforms": False,
    "instancing": False,
    "materialType": 0,
    "reportProgress": True,
    "tessLOD": 1,
    "useMaterials": False,
    "useNormals": True,
    "iUpAxis": 2,
    "dMetersPerUnit": 1.0,
    "bOptimize": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


async def _convert(force: bool) -> dict:
    from omni.kit.converter.hoops_core import get_instance, is_format_supported
    from pxr import Usd

    converter = get_instance()
    if converter is None:
        raise RuntimeError("omni.kit.converter.hoops_core did not initialize")
    arguments = {key: str(value).lower() if isinstance(value, bool) else str(value)
                 for key, value in OPTIONS.items()}
    report = {"schema_version": "rack-recall-assets/v2", "options": OPTIONS,
              "assets": {},
              "frozen_geometry": {
                  "rack_grid": "3x4",
                  "row_centers_source_mm": [40.0, 90.0, 140.0],
                  "column_centers_source_mm": [32.5, 97.5, 162.5, 227.5],
                  "row_pitch_m": 0.050,
                  "column_pitch_m": 0.065,
                  "hole_radius_m": 0.021,
                  "rack_body_bounds_source_mm": [[-10.0, 0.0, -165.0],
                                                   [190.0, 260.0, 15.0]],
                  "tube_scale": 1.20,
                  "tube_cap_scale": 1.25,
                  "tube_body_diameter_m": 0.033864,
                  "tube_cap_diameter_m": 0.043125,
                  "tube_total_length_m": 0.14040,
              }}
    for name, (source, output) in ASSETS.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        if not is_format_supported(str(source)):
            raise RuntimeError(f"unsupported CAD format: {source}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if force or not output.is_file():
            _, status = await converter.create_converter_task(
                str(source), str(output), arguments)
            if status.error_code != 0:
                raise RuntimeError(
                    f"failed converting {source.name}: "
                    f"{status.error_code} {status.error_msg}")
        stage = Usd.Stage.Open(str(output))
        if stage is None:
            raise RuntimeError(f"converted layer cannot be opened: {output}")
        meshes = [str(prim.GetPath()) for prim in stage.TraverseAll()
                  if prim.GetTypeName() == "Mesh"]
        if not meshes:
            raise RuntimeError(f"converted layer has no meshes: {output}")
        report["assets"][name] = {
            "source": str(source.relative_to(ROOT)),
            "source_sha256": _sha256(source),
            "output": str(output.relative_to(ROOT)),
            "mesh_count": len(meshes),
            "mesh_paths": meshes,
        }
        if name == "rack":
            report["assets"][name]["calibration_locator_count"] = 12
            report["assets"][name]["calibration_locators_rendered"] = False
            report["assets"][name]["calibration_locators_collidable"] = False
        else:
            report["assets"][name]["part_roles"] = {
                "cap": next(path for path in meshes if "CutExtrude1" in path),
                "body": next(path for path in meshes if "CutExtrude3" in path),
            }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    app = SimulationApp({"headless": True})
    try:
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("omni.kit.converter.hoops_core", True)
        app.update()
        app.update()
        report = asyncio.run(_convert(args.force))
        output = ROOT / "assets/asset_manifest.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 0
    finally:
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())
