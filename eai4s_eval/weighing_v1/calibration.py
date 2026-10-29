"""Unscored load-cell calibration micro-scene for repeatable force checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cli import load_config
from .core import generate_instance


def run_calibration(config: str, output: str, *, seed: int = 101) -> dict[str, Any]:
    """Move calibration loads only in this probe, never in a scored rollout."""
    source, contract = load_config(config)
    instance = generate_instance(contract, seed)
    out = Path(output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "enable_cameras": False})
    report: dict[str, Any] = {"task_version": contract.task_version,
                              "contract_hash": contract.hash,
                              "scene_revision": source["scene"]["scene_revision"],
                              "sensor_source": "physx_loadcell_joint", "seed": seed,
                              "probe_only": True, "cases": []}
    try:
        from isaacsim.core.api import World
        from .scene import SceneHandles, build_scene, initial_load_positions
        from .loadcell import PhysxLoadCell
        from .runner import _contact_report, _read_contacts

        world = World(stage_units_in_meters=1.0, physics_prim_path="/World/physicsScene", backend="numpy")
        handles = build_scene(world, contract, instance, Path(__file__).resolve().parents[2],
                              enable_cameras=False)
        sensors, _ = _contact_report(handles.stage, handles)
        world.reset()
        handles.initialize()
        for sensor in sensors.values():
            sensor.initialize()
        cell = PhysxLoadCell(handles.stage, handles.loadcell_joint_path,
                             tray_path="/World/Scale/Tray",
                             filter_tau_s=contract.readout_tau_s,
                             display_quantum_g=contract.display_quantum_g,
                             stable_window_s=contract.stable_window_s,
                             stable_range_g=contract.stable_range_g)
        cell.initialize()
        report["empty_pan_zero_force_n"] = cell.calibrate(world)
        for case_index, masses in enumerate(((), (10,), (20,), (50,), (100,),
                                             (10, 50), (20, 100))):
            for mass, body in handles.weights.items():
                SceneHandles._set_initial_pose(body, handles.object_start_poses[f"{mass}g"])
            positions = initial_load_positions(handles.tray_center, handles.tray_top_z, masses)
            for mass in masses:
                SceneHandles._set_initial_pose(handles.weights[mass], positions[f"{mass}g"])
            samples = []
            for step in range(240):
                world.step(render=False)
                reading = cell.update(case_index * 240 + step, 1.0 / contract.physics_hz)
                if step >= 210 and reading.valid and not reading.overloaded:
                    samples.append(reading.raw_grams)
            contacts, _ = _read_contacts(sensors, (), .08, .08)
            actual_supported = tuple(weight for weight in (10, 20, 50, 100)
                                     if contacts[f"{weight}g"]["tray"])
            measured = sum(samples) / len(samples) if samples else None
            expected = sum(masses)
            error = measured - expected if measured is not None else None
            passed = bool(measured is not None and abs(error) <= 1.0 and
                          actual_supported == masses and not contacts["sample"]["tray"])
            report["cases"].append({"weights_g": list(masses), "expected_g": expected,
                                    "measured_g": measured, "error_g": error,
                                    "supported_weights_g": list(actual_supported),
                                    "sample_on_pan": contacts["sample"]["tray"],
                                    "pass": passed})
        report["pass"] = all(case["pass"] for case in report["cases"])
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error_message=str(exc))
        report["pass"] = False
        raise
    finally:
        out.write_text(json.dumps(report, indent=2) + "\n")
        try:
            app.close(skip_cleanup=True)
        except TypeError:
            app.close()
    return report
