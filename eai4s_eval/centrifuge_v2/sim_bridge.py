"""Isaac Sim readback for the centrifuge contract.

Import only after SimulationApp starts. Runtime state comes from rigid-body
views and contact sensors, never from the scripted controller phase.
"""

from __future__ import annotations

import numpy as np

from .core import Contract, Observation, SLOTS, TUBES, classify_contacts, evaluate_insertion


_DIGIT_SEGMENTS = {
    "0": "abcfed", "1": "bc", "2": "abged", "3": "abgcd",
    "4": "fgbc", "5": "afgcd", "6": "afgecd", "7": "abc",
    "8": "abcdefg", "9": "abfgcd",
}
_SEGMENT_CENTERS = {
    "a": (0, .006), "g": (0, 0), "d": (0, -.006),
    "f": (-.004, .003), "b": (.004, .003),
    "e": (-.004, -.003), "c": (.004, -.003),
}


def author_slot_labels(stage, contract: Contract) -> int:
    """Add uniform, non-colliding two-digit markers to the rotor visual."""
    from pxr import Gf, UsdGeom

    root = UsdGeom.Xform.Define(stage, "/World/Centrifugal/DiagnosticLabels")
    rotor_cx, rotor_cy, _ = contract.rotor_center_offset_m
    rotor_cy = -2 * rotor_cy
    for slot in SLOTS:
        index = SLOTS.index(slot)
        x, y, _ = contract.slot_local_center(slot)
        radial = np.array([x - rotor_cx, y - rotor_cy], dtype=float)
        radial /= np.linalg.norm(radial)
        label = UsdGeom.Xform.Define(stage, f"{root.GetPath()}/{slot}")
        lx, ly = np.array([x, y]) + .046 * radial
        label.AddTranslateOp().Set(Gf.Vec3d(float(lx), float(ly), -.300))
        label.AddRotateZOp().Set(180)
        for digit_index, character in enumerate(f"{index:02d}"):
            for segment in _DIGIT_SEGMENTS[character]:
                sx, sy = _SEGMENT_CENTERS[segment]
                prim = UsdGeom.Cube.Define(stage, f"{label.GetPath()}/D{digit_index}_{segment}")
                prim.CreateSizeAttr(1.0)
                prim.AddTranslateOp().Set(Gf.Vec3d(sx + (digit_index - .5) * .012, sy, 0))
                horizontal = segment in "agd"
                prim.AddScaleOp().Set(Gf.Vec3f(.006 if horizontal else .0012,
                                               .0012 if horizontal else .005, .001))
                prim.CreateDisplayColorAttr([Gf.Vec3f(.92, .92, .90)])
    return len(SLOTS)


def author_tube_identity_marks(stage) -> int:
    from pxr import Gf, UsdGeom

    colors = {"NMR_01": Gf.Vec3f(.10, .70, .75),
              "NMR_02": Gf.Vec3f(.95, .65, .12)}
    for tube, color in colors.items():
        root_path = f"/World/{tube}"
        if not stage.GetPrimAtPath(root_path).IsValid():
            raise ValueError(f"missing tube for identity mark: {root_path}")
        mark = UsdGeom.Cylinder.Define(stage, f"{root_path}/IdentityBand")
        mark.CreateRadiusAttr(.0078)
        mark.CreateHeightAttr(.004)
        mark.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.066))
        mark.CreateDisplayColorAttr([color])
    return len(colors)


def slot_frames(stage, contract: Contract, rotor_path: str = "/World/Centrifugal") -> tuple[dict, tuple]:
    from pxr import Gf, Usd, UsdGeom

    rotor = stage.GetPrimAtPath(rotor_path)
    if not rotor.IsValid():
        raise ValueError(f"missing rotor: {rotor_path}")
    matrix = UsdGeom.Xformable(rotor).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    frames = {}
    for slot in SLOTS:
        point = matrix.Transform(Gf.Vec3d(*contract.slot_local_center(slot)))
        frames[slot] = tuple(float(v) for v in point)
    axis_point = matrix.Transform(Gf.Vec3d(0, 0, 1))
    origin = matrix.Transform(Gf.Vec3d(0, 0, 0))
    axis = np.array(axis_point) - np.array(origin)
    axis = axis / np.linalg.norm(axis)
    return frames, tuple(float(v) for v in axis)


class PhysxReadback:
    def __init__(self, contract: Contract, stage) -> None:
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.sensors.physics import ContactSensor

        self.contract = contract
        self.slots_world, self.insertion_axis_world = slot_frames(stage, contract)
        self.rigids = {}
        self.contacts = {}
        self.previous_positions = {}
        for tube in TUBES:
            path = f"/World/{tube}"
            if not stage.GetPrimAtPath(path).IsValid():
                raise ValueError(f"missing tube rigid body: {path}")
            self.rigids[tube] = SingleRigidPrim(prim_path=path, name=f"diagnostic_{tube}")
            self.contacts[tube] = ContactSensor(
                prim_path=f"{path}/DiagnosticContact", name=f"diagnostic_contact_{tube}",
                frequency=60, translation=np.zeros(3), min_threshold=0,
                max_threshold=1e7, radius=-1,
            )
            self.contacts[tube].add_raw_contact_data_to_frame()

    def initialize(self) -> None:
        for tube in TUBES:
            self.rigids[tube].initialize()
            self.contacts[tube].initialize()

    def measure(self, tube: str) -> dict:
        if tube not in TUBES:
            raise ValueError(f"unknown tube: {tube}")
        position, quat = self.rigids[tube].get_world_pose()
        old_position = self.previous_positions.get(tube)
        position_delta = float(np.linalg.norm(np.asarray(position) - old_position)) if old_position is not None else None
        self.previous_positions[tube] = np.asarray(position).copy()
        velocity = self.rigids[tube].get_linear_velocity()
        from scipy.spatial.transform import Rotation
        # Rigid prim quaternions are WXYZ; USD tube's long axis is local -Z.
        q = np.asarray(quat)
        axis = Rotation.from_quat([q[1], q[2], q[3], q[0]]).apply([0, 0, -1])
        frame = self.contacts[tube].get_current_frame()
        contact_classes = classify_contacts(frame.get("contacts", []))
        supported = contact_classes["rotor"]
        released = not contact_classes["gripper"]
        obs = Observation(tube, tuple(map(float, position)), tuple(map(float, axis)),
                          float(np.linalg.norm(velocity)), released, supported, position_delta)
        evidence = evaluate_insertion(self.contract, obs, self.slots_world, self.insertion_axis_world)
        evidence["contact_supported"] = supported
        evidence["gripper_contact"] = contact_classes["gripper"]
        evidence["left_finger_contact"] = contact_classes["left_finger"]
        evidence["right_finger_contact"] = contact_classes["right_finger"]
        evidence["both_fingers_contact"] = contact_classes["left_finger"] and contact_classes["right_finger"]
        evidence["contact_pair_count"] = len(frame.get("contacts", []))
        evidence["contact_force_n"] = float(frame.get("force", 0.0))
        evidence["position_world_m"] = list(map(float, position))
        evidence["orientation_wxyz"] = list(map(float, q))
        evidence["axis_world"] = list(map(float, axis))
        return evidence
