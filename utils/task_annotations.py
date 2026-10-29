"""Semantic task and per-frame subtask labels for collected trajectories.

The labels intentionally describe meaningful task phases (pick, place, pour,
insert, and so on), rather than every controller motion event.  This keeps the
annotation stable when approach trajectories or controller timing change.
"""

from __future__ import annotations

import re
from typing import Any


TASK_NAMES = {
    "PickTaskController": "Pick up the object from the table",
    "OpenArmPickTaskController": "Pick up the object from the table",
    "PickLiquidTaskController": "Pick up the filled beaker, pour the liquid, and return it upright",
    "PlaceTaskController": "Pick up the object and place it at the target",
    "PourTaskController": "Pick up the source container and pour it into the target container",
    "StirTaskController": "Pick up the glass rod and stir the liquid",
    "OpenTaskController": "Open the laboratory device",
    "CloseTaskController": "Close the laboratory device",
    "SampleTransportTaskController": "Place three vial bottles into separate sample box slots",
    "SampleOrganitionTaskController": "Place the vial into the assigned sample box",
    "ScaleWeightsTaskController": "Place the three weights onto the scale",
    "WeighingFeedbackTaskController": "Use the live scale feedback to place the required weights and reach 80 grams",
    "CentrifugeLoadTaskController": "Load the loose NMR tube into the centrifuge",
    "CentrifugeLoadHardTaskController": "Load two NMR tubes into the centrifuge",
    "NMRInjectionTaskController": "Insert the NMR object into the sample box",
    "CohortAnalysisTaskController": "Place three vials into the cohort slots",
    "WeightSelectTaskController": "Select and place the required weights on the scale",
    "RackPlaceTaskController": "Insert the tube into the assigned rack socket",
    "RackRecallTaskController": "Insert the tube into the selected rack slot, remember it, then retrieve and return the tube",
    "PickPlaceTaskController": "Insert the tube into the assigned rack socket",
    "LiquidMixingController": "Pour three beakers into the mixing target and return them",
    "DeviceOperateController": "Open the device, place both beakers, and press the button",
    "DeviceHardOperateController": "Open the drying oven, load the beaker, close and reopen the door, then return the beaker",
    "ConditionModuleTaskController": "Operate the condition module with the beaker and return it",
    "ClearResetTaskController": "Clear the bench by placing the vial and tube into their holders",
    "PressTaskController": "Read the display and press the correct button",
    "MoveTaskController": "Move the gripper above the instructed sample-box socket",
    "SinglePipettingTaskController": "Install a tip, transfer 100 microliters, and discard the tip",
    "FormulaSpottingTaskController": "Dispense the fixed formulation into wells A1, A2, B1, and B2",
    "CentrifugeLoadRTaskController": "Place the tube in the slot opposite the occupied rotor slot",
    "CentrifugeLoadR1TaskController": "Load two tubes into the first empty symmetric slot pair",
}


def _phase_value(controller: Any) -> str:
    phase = getattr(controller, "current_phase", None)
    value = getattr(phase, "value", phase)
    return str(value or "").lower()


def _clean_name(value: Any, fallback: str = "object") -> str:
    if value is None:
        return fallback
    name = str(value).rsplit("/", 1)[-1]
    name = re.sub(r"[_-]+", " ", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name or fallback


def task_name_for(controller: Any) -> str:
    """Return the normalized overall instruction for one episode."""
    class_name = controller.__class__.__name__
    state = getattr(controller, "state", None) or {}

    if class_name in {"PickTaskController", "OpenArmPickTaskController"}:
        return f"Pick up the {_clean_name(state.get('object_name'))} from the table"
    if class_name == "SampleOrganitionTaskController":
        box = state.get("target_box_name")
        if box:
            return f"Place the vial into the {_clean_name(box)} sample box"
    if class_name in {"NMRInjectionTaskController", "PressTaskController", "MoveTaskController"}:
        instruction = state.get("task_instruction")
        if instruction:
            return str(instruction)

    configured = TASK_NAMES.get(class_name)
    if configured:
        return configured

    legacy = getattr(controller, "_language_instruction", None)
    if isinstance(legacy, str) and legacy.strip():
        return legacy.strip()
    return class_name.replace("TaskController", "").replace("Controller", "")


def _multi_object_subtask(controller: Any) -> str | None:
    specs = getattr(controller, "PHASE_SPECS", ())
    index = int(getattr(controller, "phase_index", 0))
    if not specs or index < 0 or index >= len(specs):
        return None
    spec = specs[index]
    name = str(spec.name)
    class_name = controller.__class__.__name__

    mappings = {
        "ScaleWeightsTaskController": {
            "pick_weight_0": "Pick up the 10 g weight",
            "place_weight_0": "Move and place the 10 g weight on the scale",
            "pick_weight_1": "Pick up the 20 g weight",
            "place_weight_1": "Move and place the 20 g weight on the scale",
            "pick_weight_2": "Pick up the 50 g weight",
            "place_weight_2": "Move and place the 50 g weight on the scale",
        },
        "WeighingFeedbackTaskController": {
            "pick_weight_0": "Pick up the 10 g weight",
            "place_weight_0": "Place the 10 g weight and observe the scale feedback",
            "pick_weight_1": "Pick up the 20 g weight",
            "place_weight_1": "Place the 20 g weight and observe the scale feedback",
            "pick_weight_2": "Pick up the 50 g weight",
            "place_weight_2": "Place the 50 g weight and wait for a stable 80 g reading",
        },
        "RackRecallTaskController": {
            "pick_for_insertion": "Pick up the target tube",
            "insert_in_remembered_slot": "Insert the tube into the selected rack slot and remember it",
            "recall_and_pick": "Recall the selected slot and retrieve the tube",
            "return_to_start": "Return the tube to its original position",
        },
        "CentrifugeLoadHardTaskController": {
            "pick_nmr_01": "Pick up NMR tube 1",
            "place_nmr_01": "Move and load NMR tube 1 into centrifuge slot 1",
            "pick_nmr_02": "Pick up NMR tube 2",
            "place_nmr_02": "Move and load NMR tube 2 into centrifuge slot 2",
        },
        "CohortAnalysisTaskController": {
            "pick_vial_00": "Pick up vial 1",
            "place_vial_00": "Move and place vial 1 into cohort slot 1",
            "pick_vial_01": "Pick up vial 2",
            "place_vial_01": "Move and place vial 2 into cohort slot 2",
            "pick_vial_02": "Pick up vial 3",
            "place_vial_02": "Move and place vial 3 into cohort slot 3",
        },
        "ClearResetTaskController": {
            "pick_vial": "Pick up the vial",
            "place_vial": "Move and insert the vial into the sample box",
            "pick_tube": "Pick up the tube",
            "place_tube": "Move and insert the tube into the rack socket",
        },
        "ConditionModuleTaskController": {
            "pick_beaker": "Pick up the beaker",
            "place_on_device": "Place the beaker on the condition module",
            "press_button": "Press the condition module button",
            "pick_beaker_back": "Pick up the beaker from the condition module",
            "place_beaker_back": "Return the beaker to its original place",
        },
    }
    if name in mappings.get(class_name, {}):
        return mappings[class_name][name]

    words = _clean_name(name)
    if spec.kind == "pick":
        return f"Pick up the {words.removeprefix('pick ')}"
    verb = "insert" if spec.kind == "insert" else spec.kind
    return f"Move and {verb} the {words.removeprefix('place ')}"


def subtask_name_for(controller: Any) -> str:
    """Return the semantic subtask label for the frame being cached."""
    class_name = controller.__class__.__name__
    phase = _phase_value(controller)
    state = getattr(controller, "state", None) or {}

    multi = _multi_object_subtask(controller)
    if multi:
        return multi

    if class_name in {"PickTaskController", "OpenArmPickTaskController"}:
        return f"Pick up the {_clean_name(state.get('object_name'))}"
    if class_name == "PickLiquidTaskController":
        atomic = getattr(controller, "pick_controller", None)
        event = int(getattr(atomic, "event_index", getattr(atomic, "_event", 0)))
        return "Pick up the filled beaker" if event <= 5 else "Pour the liquid and return the beaker upright"
    if class_name == "StirTaskController":
        pick = getattr(controller, "pick_controller", None)
        return "Pick up the glass rod" if pick is not None and not pick.is_done() else "Move the rod into the beaker and stir the liquid"
    if class_name == "PlaceTaskController":
        return "Pick up the object" if phase == "picking" else "Move and place the object at the target"
    if class_name == "PourTaskController":
        return "Pick up the source container" if phase == "picking" else "Move and pour into the target container"
    if class_name in {"PickPlaceTaskController", "RackPlaceTaskController"}:
        return "Pick up the tube" if phase == "picking" else "Move and insert the tube into the rack socket"
    if class_name == "NMRInjectionTaskController":
        target = _clean_name(state.get("target_label"), "selected socket")
        return "Pick up the NMR tube" if phase == "picking" else f"Move and insert the NMR tube into {target}"
    if class_name == "PressTaskController":
        button = _clean_name(state.get("selected_button"), "selected")
        return f"Press the {button} button according to the display"
    if class_name == "MoveTaskController":
        target = _clean_name(state.get("target_label"), "selected socket")
        return f"Move the gripper above {target}"
    if class_name == "SinglePipettingTaskController":
        return {
            "picking_object": "Install the pipette tip",
            "moving_object": "Aspirate 100 microliters from the source",
            "placing_object": "Dispense into the target and discard the tip",
        }.get(phase, "Complete the single pipetting transfer")
    if class_name == "FormulaSpottingTaskController":
        well = {
            "moving_object_to_target_01": "A1",
            "moving_object_to_target_02": "A2",
            "moving_object_to_target_03": "B1",
            "moving_object_to_target_04": "B2",
        }.get(phase)
        return "Pick up the pipetting tool" if phase == "picking_object" else f"Dispense into well {well or 'target'}"
    if class_name in {"CentrifugeLoadRTaskController", "CentrifugeLoadR1TaskController"}:
        return "Pick up the next tube" if phase.startswith("picking") else "Place the tube into the selected symmetric rotor slot"
    if class_name == "SampleOrganitionTaskController":
        box = _clean_name(state.get("target_box_name"), "target")
        return "Pick up the vial" if phase == "picking" else f"Move and place the vial into the {box} sample box"
    if class_name == "SampleTransportTaskController":
        match = re.search(r"(\d+)$", phase)
        index = int(match.group(1)) if match else 1
        return f"Pick up vial {index}" if phase.startswith("picking") else f"Move and place vial {index} into sample box slot {index}"
    if class_name == "WeightSelectTaskController":
        match = re.search(r"(\d+)$", phase)
        index = int(match.group(1)) if match else 1
        weight = {1: "10 g", 2: "20 g", 3: "50 g", 4: "100 g"}.get(
            index, f"selected {index}"
        )
        return f"Pick up the {weight} weight" if phase.startswith("picking") else f"Move and place the {weight} weight on the scale"
    if class_name == "LiquidMixingController":
        match = re.search(r"(\d+)$", phase)
        index = int(match.group(1)) if match else 1
        if phase.startswith("picking"):
            return f"Pick up beaker {index}"
        if phase.startswith("pouring"):
            return f"Pour beaker {index} into the mixing target"
        return f"Return beaker {index} to its place"
    if class_name == "DeviceOperateController":
        return {
            "opening_door": "Open the device door",
            "move_higher": "Move the arm to the clearance position",
            "picking_beaker": "Pick up beaker 1",
            "placing_beaker": "Place beaker 1 in the device",
            "picking_beaker3": "Pick up beaker 2",
            "placing_beaker3": "Place beaker 2 in the device",
            "pressing_button": "Press the device button",
        }.get(phase, "Operate the device")
    if class_name == "DeviceHardOperateController":
        return {
            "opening_door": "Open the drying oven door",
            "move_higher": "Move the arm clear of the open door",
            "picking_beaker3": "Pick up the loose beaker",
            "placing_beaker3": "Place the loose beaker inside the drying oven",
            "move_higher2": "Move the arm clear of the drying oven",
            "closing_door": "Close the drying oven door",
            "reopening_door": "Reopen the drying oven door",
            "picking_again_beaker3": "Retrieve the beaker from the drying oven",
            "placing_again_beaker3": "Return the beaker to its initial position",
        }.get(phase, "Operate the drying oven and return the beaker")
    if class_name == "ConditionModuleTaskController":
        return {
            "picking1": "Pick up the beaker",
            "placing1": "Place the beaker on the condition module",
            "press": "Press the condition module button",
            "picking2": "Pick up the beaker from the condition module",
            "placing2": "Return the beaker to its original place",
        }.get(phase, "Operate the condition module")
    if class_name == "OpenTaskController":
        return "Open the laboratory device"
    if class_name == "CloseTaskController":
        return "Close the laboratory device"

    return task_name_for(controller)
