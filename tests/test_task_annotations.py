import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from utils.task_annotations import subtask_name_for, task_name_for


def controller(class_name, **attributes):
    instance = type(class_name, (), {})()
    for name, value in attributes.items():
        setattr(instance, name, value)
    return instance


@dataclass
class PhaseSpec:
    name: str
    kind: str


class TaskAnnotationTest(unittest.TestCase):
    def test_place_has_pick_and_place_segments(self):
        item = controller(
            "PlaceTaskController",
            current_phase=SimpleNamespace(value="picking"),
            state={},
        )
        self.assertEqual(subtask_name_for(item), "Pick up the object")
        item.current_phase.value = "placing"
        self.assertEqual(
            subtask_name_for(item), "Move and place the object at the target"
        )

    def test_pick_liquid_boundary_uses_atomic_event(self):
        item = controller(
            "PickLiquidTaskController",
            pick_controller=SimpleNamespace(event_index=5),
            state={},
        )
        self.assertEqual(subtask_name_for(item), "Pick up the filled beaker")
        item.pick_controller.event_index = 6
        self.assertEqual(
            subtask_name_for(item),
            "Pour the liquid and return the beaker upright",
        )

    def test_multi_object_weight_labels_follow_phase_index(self):
        item = controller(
            "ScaleWeightsTaskController",
            PHASE_SPECS=(
                PhaseSpec("pick_weight_0", "pick"),
                PhaseSpec("place_weight_0", "place"),
            ),
            phase_index=0,
            state={},
        )
        self.assertEqual(subtask_name_for(item), "Pick up the 10 g weight")
        item.phase_index = 1
        self.assertEqual(
            subtask_name_for(item),
            "Move and place the 10 g weight on the scale",
        )

    def test_device_clearance_move_remains_explicit(self):
        item = controller(
            "DeviceOperateController",
            current_phase=SimpleNamespace(value="move_higher"),
            state={},
        )
        self.assertEqual(
            subtask_name_for(item), "Move the arm to the clearance position"
        )

    def test_liquid_mixing_has_pick_pour_and_return_for_each_beaker(self):
        item = controller(
            "LiquidMixingController",
            current_phase=SimpleNamespace(value="picking3"),
            state={},
        )
        self.assertEqual(subtask_name_for(item), "Pick up beaker 3")
        item.current_phase.value = "pouring3"
        self.assertEqual(
            subtask_name_for(item), "Pour beaker 3 into the mixing target"
        )
        item.current_phase.value = "placing3"
        self.assertEqual(subtask_name_for(item), "Return beaker 3 to its place")

    def test_dynamic_box_is_used_in_task_and_subtask(self):
        item = controller(
            "SampleOrganitionTaskController",
            current_phase=SimpleNamespace(value="placing"),
            state={"target_box_name": "box_B"},
        )
        self.assertEqual(task_name_for(item), "Place the vial into the box B sample box")
        self.assertEqual(
            subtask_name_for(item),
            "Move and place the vial into the box B sample box",
        )

    def test_weight_select_uses_the_physical_weight_for_sparse_phases(self):
        item = controller(
            "WeightSelectTaskController",
            current_phase=SimpleNamespace(value="picking_object_03"),
            state={"weight_sum": ["10g", "50g"]},
        )
        self.assertEqual(subtask_name_for(item), "Pick up the 50 g weight")
        item.current_phase.value = "placing_object_03"
        self.assertEqual(
            subtask_name_for(item), "Move and place the 50 g weight on the scale"
        )


if __name__ == "__main__":
    unittest.main()
