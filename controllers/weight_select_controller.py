import re
import numpy as np
from enum import Enum

from .atomic_actions.pickr_controller import PickRController
from .atomic_actions.place_controller import PlaceController
from .base_controller import BaseController
from utils.rotation_utils import isaac_quaternion_from_euler_deg


class Phase(Enum):
    PICKING_OBJECT1 = "picking_object_01"
    PLACING_OBJECT1 = "placing_object_01"
    PICKING_OBJECT2 = "picking_object_02"
    PLACING_OBJECT2 = "placing_object_02"
    PICKING_OBJECT3 = "picking_object_03"
    PLACING_OBJECT3 = "placing_object_03"
    PICKING_OBJECT4 = "picking_object_04"
    PLACING_OBJECT4 = "placing_object_04"

    FINISHED = "finished"

class WeightSelectTaskController(BaseController):
    def __init__(self, cfg, robot):
        """Initialize the pick and pour task controller.

        Args:
            cfg: Configuration object containing controller settings
            robot: Robot instance to control
        """
        super().__init__(cfg, robot)
        self.initial_object_position = None
        self.initial_object_01_position = None
        self.initial_object_02_position = None
        self.initial_object_03_position = None

        self.initial_size = None
        self.current_phase = Phase.PICKING_OBJECT1
        self.success_steps = set()
        self.weight_map = {"10g": 1, "20g": 2, "50g": 3,  "100g": 4}
        self.phase_sequence = None

    def _init_collect_mode(self, cfg, robot):
        """Initialize controller for data collection mode."""
        super()._init_collect_mode(cfg, robot)

        success_cfg = cfg.get("weight_select_success", {})
        self.pick_event_0_timeout_steps = int(
            success_cfg.get("pick_event_0_timeout_steps", 1200)
        )
        if self.pick_event_0_timeout_steps <= 0:
            raise ValueError(
                "weight_select_success.pick_event_0_timeout_steps must be positive"
            )

        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
        )
        self.pick_controller = PickRController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            # PickRController's motion timeout is reached when accumulated
            # event dt reaches 1.0.  1/1200 therefore gives event 0 an exact
            # 1200-step failure timeout while preserving reach-based advance.
            events_dt=[
                1.0 / self.pick_event_0_timeout_steps,
                0.002,
                0.01,
                0.02,
                0.01,
                0.004,
                0.01,
            ],
        )
        print(
            "[WeightSelectTaskController] pick event 0 timeout: "
            f"{self.pick_event_0_timeout_steps} steps",
            flush=True,
        )
        self.active_controller = self.pick_controller

    def _init_infer_mode(self, cfg, robot):
        """Initialize controller for inference mode."""
        self.pick_controller = PickRController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )
        super()._init_infer_mode(cfg, robot)

    def reset(self):
        """Reset controller state and phase."""
        super().reset()
        self.current_phase = Phase.PICKING_OBJECT1
        self.initial_object_position = None
        self.initial_object_01_position = None
        self.initial_object_02_position = None
        self.initial_object_03_position = None

        self.initial_size = None
        self.pick_controller.reset()
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.place_controller.reset()
        else:
            pass

        self.phase_sequence = None
        self.success_steps.clear()

    def _check_phase_success(self):
        """Check if current phase is successful based on object position."""
        object_pos = self.state['object_position']
        target_position = self.state['target_position']

        target_01_position = self.state['target_01_position']
        target_02_position = self.state['target_02_position']
        target_03_position = self.state['target_03_position']
        object_01_pos = self.state['object_01_position']
        object_02_pos = self.state['object_02_position']
        object_03_pos = self.state['object_03_position']

        if self.current_phase == Phase.PICKING_OBJECT1:
            return object_pos[2] > self.initial_object_position[2] + 0.12
        elif self.current_phase == Phase.PLACING_OBJECT1:
            success = (np.linalg.norm(object_pos[:2] - target_position[:2]) < 0.02 and abs(object_pos[2] - self.initial_object_position[2]) < 0.05)
            return success
        elif self.current_phase == Phase.PICKING_OBJECT2:
            return object_01_pos[2] > self.initial_object_01_position[2] + 0.12
        elif self.current_phase == Phase.PLACING_OBJECT2:
            success = (np.linalg.norm(object_01_pos[:2] - target_01_position[:2]) < 0.02 and abs(object_01_pos[2] - self.initial_object_01_position[2]) < 0.05)
            return success
        elif self.current_phase == Phase.PICKING_OBJECT3:
            return object_02_pos[2] > self.initial_object_02_position[2] + 0.12
        elif self.current_phase == Phase.PLACING_OBJECT3:
            success = (np.linalg.norm(object_02_pos[:2] - target_02_position[:2]) < 0.02 and abs(object_02_pos[2] - self.initial_object_02_position[2]) < 0.05)
            return success

        elif self.current_phase == Phase.PICKING_OBJECT4:
            return object_03_pos[2] > self.initial_object_03_position[2] + 0.12
        elif self.current_phase == Phase.PLACING_OBJECT4:
            success = (np.linalg.norm(object_03_pos[:2] - target_03_position[:2]) < 0.02 and abs(object_03_pos[2] - self.initial_object_03_position[2]) < 0.05)
            return success


    def _advance_to_next_phase(self):
        """Advance to the next phase of the task."""
        self.success_steps.add(self.current_phase)

        self.last_phase = self.current_phase
        self.current_phase = self.phase_sequence.get(self.current_phase, Phase.FINISHED)

        if self.mode == "collect":
            controller_map = {
                Phase.PICKING_OBJECT1: self.pick_controller,
                Phase.PLACING_OBJECT1: self.place_controller,
                Phase.PICKING_OBJECT2: self.pick_controller,
                Phase.PLACING_OBJECT2: self.place_controller,
                Phase.PICKING_OBJECT3: self.pick_controller,
                Phase.PLACING_OBJECT3: self.place_controller,
                Phase.PICKING_OBJECT4: self.pick_controller,
                Phase.PLACING_OBJECT4: self.place_controller,
            }
            self.active_controller = controller_map.get(self.current_phase)
            if self.active_controller:
                self.active_controller.reset()

    def step(self, state):
        """Execute one step of control.

        Args:
            state: Current state dictionary containing sensor data and robot state

        Returns:
            Tuple containing action, done flag, and success flag
        """
        self.state = state
        if self.initial_object_position is None:
            self.initial_object_position = state['object_position']
        if self.initial_object_01_position is None:
            self.initial_object_01_position = state['object_01_position']
        if self.initial_object_02_position is None:
            self.initial_object_02_position = state['object_02_position']
        if self.initial_object_03_position is None:
            self.initial_object_03_position = state['object_03_position']
        if self.phase_sequence is None:
            self._collect_phase(state['weight_sum'])

        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _collect_phase(self, weight_sum):
        self.phase_sequence = {}
        weight_index = [self.weight_map[w] for w in weight_sum]
        for i, wi in enumerate(weight_index):
            picking = getattr(Phase, f"PICKING_OBJECT{wi}")
            placing = getattr(Phase, f"PLACING_OBJECT{wi}")
            self.phase_sequence[picking] = placing
            if i < len(weight_index) - 1:
                next_picking = getattr(Phase, f"PICKING_OBJECT{weight_index[i+1]}")
                self.phase_sequence[placing] = next_picking
            else:
                self.phase_sequence[placing] = Phase.FINISHED

    def _step_collect(self, state):
        """Execute collection mode step."""
        success = self._check_phase_success()
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            self._last_success = len(self.success_steps) == len(state['weight_sum']) * 2
            return None, True, self._last_success

        if not self.active_controller.is_done():
            action = None
            if self.current_phase == Phase.PICKING_OBJECT1:
                action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=isaac_quaternion_from_euler_deg((0.0, 180.0, 0.0)),
                    pre_offset_x=0.03,
                    pre_offset_z=0.1,
                    after_offset_z=0.3
                )
            elif self.current_phase == Phase.PLACING_OBJECT1:
                action = self.place_controller.forward(
                    place_position = state['target_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=isaac_quaternion_from_euler_deg((0.0, 180.0, 0.0)),
                    gripper_position=state['gripper_position'],
                    pre_place_z = 0.3,
                    place_offset_z= 0.01
                )

            elif self.current_phase == Phase.PICKING_OBJECT2:

                action = self.pick_controller.forward(
                    picking_position=state['object_01_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_01_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=isaac_quaternion_from_euler_deg((0.0, 180.0, 0.0)),
                    pre_offset_x=0.03,
                    pre_offset_z=0.1,
                    after_offset_z=0.3
                )
            elif self.current_phase == Phase.PLACING_OBJECT2:
                action = self.place_controller.forward(
                    place_position = state['target_01_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=isaac_quaternion_from_euler_deg((0.0, 180.0, 0.0)),
                    gripper_position=state['gripper_position'],
                    pre_place_z = 0.3,
                    place_offset_z= 0.02
                )

            elif self.current_phase == Phase.PICKING_OBJECT3:
                action = self.pick_controller.forward(
                    picking_position=state['object_02_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_02_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=isaac_quaternion_from_euler_deg((0.0, 180.0, 0.0)),
                    pre_offset_x=0.03,
                    pre_offset_z=0.1,
                    after_offset_z=0.3
                )
            elif self.current_phase == Phase.PLACING_OBJECT3:
                action = self.place_controller.forward(
                    place_position = state['target_02_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=isaac_quaternion_from_euler_deg((0.0, 180.0, 0.0)),
                    gripper_position=state['gripper_position'],
                    pre_place_z = 0.3,
                    place_offset_z= 0.05
                )
            elif self.current_phase == Phase.PICKING_OBJECT4:
                action = self.pick_controller.forward(
                    picking_position=state['object_03_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_03_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=isaac_quaternion_from_euler_deg((0.0, 180.0, 0.0)),
                    pre_offset_x=0.03,
                    pre_offset_z=0.1,
                    after_offset_z=0.3
                )
            elif self.current_phase == Phase.PLACING_OBJECT4:
                action = self.place_controller.forward(
                    place_position = state['target_03_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=isaac_quaternion_from_euler_deg((0.0, 180.0, 0.0)),
                    gripper_position=state['gripper_position'],
                    pre_place_z = 0.3,
                    place_offset_z= 0.05
                )
            self.cache_collection_step(state, action)

            return action, False, False

        if success:
            print(f"{self.current_phase.value} success!")
            self._advance_to_next_phase()
            if self.current_phase == Phase.FINISHED:
                self.write_cached_episode(state)
            return None, False, False
        else:
            print(f"{self.current_phase.value} failed!")
            self.data_collector.clear_cache()
            self.current_phase = Phase.FINISHED
            return None, True, False

        return None, False, False

    def _step_infer(self, state):
        """Execute inference mode step."""
        self.state = state
        if self.current_phase != Phase.FINISHED:
            success = self._check_phase_success()
            if success:
                print(f"Inference: {self.current_phase.value} success!")
                self.success_steps.add(self.current_phase)
                self._advance_to_next_phase()
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            expected = len(self.phase_sequence) if self.phase_sequence else 8
            self._last_success = len(self.success_steps) == expected
            return None, True, self._last_success

        state['language_instruction'] = self.get_language_instruction()
        action = self.inference_engine.step_inference(state)
        return action, False, False

    def is_success(self):
        """Check if task was successful."""
        expected = len(self.phase_sequence) if self.phase_sequence else 8
        return len(self.success_steps) == expected

    def get_language_instruction(self) -> str:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.

        Returns:
            Optional[str]: The language instruction or None if not available
        """
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').replace('  ',' ').lower()
        self._language_instruction = f"Pick up the {object_name} from the table and place it at the target"
        return self._language_instruction
