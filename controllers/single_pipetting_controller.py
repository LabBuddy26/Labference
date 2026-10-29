import re
from scipy.spatial.transform import Rotation as R
import numpy as np
from enum import Enum

from .atomic_actions.pickr_controller import PickRController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.move_controller import MoveController
from .base_controller import BaseController

class Phase(Enum):
    PICKING_OBJECT = "picking_object_01"
    MOVING_OBJECT = "moving_object_01"
    PLACING_OBJECT = "placing_object_01"
    FINISHED = "finished"

class SinglePipettingTaskController(BaseController):
    def __init__(self, cfg, robot):
        """Initialize the pick and pour task controller.
        
        Args:
            cfg: Configuration object containing controller settings
            robot: Robot instance to control
        """
        super().__init__(cfg, robot)
        self.initial_position = None

        self.initial_size = None
        self.current_phase = Phase.PICKING_OBJECT

    def _init_collect_mode(self, cfg, robot):
        """Initialize controller for data collection mode."""
        super()._init_collect_mode(cfg, robot)
        
        self.place_controller = PlaceController(
            name="place_controller",
            cspace_controller=self.rmp_controller,
            gripper=robot.gripper,
            legacy_timed_progression=self._main_compatible_collection,
        )
        self.pick_controller = PickRController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.01, 0.004, 0.01],
            legacy_timed_progression=self._main_compatible_collection,
        )
        self.move_controller = MoveController(
            name="move_controller",
            cspace_controller=self.rmp_controller,
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
        self.current_phase = Phase.PICKING_OBJECT
        self.initial_position = None

        self.initial_size = None
        self.pick_controller.reset()
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.place_controller.reset()
        else:
            self.inference_engine.reset()

    def _check_phase_success(self):
        """Check if current phase is successful based on object position."""
        object_pos = self.state['object_position']
        target_position = self.state['target_position'] 

        target_01_position = self.state['target_01_position']
        gripper_pos = self.state['gripper_position']

        if self.current_phase == Phase.PICKING_OBJECT:
            return object_pos[2] > self.initial_position[2] + 0.12
        elif self.current_phase == Phase.PLACING_OBJECT:
            success = (np.linalg.norm(object_pos[:2] - target_01_position[:2]) < 0.02 and abs(object_pos[2] - self.initial_position[2]) < 0.05)
            # print("xy", np.linalg.norm(object_pos[:2] - target_01_position[:2]), "z", abs(object_pos[2] - self.initial_position[2]))
            return success
        elif self.current_phase == Phase.MOVING_OBJECT:
            success = (np.linalg.norm(gripper_pos[:2] - target_position[:2]) < 0.02)
            return success
    
    def step(self, state):
        """Execute one step of control.
        
        Args:
            state: Current state dictionary containing sensor data and robot state
            
        Returns:
            Tuple containing action, done flag, and success flag
        """
        self.state = state
        if self.initial_position is None:
            self.initial_position = state['object_position']
  
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _step_collect(self, state):
        """Execute collection mode step."""
        success = self._check_phase_success()
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.active_controller.is_done():
            action = None
            if self.current_phase == Phase.PICKING_OBJECT:
                action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 180, 0])).as_quat(),
                    pre_offset_x=0.03,
                    pre_offset_z=0.1,
                    after_offset_z=0.3,
                    gripper_distances=float(self.cfg.collection_control.grasp_width_m),
                )
            elif self.current_phase == Phase.MOVING_OBJECT:
                action = self.move_controller.forward(
                    target_position=state['target_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_position=state['gripper_position'],
                    target_orientation=R.from_euler('xyz', np.radians([0, 180, 0])).as_quat(),
                )
            
            elif self.current_phase == Phase.PLACING_OBJECT:
                action = self.place_controller.forward(
                    place_position = state['target_01_position'],
                    current_joint_positions=state['joint_positions'],
                    gripper_control=self.gripper_control,
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 180, 0])).as_quat(),
                    gripper_position=state['gripper_position'],
                    pre_place_z = 0.3,
                    place_offset_z= 0.05
                )
            if 'camera_data' in state:
                self.cache_collection_step(state, action)
            
            return action, False, False

        if getattr(self.active_controller, "has_failed", lambda: False)():
            self._last_failure_reason = getattr(
                self.active_controller, "last_failure_reason", "atomic action failed"
            )
            self.data_collector.clear_cache()
            self._last_success = False
            self.current_phase = Phase.FINISHED
            self.reset_needed = True
            return None, True, False

        if success:
            if self.current_phase == Phase.PICKING_OBJECT:
                print("Pick task success! Switching to move...")
                self.current_phase = Phase.MOVING_OBJECT
                self.active_controller.reset()
                self.active_controller = self.move_controller
                return None, False, False
            
            elif self.current_phase == Phase.MOVING_OBJECT:
                print("move task success! Switching to place...")
                self.current_phase = Phase.PLACING_OBJECT
                self.active_controller.reset()
                self.active_controller = self.place_controller
                return None, False, False
            elif self.current_phase == Phase.PLACING_OBJECT:
                print("placing task success! Switching to finished")
                print("task success!")
                self.write_cached_episode(state)
                self._last_success = True
                self.current_phase = Phase.FINISHED
                return None, True, True
            else:
                print(f"{self.current_phase.value} task failed!")
                self.data_collector.clear_cache()
                self._last_success = False
                self.current_phase = Phase.FINISHED
                return None, True, False
        
        self._last_failure_reason = (
            f"{self.current_phase.value} finished without satisfying geometric success; "
            f"object={np.asarray(state['object_position']).tolist()}, "
            f"initial={np.asarray(self.initial_position).tolist()}, "
            f"gripper={np.asarray(state['gripper_position']).tolist()}, "
            f"target_01={np.asarray(state.get('target_01_position')).tolist()}"
        )
        self.data_collector.clear_cache()
        self._last_success = False
        self.current_phase = Phase.FINISHED
        self.reset_needed = True
        return None, True, False

    def _step_infer(self, state):
        """Execute inference mode step."""
        self.state = state
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success
        
        if self.current_phase == Phase.PICKING_OBJECT:
            action = None
            action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                )    
        else:
            state['language_instruction'] = self.get_language_instruction()
            action = self.inference_engine.step_inference(state)
        return action, False, self.is_success()

    def is_success(self):
        object_pos = self.state["object_position"]
        target_position = self.state['target_position']
        if (np.linalg.norm(object_pos[:2] - target_position[:2]) < 0.05 and abs(object_pos[2] - self.initial_position[2]) < 0.05):
            self._last_success = True
            self.current_phase = Phase.FINISHED
            return True
        return False

    def get_language_instruction(self) -> str:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').replace('  ',' ').lower()
        self._language_instruction = f"Pick up the {object_name} from the table and place it at the target"
        return self._language_instruction
