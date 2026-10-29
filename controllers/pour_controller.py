import re
from typing import Optional
from scipy.spatial.transform import Rotation as R
import numpy as np
from enum import Enum
from utils.task_utils import TaskUtils

# from .atomic_actions.pickliquid_controller import PickController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.pour_controller import PourController
from .base_controller import BaseController

class Phase(Enum):
    PICKING = "picking"
    POURING = "pouring"
    FINISHED = "finished"

class PourTaskController(BaseController):
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
        self.initial_size = None
        self.task_utils = TaskUtils.get_instance()
        self.initial_quaternion = None
        self.initial_wrist_joint6 = None
        self.pour_timer = 0
        self.pour_complete = False
        self.return_complete = False
        self.return_timer = 0
        self.last_error_info = None
        self.current_phase = Phase.PICKING
        success_cfg = cfg.get("pour_success", {})
        self.min_pour_rotation_degrees = float(success_cfg.get("min_tilt_degrees", 70.0))
        self.return_rotation_tolerance_degrees = float(
            success_cfg.get("return_upright_tolerance_degrees", 15.0)
        )
        self.return_hold_steps = int(success_cfg.get("return_hold_steps", 60))
        configured_distance = success_cfg.get("max_xy_distance")
        self.max_pour_xy_distance = (
            None if configured_distance is None else float(configured_distance)
        )
        grasp_cfg = cfg.get("pour_grasp", {})
        self.pour_gripper_target_distance = float(
            grasp_cfg.get("gripper_target_distance", 0.012)
        )
        self.peak_pour_rotation_degrees = 0.0
        self.peak_wrist_rotation_degrees = 0.0
            
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        """Initialize controller for data collection mode."""
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, 0.02, 0.004, 0.008]
        )

        self.pour_controller = PourController(
            name="pour_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.006, 0.002, 0.009, 0.01, 0.009, 0.01]
        )
        self.active_controller = self.pick_controller

    def _init_infer_mode(self, cfg, robot=None):
        super()._init_infer_mode(cfg, robot)
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, 0.05, 0.01, 0.02]
        )

    def reset(self):
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self.initial_quaternion = None
        self.initial_wrist_joint6 = None
        self.pour_timer = 0
        self.pour_complete = False
        self.return_complete = False
        self.return_timer = 0
        self.peak_pour_rotation_degrees = 0.0
        self.peak_wrist_rotation_degrees = 0.0
        self.last_error_info = None
        self.pick_controller.reset()
        if self.mode == "collect":
            self.active_controller = self.pick_controller
            self.pour_controller.reset()
        else:
            self.inference_engine.reset()

    def _check_phase_success(self):
        """Check if current phase is successful."""
        object_pos = self.state['object_position']
        self.last_error_info = None 
        
        if self.initial_position is None:
            raise ValueError("initial_position not set")

        if self.current_phase == Phase.PICKING:
            required_height = self.initial_position[2] + 0.12
            success = object_pos[2] > required_height
            if not success:
                self.last_error_info = {
                    'phase': 'PICKING',
                    'current_height': object_pos[2],
                    'required_height': required_height,
                    'height_diff': object_pos[2] - required_height
                }
            return success
            
        elif self.current_phase == Phase.POURING:
            # The source centre is intentionally offset from the target centre
            # while its rim is placed over the target.  Keep the legacy
            # geometry-derived limit as a fallback for configs without an
            # explicit task-level tolerance.
            xy_dist = np.linalg.norm(object_pos[:2] - self.state['target_position'][:2])
            pour_threshold = self.max_pour_xy_distance
            if pour_threshold is None:
                pour_threshold = (
                    self.task_utils.get_pour_threshold(
                        self.state['object_name'], self.state['object_size']
                    )
                    + 0.05
                )

            if xy_dist > pour_threshold:
                self.last_error_info = {
                    'phase': 'POURING',
                    'current_distance': xy_dist,
                    'required_distance': pour_threshold,
                    'distance_diff': xy_dist - pour_threshold
                }
                return False

            # Do not use the pose at the start of the approach as the
            # reference.  Moving the wrist to the target can itself rotate a
            # held cup by more than 50 degrees and used to falsely mark an
            # episode successful before event 2 ever ran.
            if not self.pour_controller.has_started_pouring():
                self.last_error_info = {
                    'phase': 'POURING',
                    'error': 'Velocity-driven pour state has not started',
                    'controller_event': self.pour_controller.event_index,
                }
                return False

            current_quat = np.asarray(self.state['object_quaternion'], dtype=np.float64)
            current_wrist_joint6 = float(self.state['joint_positions'][6])
            if self.initial_quaternion is None or self.initial_wrist_joint6 is None:
                self.initial_quaternion = current_quat.copy()
                self.initial_wrist_joint6 = current_wrist_joint6
                self.last_error_info = {
                    'phase': 'POURING',
                    'error': 'Captured source and wrist pose at pour start',
                    'controller_event': self.pour_controller.event_index,
                }
                return False

            object_rotation_degrees = self.task_utils.rotation_angle_degrees(
                self.initial_quaternion, current_quat
            )
            wrist_rotation_degrees = abs(
                np.degrees(current_wrist_joint6 - self.initial_wrist_joint6)
            )
            # The commanded Franka wrist is the reliable kinematic evidence
            # for a pour.  Some referenced cups expose a static parent pose
            # even when their visual/rigid child rotates, so treating the
            # child quaternion as the sole signal incorrectly rejects videos
            # that visibly complete the pour.
            rotation_degrees = max(object_rotation_degrees, wrist_rotation_degrees)
            self.peak_pour_rotation_degrees = max(
                self.peak_pour_rotation_degrees, rotation_degrees
            )
            self.peak_wrist_rotation_degrees = max(
                self.peak_wrist_rotation_degrees, wrist_rotation_degrees
            )

            if not self.pour_complete:
                self.pour_complete = rotation_degrees >= self.min_pour_rotation_degrees
                if not self.pour_complete:
                    self.last_error_info = {
                        'phase': 'POURING',
                        'error': 'Pour rotation not complete yet',
                        'current_rotation_degrees': rotation_degrees,
                        'current_object_rotation_degrees': object_rotation_degrees,
                        'current_wrist_rotation_degrees': wrist_rotation_degrees,
                        'required_rotation_degrees': self.min_pour_rotation_degrees,
                        'peak_rotation_degrees': self.peak_pour_rotation_degrees,
                        'peak_wrist_rotation_degrees': self.peak_wrist_rotation_degrees,
                    }
                return False

            # A high rotation is only evidence of an attempted pour.  Require
            # the complete velocity sequence (tilt, pause, reverse, pause) so
            # an approach motion cannot be labeled as a successful pour.
            if not self.pour_controller.is_done():
                self.last_error_info = {
                    'phase': 'POURING',
                    'error': 'Pour controller has not completed its full cycle',
                    'controller_event': self.pour_controller.event_index,
                    'peak_rotation_degrees': self.peak_pour_rotation_degrees,
                }
                return False

            # After a completed cycle, require a tight return to the exact
            # orientation measured when pouring started.
            if not self.return_complete:
                if wrist_rotation_degrees <= self.return_rotation_tolerance_degrees:
                    self.return_complete = True
                    self.return_timer = 0
                else:
                    self.last_error_info = {
                        'phase': 'POURING',
                        'error': 'Return rotation not complete yet',
                        'current_wrist_rotation_degrees': wrist_rotation_degrees,
                        'required_rotation_degrees': self.return_rotation_tolerance_degrees,
                    }
                return False

            # Hold the returned, elevated cup for one simulation second before
            # labeling the sample.  Count frames rather than a hard-coded
            # fractional timer so the threshold is reproducible at 60 Hz.
            if self.return_complete and object_pos[2] > self.initial_position[2] + 0.05:
                self.return_timer += 1
                success = self.return_timer >= self.return_hold_steps
                if not success:
                    self.last_error_info = {
                        'phase': 'POURING',
                        'error': 'Waiting for return timer',
                        'return_timer': self.return_timer,
                        'required_steps': self.return_hold_steps,
                    }
                return success
            else:
                self.last_error_info = {
                    'phase': 'POURING',
                    'error': 'Object not in correct position for return timer',
                    'current_height': object_pos[2],
                    'required_height': self.initial_position[2] + 0.05,
                    'return_complete': self.return_complete
                }
                return False
        
        return False
    def step(self, state):
        """Execute one step of control.
        
        Args:
            state: Current state dictionary containing sensor data and robot state
            
        Returns:
            Tuple containing action, done flag, and success flag
        """
        self.state = state
        if self.initial_position is None:
            self.initial_position = self.state['object_position']
        if self.initial_size is None:
            self.initial_size = self.state['object_size']
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)

    def _step_collect(self, state):
        """Execute collection mode step."""
        success = self._check_phase_success()
        if success:
            if self.current_phase == Phase.PICKING:
                print("Pick task success! Switching to pour...")
                self.current_phase = Phase.POURING
                self.active_controller = self.pour_controller
                return None, False, False
            elif self.current_phase == Phase.POURING:
                print("Pour task success!")
                self.write_cached_episode(state)
                self._last_success = True
                self.current_phase = Phase.FINISHED
                return None, True, True
        
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.active_controller.is_done():
            action = None
            if self.current_phase == Phase.PICKING:
                action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                    pre_offset_x=0.05,
                    pre_offset_z=0.05,
                    after_offset_z=0.5,
                    gripper_distances=self.pour_gripper_target_distance,
                )
            else:
                action = self.pour_controller.forward(
                    articulation_controller=self.robot.get_articulation_controller(),
                    source_size=self.initial_size,
                    target_position=state['target_position'],
                    current_joint_velocities=self.robot.get_joint_velocities(),
                    pour_speed=-1,
                    source_name=state['object_name'],
                    gripper_position=state['gripper_position'],
                )
            self.cache_collection_step(state, action)

            return action, False, False

        if self.current_phase == Phase.POURING and self.pour_complete:
            # PourController finishes issuing joint commands before the task
            # predicate's post-return hold window is complete.  Previously we
            # fell through to the generic failure below on that first frame,
            # so return_hold_steps could never reach its required count even
            # for a visibly correct pour.  Keep the final wrist command in
            # effect and let _check_phase_success() accumulate stable frames.
            return None, False, False

        print(f"{self.current_phase.value} task failed!")
        if self.last_error_info is not None:
            print(f"Phase failure details: {self.last_error_info}")
        self.data_collector.clear_cache()
        self._last_success = False
        self.current_phase = Phase.FINISHED
        return None, True, False

    def _step_infer(self, state):
        """Execute inference mode step."""
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success

        if not self.pick_controller.is_done():
            action = None
            action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 15])).as_quat(),
                    gripper_distances=self.pour_gripper_target_distance,
                )
            
        else:
            state['language_instruction'] = self.get_language_instruction()

            action = self.inference_engine.step_inference(state)
        success = self._check_phase_success()
        if success and self.current_phase == Phase.PICKING:
            print("Pick task success! Switching to pour...")
            self.current_phase = Phase.POURING
        elif success and self.current_phase == Phase.POURING:
            print("Pour task success!")
            self._last_success = True
            self.current_phase = Phase.FINISHED
            return None, True, True
               
        return action, False, False

    def get_language_instruction(self) -> Optional[str]:
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').lower()
        self.language_instruction = f"Pick up the {object_name} from the table and pour it into the target".replace("  ", " ")
        return self.language_instruction
