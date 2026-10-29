import re
from typing import Optional
import numpy as np
from robots.franka.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pickliquid_controller import PickController
# from .robot_controllers.trajectory_controller import FrankaTrajectoryController
# from .inference_engines.inference_engine_factory import InferenceEngineFactory
class PickLiquidTaskController(BaseController):
    """
    Controller for pick-and-place tasks with two operation modes:
    - Collection mode: Gathers training data through demonstrations
    - Inference mode: Executes learned policies for autonomous picking

    Attributes:
        mode (str): Operation mode ("collect" or "infer")
        REQUIRED_SUCCESS_STEPS (int): Number of consecutive steps needed for success
        success_counter (int): Counter for tracking successful steps
    """
    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
        pour_cfg = cfg.get("pick_liquid_pour", {})
        self.min_lift_m = float(pour_cfg.get("min_lift_m", 0.08))
        self.return_hold_steps = int(pour_cfg.get("return_hold_steps", 30))
        self.peak_lift_m = 0.0
        self.return_stable_counter = 0
     
    def _init_collect_mode(self, cfg, robot):
        """
        Initializes components for data collection mode.
        Sets up pick controller, gripper control, and data collector.

        Args:
            cfg: Configuration object containing collection settings
            robot: Robot instance to control
        """
        super()._init_collect_mode(cfg, robot)
        pour_cfg = cfg.get("pick_liquid_pour", {})
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=list(pour_cfg.get(
                "controller_events_dt",
                [0.006, 0.003, 0.01, 0.02, 0.02, 0.0012, 0.01, 0.01, 0.01, 0.0125],
            )),
            pour_angle_degrees=float(pour_cfg.get("pour_angle_degrees", 120.0)),
            pour_joint_velocity=float(pour_cfg.get("pour_joint_velocity", -0.45)),
            pour_hold_steps=int(pour_cfg.get("pour_hold_steps", 240)),
            return_joint_velocity=float(pour_cfg.get("return_joint_velocity", 0.30)),
            return_tolerance_degrees=float(
                pour_cfg.get("return_tolerance_degrees", 2.0)
            ),
        )

    def reset(self):
        super().reset()
        if self.mode == "collect":
            self.pick_controller.reset()
        else:
            self.inference_engine.reset()
        self.initial_position = None
        self.peak_lift_m = 0.0
        self.return_stable_counter = 0
    
    def step(self, state):
        if self.initial_position is None:
            self.initial_position = state['object_position']
        self.state = state
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
            
    def _check_success(self):
        current_lift_m = float(self.state['object_position'][2] - self.initial_position[2])
        self.peak_lift_m = max(self.peak_lift_m, current_lift_m)

        # Learned-policy inference has no scripted pour state machine.  Keep
        # its historical lift-only completion criterion, but make it match the
        # collection threshold used for this task.
        if self.mode != "collect":
            return current_lift_m >= self.min_lift_m

        # In collection, successful pouring is defined by the completed
        # tilt/reverse/hold cycle plus a real lift.  Testing it only after the
        # controller is finished avoids accepting a cup while it is still
        # tilted, and avoids the old 1,000-frame idle phases that hit the task
        # frame limit before success was ever evaluated.
        return (
            self.pick_controller.has_completed_pour_cycle()
            and current_lift_m >= self.min_lift_m
            and self.peak_lift_m >= self.min_lift_m
        )

    def _init_infer_mode(self, cfg, robot):
        """
        Initializes components for inference mode.
        Creates inference engine and trajectory controller.

        Args:
            cfg: Configuration object containing model paths and settings
            robot: Robot instance to control
        """
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot
        )
        
        self.inference_engine = InferenceEngineFactory.create_inference_engine(
            cfg, self.trajectory_controller
        )
        
    def _step_collect(self, state):
        """
        Executes one step in collection mode.
        Records demonstrations and manages episode transitions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        if not self.pick_controller.is_done():
            action = self.pick_controller.forward(
                articulation_controller=self.robot.get_articulation_controller(),
                current_joint_velocities=self.robot.get_joint_velocities(),
                picking_position=state['object_position'],
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name=state['object_name'],
                gripper_control=self.gripper_control,
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 25])).as_quat(),
                gripper_position=state['gripper_position'],
                pre_offset_x=0.05,
                after_offset_z=0.25
            )
            self.cache_collection_step(state, action)
            
            return action, False, False

        # The final atomic phase has restored the pre-pour wrist angle.  Keep
        # the robot at that target for a short, explicit settle window instead
        # of requiring a long generic success counter while the cup is tilted.
        if self._check_success():
            self.return_stable_counter += 1
        else:
            self.return_stable_counter = 0

        self._last_success = self.return_stable_counter >= self.return_hold_steps

        # Preserve the short upright settle interval in the recorded
        # demonstration.  There is no new arm command during this interval;
        # the tracker deliberately retains the last (closed) gripper command.
        self.cache_collection_step(state, None)

        if self._last_success:
            self.write_cached_episode(state)
            self.reset_needed = True
            return None, True, True

        return None, False, False
        
    def _step_infer(self, state):
        """
        Executes one step in inference mode.
        Uses inference engine to process observations and generate actions.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        language_instruction = self.get_language_instruction()
        state['language_instruction'] = language_instruction
            
        action = self.inference_engine.step_inference(state)
        
        if self._check_success():
            self.check_success_counter += 1
        else:
            self.check_success_counter = 0
            
        self._last_success = self.check_success_counter >= self.REQUIRED_SUCCESS_STEPS
        if self._last_success:
            self.reset_needed = True
            return action, True, True
        return action, False, False

    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        object_name = re.sub(r'\d+', '', self.state['object_name']).replace('_', ' ').replace('  ', ' ').lower()
        self._language_instruction = f"Pick up the {object_name} with water from the table and pour the water"
        return self._language_instruction
