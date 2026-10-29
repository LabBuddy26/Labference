from typing import Optional
import numpy as np

from isaacsim.core.utils.types import ArticulationAction
from robots.franka.rmpflow_controller import RMPFlowController
from scipy.spatial.transform import Rotation as R

from .base_controller import BaseController
from .atomic_actions.pick_controller import PickController
from .atomic_actions.stir_controller import StirController
class StirTaskController(BaseController):

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self.initial_position = None
            
    def _init_collect_mode(self, cfg, robot):
        super()._init_collect_mode(cfg, robot)
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            position_threshold=0.005,
            events_dt = [0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02],
            enable_scripted_attachment=True,
            require_attachment_proximity=False,
        )
        
        self.stir_controller = StirController(
            name="stir_controller",
            cspace_controller=self.rmp_controller,
        )

        self.gripper_control.release_object()
        self._last_pick_joint_data = None

    def _init_infer_mode(self, cfg, robot):
        super()._init_infer_mode(cfg, robot)
        
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            position_threshold=0.005,
            events_dt = [0.004, 0.002, 0.005, 0.02, 0.05, 0.004, 0.02],
            require_pose_reached=True,
            max_motion_phase_steps=1200,
            enable_scripted_attachment=True,
        )
        self.use_stir_model = False
        self.frame_count = 0
        self._scripted_pick_announced = False
        
    def reset(self):
        super().reset()
        self.gripper_control.release_object()
        self.pick_controller.reset()
        if self.mode == "collect":
            self.stir_controller.reset()
        else:
            self.inference_engine.reset()
        self.initial_position = None
        self.use_stir_model = False
        self.frame_count = 0
        self._scripted_pick_announced = False
    
    def step(self, state):
        if self.initial_position is None:
            self.initial_position = state['glass_rod_position']
        self.state = state
        if self.mode == "collect":
            return self._step_collect(state)
        else:
            return self._step_infer(state)
        
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
                picking_position=state.get('grasp_position', state['object_position']),
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="glass_rod",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                after_offset_z= 0.15,
                gripper_distances=0.002
            )
            self.cache_collection_step(state, action)

            self.gripper_control.update_grasped_object_position()
            return action, False, False
            
        elif not self.stir_controller.is_done():
            target_position = self.object_utils.get_object_xform_position(
                object_path=state['target_beaker']
            )
            if target_position is None:
                target_position = state['target_position']
            
            action = self.stir_controller.forward(
                center_position=target_position,
                current_joint_positions=state['joint_positions'],
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, -10])).as_quat(),
            )
            self.cache_collection_step(state, action)

            self.gripper_control.update_grasped_object_position()
            return action, False, False
            
        else:
            self.reset_needed = True
            final_object_position = state['glass_rod_position']
            target_position = state['target_position']
            self.gripper_control.release_object()
            if (final_object_position is not None and 
                final_object_position[2] > 0.85 and
                np.linalg.norm(final_object_position[0:2] - target_position[0:2]) < 0.04):
                self.write_cached_episode(state)
                self._last_success = True
                return None, True, True
            else:
                self.data_collector.clear_cache()
                self._last_success = False
                return None, True, False

    def _step_infer(self, state):
        """
        Executes one step in inference mode.
        Processes observations and generates control actions using learned policy.

        Args:
            state (dict): Current environment state

        Returns:
            tuple: (action, done, success) indicating control output and episode status
        """
        if self.pick_controller.has_failed():
            self._last_success = False
            self.reset_needed = True
            print(
                "Scripted stir pick failed: end effector could not reach the "
                "required pre-grasp/grasp pose.",
                flush=True,
            )
            return None, True, False

        if not self.pick_controller.is_done():
            if not self._scripted_pick_announced:
                print(
                    "Evaluation precondition: automatically grasping the "
                    "glass rod with the scripted controller; policy action "
                    "count remains zero.",
                    flush=True,
                )
                self._scripted_pick_announced = True
            action = self.pick_controller.forward(
                picking_position=state.get('grasp_position', state['object_position']),
                current_joint_positions=state['joint_positions'],
                object_size=state['object_size'],
                object_name="glass_rod",
                gripper_control=self.gripper_control,
                gripper_position=state['gripper_position'],
                end_effector_orientation=R.from_euler('xyz', np.radians([0, 90, 30])).as_quat(),
                after_offset_z=0.15,
                gripper_distances=0.002,
            )
            self.gripper_control.update_grasped_object_position()
            return action, False, False

        if not self.use_stir_model:
            # The deterministic pre-pick is valid only after the hand reached
            # the demonstrated grasp pose, closed, and lifted the attached rod.
            rod_position = np.asarray(state.get('glass_rod_position'), dtype=float)
            hand_position = np.asarray(state.get('gripper_position'), dtype=float)
            initial_position = np.asarray(self.initial_position, dtype=float)
            joints = np.asarray(state.get('joint_positions', []), dtype=float).reshape(-1)
            gripper_closed = bool(joints.size >= 9 and np.max(joints[7:9]) < 0.015)
            grasp_valid = bool(
                rod_position.shape == (3,)
                and hand_position.shape == (3,)
                and np.all(np.isfinite(rod_position))
                and gripper_closed
                and np.linalg.norm(rod_position - hand_position) < 0.12
                and rod_position[2] - initial_position[2] > 0.03
            )
            if not grasp_valid:
                self._last_success = False
                self.reset_needed = True
                print(
                    "Scripted stir pick failed: glass rod was not physically "
                    f"grasped (rod={rod_position.tolist()}, "
                    f"hand={hand_position.tolist()})."
                )
                return None, True, False
            self.use_stir_model = True
            print(
                "Scripted glass-rod grasp complete — switching to "
                "policy-driven stirring.",
                flush=True,
            )
        
        state['language_instruction'] = self.get_language_instruction()
        
        if self.use_stir_model:
            action = self.inference_engine.step_inference(state)
            self.gripper_control.update_grasped_object_position()
            return action, False, self._check_success()
        
        return ArticulationAction(), False, False
    
    def _check_success(self):
        object_pos = self.state['glass_rod_position']
        target_position = self.state['target_position']
        if object_pos[2] > 0.85 and np.linalg.norm(object_pos[0:2] - target_position[0:2]) < 0.04:
            self.check_success_counter += 1
            if self.check_success_counter > 240:
                self._last_success = True
                return True
        return False

    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        Override to provide dynamic instructions based on the current state.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        # Default instruction for shake tasks
        self._language_instruction = "Use the glass rod to stir the liquid."
        return self._language_instruction
