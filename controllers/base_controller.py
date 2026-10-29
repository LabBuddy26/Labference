from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from omegaconf import OmegaConf
from controllers.inference_engines.inference_engine_factory import InferenceEngineFactory
from controllers.robot_controllers.grapper_manager import Gripper
from controllers.robot_controllers.trajectory_controller import FrankaTrajectoryController
from controllers.gripper_action_tracker import GripperActionTracker
from factories.collector_factory import create_collector
from utils.object_utils import ObjectUtils
from isaacsim.core.experimental.utils import transform as transform_utils
from robots.franka.rmpflow_controller import RMPFlowController as FrankaRMPFlowController
from robots.openarm.rmpflow_controller import OpenArmRMPFlowController
from utils.task_annotations import subtask_name_for, task_name_for
class BaseController(ABC):
    """Base class for all controllers in the chemistry lab simulator.
    
    Provides common functionality for robot control, state management,
    and episode tracking.
    """
    
    def __init__(self, cfg, robot, use_default_config=True):
        """Initialize the base controller.
        
        Args:
            cfg: Configuration object containing controller settings
            robot: Robot instance to control
            object_utils: Utility class for object manipulation
        """
        self.cfg = cfg
        self.robot = robot
        self.object_utils = ObjectUtils.get_instance()
        self.reset_needed = False
        self._last_success = False
        self._episode_num = 0
        self.success_count = 0 
        self._language_instruction = ""
        self.gripper_control = Gripper()
        collection_control = cfg.get("collection_control", {})
        self._main_compatible_collection = bool(
            collection_control.get("main_compatible", False)
        )
        robot_type = str(getattr(cfg.robot, "type", "franka")).lower()
        self._robot_type = robot_type
        gripper_joint_indices = (14, 15) if robot_type == "openarm" else (7, 8)
        self._gripper_action_tracker = GripperActionTracker(gripper_joint_indices)
        # self.gripper_control = None
        self.REQUIRED_SUCCESS_STEPS = 60
        self.check_success_counter = 0
        self.rmp_controller = None
        self._last_failure_reason = ""

        if self.cfg.robot.type == "franka":

            self.rmp_controller = FrankaRMPFlowController(
                name="target_follower_controller",
                robot_articulation=robot,
                # use_default_config=use_default_config
            )
        else:
            self.rmp_controller = OpenArmRMPFlowController(
                name="target_follower_controller",
                robot_articulation=robot,
                # use_default_config=use_default_config
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        if hasattr(cfg, "mode"):
            self.mode = cfg.mode # "collect", "infer", or benchmark "eval"
            if self.mode == "collect":
                self._init_collect_mode(cfg, robot)
            elif self.mode in {"infer", "eval"}:
                # ``eval`` is deliberately the same low-level control path as
                # inference.  The evaluation runner replaces this deferred
                # engine with its validated policy bridge before stepping.
                self._init_infer_mode(cfg, robot)
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Expected 'collect', 'infer', or 'eval'.")

    @property
    def language_instruction(self) -> Optional[str]:
        """Get the current language instruction for the task.
        
        Returns:
            Optional[str]: The language instruction or None if not set
        """
        return self._language_instruction
    
    @language_instruction.setter
    def language_instruction(self, instruction: Optional[str]):
        """Set the language instruction for the task.
        
        Args:
            instruction: The language instruction to set, or None to clear
        """
        self._language_instruction = instruction
    
    def get_language_instruction(self) -> Optional[str]:
        """Get the language instruction for the current task.
        This method can be overridden by subclasses to provide dynamic instructions.
        
        Returns:
            Optional[str]: The language instruction or None if not available
        """
        return self._language_instruction

    def get_task_name(self) -> str:
        """Return the normalized overall instruction stored for the episode."""
        return task_name_for(self)

    def get_subtask_name(self) -> str:
        """Return the semantic phase label stored for the current frame."""
        return subtask_name_for(self)

    def transform_ee_pose(self, target_joint_position) -> np.ndarray:
        """Compute the Franka end-effector action for a robot joint state.

        Returns [x, y, z, qw, qx, qy, qz, finger], with the scalar-first
        (w, x, y, z) quaternion convention used by Isaac Sim and matched by
        the evaluator's EE execution path.
        """
        if self._robot_type != "franka":
            raise ValueError(
                "actions_ee collection is currently defined only for the Franka robot"
            )
        joint_state = np.asarray(target_joint_position, dtype=np.float64).reshape(-1)
        if joint_state.size < 8:
            raise ValueError(
                f"Franka EE conversion requires at least 8 joint values, got {joint_state.size}"
            )
        # Task states contain all nine Franka DOFs (7 arm + 2 fingers), while
        # the historical collection action contains 7 arm joints + one finger.
        compact_state = np.concatenate((joint_state[:7], joint_state[7:8]))
        rmpflow = self.rmp_controller.get_motion_policy()
        ee_pos, ee_rot = rmpflow.get_end_effector_pose(compact_state[:7])
        ee_quat = transform_utils.rotation_matrix_to_quaternion(ee_rot)
        ee_quat = np.asarray(
            ee_quat.numpy() if hasattr(ee_quat, "numpy") else ee_quat,
            dtype=np.float64,
        ).reshape(4)
        quaternion_norm = float(np.linalg.norm(ee_quat))
        if not np.isfinite(quaternion_norm) or quaternion_norm < 1e-9:
            raise ValueError("RMPFlow returned an invalid end-effector quaternion")
        ee_quat = ee_quat / quaternion_norm
        return np.asarray(
            np.concatenate((np.asarray(ee_pos).reshape(3), ee_quat, compact_state[7:8])),
            dtype=np.float32,
        )

    def cache_collection_step(
        self, state: Dict[str, Any], action: Any, ee_pose: Optional[np.ndarray] = None
    ) -> None:
        """Cache one aligned collection frame with automatic annotations."""
        if self.mode != "collect":
            return
        action_gripper = self.get_gripper_action_state(action)
        if "camera_data" not in state:
            return
        if ee_pose is None and self._robot_type == "franka":
            ee_pose = self.transform_ee_pose(state["joint_positions"])
        self.data_collector.cache_step(
            camera_images=state["camera_data"],
            joint_angles=state["joint_positions"][:-1],
            action_gripper=action_gripper,
            subtask=self.get_subtask_name(),
            language_instruction=self.get_task_name(),
            ee_pose=ee_pose,
            sim_step=state.get("sim_step"),
        )

    def write_cached_episode(self, state: Dict[str, Any]) -> None:
        """Finalize one collected episode with aligned joint and EE actions.

        Franka episodes always include ``actions_ee``.  OpenArm deliberately
        retains its existing joint-only schema until a validated bimanual EE
        action contract is available.
        """
        if self.mode != "collect":
            return
        final_joint_positions = np.asarray(state["joint_positions"][:-1])
        if self._robot_type == "franka":
            self.data_collector.write_cached_data(
                final_joint_positions,
                final_ee_pose=self.transform_ee_pose(state["joint_positions"]),
            )
            return
        self.data_collector.write_cached_data(final_joint_positions)
    
    @abstractmethod
    def step(self, state: Dict[str, Any]) -> Tuple[Any, bool, bool]:
        """Execute one step of control.
        
        Args:
            state: Current state dictionary containing sensor data and robot state
            
        Returns:
            Tuple containing:
            - action: Control action to execute
            - done: Whether the episode is complete
            - is_success: Whether the task was completed successfully
        """
        pass
    
    def _init_collect_mode(self, cfg, robot=None):
        """Initialize the controller for collect mode."""
        self.data_collector = create_collector(
            cfg.collector.type,
            camera_configs=cfg.cameras,
            save_dir=cfg.multi_run.run_dir,
            max_episodes=cfg.max_episodes,
            compression=cfg.collector.compression,
            queue_capacity=int(cfg.collector.get("queue_capacity", 2)),
            sample_hz=int(cfg.collector.get("sample_hz", 10)),
            source_hz=60,
            background_write=bool(cfg.collector.get("background_write", True)),
        )
    
    def _init_infer_mode(self, cfg, robot=None): 
        """Initialize the controller for infer mode."""
        self.trajectory_controller = FrankaTrajectoryController(
            name="trajectory_controller",
            robot_articulation=robot,
            use_interpolation=False
        )
        
        self.inference_engine = InferenceEngineFactory.create_inference_engine(
            cfg, self.trajectory_controller
        )
    
    def episode_num(self) -> int:
        """Get the current episode number.
        
        Returns:
            int: Current episode number
        """
        if self.mode == "collect":
            return self.data_collector.episode_count
        return self._episode_num
    
    def print_failure_reason(self) -> None:
        """Print the last failure reason if it exists."""
        if self._last_failure_reason:
            print(f"Failure Reason: {self._last_failure_reason}")
    
    def reset(self) -> None:
        """Reset controller state between episodes."""
        # The benchmark runner owns official episode outcomes.  This method is
        # called before an eval episode starts, so the legacy counter below
        # would misleadingly print 0/1 before the policy executes any action.
        if self.mode == "eval":
            self.check_success_counter = 0
            self.reset_needed = False
            self._last_success = False
            self._last_failure_reason = ""
            self._gripper_action_tracker.reset()
            return
        if self._last_success:
            self.success_count += 1
        self._episode_num += 1
        print(f"Episode Stats: Success Rate = {self.success_count}/{self._episode_num} ({(self.success_count/self._episode_num)*100:.2f}%)")
        self.check_success_counter = 0
        self.reset_needed = False
        self._last_success = False
        self._last_failure_reason = ""
        self._gripper_action_tracker.reset()
        if self.mode == "collect":
            self.data_collector.clear_cache()

    def get_gripper_action_state(self, action) -> int:
        """Return the current binary gripper target from a control command.

        This is based on the command produced by the scripted controller,
        rather than the raw gripper position stored in robot state.
        """
        return self._gripper_action_tracker.update_from_action(action)

        
    def close(self) -> None:
        """Clean up resources used by the controller."""
        if self.mode == "collect" and hasattr(self, 'data_collector'):
            self.data_collector.close()
        if self.mode in {"infer", "eval"} and hasattr(self, 'inference_engine'):
            self.inference_engine.close()
        
    def need_reset(self) -> bool:
        """Check if the controller needs to be reset.
        
        Returns:
            bool: True if reset is needed, False otherwise
        """
        return self.reset_needed

    def is_success(self):
        return self._last_success
