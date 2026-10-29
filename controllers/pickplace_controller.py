import re
import numpy as np
from enum import Enum

from .atomic_actions.pick_controller import PickController
from .atomic_actions.place_controller import PlaceController
from .atomic_actions.tube_insert_controller import TubeInsertController
from .base_controller import BaseController
from isaacsim.core.utils.rotations import euler_angles_to_quat

class Phase(Enum):
    PICKING = "picking"
    PLACING = "placing"
    FINISHED = "finished"

class PickPlaceTaskController(BaseController):
    def __init__(self, cfg, robot):
        """Initialize the pick and pour task controller.
        
        Args:
            cfg: Configuration object containing controller settings
            robot: Robot instance to control
        """
        # BaseController invokes _init_collect_mode() from its constructor, so
        # this task-specific configuration must exist before super().__init__.
        self.tube_insertion_cfg = getattr(cfg, "tube_insertion", None)
        self._tube_success_counter = 0
        self._pick_retry_count = 0
        self._pick_retry_wait_counter = 0
        self._pick_retry_pending = False
        super().__init__(cfg, robot)
        self.initial_position = None
        self.initial_size = None
        self.current_phase = Phase.PICKING
        # `self.state` is read by is_success() before the first step() call (the
        # runtime probes success via getattr(self.controller, "is_success") before
        # calling step()), so it must exist (as None) from construction time.
        self.state = None

    @property
    def _uses_tube_insertion(self) -> bool:
        return self.tube_insertion_cfg is not None

    def _tube_vertical_orientation(self) -> np.ndarray:
        """Return the scalar-first RMPFlow orientation for a vertical tube.

        RMPFlow targets ``right_gripper``.  Its local +Z is the finger-length
        axis, and ``tube_v1`` is long along its own local +Z.  The scene's
        tube orientation maps that tube axis to world -Z, so rotating the end
        effector by pi about Y keeps both axes collinear and vertical.
        """
        yaw = np.deg2rad(float(self.tube_insertion_cfg.end_effector_yaw_degrees))
        return euler_angles_to_quat(np.array([0.0, np.pi, yaw]))

    def _pick_orientation(self) -> np.ndarray:
        if self._uses_tube_insertion:
            return self._tube_vertical_orientation()
        # Isaac Sim expects [w, x, y, z]; using its helper avoids the former
        # scalar-last SciPy quaternion mismatch.
        return euler_angles_to_quat(np.radians([0.0, 90.0, 30.0]))

    def _init_collect_mode(self, cfg, robot):
        """Initialize controller for data collection mode."""
        super()._init_collect_mode(cfg, robot)
        
        if self._uses_tube_insertion:
            self.place_controller = TubeInsertController(
                name="tube_insert_controller",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
                insertion_cfg=self.tube_insertion_cfg,
            )
        else:
            self.place_controller = PlaceController(
                name="place_controller",
                cspace_controller=self.rmp_controller,
                gripper=robot.gripper,
            )
        close_hold_dt = (
            1.0 / float(self.tube_insertion_cfg.pick_close_hold_frames)
            if self._uses_tube_insertion else 0.01
        )
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.004, 0.002, 0.01, 0.02, close_hold_dt, 0.004, 0.008],
            lift_speed=(
                float(self.tube_insertion_cfg.pick_lift_speed)
                if self._uses_tube_insertion else 0.02
            ),
            incremental_lift=self._uses_tube_insertion,
        )
        self.active_controller = self.pick_controller

    def _init_infer_mode(self, cfg, robot):
        """Initialize controller for inference mode."""
        close_hold_dt = (
            1.0 / float(self.tube_insertion_cfg.pick_close_hold_frames)
            if self._uses_tube_insertion else 0.05
        )
        self.pick_controller = PickController(
            name="pick_controller",
            cspace_controller=self.rmp_controller,
            events_dt=[0.002, 0.002, 0.005, 0.02, close_hold_dt, 0.01, 0.02],
            lift_speed=(
                float(self.tube_insertion_cfg.pick_lift_speed)
                if self._uses_tube_insertion else 0.02
            ),
            incremental_lift=self._uses_tube_insertion,
        )
        super()._init_infer_mode(cfg, robot)

    def reset(self):
        """Reset controller state and phase."""
        super().reset()
        self.current_phase = Phase.PICKING
        self.initial_position = None
        self.initial_size = None
        self.state = None
        self._tube_success_counter = 0
        self._pick_retry_count = 0
        self._pick_retry_wait_counter = 0
        self._pick_retry_pending = False
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

        if self.current_phase == Phase.PICKING:
            return object_pos[2] > self.initial_position[2] + 0.12
        elif self.current_phase == Phase.PLACING:
            if self._uses_tube_insertion:
                # A tube may only be declared inserted after it has been
                # released.  This prevents a gripper-held tube at the socket
                # from being incorrectly recorded as a completed placement.
                released = self.mode != "collect" or self.place_controller.object_released
                if not released:
                    self._tube_success_counter = 0
                    return False

                target_axis = np.asarray(self.state['tube_axis'], dtype=np.float64)
                object_axis = np.asarray(self.state['object_axis'], dtype=np.float64)
                axis_alignment = abs(float(np.dot(target_axis, object_axis)))
                max_axis_error = np.deg2rad(
                    float(self.tube_insertion_cfg.success_axis_tolerance_degrees)
                )
                position_ok = (
                    np.linalg.norm(object_pos[:2] - target_position[:2])
                    < float(self.tube_insertion_cfg.success_xy_tolerance)
                    and abs(object_pos[2] - target_position[2])
                    < float(self.tube_insertion_cfg.success_z_tolerance)
                )
                axis_ok = axis_alignment >= np.cos(max_axis_error)
                if position_ok and axis_ok:
                    self._tube_success_counter += 1
                else:
                    self._tube_success_counter = 0
                return self._tube_success_counter >= int(self.tube_insertion_cfg.stable_frames)

            success = (np.linalg.norm(object_pos[:2] - target_position[:2]) < 0.05 and abs(object_pos[2] - self.initial_position[2]) < 0.05)
            return success

    def _tube_completion_diagnostics(self) -> str:
        object_pos = np.asarray(self.state['object_position'], dtype=np.float64)
        target_position = np.asarray(self.state['target_position'], dtype=np.float64)
        target_axis = np.asarray(self.state['tube_axis'], dtype=np.float64)
        object_axis = np.asarray(self.state['object_axis'], dtype=np.float64)
        target_axis /= max(float(np.linalg.norm(target_axis)), 1e-12)
        object_axis /= max(float(np.linalg.norm(object_axis)), 1e-12)
        alignment = abs(float(np.dot(target_axis, object_axis)))
        axis_error_degrees = float(
            np.degrees(np.arccos(np.clip(alignment, 0.0, 1.0)))
        )
        xy_error = float(np.linalg.norm(object_pos[:2] - target_position[:2]))
        z_error = abs(float(object_pos[2] - target_position[2]))
        released = self.mode != "collect" or self.place_controller.object_released
        return (
            f"target_center={np.array2string(target_position, precision=6, separator=', ')}, "
            f"actual_center={np.array2string(object_pos, precision=6, separator=', ')}, "
            f"xy_error={xy_error:.6f} m/"
            f"{float(self.tube_insertion_cfg.success_xy_tolerance):.6f} m, "
            f"z_error={z_error:.6f} m/"
            f"{float(self.tube_insertion_cfg.success_z_tolerance):.6f} m, "
            f"axis_error={axis_error_degrees:.3f} deg/"
            f"{float(self.tube_insertion_cfg.success_axis_tolerance_degrees):.3f} deg, "
            f"stable_counter={self._tube_success_counter}/"
            f"{int(self.tube_insertion_cfg.stable_frames)}, "
            f"released={released}"
        )

    def _handle_pick_failure(self, state):
        max_retries = int(getattr(self.tube_insertion_cfg, "pick_max_retries", 0))
        wait_steps = max(
            0,
            int(getattr(self.tube_insertion_cfg, "pick_retry_wait_steps", 0)),
        )
        current_position = np.asarray(state['object_position'], dtype=np.float64)
        lift_height = float(current_position[2] - self.initial_position[2])
        if self._pick_retry_count >= max_retries:
            self._last_failure_reason = (
                f"tube pick failed after {self._pick_retry_count + 1} attempts; "
                f"initial={np.array2string(self.initial_position, precision=6)}, "
                f"current={np.array2string(current_position, precision=6)}, "
                f"lift_height={lift_height:.6f} m, required=0.120000 m"
            )
            self.data_collector.clear_cache()
            self._last_success = False
            self.current_phase = Phase.FINISHED
            self.reset_needed = True
            print(f"Pick task failed: {self._last_failure_reason}", flush=True)
            return None, True, False

        if not self._pick_retry_pending:
            self._pick_retry_pending = True
            self._pick_retry_wait_counter = 0
            print(
                f"Pick attempt {self._pick_retry_count + 1} did not lift the tube "
                f"0.12 m; retrying after {wait_steps} physics steps.",
                flush=True,
            )

        self._pick_retry_wait_counter += 1
        if self._pick_retry_wait_counter >= max(1, wait_steps):
            self._pick_retry_count += 1
            self._pick_retry_pending = False
            self._pick_retry_wait_counter = 0
            self.initial_position = current_position.copy()
            self.pick_controller.reset()
            self.active_controller = self.pick_controller
            print(
                f"Starting tube pick retry {self._pick_retry_count}/"
                f"{max_retries} from {np.array2string(current_position, precision=6)}.",
                flush=True,
            )
        return None, False, False


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
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success
        if not self.active_controller.is_done():
            action = None
            if self.current_phase == Phase.PICKING:
                pick_kwargs = {}
                if self._uses_tube_insertion:
                    # Hold the upper portion of the tube so the hand stays
                    # above the rack's top plate during the final insertion.
                    pick_kwargs['pick_z_offset'] = float(
                        self.tube_insertion_cfg.grasp_height_above_center
                    )
                    pick_kwargs['gripper_distances'] = float(
                        self.tube_insertion_cfg.gripper_target_distance
                    )
                action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=self._pick_orientation(),
                    pre_offset_x=0.03,
                    pre_offset_z=0.05,
                    after_offset_z=(
                        float(self.tube_insertion_cfg.pick_lift_height)
                        if self._uses_tube_insertion else 0.5
                    ),
                    **pick_kwargs,
                )
            else:
                if self._uses_tube_insertion:
                    action = self.place_controller.forward(
                        socket_position=state['tube_socket_position'],
                        inserted_tube_center=state['tube_pre_release_center'],
                        insertion_axis=state['tube_axis'],
                        current_joint_positions=state['joint_positions'],
                        gripper_position=state['gripper_position'],
                        object_position=state['object_position'],
                        end_effector_orientation=self._tube_vertical_orientation(),
                    )
                else:
                    action = self.place_controller.forward(
                        place_position=state['target_position'],
                        current_joint_positions=state['joint_positions'],
                        gripper_control=self.gripper_control,
                        end_effector_orientation=euler_angles_to_quat(
                            np.radians([0.0, 90.0, 25.0])
                        ),
                        gripper_position=state['gripper_position'],
                        pre_place_z=0.5,
                        place_offset_z=0.05,
                    )
            self.cache_collection_step(state, action)
            
            return action, False, False

        if (
            self.current_phase == Phase.PLACING
            and getattr(self.active_controller, "has_failed", lambda: False)()
        ):
            self._last_failure_reason = getattr(
                self.active_controller,
                "last_failure_reason",
                "tube insertion phase timed out",
            )
            self.data_collector.clear_cache()
            self._last_success = False
            self.current_phase = Phase.FINISHED
            self.reset_needed = True
            print(f"Tube insertion task failed: {self._last_failure_reason}", flush=True)
            return None, True, False

        if self.current_phase == Phase.PICKING and not success:
            return self._handle_pick_failure(state)

        if success:
            if self.current_phase == Phase.PICKING:
                print("Pick task success! Switching to place...")
                self.current_phase = Phase.PLACING
                self.active_controller = self.place_controller
                return None, False, False
            elif self.current_phase == Phase.PLACING:
                print("Tube insertion task success!" if self._uses_tube_insertion else "Place task success!")
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

        if self._uses_tube_insertion and self.current_phase == Phase.PLACING:
            # The scripted insertion has released and waited for stability.
            # If the physical tube did not meet the precise socket predicate,
            # retain no partial trajectory as a successful demonstration.
            self._last_failure_reason = (
                "tube did not settle in the selected socket; "
                f"{self._tube_completion_diagnostics()}"
            )
            print(f"Tube insertion task failed: {self._last_failure_reason}", flush=True)
            self.data_collector.clear_cache()
            self._last_success = False
            self.current_phase = Phase.FINISHED
            return None, True, False
        
        return None, False, False

    def _step_infer(self, state):
        """Execute inference mode step."""
        self.state = state
        if self.current_phase == Phase.FINISHED:
            self.reset_needed = True
            return None, True, self._last_success
        
        if self.current_phase == Phase.PICKING:
            if self.pick_controller.is_done():
                # Scripted pick completed; hand control to the policy for placing.
                # Without this transition the controller stays in PICKING forever,
                # the policy is never called, and the episode ends in a timeout
                # flagged as an invalid scripted precondition (policy_calls == 0).
                print("Scripted pick complete — switching to policy-driven placing.")
                self.current_phase = Phase.PLACING
                action = None
            else:
                action = self.pick_controller.forward(
                    picking_position=state['object_position'],
                    current_joint_positions=state['joint_positions'],
                    object_size=state['object_size'],
                    object_name=state['object_name'],
                    gripper_control=self.gripper_control,
                    gripper_position=state['gripper_position'],
                    end_effector_orientation=self._pick_orientation(),
                    gripper_distances=(
                        float(self.tube_insertion_cfg.gripper_target_distance)
                        if self._uses_tube_insertion else None
                    ),
                    after_offset_z=(
                        float(self.tube_insertion_cfg.pick_lift_height)
                        if self._uses_tube_insertion else 0.15
                    ),
                )
        else:
            state['language_instruction'] = self.get_language_instruction()
            action = self.inference_engine.step_inference(state)
        return action, False, self.is_success()

    def is_success(self):
        if self.state is None:
            # The runtime probes success before the first step() of an episode.
            return False
        if self._uses_tube_insertion:
            if self.current_phase != Phase.PLACING:
                return False
            if self._check_phase_success():
                self._last_success = True
                self.current_phase = Phase.FINISHED
                return True
            return False
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
