from controllers.robot_controllers.grapper_manager import Gripper
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
import numpy as np
import typing
from isaacsim.core.utils.rotations import euler_angles_to_quat
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

class CloseController(BaseController):
    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        furniture_type: str = "drawer",
        door_width: float = 0.3,
        door_open_direction: str = None,
        contact_standoff: float = 0.055,
        approach_height: float = 0.08,
        close_hold_steps: int = 90,
        retreat_distance: float = 0.15,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._cspace_controller = cspace_controller
        self.furniture_type = furniture_type
        self.door_width = door_width
        self.position_rotation_interp_iter = None
        self.door_open_direction = door_open_direction
        self.init_handle_position = None
        self.contact_standoff = float(contact_standoff)
        self.approach_height = float(approach_height)
        self.close_hold_steps = int(close_hold_steps)
        self.retreat_distance = float(retreat_distance)
        self._door_close_angle_degrees = None
        self._final_contact_position = None
        self._final_closing_tangent = None
        self._hold_step_count = 0
        
        if self.furniture_type == "drawer":
            self._events_dt = [0.0005, 0.002, 0.05, 0.008]
        else:
            # Approach, arc push, hold pressure, then retreat for release test.
            self._events_dt = [0.0025, 0.005, 0.012, 0.008]
            
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events_dt must be a list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = events_dt.tolist()
            if len(self._events_dt) != 4 and self.furniture_type == "drawer":
                raise Exception(f"events_dt length must be 4, got {len(self._events_dt)}")
            if len(self._events_dt) != 4 and self.furniture_type == "door":
                raise Exception(f"events_dt length must be 4, got {len(self._events_dt)}")
        
        self._position_threshold = 0.01 / get_stage_units()

    def forward(
        self,
        handle_position: np.ndarray,
        current_joint_positions: np.ndarray,
        gripper_position: np.ndarray,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        angle: float = 50.0,
        revolute_joint_position: np.ndarray = None,
        push_distance: float = None
    ) -> ArticulationAction:        
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat([0, 110, 0], degrees=True, extrinsic=False)
        
        if self.init_handle_position is None:
            self.init_handle_position = handle_position.copy()
        
        # Door push interpolation and contact holding own their transitions;
        # a generic duration timeout must not skip either phase mid-motion.
        door_phase_has_explicit_completion = (
            self.furniture_type == "door" and self._event in (1, 2, 3)
        )
        if not door_phase_has_explicit_completion:
            self._t += self._events_dt[self._event]
        
        target_joint_positions = self._execute_phase(
            handle_position, 
            end_effector_orientation, 
            current_joint_positions,
            revolute_joint_position,
            gripper_position,
            angle,
            push_distance
        )
        
        if not door_phase_has_explicit_completion and self._t >= 1.0:
            self._event += 1
            self._t = 0
        return target_joint_positions

    def _execute_phase(self, handle_position, end_effector_orientation, current_joint_positions, revolute_joint_position, gripper_position, angle = 50, push_distance = None):
        if self.furniture_type == "drawer":
            return self._execute_drawer_phase(handle_position, end_effector_orientation, current_joint_positions, gripper_position, push_distance)
        else:
            return self._execute_door_phase(handle_position, end_effector_orientation, current_joint_positions, revolute_joint_position, gripper_position, angle)

    def _execute_drawer_phase(self, handle_position, end_effector_orientation, current_joint_positions, gripper_position, push_distance):
        if self._event == 0:
            target_handle_position = handle_position.copy()
            target_handle_position[0] -= 0.1 / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_handle_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - target_handle_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
        elif self._event == 1:
            target_handle_position = handle_position.copy()
            target_handle_position[0] += 0.05 / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_handle_position,
                target_end_effector_orientation=end_effector_orientation
            )
            if push_distance is not None and np.linalg.norm(handle_position - self.init_handle_position) > push_distance:
                self._event += 1
                self._t = 0
        elif self._event == 2:
            target_handle_position = handle_position.copy()
            target_handle_position[0] -= 0.1 / get_stage_units()
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=target_handle_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - handle_position[:2])
            if xy_distance > 0.05:
                self._event += 1
                self._t = 0
        else:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        return target_joint_positions

    def _execute_door_phase(self, handle_position, end_effector_orientation, current_joint_positions, revolute_joint_position, gripper_position, angle = 50):
        if self._event == 0:
            self.start_position = handle_position.copy()
            if revolute_joint_position[1] > self.start_position[1]:
                angle = -angle
            self._door_close_angle_degrees = -angle
            contact_position, _ = self._door_contact_position(
                self.start_position,
                revolute_joint_position,
                self._door_close_angle_degrees,
            )
            approach_position = contact_position.copy()
            approach_position[2] += self.approach_height
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=approach_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - approach_position[:2])
            if xy_distance < self._position_threshold:
                self._event += 1
                self._t = 0
        elif self._event == 1:
            if self.position_rotation_interp_iter is None:
                self.target_position = self.rotate_around_z_axis(
                    self.start_position,
                    revolute_joint_position,
                    self._door_close_angle_degrees,
                )
                num_interpolation = max(
                    2, int(600 * np.linalg.norm(self.start_position - self.target_position))
                )
                alphas = np.linspace(start=0, stop=1, num=num_interpolation)[1:]
                position_rotation_interp_list = self.action_interpolation(
                    self.start_position,
                    end_effector_orientation,
                    self.target_position,
                    end_effector_orientation,
                    alphas,
                    joint_pos=revolute_joint_position,
                    contact_standoff=self.contact_standoff,
                    closing_angle_degrees=self._door_close_angle_degrees,
                )
                self.position_rotation_interp_iter = iter(position_rotation_interp_list)
            try:
                self.trans_interp, self.rotation_interp = next(self.position_rotation_interp_iter)
                target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=self.trans_interp,
                    target_end_effector_orientation=self.rotation_interp
                )
            except:
                self._event += 1
                self._t = 0
                self._final_contact_position, self._final_closing_tangent = self._door_contact_position(
                    self.target_position,
                    revolute_joint_position,
                    self._door_close_angle_degrees,
                )
                target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=self._final_contact_position,
                    target_end_effector_orientation=end_effector_orientation,
                )
        elif self._event == 2:
            # Hold the final contact pose long enough for the zero-restitution
            # joint limit to settle before the hand withdraws.
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=self._final_contact_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            self._hold_step_count += 1
            if self._hold_step_count >= self.close_hold_steps:
                self._event += 1
                self._t = 0
        elif self._event == 3:
            # Use the measured handle pose rather than the ideal arc endpoint:
            # contact and joint compliance can leave the physical door a few
            # centimetres from that endpoint.  Retiring from the ideal point
            # was the reason the previous hand target remained beside the
            # actual handle after closing.
            current_contact_position, current_closing_tangent = self._door_contact_position(
                handle_position,
                revolute_joint_position,
                self._door_close_angle_degrees,
            )
            retreat_position = current_contact_position.copy()
            retreat_position[:2] -= current_closing_tangent * self.retreat_distance
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=retreat_position,
                target_end_effector_orientation=end_effector_orientation
            )
            xy_distance = np.linalg.norm(gripper_position[:2] - retreat_position[:2])
            if xy_distance < 0.02:
                self._event += 1
                self._t = 0
        else:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        return target_joint_positions
    
    def reset(self) -> None:
        """Reset controller state"""
        BaseController.reset(self)
        self._event = 0
        self._t = 0
        self.position_rotation_interp_iter = None
        self._door_close_angle_degrees = None
        self._final_contact_position = None
        self._final_closing_tangent = None
        self._hold_step_count = 0

    def is_done(self) -> bool:
        """Check if controller has completed all states"""
        return self._event >= len(self._events_dt)
        
    def action_interpolation(
        self,
        trans_previous,
        rotation_previous,
        trans_target,
        rotation_target,
        alphas,
        joint_pos=None,
        contact_standoff: float = 0.0,
        closing_angle_degrees: float = 0.0,
    ):
        """
        Interpolate between two poses for smooth motion.

        Args:
            trans_previous: Starting position
            rotation_previous: Starting orientation (quaternion)
            trans_target: Target position
            rotation_target: Target orientation (quaternion)
            alphas: Interpolation points
            joint_pos: Joint position for door rotation center

        Returns:
            List of interpolated (position, orientation) pairs
        """
        action_list = []
        # RMPFlow and scipy both use XYZW quaternions here.
        key_rots = R.from_quat(np.stack([rotation_previous, rotation_target]))
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        interp_rots = slerp(alphas).as_quat()

        # Calculate polar coordinates relative to joint_pos
        r_0 = trans_previous[:2] - joint_pos[:2]
        r_1 = trans_target[:2] - joint_pos[:2]

        # Calculate radii
        radius_0 = np.linalg.norm(r_0)
        radius_1 = np.linalg.norm(r_1)
        radii = np.linspace(radius_0, radius_1, len(alphas) + 1)[1:]

        # Calculate start and target angles
        theta_0 = np.arctan2(r_0[1], r_0[0])
        theta_1 = np.arctan2(r_1[1], r_1[0])

        # Ensure interpolation angle range is within 1/4 circle (90°)
        # Choose shortest path and limit to π/2
        delta_theta = theta_1 - theta_0
        if delta_theta > np.pi:
            delta_theta -= 2 * np.pi
        elif delta_theta < -np.pi:
            delta_theta += 2 * np.pi
        # Limit interpolation to 1/4 circle
        delta_theta = np.clip(delta_theta, -np.pi / 2, np.pi / 2)
        thetas = np.linspace(theta_0, theta_0 + delta_theta, len(alphas) + 1)[1:]

        for alpha, radius, theta, interp_rot in zip(alphas, radii, thetas, interp_rots):
            handle_interp = np.array([
                joint_pos[0] + radius * np.cos(theta),  # Directly use theta
                joint_pos[1] + radius * np.sin(theta),  # Directly use theta
                alpha * trans_target[2] + (1 - alpha) * trans_previous[2],  # Linear interpolation for Z axis
            ])
            tangent = np.array([-np.sin(theta), np.cos(theta)])
            if closing_angle_degrees < 0:
                tangent = -tangent
            trans_interp = handle_interp.copy()
            trans_interp[:2] -= tangent * contact_standoff
            action_list.append((trans_interp, interp_rot))

        return action_list

    def _door_contact_position(self, handle_position, revolute_joint_position, closing_angle_degrees):
        """Return a handle-side contact pose and the tangential closing direction."""
        radial = np.asarray(handle_position[:2]) - np.asarray(revolute_joint_position[:2])
        radius = np.linalg.norm(radial)
        if radius < 1e-6:
            raise ValueError("Door handle and revolute joint cannot share the same XY position")
        tangent = np.array([-radial[1], radial[0]]) / radius
        if closing_angle_degrees < 0:
            tangent = -tangent
        contact_position = np.asarray(handle_position, dtype=np.float64).copy()
        contact_position[:2] -= tangent * self.contact_standoff
        return contact_position, tangent
    
    def rotate_around_z_axis(self, p1, p2, angle_deg):
        """
        Rotate point p1 around point p2 by angle_deg degrees around Z axis.

        Args:
            p1: Point to rotate
            p2: Center of rotation
            angle_deg: Rotation angle in degrees

        Returns:
            Rotated point
        """
        angle_rad = np.deg2rad(angle_deg)
        # Translate: Move p1 to coordinate system with p2 as origin
        p1_relative = p1 - p2
        # Counterclockwise rotation matrix around Z axis by angle_deg
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0,                 0,                1]
        ])
        # Apply rotation matrix
        p1_rotated_relative = rotation_matrix @ p1_relative
        # Translate back: Add p2 coordinates
        p1_rotated = p1_rotated_relative + p2
        return p1_rotated

    def rotate_quaternion_around_x(self, q, angle_deg):
        """
        Rotate quaternion around X axis by specified angle.

        Args:
            q: Input quaternion
            angle_deg: Rotation angle in degrees

        Returns:
            Rotated quaternion
        """
        angle_rad = np.deg2rad(angle_deg)
        q_rot = np.array([
            -np.sin(angle_rad / 2), 0, 0, np.cos(angle_rad / 2)
        ])
        r = R.from_quat(q)
        r_rot = R.from_quat(q_rot)
        r_new = r * r_rot
        return r_new.as_quat()
