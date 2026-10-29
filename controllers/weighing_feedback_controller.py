from __future__ import annotations

from isaacsim.core.utils.types import ArticulationAction

from .scale_weights_controller import ScaleWeightsTaskController


class WeighingFeedbackTaskController(ScaleWeightsTaskController):
    LANGUAGE_INSTRUCTION = (
        "Use the live scale feedback to place the required weights and reach 80 grams"
    )

    def __init__(self, cfg, robot):
        super().__init__(cfg, robot)
        self._feedback_wait_steps = 0

    def reset(self):
        super().reset()
        self._feedback_wait_steps = 0

    def _step_collect(self, state):
        """Hold after the last placement while the scale reading stabilizes.

        The generic multi-object controller treats a completed atomic motion
        whose predicate is not immediately true as a failed episode.  A scale
        deliberately needs several subsequent physics frames to establish a
        stable reading, so keep the current robot pose and continue recording
        during that dwell instead of failing the episode.
        """
        if self.phase_index < len(self.PHASE_SPECS):
            spec = self.PHASE_SPECS[self.phase_index]
            if spec.name == "place_weight_2" and self.active_controller.is_done():
                geometric_success = ScaleWeightsTaskController._check_phase_success(
                    self, state, spec
                )
                feedback_success = self._feedback_success(state)
                if geometric_success and not feedback_success:
                    self._feedback_wait_steps += 1
                    max_wait = int(
                        self.cfg.weighing_feedback.get("max_stabilization_steps", 300)
                    )
                    if self._feedback_wait_steps <= max_wait:
                        action = ArticulationAction(
                            joint_positions=list(state["joint_positions"])
                        )
                        self.cache_collection_step(state, action)
                        return action, False, False
        return super()._step_collect(state)

    def _feedback_success(self, state):
        tolerance = float(self.cfg.weighing_feedback.stable_tolerance_g)
        return bool(
            state.get("load_cell_stable", False)
            and abs(
                float(state.get("display_mass_g", 0.0))
                - float(state.get("target_mass_g", 80.0))
            ) <= tolerance
        )

    def _check_phase_success(self, state, spec):
        if not super()._check_phase_success(state, spec):
            return False
        if spec.name != "place_weight_2":
            return True
        return self._feedback_success(state)
