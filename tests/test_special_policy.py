import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from eai4s_eval.oracle_runtime.contracts import PolicyProfile, InterventionSpec
from eai4s_eval.special_policy import BranchArrayPolicy, joint_waypoint


def test_branch_policy_preserves_images_and_instruction_without_evaluator_fields():
    profile = PolicyProfile.legacy({"name": "openpi_array", "action_space": "array"}, ("front", "wrist"))
    frame = BytesIO()
    Image.fromarray(np.full((4, 6, 3), 120, dtype=np.uint8)).save(frame, format="PNG")
    encoded = base64.b64encode(frame.getvalue()).decode()
    with patch("eai4s_eval.policy.JointPolicyAdapter") as factory:
        adapter = factory.return_value
        adapter.server_metadata = {}
        adapter.predict.return_value = SimpleNamespace(joint_positions=np.ones((3, 8)), latency_ms=2)
        policy = BranchArrayPolicy("localhost", 8000, profile=profile, intervention=InterventionSpec(),
                                   options={"execution_horizon": 2})
        result = policy.infer_full({"images": {"front": {"base64": encoded}, "wrist": {"base64": encoded}},
                                    "joint_positions": [0] * 9, "task_rule": "remember the slot",
                                    "evaluator_only_slot": "A1"})
        state = adapter.predict.call_args.args[0]
        assert state["language_instruction"] == "remember the slot"
        assert "evaluator_only_slot" not in state
        assert set(state["camera_data"]) == {"camera_1_rgb", "camera_3_rgb"}
        assert len(result["action_chunk"]) == 2


def test_joint_mode_is_not_reinterpreted_as_ee():
    waypoint = [0.] * 8
    assert joint_waypoint(None, waypoint, {"action_space": "joint"}) is waypoint


def test_invalid_ee_quaternion_is_rejected_before_ik():
    with pytest.raises(ValueError, match="quaternion"):
        joint_waypoint(None, [0.] * 8, {"action_space": "ee_pos"})
