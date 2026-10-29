import numpy as np

from eai4s_eval.centrifuge_v2.motion_planning import plan_joint_path


def test_joint_path_goes_around_obstacle_and_checks_every_command():
    collision = lambda point: np.linalg.norm(point) < .35
    path = plan_joint_path([-1., 0.], [1., 0.], [-1.5, -1.5], [1.5, 1.5],
                           collision, seed=108)
    assert path is not None
    assert np.allclose(path[0], [-1., 0.])
    assert np.allclose(path[-1], [1., 0.])
    assert all(not collision(point) for point in path)
    assert np.max(np.abs(np.diff(path, axis=0))) <= .060001


def test_colliding_goal_is_not_planned_through():
    assert plan_joint_path([0.], [.8], [-1.], [1.], lambda point: point[0] > .5) is None


def test_joint_path_is_reproducible():
    collision = lambda point: np.linalg.norm(point) < .35
    first = plan_joint_path([-1., 0.], [1., 0.], [-1.5, -1.5], [1.5, 1.5], collision, seed=9)
    second = plan_joint_path([-1., 0.], [1., 0.], [-1.5, -1.5], [1.5, 1.5], collision, seed=9)
    assert np.array_equal(first, second)
