"""Deterministic joint-space planning against a caller-owned collision query."""
from __future__ import annotations

import numpy as np


def plan_joint_path(start, goal, lower, upper, collision, *, seed=0,
                    max_iterations=2000, step_size=.25, resolution=.06):
    """Return collision-checked, densely sampled waypoints, or None.

    This plans commands only; it never changes simulator or evaluator state.
    """
    start, goal, lower, upper = [np.asarray(value, dtype=float) for value in
                                  (start, goal, lower, upper)]
    if (start.ndim != 1 or any(value.shape != start.shape for value in (goal, lower, upper))
            or not all(np.all(np.isfinite(value)) for value in (start, goal, lower, upper))
            or np.any(lower >= upper) or resolution <= 0 or step_size <= 0):
        raise ValueError("invalid joint planning inputs")
    if np.any(start < lower) or np.any(start > upper) or np.any(goal < lower) or np.any(goal > upper):
        return None

    def segment(first, second):
        count = max(1, int(np.ceil(np.max(np.abs(second - first)) / resolution)))
        return [first + (second - first) * (index / count) for index in range(1, count + 1)]

    def clear(first, second):
        return not any(collision(point) for point in segment(first, second))

    if collision(start) or collision(goal):
        return None
    if clear(start, goal):
        return [start.copy(), *segment(start, goal)]
    rng = np.random.default_rng(seed)
    trees = [([start.copy()], [-1]), ([goal.copy()], [-1])]

    def extend(tree, target):
        nodes, parents = tree
        nearest = int(np.argmin([np.linalg.norm(point - target) for point in nodes]))
        delta = target - nodes[nearest]
        distance = np.linalg.norm(delta)
        point = target.copy() if distance <= step_size else nodes[nearest] + delta * step_size / distance
        if not clear(nodes[nearest], point):
            return None, False
        nodes.append(point)
        parents.append(nearest)
        return len(nodes) - 1, distance <= step_size

    def chain(tree, index):
        nodes, parents = tree
        path = []
        while index >= 0:
            path.append(nodes[index])
            index = parents[index]
        return path[::-1]

    for iteration in range(max_iterations):
        active = iteration % 2
        first, second = trees[active], trees[1 - active]
        target = second[0][0] if rng.random() < .15 else rng.uniform(lower, upper)
        index, _ = extend(first, target)
        if index is None:
            continue
        while True:
            other_index, reached = extend(second, first[0][index])
            if other_index is None:
                break
            if reached:
                left, right = chain(first, index), chain(second, other_index)
                coarse = left + right[-2::-1]
                if active:
                    coarse.reverse()
                # Shorten only when the entire shortcut is collision-free.
                simplified = [coarse[0]]
                cursor = 0
                while cursor < len(coarse) - 1:
                    following = len(coarse) - 1
                    while following > cursor + 1 and not clear(coarse[cursor], coarse[following]):
                        following -= 1
                    simplified.append(coarse[following])
                    cursor = following
                dense = [simplified[0].copy()]
                for first_point, second_point in zip(simplified, simplified[1:]):
                    dense.extend(segment(first_point, second_point))
                return dense
    return None
