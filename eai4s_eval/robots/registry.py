from __future__ import annotations

from .base import RobotAdapter
from .franka_panda import FrankaPandaAdapter
from .ur5_single import UR5SingleAdapter


_REGISTRY: dict[str, type[RobotAdapter]] = {
    FrankaPandaAdapter.robot_id: FrankaPandaAdapter,
    UR5SingleAdapter.robot_id: UR5SingleAdapter,
}


def available_robots() -> list[str]:
    return sorted(_REGISTRY)


def get_robot_adapter(robot_id: str) -> RobotAdapter:
    try:
        return _REGISTRY[robot_id]()
    except KeyError as exc:
        valid = ", ".join(available_robots())
        raise ValueError(f"Unknown robot '{robot_id}'. Available robots: {valid}") from exc
