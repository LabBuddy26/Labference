from typing import Dict, Type
from isaacsim.core.api.robots.robot import Robot
from robots.franka.franka import Franka
from robots.franka.franka import Franka
from robots.openarm.openarm import OpenArm

_robot_registry: Dict[str, Type[Robot]] = {}

def register_robot(name: str, robot_class: Type[Robot]):
    
    _robot_registry[name] = robot_class

def create_robot(robot_type: str, *args, **kwargs) -> Robot:
    
    if robot_type not in _robot_registry:
        raise ValueError(f": {robot_type}")
    return _robot_registry[robot_type](*args, **kwargs)


register_robot("franka", Franka)
register_robot("openarm", OpenArm)
