import numpy as np
from .single_object_task import SingleObjectTask
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage


class PickTask(SingleObjectTask):
    """
    A task class for robotic picking operations.
    Manages object placement, material switching, and task state transitions.
    """
    def __init__(self, cfg, world, stage, robot):
        super().__init__(cfg, world, stage, robot)
    