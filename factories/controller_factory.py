from typing import Dict, Type
from controllers.base_controller import BaseController
from controllers.openarm_pick_controller import OpenArmPickTaskController
from controllers.pick_controller import PickTaskController
from controllers.pickliquid_controller import PickLiquidTaskController
from controllers.pour_controller import PourTaskController
from controllers.place_controller import PlaceTaskController
from controllers.stir_controller import StirTaskController
from controllers.pickplace_controller import PickPlaceTaskController 
from controllers.open_controller import OpenTaskController
from controllers.close_controller import CloseTaskController
from controllers.device_operate_controller import DeviceOperateController
from controllers.device_hard_operate_controller import DeviceHardOperateController
from controllers.sample_transport_controller import SampleTransportTaskController
from controllers.nmr_injection_controller import NMRInjectionTaskController
from controllers.centrifuge_load_controller import CentrifugeLoadTaskController
from controllers.centrifuge_hard_load_controller import CentrifugeLoadHardTaskController
from controllers.cohort_analysis_controller import CohortAnalysisTaskController
from controllers.sample_organition_controller import SampleOrganitionTaskController
from controllers.scale_weights_controller import ScaleWeightsTaskController
from controllers.weighing_feedback_controller import WeighingFeedbackTaskController
from controllers.condition_module_controller import ConditionModuleTaskController
from controllers.LiquidMixing_controller import LiquidMixingController
from controllers.clear_reset_controller import ClearResetTaskController
from controllers.rack_place_controller import RackPlaceTaskController
from controllers.rack_recall_controller import RackRecallTaskController
from controllers.weight_select_controller import WeightSelectTaskController
from controllers.press_controller import PressTaskController
from controllers.move_controller import MoveTaskController
from controllers.centrifuge_loadR_controller import CentrifugeLoadRTaskController
from controllers.centrifuge_loadR1_controller import CentrifugeLoadR1TaskController
from controllers.single_pipetting_controller import SinglePipettingTaskController
from controllers.formula_spotting_controller import FormulaSpottingTaskController

from controllers.openarm_pick_controller import OpenArmPickTaskController

_controller_registry: Dict[str, Type[BaseController]] = {}

def register_controller(name: str, controller_class: Type[BaseController]):
    _controller_registry[name] = controller_class

def create_controller(controller_name: str, *args, **kwargs) -> BaseController:
    if controller_name not in _controller_registry:
        raise ValueError(f": {controller_name}")
    return _controller_registry[controller_name](*args, **kwargs)

register_controller("pick", PickTaskController)
register_controller("pickliquid", PickLiquidTaskController)
register_controller("pour", PourTaskController)
register_controller("place", PlaceTaskController)
register_controller("stir", StirTaskController)
register_controller("pickplace", PickPlaceTaskController)
register_controller("open", OpenTaskController)
register_controller("close", CloseTaskController)
register_controller("openarm_pick", OpenArmPickTaskController)
register_controller("device_operate", DeviceOperateController)
register_controller("device_hard_operate", DeviceHardOperateController)
register_controller("sample_transport", SampleTransportTaskController)
register_controller("nmr_injection", NMRInjectionTaskController)
register_controller("centrifuge_load", CentrifugeLoadTaskController)
register_controller("centrifuge_hard_load", CentrifugeLoadHardTaskController)
register_controller("cohort_analysis", CohortAnalysisTaskController)
register_controller("sample_organition", SampleOrganitionTaskController)
register_controller("scale_weights", ScaleWeightsTaskController)
register_controller("weighing_feedback", WeighingFeedbackTaskController)
register_controller("ConditionModule", ConditionModuleTaskController)
register_controller("LiquidMixing", LiquidMixingController)
register_controller("clear_reset", ClearResetTaskController)
register_controller("rack_place", RackPlaceTaskController)
register_controller("rack_recall", RackRecallTaskController)
register_controller("weight_select", WeightSelectTaskController)
register_controller("press", PressTaskController)
register_controller("move", MoveTaskController)
register_controller("centrifuge_loadR", CentrifugeLoadRTaskController)
register_controller("centrifuge_loadR1", CentrifugeLoadR1TaskController)
register_controller("single_pipetting", SinglePipettingTaskController)
register_controller("formula_spotting", FormulaSpottingTaskController)
