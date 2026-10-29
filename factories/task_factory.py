from typing import Dict, Type
from tasks.base_task import BaseTask
from tasks.pick_task import PickTask
from tasks.pick_liquid_task import PickLiquidTask
from tasks.pickpour_task import PickPourTask
from tasks.place_task import PlaceTask
from tasks.stir_task import StirTask
from tasks.pickpourv2_task import PickPourV2Task
from tasks.pickplace_task import PickPlaceTask
from tasks.open_close_task import OpenCloseTask
from tasks.device_operate_task import DeviceOperateTask
from tasks.device_hard_operate_task import DeviceHardOperateTask
from tasks.sampletransport_task import SampleTransportTask
from tasks.nmrinjection_task import NMRInjectionTask
from tasks.centrifugeload_task import CentrifugeLoadTask
from tasks.centrifugeload_hard_task import CentrifugeLoadHardTask
from tasks.cohortanalysis_task import CohortAnalysisTask
from tasks.sampleorganition_task import SampleOrganitionTask
from tasks.scaleweights_task import ScaleWeightsTask
from tasks.weighing_feedback_task import WeighingFeedbackTask
from tasks.condition_module_task import ConditionModuleTask
from tasks.LiquidMixing_task import LiquidMixing
from tasks.clearreset_task import ClearResetTask
from tasks.rackplace_task import RackPlaceTask
from tasks.rack_recall_task import RackRecallTask
from tasks.weightselect_task import WeightSelectTask
from tasks.nmrsample_task import NMRSampleTask
from tasks.nmrrack_task import NMRRackTask
from tasks.temperaturcontrol_task import TemperaturControlTask
from tasks.devicestatus_task import DeviceStatusTask
from tasks.centrifugecontrol_task import CentrifugeControlTask
from tasks.scaledigital_task import ScaleDigitalTask
from tasks.plateposition_task import PlatePositionTask
from tasks.centrifugeloadR_task import CentrifugeLoadRTask
from tasks.centrifugeloadR1_task import CentrifugeLoadR1Task
from tasks.devicecover_status_task import DeviceCoverStatusTask
from tasks.singlepipetting_task import SinglePipettingTask
from tasks.formulaspotting_task import FormulaSpottingTask

_task_registry: Dict[str, Type[BaseTask]] = {}

def register_task(name: str, task_class: Type[BaseTask]):
    
    _task_registry[name] = task_class

def create_task(task_name: str, *args, **kwargs) -> BaseTask:
    
    if task_name not in _task_registry:
        raise ValueError(f": {task_name}")
    return _task_registry[task_name](*args, **kwargs)


register_task("pick", PickTask)
register_task("pickliquid", PickLiquidTask)
register_task("pickpour", PickPourTask)
register_task("place", PlaceTask)
register_task("stir", StirTask)
register_task("pickpourv2", PickPourV2Task)
register_task("pickplace", PickPlaceTask)
register_task("openclose", OpenCloseTask)
register_task("device_operate", DeviceOperateTask)
register_task("device_hard_operate", DeviceHardOperateTask)
register_task("sampletransport", SampleTransportTask)
register_task("nmrinjection", NMRInjectionTask)
register_task("centrifugeload", CentrifugeLoadTask)
register_task("centrifugeload_H", CentrifugeLoadHardTask)
# LabBench registered this as "cohortanalysisTask" (a typo); the canonical
# registry name is the plain task type "cohortanalysis".
register_task("cohortanalysis", CohortAnalysisTask)
register_task("sampleorganition", SampleOrganitionTask)
register_task("scaleweights", ScaleWeightsTask)
register_task("weighing_feedback", WeighingFeedbackTask)
register_task("ConditionModule", ConditionModuleTask)
register_task("LiquidMixing", LiquidMixing)
register_task("clearreset", ClearResetTask)
register_task("rackplace", RackPlaceTask)
register_task("rack_recall", RackRecallTask)
register_task("weightselect", WeightSelectTask)
register_task("nmrsample", NMRSampleTask)
register_task("nmrrack", NMRRackTask)
register_task("temperaturcontrol", TemperaturControlTask)
register_task("devicestatus", DeviceStatusTask)
register_task("centrifugecontrol", CentrifugeControlTask)
register_task("scaledigital", ScaleDigitalTask)
register_task("plateposition", PlatePositionTask)
register_task("centrifugeloadR", CentrifugeLoadRTask)
register_task("centrifugeloadR1", CentrifugeLoadR1Task)
register_task("devicecover_status", DeviceCoverStatusTask)
register_task("singlepipetting", SinglePipettingTask)
register_task("formulaspotting", FormulaSpottingTask)
