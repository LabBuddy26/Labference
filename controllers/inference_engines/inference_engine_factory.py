# from .local_model_inference_engine import LocalModelInferenceEngine
from .remote_inference_engine import RemoteInferenceEngine
from .replay_inference_engine import ReplayInferenceEngine
from .base_inference_engine import BaseInferenceEngine
from eai4s_eval.eval.external_inference_engine import ExternalInferenceEngine


class InferenceEngineFactory:
    """Inference engine factory class"""
    
    @staticmethod
    def create_inference_engine(cfg, trajectory_controller) -> BaseInferenceEngine:
        """
        Create inference engine based on configuration
        
        Args:
            cfg: Configuration object
            trajectory_controller: Trajectory controller
            
        Returns:
            Corresponding inference engine instance
        """
        inference_type = getattr(cfg.infer, 'type', 'local')
        
        if inference_type == 'local':
            # return LocalModelInferenceEngine(cfg, trajectory_controller)
            pass
        elif inference_type == 'remote':
            return RemoteInferenceEngine(cfg, trajectory_controller)
        elif inference_type == 'replay':
            return ReplayInferenceEngine(cfg, trajectory_controller)
        elif inference_type == 'external':
            # Benchmark evaluation replaces this deferred engine immediately
            # after the task/controller runtime has been constructed.
            return ExternalInferenceEngine(cfg, trajectory_controller)
        else:
            raise ValueError(f"Unsupported inference engine type: {inference_type}")
    
    @staticmethod
    def get_supported_types():
        """Get supported inference engine types"""
        return ['local', 'remote', 'replay', 'external']
