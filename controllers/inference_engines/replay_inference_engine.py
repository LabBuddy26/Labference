import h5py
import numpy as np
from typing import Dict
import torch

from .base_inference_engine import BaseInferenceEngine


class ReplayInferenceEngine(BaseInferenceEngine):
    """
    Replay inference engine
    
    Load pre-collected action data from HDF5 file and replay them sequentially
    """
    
    def _get_n_obs_steps(self) -> int:
        """Get observation steps"""
        return getattr(self.cfg.infer, 'n_obs_steps', 1)
    
    def _init_inference_engine(self):
        """Initialize replay inference engine"""
        # Get HDF5 file path from configuration
        self.h5_path = self.cfg.infer.replay_data_path
        self.episode_index = getattr(self.cfg.infer, 'episode_index', 0)
        
        self.action_dataset = getattr(
            self.cfg.infer,
            "action_dataset",
            "actions_ee" if self.action_space == "ee_pos" else "actions",
        )
        # Load actions from HDF5 file
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if self.action_dataset in f:
                    # Single episode file
                    self.actions = f[self.action_dataset][:]
                else:
                    # Multi-episode file
                    episode_name = f"episode_{self.episode_index:04d}"
                    if episode_name in f and self.action_dataset in f[episode_name]:
                        self.actions = f[episode_name][self.action_dataset][:]
                    else:
                        raise KeyError(
                            f"Dataset {self.action_dataset!r} was not found for episode "
                            f"{episode_name} in {self.h5_path}"
                        )
            
            self.current_step = 0
            self.total_steps = len(self.actions)
            
            print(f"✓ Replay inference engine initialized")
            print(f"  - Data path: {self.h5_path}")
            print(f"  - Episode index: {self.episode_index}")
            print(f"  - Action dataset: {self.action_dataset}")
            print(f"  - Total steps: {self.total_steps}")
            print(f"  - Action shape: {self.actions.shape}")
            
        except Exception as e:
            print(f"❌ Failed to load replay data: {e}")
            raise
    
    def _predict_action(self, obs_dict: Dict[str, torch.Tensor], language_instruction: str = "") -> np.ndarray:
        """
        Get next action from pre-recorded trajectory
        
        Args:
            obs_dict: Observation data dictionary (not used in replay)
            language_instruction: Language instruction (not used in replay)
            
        Returns:
            Next action from recorded trajectory, or zeros if replay is complete
        """
        
        if self.current_step >= self.total_steps:
            print(f"⚠ Replay complete: all {self.total_steps} actions have been used")
            # Return zero action to indicate completion
            return np.zeros_like(self.actions[0:1])
        
        # Get current action and increment step counter
        action = self.actions[self.current_step:self.current_step + 1]
        self.current_step += 1
        
        return action
    
    def reset(self):
        """Reset the replay engine to start from beginning"""
        super().reset()
        self.current_step = 0
        print(f"✓ Replay engine reset to step 0")
    
    def close(self):
        """Close replay inference engine"""
        if hasattr(self, 'actions'):
            del self.actions
        print("✓ Replay inference engine closed")
