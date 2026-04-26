"""OpenPI policy model loader using high-level API."""
import numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config

# OpenPI config name from training
CONFIG_NAME = "pi05_hsr_task6891011_level12_v2.5_train_adaptive"


def load_policy_model(checkpoint_path: str, device: str = "cuda"):
    """Load OpenPI policy using high-level API.
    
    Args:
        checkpoint_path: Path to checkpoint directory (step directory with assets/, params/, train_state/)
        device: Device string (ignored for JAX, kept for API compatibility)
    
    Returns:
        Policy object with infer(obs) method
    """
    # Load config
    cfg = _config.get_config(CONFIG_NAME)
    
    # Create policy from checkpoint
    # checkpoint_path should point to the step directory containing assets/, params/, train_state/
    policy = policy_config.create_trained_policy(cfg, checkpoint_path)
    
    return policy
