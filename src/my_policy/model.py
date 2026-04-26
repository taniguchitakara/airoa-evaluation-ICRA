"""OpenPI policy model loader (JAX/Flax based)."""
import jax
import jax.numpy as jnp
from openpi.training.checkpoints import CheckpointManager


def load_policy_model(checkpoint_path: str, device: str = "cuda"):
    """Load OpenPI model from Orbax checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory (e.g., download/baseline/50000)
        device: Device string (cuda or cpu) - JAX will use this
    
    Returns:
        Dict with model parameters and metadata
    """
    # Load checkpoint using Orbax CheckpointManager
    checkpoint_manager = CheckpointManager(checkpoint_path)
    checkpoint = checkpoint_manager.restore(0)  # Restore latest checkpoint
    
    return checkpoint
