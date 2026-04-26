"""OpenPI policy model loader (JAX/Flax based)."""
import os
import json
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from orbax.checkpoint.pytree_checkpoint_handler import PyTreeCheckpointHandler


def load_policy_model(checkpoint_path: str, device: str = "cuda"):
    """Load OpenPI model from Orbax checkpoint.
    
    Checkpoint structure:
        checkpoint_path/
        ├── params/              (PyTree of JAX params)
        ├── assets/              (OpenPI CallbackHandler)
        └── train_state/         (PyTree of training state)
    
    Args:
        checkpoint_path: Path to checkpoint directory (e.g., my_checkpoint/)
        device: Device string (cuda or cpu)
    
    Returns:
        Dict with 'params', 'assets', and 'train_state'
    """
    checkpoint_path = os.path.expanduser(checkpoint_path)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load params using PyTreeCheckpointHandler
    params_dir = os.path.join(checkpoint_path, "params")
    if os.path.exists(params_dir):
        params_handler = PyTreeCheckpointHandler()
        params = params_handler.restore(params_dir)
    else:
        params = {}
    
    # Load train_state using PyTreeCheckpointHandler
    train_state_dir = os.path.join(checkpoint_path, "train_state")
    if os.path.exists(train_state_dir):
        train_state_handler = PyTreeCheckpointHandler()
        train_state = train_state_handler.restore(train_state_dir)
    else:
        train_state = {}
    
    # Load assets (OpenPI-specific)
    assets_dir = os.path.join(checkpoint_path, "assets")
    assets = {}
    if os.path.exists(assets_dir):
        # Assets typically contain dataset info, metadata, etc.
        # Load any JSON/YAML files if present
        for fname in os.listdir(assets_dir):
            if fname.endswith('.json'):
                asset_path = os.path.join(assets_dir, fname)
                with open(asset_path, 'r') as f:
                    assets[fname[:-5]] = json.load(f)
    
    return {
        "params": params,
        "assets": assets,
        "train_state": train_state,
        "checkpoint_path": checkpoint_path,
    }
