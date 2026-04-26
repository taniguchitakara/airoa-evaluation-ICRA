# src/my_policy/adapter.py
import numpy as np
import jax
import jax.numpy as jnp
from .model import load_policy_model


class MyPolicyAdapter:
    """Adapter for OpenPI policy (JAX) to AIRoA WebSocket contract."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """Initialize the policy adapter.
        
        Args:
            checkpoint_path: Path to checkpoint directory (e.g., download/baseline/50000)
            device: Device string (cuda or cpu)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        # Load OpenPI checkpoint
        self.checkpoint = load_policy_model(checkpoint_path, device)
        
        # Extract model parameters and config
        self.params = self.checkpoint.get("params", {})
        self.train_state = self.checkpoint.get("train_state", {})

    def infer(self, obs: dict) -> dict:
        """Run inference on observations and return actions.
        
        Contract (see README §5):
            Input:
                obs["head_rgb"]:  (480, 640, 3) uint8
                obs["hand_rgb"]:  (480, 640, 3) uint8
                obs["state"]:     (8,) float32
                obs["prompt"]:    str (optional)
            Output:
                {"actions": np.ndarray of shape (T, 11), dtype=float32}
        """
        # Extract inputs
        head_rgb = obs["head_rgb"]                      # (480, 640, 3) uint8
        hand_rgb = obs["hand_rgb"]                      # (480, 640, 3) uint8
        state = obs["state"].astype(np.float32)        # (8,) float32
        prompt = obs.get("prompt", "")                  # str
        
        # Normalize images from uint8 [0, 255] to float32 [0, 1]
        head_rgb_normalized = head_rgb.astype(np.float32) / 255.0
        hand_rgb_normalized = hand_rgb.astype(np.float32) / 255.0
        
        # Convert to JAX arrays
        head_rgb_jax = jnp.array(head_rgb_normalized)
        hand_rgb_jax = jnp.array(hand_rgb_normalized)
        state_jax = jnp.array(state)
        
        # Prepare observation dict for OpenPI
        obs_dict = {
            "observation": {
                "head_rgb": head_rgb_jax,
                "hand_rgb": hand_rgb_jax,
                "state": state_jax,
            },
            "task_description": prompt,
        }
        
        # Run policy inference (OpenPI model forward pass)
        # Note: The exact API depends on OpenPI's policy implementation
        # This is a placeholder - adjust based on actual OpenPI API
        try:
            actions = self._run_openpi_inference(obs_dict)
        except Exception as e:
            # Fallback: return zero actions if inference fails
            print(f"Inference error: {e}")
            actions = np.zeros((1, 11), dtype=np.float32)
        
        # Ensure output is correct shape and dtype
        actions = np.asarray(actions, dtype=np.float32)
        
        # Validate shape: must be (T, 11) where T >= 1
        assert actions.ndim == 2, f"Expected 2D actions, got {actions.ndim}D"
        assert actions.shape[1] == 11, f"Expected 11 action dims, got {actions.shape[1]}"
        assert actions.shape[0] >= 1, f"Expected T >= 1, got {actions.shape[0]}"
        
        return {"actions": actions}
    
    def _run_openpi_inference(self, obs_dict: dict) -> np.ndarray:
        """Run OpenPI model inference.
        
        This is a placeholder. The actual implementation depends on:
        - OpenPI's policy module structure
        - How to call the model with params
        - Expected output shape
        """
        # TODO: Implement actual OpenPI inference
        # This might look like:
        # actions = openpi_policy_fn(self.params, obs_dict)
        raise NotImplementedError(
            "OpenPI inference not yet implemented. "
            "Need to inspect OpenPI API and checkpoint structure."
        )