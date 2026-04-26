"""Adapter for OpenPI policy to AIRoA WebSocket contract."""
import numpy as np
from .model import load_policy_model


class MyPolicyAdapter:
    """OpenPI policy adapter for AIRoA evaluation."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """Initialize the policy adapter.
        
        Args:
            checkpoint_path: Path to checkpoint directory (step directory)
            device: Device string (kept for API compatibility, JAX handles internally)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        # Load OpenPI policy using high-level API
        self.policy = load_policy_model(checkpoint_path, device)
    
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
        # Extract inputs from AIRoA contract
        head_rgb = obs["head_rgb"]                      # (480, 640, 3) uint8
        hand_rgb = obs["hand_rgb"]                      # (480, 640, 3) uint8
        state = obs["state"].astype(np.float32)        # (8,) float32
        prompt = obs.get("prompt", "")                  # str
        
        # Prepare observation dict for OpenPI
        # OpenPI expects the exact keys from its config transforms
        obs_dict = {
            "head_rgb": np.asarray(head_rgb, dtype=np.uint8),
            "hand_rgb": np.asarray(hand_rgb, dtype=np.uint8),
            "state": np.asarray(state, dtype=np.float32),
            "prompt": prompt,
        }
        
        # Run OpenPI inference
        output = self.policy.infer(obs_dict)
        
        # Extract and validate actions
        actions = np.asarray(output["actions"], dtype=np.float32)
        
        # Validate shape: must be (T, 11) where T >= 1
        assert actions.ndim == 2, f"Expected 2D actions, got {actions.ndim}D with shape {actions.shape}"
        assert actions.shape[1] == 11, f"Expected 11 action dims, got {actions.shape[1]}"
        assert actions.shape[0] >= 1, f"Expected T >= 1, got {actions.shape[0]}"
        
        return {"actions": actions}