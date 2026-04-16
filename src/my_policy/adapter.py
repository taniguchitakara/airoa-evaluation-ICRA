from __future__ import annotations

from typing import Any

import numpy as np

from .model import MyModel


ACTION_ACTIVE_IDX = np.array([0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15], dtype=np.int64)


class MyPolicyAdapter:
    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        checkpoint_dir: str | None = None,
    ):
        self.checkpoint_path = checkpoint_path or checkpoint_dir
        if not self.checkpoint_path:
            raise ValueError("Either checkpoint_path or checkpoint_dir must be provided.")

        self.model = MyModel(
            checkpoint_path=self.checkpoint_path,
            device=device,
            strict=False,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "lerobot-pi05",
            "checkpoint_path": self.checkpoint_path,
            "action_dim": int(self.model.action_dim),
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        actions = self.model.predict_actions(obs)
        actions = self._postprocess_actions(actions)
        if not np.isfinite(actions).all():
            raise ValueError("Policy produced non-finite actions.")
        return {"actions": actions}

    def _postprocess_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]

        if actions.ndim != 2 or actions.shape[0] < 1:
            raise ValueError(f"Expected action array with shape (T, D), got {actions.shape}")

        if actions.shape[1] == 32:
            actions = actions[:, ACTION_ACTIVE_IDX]

        if actions.shape[1] != 11:
            raise ValueError(f"Expected action dim 11 (or 32 before projection), got {actions.shape[1]}")

        return actions.astype(np.float32, copy=False)