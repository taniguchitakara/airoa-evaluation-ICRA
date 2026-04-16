from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


STATE_ACTIVE_IDX = np.array([0, 1, 2, 3, 4, 6, 11, 12], dtype=np.int64)


def _maybe_add_lerobot_src_to_syspath() -> None:
    env_path = os.environ.get("LEROBOT_SRC_PATH")
    if env_path and os.path.isdir(env_path) and env_path not in sys.path:
        sys.path.insert(0, env_path)

    # .../airoa-evaluation-ICRA/src/my_policy/model.py -> .../airoa/lerobot/src
    candidate = Path(__file__).resolve().parents[4] / "lerobot" / "src"
    candidate_str = str(candidate)
    if candidate.is_dir() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _feature_type_name(feature: Any) -> str:
    ftype = getattr(feature, "type", None)
    name = getattr(ftype, "name", None)
    return str(name or ftype or "")


def _feature_first_dim(feature: Any, default: int) -> int:
    shape = getattr(feature, "shape", None)
    if isinstance(shape, (tuple, list)) and len(shape) > 0:
        return int(shape[0])
    return int(default)


class MyModel:
    """Thin wrapper around LeRobot PI05Policy for AIRoA websocket inference."""

    def __init__(self, checkpoint_path: str, device: str = "cuda", strict: bool = False) -> None:
        _maybe_add_lerobot_src_to_syspath()

        try:
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.pi05 import PI05Policy
            from lerobot.utils.constants import ACTION, OBS_STATE
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import LeRobot PI05 modules. "
                "Set LEROBOT_SRC_PATH to your lerobot/src path and install dependencies."
            ) from exc

        self._ACTION = ACTION
        self._OBS_STATE = OBS_STATE
        self.checkpoint_path = checkpoint_path
        self.device = self._resolve_device(device)

        self.policy = PI05Policy.from_pretrained(checkpoint_path, strict=strict)
        self.policy.to(self.device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        self._image_keys = self._infer_image_keys()
        self._state_key = self._infer_state_key()
        state_feature = self.policy.config.input_features[self._state_key]
        self._state_dim = _feature_first_dim(state_feature, default=8)

        action_feature = self.policy.config.output_features.get(self._ACTION)
        self._action_dim = _feature_first_dim(action_feature, default=11) if action_feature else 11

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested)
        if requested.startswith("mps") and torch.backends.mps.is_available():
            return torch.device(requested)
        return torch.device("cpu")

    def _infer_image_keys(self) -> list[str]:
        keys: list[str] = []
        for key, feature in self.policy.config.input_features.items():
            if _feature_type_name(feature).upper() == "VISUAL":
                keys.append(key)
        if not keys:
            raise ValueError("PI05 config has no visual input features.")
        return keys

    def _infer_state_key(self) -> str:
        if self._OBS_STATE in self.policy.config.input_features:
            return self._OBS_STATE
        for key, feature in self.policy.config.input_features.items():
            if _feature_type_name(feature).upper() == "STATE":
                return key
        raise ValueError("PI05 config has no state input feature.")

    def _pick_camera(self, key: str, head_chw: torch.Tensor, hand_chw: torch.Tensor) -> torch.Tensor:
        key_l = key.lower()
        if "hand" in key_l or "wrist" in key_l or "gripper" in key_l:
            return hand_chw
        return head_chw

    def _build_state(self, state8: np.ndarray) -> torch.Tensor:
        if state8.shape != (8,):
            raise ValueError(f"Expected obs['state'] shape (8,), got {state8.shape}")

        if self._state_dim == 8:
            out = state8
        else:
            out = np.zeros((self._state_dim,), dtype=np.float32)
            if self._state_dim > int(STATE_ACTIVE_IDX.max()):
                out[STATE_ACTIVE_IDX] = state8
            else:
                out[:8] = state8
        return torch.from_numpy(out)

    @staticmethod
    def _to_chw_float01(image_hwc: np.ndarray) -> torch.Tensor:
        if image_hwc.ndim != 3 or image_hwc.shape[2] != 3:
            raise ValueError(f"Expected HWC image with 3 channels, got {image_hwc.shape}")
        image = image_hwc.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1).contiguous()

    def _build_batch(self, obs: dict[str, Any]) -> dict[str, Any]:
        head_rgb = np.asarray(obs["head_rgb"], dtype=np.uint8)
        hand_rgb = np.asarray(obs["hand_rgb"], dtype=np.uint8)
        state = np.asarray(obs["state"], dtype=np.float32)
        prompt = str(obs.get("prompt", ""))

        head_chw = self._to_chw_float01(head_rgb)
        hand_chw = self._to_chw_float01(hand_rgb)
        state_t = self._build_state(state)

        batch: dict[str, Any] = {
            self._state_key: state_t,
            "task": [prompt],
        }
        for image_key in self._image_keys:
            batch[image_key] = self._pick_camera(image_key, head_chw, hand_chw)
        return batch

    def predict_actions(self, obs: dict[str, Any]) -> np.ndarray:
        batch = self._build_batch(obs)
        with torch.inference_mode():
            processed = self.preprocessor(batch)
            actions = self.policy.predict_action_chunk(processed)
            actions = self.postprocessor(actions)

        actions_np = np.asarray(actions.detach().cpu().numpy(), dtype=np.float32)
        if actions_np.ndim == 3:
            actions_np = actions_np[0]
        if actions_np.ndim == 1:
            actions_np = actions_np[None, :]
        return actions_np

    @property
    def action_dim(self) -> int:
        return self._action_dim