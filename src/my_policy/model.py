from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


LOGGER = logging.getLogger(__name__)
STATE_ACTIVE_IDX = np.array([0, 1, 2, 3, 4, 6, 11, 12], dtype=np.int64)
TOKENIZER_CONFIG_NAME = "policy_preprocessor.json"
GATED_PALIGEMMA_REPO = "google/paligemma-3b-pt-224"
TOKENIZER_SENTINEL_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _maybe_add_lerobot_src_to_syspath() -> None:
    env_path = os.environ.get("LEROBOT_SRC_PATH")
    if env_path:
        env_candidate = Path(env_path).expanduser().resolve()
        if env_candidate.name != "src":
            env_candidate = env_candidate / "src"
        env_candidate_str = str(env_candidate)
        if env_candidate.is_dir() and env_candidate_str not in sys.path:
            sys.path.insert(0, env_candidate_str)

    if importlib.util.find_spec("lerobot") is not None:
        return

    # Search upward for either a sibling `lerobot/src` checkout or a direct
    # `src` directory inside a lerobot repo instead of assuming a fixed depth.
    for parent in Path(__file__).resolve().parents:
        candidates = (
            parent / "lerobot" / "src",
            parent / "src" if parent.name == "lerobot" else None,
        )
        for candidate in candidates:
            if candidate is None:
                continue
            candidate_str = str(candidate)
            if candidate.is_dir() and candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
                return


def _feature_type_name(feature: Any) -> str:
    ftype = getattr(feature, "type", None)
    name = getattr(ftype, "name", None)
    return str(name or ftype or "")


def _feature_first_dim(feature: Any, default: int) -> int:
    shape = getattr(feature, "shape", None)
    if isinstance(shape, (tuple, list)) and len(shape) > 0:
        return int(shape[0])
    return int(default)


def _looks_like_tokenizer_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and any(
        (path / filename).is_file() for filename in TOKENIZER_SENTINEL_FILES
    )


def _resolve_tokenizer_dir_candidate(path: Path) -> Path | None:
    path = path.expanduser().resolve()
    if _looks_like_tokenizer_dir(path):
        return path

    snapshots_dir = path / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    snapshot_dirs = sorted(
        (candidate for candidate in snapshots_dir.iterdir() if candidate.is_dir()),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    for snapshot_dir in snapshot_dirs:
        if _looks_like_tokenizer_dir(snapshot_dir):
            return snapshot_dir
    return None


def _load_tokenizer_repo_id(checkpoint_path: str) -> str | None:
    config_path = Path(checkpoint_path) / TOKENIZER_CONFIG_NAME
    if not config_path.is_file():
        return None

    config = json.loads(config_path.read_text())
    for step in config.get("steps", []):
        if step.get("registry_name") != "tokenizer_processor":
            continue
        tokenizer_name = step.get("config", {}).get("tokenizer_name")
        return str(tokenizer_name) if tokenizer_name else None
    return None


def _find_local_tokenizer_dir(repo_id: str) -> Path | None:
    env_path = os.environ.get("PALIGEMMA_TOKENIZER_PATH")
    if env_path:
        candidate = _resolve_tokenizer_dir_candidate(Path(env_path))
        if candidate is None:
            raise RuntimeError(
                "PALIGEMMA_TOKENIZER_PATH is set, but no tokenizer files were found there. "
                "Point it at a local snapshot directory that contains config.json and tokenizer files."
            )
        return candidate

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_cache_dir = hf_home / f"models--{repo_id.replace('/', '--')}"
    return _resolve_tokenizer_dir_candidate(repo_cache_dir)


def _resolve_tokenizer_override(checkpoint_path: str) -> str | None:
    tokenizer_name = _load_tokenizer_repo_id(checkpoint_path)
    if not tokenizer_name:
        return None

    tokenizer_path = Path(tokenizer_name).expanduser()
    if tokenizer_path.exists():
        return str(tokenizer_path.resolve())

    local_dir = _find_local_tokenizer_dir(tokenizer_name)
    if local_dir is not None:
        LOGGER.info("Using local tokenizer files from %s", local_dir)
        return str(local_dir)

    if tokenizer_name == GATED_PALIGEMMA_REPO:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        raise RuntimeError(
            "Tokenizer repo 'google/paligemma-3b-pt-224' is gated and no local snapshot was found. "
            f"Download it into HF_HOME ({hf_home}) or set PALIGEMMA_TOKENIZER_PATH to the local snapshot directory."
        )

    return None


class MyModel:
    """Thin wrapper around current LeRobot pretrained policies for AIRoA inference."""

    def __init__(self, checkpoint_path: str, device: str = "cuda", strict: bool = False) -> None:
        _maybe_add_lerobot_src_to_syspath()
        self.checkpoint_path = checkpoint_path
        self.device = self._resolve_device(device)
        self._last_prompt: str | None = None
        self._legacy_api = False

        if importlib.util.find_spec("lerobot.policies.pi05") is not None:
            try:
                from lerobot.policies.factory import make_pre_post_processors
                from lerobot.policies.pi05 import PI05Policy
                from lerobot.utils.constants import ACTION
                from lerobot.utils.constants import OBS_STATE
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Found a LeRobot installation with PI05 support, but importing it failed."
                ) from exc

            self._legacy_api = True
            self._ACTION = ACTION
            self._OBS_STATE = OBS_STATE
            self.policy = PI05Policy.from_pretrained(checkpoint_path, strict=strict)
            self.policy.to(self.device)
            self.policy.eval()
            preprocessor_overrides = {
                "device_processor": {"device": str(self.device)},
            }
            tokenizer_override = _resolve_tokenizer_override(checkpoint_path)
            if tokenizer_override is not None:
                preprocessor_overrides["tokenizer_processor"] = {
                    "tokenizer_name": tokenizer_override,
                }
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                checkpoint_path,
                preprocessor_overrides=preprocessor_overrides,
            )
        else:
            try:
                from lerobot.common.constants import ACTION
                from lerobot.common.constants import OBS_ROBOT
                from lerobot.common.policies.factory import get_policy_class
                from lerobot.configs.policies import PreTrainedConfig
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Failed to import LeRobot policy modules. "
                    "Set LEROBOT_SRC_PATH to your lerobot/src path and install dependencies."
                ) from exc

            self._ACTION = ACTION
            self._OBS_STATE = OBS_ROBOT
            self.config = PreTrainedConfig.from_pretrained(checkpoint_path)
            self.config.device = str(self.device)
            if self.config.type == "pi05":
                raise RuntimeError(
                    "This checkpoint requires PI05 support, but the installed `lerobot` package "
                    "does not provide `lerobot.policies.pi05`. Install `lerobot[pi]` from the "
                    "upstream Hugging Face repository in the Docker image."
                )
            policy_cls = get_policy_class(self.config.type)
            self.policy = policy_cls.from_pretrained(checkpoint_path, config=self.config, strict=strict)
            self.policy.to(self.device)
            self.policy.eval()

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

    def _build_legacy_sample(self, obs: dict[str, Any]) -> dict[str, Any]:
        head_rgb = np.asarray(obs["head_rgb"], dtype=np.uint8)
        hand_rgb = np.asarray(obs["hand_rgb"], dtype=np.uint8)
        state = np.asarray(obs["state"], dtype=np.float32)
        prompt = str(obs.get("prompt", ""))

        head_chw = self._to_chw_float01(head_rgb)
        hand_chw = self._to_chw_float01(hand_rgb)
        state_t = self._build_state(state)

        batch: dict[str, Any] = {
            self._state_key: state_t,
            "task": prompt,
        }
        for image_key in self._image_keys:
            batch[image_key] = self._pick_camera(image_key, head_chw, hand_chw)
        return batch

    def _build_batch(self, obs: dict[str, Any]) -> dict[str, Any]:
        sample = self._build_legacy_sample(obs)
        batch: dict[str, Any] = {
            self._state_key: sample[self._state_key].unsqueeze(0).to(self.device),
            "task": [str(sample["task"])],
        }
        for image_key in self._image_keys:
            batch[image_key] = sample[image_key].unsqueeze(0).to(self.device)
        return batch

    def predict_actions(self, obs: dict[str, Any]) -> np.ndarray:
        prompt = str(obs.get("prompt", ""))
        if prompt != self._last_prompt and hasattr(self.policy, "reset"):
            self.policy.reset()
        self._last_prompt = prompt

        with torch.inference_mode():
            if self._legacy_api:
                sample = self._build_legacy_sample(obs)
                processed = self.preprocessor(sample)
                actions = self.policy.select_action(processed)
                actions = self.postprocessor(actions)
            else:
                batch = self._build_batch(obs)
                actions = self.policy.select_action(batch)

        actions_np = np.asarray(actions.detach().cpu().numpy(), dtype=np.float32)
        if actions_np.ndim == 1:
            actions_np = actions_np[None, :]
        return actions_np

    @property
    def action_dim(self) -> int:
        return self._action_dim
