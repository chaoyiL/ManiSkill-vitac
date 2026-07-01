from __future__ import annotations

import dataclasses
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY_SRC = ROOT / "policy" / "src"
if str(POLICY_SRC) not in sys.path:
    sys.path.insert(0, str(POLICY_SRC))

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass(frozen=True)
class EpisodeData:
    """One transformed episode, ready for model calls."""

    indices: tuple[int, ...]
    frames: tuple[int, ...]
    raw_samples: tuple[dict[str, Any], ...]
    observations: tuple[_model.Observation, ...]
    actions: tuple[jax.Array, ...]
    prompts: tuple[str | None, ...]


def _as_scalar(value: Any) -> Any:
    value = np.asarray(value)
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(()).item()
    return value


def _scalar(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)).reshape(-1)[0])


def _copy_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _copy_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_tree(v) for v in value)
    if hasattr(value, "copy"):
        return value.copy()
    return value


def _add_batch_dim(data: Any) -> Any:
    """Add batch dimension (None, ...) to arrays or pytrees."""
    return jax.tree.map(lambda x: jnp.asarray(x)[None, ...] if x is not None else None, data)


def _batch_observation(observation: _model.Observation) -> _model.Observation:
    """Add batch dimension to observation."""
    return jax.tree.map(lambda x: jnp.asarray(x)[None, ...] if x is not None else None, observation)


def _batch_actions(actions: jax.Array) -> jax.Array:
    """Add batch dimension to actions."""
    return jnp.asarray(actions)[None, ...]


def _prompt_from_raw(raw: dict[str, Any]) -> str | None:
    prompt = raw.get("prompt", raw.get("task"))
    if prompt is None:
        return None
    if isinstance(prompt, bytes):
        return prompt.decode("utf-8")
    if not isinstance(prompt, str):
        prompt = np.asarray(prompt).item()
    if isinstance(prompt, bytes):
        return prompt.decode("utf-8")
    return str(prompt)


def _normalize_observation_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Make transformed samples compatible with Observation type checks."""

    if "image_mask" in data:
        data["image_mask"] = {key: np.asarray(value, dtype=np.bool_) for key, value in data["image_mask"].items()}
    if "tactile_mask" in data:
        data["tactile_mask"] = np.asarray(data["tactile_mask"], dtype=np.bool_)
    if "tokenized_prompt_mask" in data:
        data["tokenized_prompt_mask"] = np.asarray(data["tokenized_prompt_mask"], dtype=np.bool_)
    return data


def _unwrap_dataset(dataset: Any) -> Any:
    while hasattr(dataset, "_dataset"):
        dataset = dataset._dataset
    return dataset


def _indices_for_episode_from_metadata(raw_dataset: Any, episode_index: int | str) -> tuple[int, ...] | None:
    try:
        episode_index = int(episode_index)
    except (TypeError, ValueError):
        return None

    dataset = _unwrap_dataset(raw_dataset)
    episode_data_index = getattr(dataset, "episode_data_index", None)
    if episode_data_index is None:
        return None
    if "from" not in episode_data_index or "to" not in episode_data_index:
        return None

    starts = episode_data_index["from"]
    ends = episode_data_index["to"]
    if episode_index < 0 or episode_index >= len(starts):
        raise ValueError(
            f"Episode {episode_index} is out of range for this dataset. "
            f"Available episode indices are 0..{len(starts) - 1}."
        )

    start = int(np.asarray(starts[episode_index]))
    end = int(np.asarray(ends[episode_index]))
    if end <= start:
        return None
    return tuple(range(start, end))


def _indices_for_episode(raw_dataset: Any, episode_index: int | str) -> tuple[int, ...]:
    """Get episode indices from dataset metadata."""
    metadata_indices = _indices_for_episode_from_metadata(raw_dataset, episode_index)
    if metadata_indices is None:
        raise ValueError(f"Episode {episode_index!r} not found in dataset metadata.")
    return metadata_indices


def create_transformed_dataset(
    train_config: _config.TrainConfig,
    checkpoint_dir: str | pathlib.Path,
) -> tuple[_config.DataConfig, _data_loader.Dataset, _data_loader.Dataset]:
    """Build raw and transformed datasets with the same transforms as training."""

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    assets_dir = checkpoint_dir / "assets"
    if not assets_dir.exists() and checkpoint_dir.name == "params":
        assets_dir = checkpoint_dir.parent / "assets"
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset_name = getattr(_config, "DATASET_TRAIN_NAME")
    dataset_namespace = getattr(_config, "DATASET_REPO_NAMESPACE")
    asset_id = data_config.asset_id or dataset_name
    data_config = dataclasses.replace(
        data_config,
        repo_id=f"{dataset_namespace}/{dataset_name}",
        asset_id=asset_id,
        norm_stats=_checkpoints.load_norm_stats(assets_dir, asset_id),
    )
    raw_dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=train_config.model.action_horizon,
        model_config=train_config.model,
    )
    transformed_dataset = _data_loader.transform_dataset(
        raw_dataset,
        data_config,
    )
    return data_config, raw_dataset, transformed_dataset


def load_episode(
    train_config: _config.TrainConfig,
    checkpoint_dir: str | pathlib.Path,
    episode_index: int | str,
    *,
    start_frame: int = 0,
    sample_interval: int | None = None,
    max_frames: int | None = None,
    frame_indices: Sequence[int] | None = None,
) -> EpisodeData:
    """Load one episode and return observations, language prompts, and actions."""

    _, raw_dataset, transformed_dataset = create_transformed_dataset(train_config, checkpoint_dir)
    indices = _indices_for_episode(raw_dataset, episode_index)
    if max_frames is not None:
        indices = indices[:max_frames]
    if frame_indices is not None and sample_interval is not None:
        raise ValueError("frame_indices and sample_interval cannot both be set.")
    if sample_interval is not None:
        if sample_interval <= 0:
            raise ValueError(f"sample_interval must be positive, got {sample_interval}")
        frame_indices = tuple(range(start_frame, len(indices), sample_interval))
    elif frame_indices is None:
        frame_indices = tuple(range(len(indices)))

    relative_frames = []
    if frame_indices is not None:
        selected_indices = []
        for frame_index in frame_indices:
            if frame_index < 0 or frame_index >= len(indices):
                raise ValueError(
                    f"Frame {frame_index} is out of range for episode {episode_index}; "
                    f"available relative frames are 0..{len(indices) - 1}."
                )
            selected_indices.append(indices[frame_index])
            relative_frames.append(frame_index)
        indices = tuple(selected_indices)
    if not indices:
        raise ValueError(f"No frames selected for episode {episode_index}.")

    raw_samples = []
    observations = []
    actions = []
    prompts = []
    for index in indices:
        raw = raw_dataset[index]
        transformed = _copy_tree(transformed_dataset[index])
        transformed = _normalize_observation_dict(transformed)
        raw_samples.append(raw)
        observations.append(_model.Observation.from_dict(transformed))
        actions.append(np.asarray(transformed["actions"], dtype=np.float32))
        prompts.append(_prompt_from_raw(raw))

    return EpisodeData(
        indices=tuple(indices),
        frames=tuple(relative_frames),
        raw_samples=tuple(raw_samples),
        observations=tuple(observations),
        actions=tuple(actions),
        prompts=tuple(prompts),
    )


def load_model(train_config: _config.TrainConfig, checkpoint_dir: str | pathlib.Path):
    """Load a pi0/pi05 model from a checkpoint step directory or its params subdir."""

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    params_dir = checkpoint_dir if checkpoint_dir.name == "params" else checkpoint_dir / "params"
    if not params_dir.exists():
        raise FileNotFoundError(f"Checkpoint params directory not found: {params_dir}")
    params = _model.restore_params(params_dir, dtype=jnp.bfloat16)
    model_config = train_config.model
    if hasattr(model_config, "dtype"):
        model_config = dataclasses.replace(model_config, dtype="bfloat16")
    try:
        model = model_config.load(params)
    except ValueError as exc:
        message = str(exc)
        if "anytouch" in message or "tactile_proj" in message:
            raise ValueError(
                "Checkpoint/model config mismatch: the selected config expects tactile AnyTouch parameters "
                "(`anytouch` and `tactile_proj`), but this checkpoint does not contain them. "
                "Use a checkpoint trained with `pi05_bi_vitac` tactile support, or use the matching visual-only "
                "config/checkpoint. Tactile contribution cannot be evaluated from a checkpoint without tactile "
                f"parameters. Params path: {params_dir}"
            ) from exc
        raise
    model.eval()
    return model


def ablate_modality_observation(
    observation: _model.Observation,
    *,
    modality: str,
    prompt: str | None = None,
    prompt_tokenizer: _tokenizer.PaligemmaTokenizer | None = None,
    state_in_prompt: bool = False,
) -> _model.Observation:
    if modality == "vision":
        if not observation.images:
            raise ValueError("Observation has no visual images to ablate.")
        return dataclasses.replace(
            observation,
            image_masks={
                key: np.zeros_like(np.asarray(observation.image_masks.get(key, False)), dtype=np.bool_)
                for key in observation.images
            },
        )
    if modality == "tactile":
        if observation.tactile is None:
            raise ValueError("Observation has no tactile field to ablate.")
        tactile_mask = observation.tactile_mask
        if tactile_mask is None:
            tactile_mask = False
        return dataclasses.replace(
            observation,
            tactile_mask=np.zeros_like(np.asarray(tactile_mask), dtype=np.bool_),
        )
    if modality == "state":
        if not state_in_prompt:
            raise ValueError("state ablation expects a discrete-state model with state in the prompt.")
        if prompt is None or prompt_tokenizer is None:
            raise ValueError("state ablation requires prompt and prompt_tokenizer.")
        if observation.tokenized_prompt_mask is None:
            raise ValueError("Observation has no tokenized_prompt_mask to ablate.")

        state = np.asarray(observation.state)
        token_mask = np.asarray(observation.tokenized_prompt_mask, dtype=np.bool_).copy()
        tokenizer = prompt_tokenizer._tokenizer
        max_len = prompt_tokenizer._max_len
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")

        def state_span_for_state(state_i: np.ndarray) -> tuple[int, int]:
            discretized_state = np.digitize(state_i, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            before_state = f"Task: {cleaned_text}, State: "
            through_state = f"Task: {cleaned_text}, State: {state_str}"
            start = len(tokenizer.encode(before_state, add_bos=True))
            end = len(tokenizer.encode(through_state, add_bos=True))
            return min(start, max_len), min(end, max_len)

        if state.ndim == 1:
            start, end = state_span_for_state(state)
            token_mask[start:end] = False
        else:
            for batch_index, state_i in enumerate(state):
                start, end = state_span_for_state(state_i)
                token_mask[batch_index, start:end] = False

        return dataclasses.replace(
            observation,
            tokenized_prompt_mask=token_mask,
        )
    if modality == "language_prompt":
        if not state_in_prompt:
            raise ValueError("language_prompt ablation expects a discrete-state model with state in the prompt.")
        if prompt is None or prompt_tokenizer is None:
            raise ValueError("language_prompt ablation requires prompt and prompt_tokenizer.")
        if observation.tokenized_prompt_mask is None:
            raise ValueError("Observation has no tokenized_prompt_mask to ablate.")

        token_mask = np.asarray(observation.tokenized_prompt_mask, dtype=np.bool_).copy()
        tokenizer = prompt_tokenizer._tokenizer
        max_len = prompt_tokenizer._max_len
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        before_language = "Task: "
        through_language = f"Task: {cleaned_text}"
        start = min(len(tokenizer.encode(before_language, add_bos=True)), max_len)
        end = min(len(tokenizer.encode(through_language, add_bos=True)), max_len)
        if token_mask.ndim == 1:
            token_mask[start:end] = False
        else:
            token_mask[:, start:end] = False

        return dataclasses.replace(
            observation,
            tokenized_prompt_mask=token_mask,
        )
    raise ValueError(
        f"Unsupported modality {modality!r}. Expected 'vision', 'tactile', 'state', or 'language_prompt'."
    )
