from __future__ import annotations

import argparse
import csv
import dataclasses
import pathlib
import sys
from collections.abc import Iterable, Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY_SRC = ROOT / "policy" / "src"
if str(POLICY_SRC) not in sys.path:
    sys.path.insert(0, str(POLICY_SRC))

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask
from openpi.models import tokenizer as _tokenizer
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


EPISODE_KEYS = ("episode_index", "episode_idx", "episode_id", "episode")


@dataclasses.dataclass(frozen=True)
class EpisodeData:
    """One transformed episode, ready for model calls."""

    indices: tuple[int, ...]
    frames: tuple[int, ...]
    raw_samples: tuple[dict[str, Any], ...]
    observations: tuple[_model.Observation, ...]
    actions: tuple[jax.Array, ...]
    prompts: tuple[str | None, ...]


@dataclasses.dataclass(frozen=True)
class LikelihoodIntegrationResult:
    """Result of integrating data actions to the base distribution."""

    x_base: jax.Array
    r_tot: jax.Array
    log_p_base: jax.Array
    log_likelihood: jax.Array


@dataclasses.dataclass(frozen=True)
class VelocityContext:
    """Cached observation-dependent state for repeated velocity calls."""

    observation: _model.Observation
    prefix_tokens: jax.Array
    prefix_mask: jax.Array
    kv_cache: Any


def _as_scalar(value: Any) -> Any:
    value = np.asarray(value)
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(()).item()
    return value


def _copy_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _copy_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_copy_tree(v) for v in value)
    if hasattr(value, "copy"):
        return value.copy()
    return value


def _batch_observation(observation: _model.Observation) -> _model.Observation:
    return jax.tree.map(lambda x: jnp.asarray(x)[None, ...] if x is not None else None, observation)


def _batch_actions(actions: Any) -> jax.Array:
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


def _available_keys(dataset: Any, limit: int = 1) -> list[str]:
    keys: set[str] = set()
    for index in range(min(limit, len(dataset))):
        item = dataset[index]
        if isinstance(item, dict):
            keys.update(item.keys())
    return sorted(keys)


def _episode_value(raw: dict[str, Any]) -> Any | None:
    for key in EPISODE_KEYS:
        if key in raw:
            return _as_scalar(raw[key])
    return None


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
    metadata_indices = _indices_for_episode_from_metadata(raw_dataset, episode_index)
    if metadata_indices is not None:
        return metadata_indices

    wanted: Any = episode_index
    try:
        wanted = int(episode_index)
    except (TypeError, ValueError):
        pass

    indices = []
    saw_episode_key = False
    for index in range(len(raw_dataset)):
        raw = raw_dataset[index]
        if not isinstance(raw, dict):
            continue
        current = _episode_value(raw)
        if current is None:
            continue
        saw_episode_key = True
        if current == wanted or str(current) == str(wanted):
            indices.append(index)

    if indices:
        return tuple(indices)

    keys = _available_keys(raw_dataset, limit=min(10, len(raw_dataset)))
    if not saw_episode_key:
        raise ValueError(
            "Could not find an episode field in dataset samples. "
            f"Tried {EPISODE_KEYS}; first sample keys include: {keys}"
        )
    raise ValueError(f"Episode {episode_index!r} was not found in dataset.")


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
        actions.append(jnp.asarray(transformed["actions"]))
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


def predict_velocity(
    model: _model.BaseModel,
    observation: _model.Observation,
    x: jax.Array,
    t: jax.Array | float,
) -> jax.Array:
    """Compute one flow velocity v(x, t, o), following Pi0.sample_actions."""

    if not hasattr(model, "embed_prefix") or not hasattr(model, "embed_suffix"):
        raise TypeError("predict_velocity currently expects a Pi0/Pi05-style model.")

    observation = _model.preprocess_observation(
        None,
        observation,
        train=False,
        image_keys=model.image_keys if model.image_keys is not None else list(observation.images.keys()),
    )
    batch_size = observation.state.shape[0]
    x = jnp.asarray(x)
    if x.ndim == 2:
        x = x[None, ...]
    t = jnp.asarray(t, dtype=jnp.float32)
    if t.ndim == 0:
        t = jnp.broadcast_to(t, (batch_size,))

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x, t)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    _, suffix_out = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )[0]
    return model.action_out_proj(suffix_out[:, -model.action_horizon :])


def create_velocity_context(model: _model.BaseModel, observation: _model.Observation) -> VelocityContext:
    """Precompute observation prefix and KV cache for repeated suffix velocity calls."""

    image_keys = model.image_keys if model.image_keys is not None else list(observation.images.keys())
    observation = _model.preprocess_observation(None, observation, train=False, image_keys=image_keys)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return VelocityContext(
        observation=observation,
        prefix_tokens=prefix_tokens,
        prefix_mask=prefix_mask,
        kv_cache=kv_cache,
    )


def predict_velocity_with_context(
    model: _model.BaseModel,
    context: VelocityContext,
    x: jax.Array,
    t: jax.Array,
) -> jax.Array:
    """Compute v(x,t,o) using cached prefix/KV state."""

    batch_size = context.observation.state.shape[0]
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(context.observation, x, t)
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = jnp.broadcast_to(
        context.prefix_mask[:, None, :],
        (batch_size, suffix_tokens.shape[1], context.prefix_tokens.shape[1]),
    )
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(context.prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=context.kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    del prefix_out
    return model.action_out_proj(suffix_out[:, -model.action_horizon :])


def predict_episode_velocity(
    model: _model.BaseModel,
    episode: EpisodeData,
    *,
    frame: int = 0,
    t: float = 0.5,
    x: jax.Array | None = None,
) -> jax.Array:
    """Convenience wrapper for a single frame in a loaded episode."""

    observation = _batch_observation(episode.observations[frame])
    actions = _batch_actions(episode.actions[frame])
    if x is None:
        x = actions
    return predict_velocity(model, observation, x, t)


def velocity_and_exact_divergence(
    model: _model.BaseModel,
    context: VelocityContext,
    x: jax.Array,
    t: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute velocity and exact Tr(dv/dx)."""

    x = jnp.asarray(x, dtype=jnp.float32)
    batch_size = x.shape[0]
    x_flat = jnp.reshape(x, (batch_size, -1))

    def flat_velocity(x_flat_in):
        x_in = jnp.reshape(x_flat_in, x.shape)
        velocity = predict_velocity_with_context(model, context, x_in, t)
        return jnp.reshape(velocity.astype(jnp.float32), x_flat.shape)

    velocity_flat = flat_velocity(x_flat)
    jac = jax.jacfwd(flat_velocity)(x_flat)
    divergence = jnp.einsum("bebe->b", jac)
    return jnp.reshape(velocity_flat, x.shape), divergence


def standard_normal_log_prob(x: jax.Array) -> jax.Array:
    """Compute log p0(x) for a standard Gaussian base distribution."""

    event_dims = tuple(range(1, x.ndim))
    event_size = int(np.prod(x.shape[1:]))
    return -0.5 * (jnp.sum(jnp.square(x), axis=event_dims) + event_size * jnp.log(2.0 * jnp.pi))


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


def integrate_to_base_log_likelihood(
    model: _model.BaseModel,
    observation: _model.Observation,
    reference_actions: jax.Array,
    *,
    num_steps: int,
    t_min: float,
    loglike_fn: Any | None = None,
) -> LikelihoodIntegrationResult:
    """Integrate pi0 code-time from actions at t=0 to base noise at t=1.

    In policy/src/openpi/models/pi0.py:
      x_t = t * noise + (1 - t) * actions
      v_t learns dx_t/dt = noise - actions

    Therefore actions live near t=0 and the standard Gaussian base lives at t=1.
    This integrates from t_min to 1 with a Heun predictor-corrector step to avoid
    evaluating the model exactly at t=0, which is outside the training time range.
    """

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if not 0.0 <= t_min < 1.0:
        raise ValueError(f"t_min must be in [0, 1), got {t_min}")

    x = jnp.asarray(reference_actions, dtype=jnp.float32)
    if x.ndim == 2:
        x = x[None, ...]
    observation = _batch_observation(observation)

    step_indices = jnp.arange(num_steps)

    if loglike_fn is not None:
        x_base, r_tot, log_p_base, log_likelihood = loglike_fn(
            observation,
            x,
            step_indices,
            t_min,
        )
        return LikelihoodIntegrationResult(
            x_base=x_base,
            r_tot=r_tot,
            log_p_base=log_p_base,
            log_likelihood=log_likelihood,
        )

    batch_size = x.shape[0]
    t_min = jnp.asarray(t_min, dtype=jnp.float32)
    dt = (jnp.asarray(1.0, dtype=jnp.float32) - t_min) / num_steps
    context = create_velocity_context(model, observation)

    def scan_body(carry, _):
        x, r_tot, t = carry
        v0, div0 = velocity_and_exact_divergence(model, context, x, t)
        t_next = t + dt
        x_euler = x + v0 * dt
        v1, div1 = velocity_and_exact_divergence(model, context, x_euler, t_next)
        x_next = x + 0.5 * (v0 + v1) * dt
        r_next = r_tot + 0.5 * (div0 + div1) * dt
        return (x_next, r_next, t_next), None

    @jax.jit
    def run_scan(x, step_indices):
        t = jnp.full((batch_size,), t_min, dtype=jnp.float32)
        r_tot = jnp.zeros((batch_size,), dtype=jnp.float32)
        (x, r_tot, _), _ = jax.lax.scan(scan_body, (x, r_tot, t), step_indices)
        return x, r_tot

    x, r_tot = run_scan(x, step_indices)

    log_p_base = standard_normal_log_prob(x)
    log_likelihood = log_p_base + r_tot
    return LikelihoodIntegrationResult(
        x_base=x,
        r_tot=r_tot,
        log_p_base=log_p_base,
        log_likelihood=log_likelihood,
    )


def _tree_summary(prefix: str, tree: Any) -> Iterable[str]:
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _tree_summary(f"{prefix}.{key}" if prefix else key, value)
        return
    if tree is None:
        yield f"{prefix}: None"
        return
    array = np.asarray(tree)
    yield f"{prefix}: shape={array.shape}, dtype={array.dtype}"


def _scalar(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)).reshape(-1)[0])


def compute_modality_contribution(
    model: _model.BaseModel,
    observation: _model.Observation,
    reference_actions: jax.Array,
    *,
    modality: str,
    num_steps: int,
    t_min: float,
    prompt: str | None = None,
    prompt_tokenizer: _tokenizer.PaligemmaTokenizer | None = None,
    state_in_prompt: bool = False,
    loglike_fn: Any | None = None,
) -> tuple[LikelihoodIntegrationResult, LikelihoodIntegrationResult, jax.Array]:
    ablated_observation = ablate_modality_observation(
        observation,
        modality=modality,
        prompt=prompt,
        prompt_tokenizer=prompt_tokenizer,
        state_in_prompt=state_in_prompt,
    )
    original_result = integrate_to_base_log_likelihood(
        model,
        observation,
        reference_actions,
        num_steps=num_steps,
        t_min=t_min,
        loglike_fn=loglike_fn,
    )
    ablated_result = integrate_to_base_log_likelihood(
        model,
        ablated_observation,
        reference_actions,
        num_steps=num_steps,
        t_min=t_min,
        loglike_fn=loglike_fn,
    )
    contribution = original_result.log_likelihood - ablated_result.log_likelihood
    return original_result, ablated_result, contribution


def save_contribution_curve(
    rows: Sequence[dict[str, float | int]],
    *,
    output_dir: pathlib.Path,
    modality: str,
    episode_index: str,
) -> tuple[pathlib.Path, pathlib.Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{modality}_contribution_episode_{episode_index}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "dataset_index",
                "original_log_likelihood",
                "ablated_log_likelihood",
                "original_r_tot",
                "ablated_r_tot",
                "delta_logp",
                "delta_r_tot",
                "contribution",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return csv_path, None

    plot_path = output_dir / f"{modality}_contribution_components_episode_{episode_index}.png"
    frames = [row["frame"] for row in rows]
    curves = (
        ("contribution", f"{modality} contribution"),
        ("delta_logp", "delta_logp(x_base)"),
        ("delta_r_tot", "delta_r_tot"),
    )
    fig, axes = plt.subplots(len(curves), 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"{modality} contribution components over episode {episode_index}")
    for ax, (field, ylabel) in zip(axes, curves, strict=True):
        ax.plot(frames, [row[field] for row in rows], marker="o", linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Episode frame")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return csv_path, plot_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Estimate modality contribution by attention-mask ablation.")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--episode-index", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--sample-interval", type=int, default=None)
    parser.add_argument("--num-steps", "-k", type=int, default=10)
    parser.add_argument("--t-min", type=float, default=1e-3)
    parser.add_argument("--remove-modality", choices=("vision", "tactile", "state", "language_prompt"), default="tactile")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("eval_outputs/loglike"))
    args = parser.parse_args(argv)

    train_config = _config.get_config(args.config_name)
    if args.sample_interval is None:
        episode = load_episode(
            train_config,
            args.checkpoint_dir,
            args.episode_index,
            max_frames=args.max_frames,
            frame_indices=(args.frame,),
        )
    else:
        episode = load_episode(
            train_config,
            args.checkpoint_dir,
            args.episode_index,
            start_frame=args.frame,
            sample_interval=args.sample_interval,
            max_frames=args.max_frames,
        )
    model = load_model(train_config, args.checkpoint_dir)
    if not hasattr(model, "integrate_to_base_log_likelihood"):
        raise TypeError("Optimized likelihood evaluation expects a Pi0/Pi05 model with integrate_to_base_log_likelihood.")
    loglike_fn = nnx_utils.module_jit(model.integrate_to_base_log_likelihood)
    state_in_prompt = bool(getattr(train_config.model, "discrete_state_input", False))
    prompt_tokenizer = (
        _tokenizer.PaligemmaTokenizer(train_config.model.max_token_len)
        if args.remove_modality in ("state", "language_prompt") and state_in_prompt
        else None
    )

    print(f"loaded episode={args.episode_index} frames={len(episode.indices)} dataset_indices={episode.indices[:5]}")
    print(f"prompt={episode.prompts[0]!r}")
    print(f"ablated_modality={args.remove_modality}")
    print("ablation_method=attention_mask")
    print(f"state_in_prompt={state_in_prompt}")
    print("divergence_method=exact_trace_autodiff")
    print("ode_solver=heun")
    print(f"t_min={args.t_min}")
    print("model_dtype=bfloat16")

    rows = []
    for i, (frame, dataset_index, observation, reference_actions, prompt) in enumerate(
        zip(episode.frames, episode.indices, episode.observations, episode.actions, episode.prompts, strict=True)
    ):
        original_result, ablated_result, contribution = compute_modality_contribution(
            model,
            observation,
            reference_actions,
            modality=args.remove_modality,
            num_steps=args.num_steps,
            t_min=args.t_min,
            prompt=prompt,
            prompt_tokenizer=prompt_tokenizer,
            state_in_prompt=state_in_prompt,
            loglike_fn=loglike_fn,
        )
        row = {
            "frame": int(frame),
            "dataset_index": int(dataset_index),
            "original_log_likelihood": _scalar(original_result.log_likelihood),
            "ablated_log_likelihood": _scalar(ablated_result.log_likelihood),
            "original_r_tot": _scalar(original_result.r_tot),
            "ablated_r_tot": _scalar(ablated_result.r_tot),
            "delta_logp": _scalar(original_result.log_p_base - ablated_result.log_p_base),
            "delta_r_tot": _scalar(original_result.r_tot - ablated_result.r_tot),
            "contribution": _scalar(contribution),
        }
        rows.append(row)
        print(
            f"frame={row['frame']} dataset_index={row['dataset_index']} "
            f"original_log_likelihood={row['original_log_likelihood']:.6f} "
            f"ablated_log_likelihood={row['ablated_log_likelihood']:.6f} "
            f"delta_logp(x_base)={row['delta_logp']:.6f} "
            f"delta_r_tot={row['delta_r_tot']:.6f}"
        )

    if args.sample_interval is not None:
        csv_path, plot_path = save_contribution_curve(
            rows,
            output_dir=args.output_dir,
            modality=args.remove_modality,
            episode_index=str(args.episode_index),
        )
        print(f"curve_csv={csv_path}")
        if plot_path is not None:
            print(f"curve_plot={plot_path}")


if __name__ == "__main__":
    main()
