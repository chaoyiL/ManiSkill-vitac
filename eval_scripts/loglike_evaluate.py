from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
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
from openpi.models.pi0 import make_attn_mask
from openpi.models import tokenizer as _tokenizer
from openpi.training import config as _config

DEFAULT_HUTCHINSON_SAMPLES = 1
DEFAULT_HUTCHINSON_SEED = 0

_EVAL_UTILS_PATH = pathlib.Path(__file__).resolve().parent / "utils.py"
_EVAL_UTILS_SPEC = importlib.util.spec_from_file_location("eval_scripts_utils", _EVAL_UTILS_PATH)
if _EVAL_UTILS_SPEC is None or _EVAL_UTILS_SPEC.loader is None:
    raise ImportError(f"Could not load eval script utilities from {_EVAL_UTILS_PATH}")
_eval_utils = importlib.util.module_from_spec(_EVAL_UTILS_SPEC)
sys.modules[_EVAL_UTILS_SPEC.name] = _eval_utils
_EVAL_UTILS_SPEC.loader.exec_module(_eval_utils)

EpisodeData = _eval_utils.EpisodeData
_scalar = _eval_utils._scalar
_add_batch_dim = _eval_utils._add_batch_dim
ablate_modality_observation = _eval_utils.ablate_modality_observation
load_episode = _eval_utils.load_episode
load_model = _eval_utils.load_model


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

    def tree_flatten(self):
        return ((self.observation, self.prefix_tokens, self.prefix_mask, self.kv_cache), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


jax.tree_util.register_pytree_node_class(VelocityContext)


_RUN_SCAN_CACHE: dict[tuple[int, int, int, int, int], Any] = {}


def _get_likelihood_scan(
    model: _model.BaseModel,
    *,
    batch_size: int,
    num_steps: int,
    hutchinson_samples: int,
    hutchinson_seed: int,
):
    """Return a cached compiled scan so observations are not captured as constants."""

    cache_key = (id(model), batch_size, num_steps, hutchinson_samples, hutchinson_seed)
    if cache_key in _RUN_SCAN_CACHE:
        return _RUN_SCAN_CACHE[cache_key]

    dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)
    rng_key = jax.random.PRNGKey(hutchinson_seed)

    @jax.jit
    def run_scan(context: VelocityContext, x: jax.Array, step_indices: jax.Array):
        def scan_body(carry, step_index):
            x, r_tot, t = carry
            step_rng_key = jax.random.fold_in(rng_key, step_index)
            velocity, divergence = velocity_and_hutchinson_trace(
                model,
                context,
                x,
                t,
                step_rng_key,
                num_samples=hutchinson_samples,
            )
            return (x + velocity * dt, r_tot + divergence * dt, t + dt), None

        t = jnp.zeros((batch_size,), dtype=jnp.float32)
        r_tot = jnp.zeros((batch_size,), dtype=jnp.float32)
        (x, r_tot, _), _ = jax.lax.scan(scan_body, (x, r_tot, t), step_indices)
        return x, r_tot

    _RUN_SCAN_CACHE[cache_key] = run_scan
    return run_scan


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


def velocity_and_hutchinson_trace(
    model: _model.BaseModel,
    context: VelocityContext,
    x: jax.Array,
    t: jax.Array,
    rng_key: jax.Array,
    *,
    num_samples: int,
) -> tuple[jax.Array, jax.Array]:
    """Estimate div v(x,t,o) with Rademacher Hutchinson probes."""

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    x = jnp.asarray(x, dtype=jnp.float32)

    def velocity_fn(x_arg: jax.Array) -> jax.Array:
        return predict_velocity_with_context(model, context, x_arg, t).astype(jnp.float32)

    event_axes = tuple(range(1, x.ndim))
    sample_keys = jax.random.split(rng_key, num_samples)

    def scan_body(carry, key: jax.Array):
        trace, velocity = carry
        probe = jax.random.rademacher(key, (1, *x.shape[1:]), dtype=jnp.float32)
        probe = jnp.broadcast_to(probe, x.shape)
        velocity, tangent_out = jax.jvp(velocity_fn, (x,), (probe,))
        trace_estimate = jnp.sum(probe * tangent_out, axis=event_axes)
        return (trace + trace_estimate, velocity), None

    trace0 = jnp.zeros((x.shape[0],), dtype=jnp.float32)
    velocity0 = jnp.zeros_like(x)
    (trace, velocity), _ = jax.lax.scan(scan_body, (trace0, velocity0), sample_keys)
    trace = trace / jnp.asarray(num_samples, dtype=jnp.float32)
    return velocity, trace


def standard_normal_log_prob(x: jax.Array) -> jax.Array:
    """Compute log p0(x) for a standard Gaussian base distribution."""

    event_dims = tuple(range(1, x.ndim))
    event_size = int(np.prod(x.shape[1:]))
    return -0.5 * (jnp.sum(jnp.square(x), axis=event_dims) + event_size * jnp.log(2.0 * jnp.pi))


def _stack_observations(*observations: _model.Observation) -> _model.Observation:
    return jax.tree.map(
        lambda *xs: None if xs[0] is None else jnp.concatenate([jnp.asarray(x)[None, ...] for x in xs], axis=0),
        *observations,
    )


def _select_batch_result(result: LikelihoodIntegrationResult, index: int) -> LikelihoodIntegrationResult:
    return LikelihoodIntegrationResult(
        x_base=result.x_base[index : index + 1],
        r_tot=result.r_tot[index : index + 1],
        log_p_base=result.log_p_base[index : index + 1],
        log_likelihood=result.log_likelihood[index : index + 1],
    )


def integrate_batched_to_base_log_likelihood(
    model: _model.BaseModel,
    observation: _model.Observation,
    reference_actions: jax.Array,
    *,
    num_steps: int,
    hutchinson_samples: int = DEFAULT_HUTCHINSON_SAMPLES,
    hutchinson_seed: int = DEFAULT_HUTCHINSON_SEED,
) -> LikelihoodIntegrationResult:
    """Integrate batched pi0 code-time from actions at t=0 to base noise at t=1.

    In policy/src/openpi/models/pi0.py:
      x_t = t * noise + (1 - t) * actions
      v_t learns dx_t/dt = noise - actions

    Therefore actions live at t=0 and the standard Gaussian base lives at t=1.
    ``sample_actions`` denoises from t=1 to t=0 with negative dt, so this likelihood
    path uses positive dt to map data actions back to the Gaussian base.
    """

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if hutchinson_samples <= 0:
        raise ValueError(f"hutchinson_samples must be positive, got {hutchinson_samples}")

    x = jnp.asarray(reference_actions, dtype=jnp.float32)
    if x.ndim == 2:
        x = x[None, ...]

    batch_size = x.shape[0]
    context = create_velocity_context(model, observation)
    step_indices = jnp.arange(num_steps, dtype=jnp.int32)

    run_scan = _get_likelihood_scan(
        model,
        batch_size=batch_size,
        num_steps=num_steps,
        hutchinson_samples=hutchinson_samples,
        hutchinson_seed=hutchinson_seed,
    )
    x, r_tot = run_scan(context, x, step_indices)

    log_p_base = standard_normal_log_prob(x)
    log_likelihood = log_p_base + r_tot
    return LikelihoodIntegrationResult(
        x_base=x,
        r_tot=r_tot,
        log_p_base=log_p_base,
        log_likelihood=log_likelihood,
    )


def integrate_to_base_log_likelihood(
    model: _model.BaseModel,
    observation: _model.Observation,
    reference_actions: jax.Array,
    *,
    num_steps: int,
    hutchinson_samples: int = DEFAULT_HUTCHINSON_SAMPLES,
    hutchinson_seed: int = DEFAULT_HUTCHINSON_SEED,
) -> LikelihoodIntegrationResult:
    return integrate_batched_to_base_log_likelihood(
        model,
        _add_batch_dim(observation),
        reference_actions,
        num_steps=num_steps,
        hutchinson_samples=hutchinson_samples,
        hutchinson_seed=hutchinson_seed,
    )


def compute_modality_contribution(
    model: _model.BaseModel,
    observation: _model.Observation,
    reference_actions: jax.Array,
    *,
    modality: str,
    num_steps: int,
    hutchinson_samples: int = DEFAULT_HUTCHINSON_SAMPLES,
    hutchinson_seed: int = DEFAULT_HUTCHINSON_SEED,
    prompt: str | None = None,
    prompt_tokenizer: _tokenizer.PaligemmaTokenizer | None = None,
    state_in_prompt: bool = False,
) -> tuple[LikelihoodIntegrationResult, LikelihoodIntegrationResult, jax.Array]:
    ablated_observation = ablate_modality_observation(
        observation,
        modality=modality,
        prompt=prompt,
        prompt_tokenizer=prompt_tokenizer,
        state_in_prompt=state_in_prompt,
    )
    batched_observation = _stack_observations(observation, ablated_observation)
    batched_actions = jnp.stack(
        [
            jnp.asarray(reference_actions, dtype=jnp.float32),
            jnp.asarray(reference_actions, dtype=jnp.float32),
        ],
        axis=0,
    )
    batched_result = integrate_batched_to_base_log_likelihood(
        model,
        batched_observation,
        batched_actions,
        num_steps=num_steps,
        hutchinson_samples=hutchinson_samples,
        hutchinson_seed=hutchinson_seed,
    )
    original_result = _select_batch_result(batched_result, 0)
    ablated_result = _select_batch_result(batched_result, 1)
    contribution = original_result.log_likelihood - ablated_result.log_likelihood
    return original_result, ablated_result, contribution


def compute_episode_modality_contributions(
    model: _model.BaseModel,
    frames: Sequence[int],
    dataset_indices: Sequence[int],
    observations: Sequence[_model.Observation],
    reference_actions: Sequence[jax.Array],
    prompts: Sequence[str | None],
    *,
    modality: str,
    num_steps: int,
    hutchinson_samples: int = DEFAULT_HUTCHINSON_SAMPLES,
    hutchinson_seed: int = DEFAULT_HUTCHINSON_SEED,
    prompt_tokenizer: _tokenizer.PaligemmaTokenizer | None = None,
    state_in_prompt: bool = False,
    eval_batch_size: int = 4,
) -> list[dict[str, float | int]]:
    """Compute contribution rows in frame chunks to reduce repeated compilation/dispatch."""

    if eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {eval_batch_size}")

    rows: list[dict[str, float | int]] = []
    total_frames = len(frames)
    for start in range(0, total_frames, eval_batch_size):
        stop = min(start + eval_batch_size, total_frames)
        chunk_observations = observations[start:stop]
        chunk_actions = reference_actions[start:stop]
        chunk_prompts = prompts[start:stop]

        batched_observations = []
        batched_actions = []
        for observation, actions, prompt in zip(chunk_observations, chunk_actions, chunk_prompts, strict=True):
            ablated_observation = ablate_modality_observation(
                observation,
                modality=modality,
                prompt=prompt,
                prompt_tokenizer=prompt_tokenizer,
                state_in_prompt=state_in_prompt,
            )
            batched_observations.extend((observation, ablated_observation))
            action = jnp.asarray(actions, dtype=jnp.float32)
            batched_actions.extend((action, action))

        batched_result = integrate_batched_to_base_log_likelihood(
            model,
            _stack_observations(*batched_observations),
            jnp.stack(batched_actions, axis=0),
            num_steps=num_steps,
            hutchinson_samples=hutchinson_samples,
            hutchinson_seed=hutchinson_seed,
        )

        for chunk_offset, (frame, dataset_index) in enumerate(
            zip(frames[start:stop], dataset_indices[start:stop], strict=True)
        ):
            original_index = 2 * chunk_offset
            ablated_index = original_index + 1
            row = {
                "frame": int(frame),
                "dataset_index": int(dataset_index),
                "original_log_likelihood": _scalar(batched_result.log_likelihood[original_index]),
                "ablated_log_likelihood": _scalar(batched_result.log_likelihood[ablated_index]),
                "original_r_tot": _scalar(batched_result.r_tot[original_index]),
                "ablated_r_tot": _scalar(batched_result.r_tot[ablated_index]),
                "delta_logp": _scalar(
                    batched_result.log_p_base[original_index] - batched_result.log_p_base[ablated_index]
                ),
                "delta_r_tot": _scalar(batched_result.r_tot[original_index] - batched_result.r_tot[ablated_index]),
                "contribution": _scalar(
                    batched_result.log_likelihood[original_index] - batched_result.log_likelihood[ablated_index]
                ),
            }
            rows.append(row)
            print(
                f"frame={row['frame']} dataset_index={row['dataset_index']} "
                f"original_log_likelihood={row['original_log_likelihood']:.6f} "
                f"ablated_log_likelihood={row['ablated_log_likelihood']:.6f} "
                f"delta_logp(x_base)={row['delta_logp']:.6f} "
                f"delta_r_tot={row['delta_r_tot']:.6f}"
            )

    return rows


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
    parser.add_argument("--config-name", default="pi05_bi_vitac")
    parser.add_argument("--checkpoint-dir", default="/home/rvsa/codehub/ManiSkill-vitac/checkpoints/11999")
    parser.add_argument("--episode-index", default=10)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--sample-interval", type=int, default=3)
    parser.add_argument("--num-steps", "-k", type=int, default=120)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="Number of episode frames to integrate per batch. Actual model batch is twice this value.",
    )
    parser.add_argument(
        "--hutchinson-samples",
        type=int,
        default=DEFAULT_HUTCHINSON_SAMPLES,
        help="Number of Hutchinson probes per Euler step.",
    )
    parser.add_argument(
        "--hutchinson-seed",
        type=int,
        default=DEFAULT_HUTCHINSON_SEED,
        help="Random seed for Hutchinson probes.",
    )
    parser.add_argument("--remove-modality", choices=("vision", "tactile", "state", "language_prompt"), default="vision")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("eval_outputs/loglike"))
    args = parser.parse_args(argv)

    if args.hutchinson_samples <= 0:
        raise ValueError(f"--hutchinson-samples must be positive, got {args.hutchinson_samples}.")
    if args.eval_batch_size <= 0:
        raise ValueError(f"--eval-batch-size must be positive, got {args.eval_batch_size}.")

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
    print("divergence_method=hutchinson_rademacher_jvp")
    print(f"hutchinson_samples={args.hutchinson_samples}")
    print(f"hutchinson_seed={args.hutchinson_seed}")
    print(f"eval_batch_size={args.eval_batch_size}")
    print("ode_solver=euler")
    print("model_dtype=bfloat16")

    rows = compute_episode_modality_contributions(
        model,
        episode.frames,
        episode.indices,
        episode.observations,
        episode.actions,
        episode.prompts,
        modality=args.remove_modality,
        num_steps=args.num_steps,
        prompt_tokenizer=prompt_tokenizer,
        state_in_prompt=state_in_prompt,
        hutchinson_samples=args.hutchinson_samples,
        hutchinson_seed=args.hutchinson_seed,
        eval_batch_size=args.eval_batch_size,
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
