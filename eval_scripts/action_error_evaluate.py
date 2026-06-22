from __future__ import annotations

import argparse
import csv
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
from openpi.shared import nnx_utils
from openpi.training import config as _config

from utils import (
    EpisodeData,
    _scalar,
    _batch_observation,
    _batch_actions,
    ablate_modality_observation,
    load_episode,
    load_model,
)


@dataclasses.dataclass(frozen=True)
class ActionErrorResult:
    actions: jax.Array
    mse: jax.Array
    rmse: jax.Array
    mae: jax.Array


def _prediction_error(predicted_actions: jax.Array, reference_actions: jax.Array) -> ActionErrorResult:
    predicted_actions = jnp.asarray(predicted_actions, dtype=jnp.float32)
    reference_actions = jnp.asarray(reference_actions, dtype=jnp.float32)
    diff = predicted_actions - reference_actions
    mse = jnp.mean(jnp.square(diff), axis=tuple(range(1, diff.ndim)))
    mae = jnp.mean(jnp.abs(diff), axis=tuple(range(1, diff.ndim)))
    return ActionErrorResult(
        actions=predicted_actions,
        mse=mse,
        rmse=jnp.sqrt(mse),
        mae=mae,
    )


def evaluate_modality_error_change(
    model: _model.BaseModel,
    sample_actions_fn: Any,
    observation: _model.Observation,
    reference_actions: jax.Array,
    *,
    modality: str,
    num_steps: int,
    rng: jax.Array,
    prompt: str | None = None,
    prompt_tokenizer: _tokenizer.PaligemmaTokenizer | None = None,
    state_in_prompt: bool = False,
) -> tuple[ActionErrorResult, ActionErrorResult, jax.Array]:
    """Compare action prediction error before/after attention-mask ablation.

    The original and ablated calls use the same initial diffusion noise so the
    error change isolates the modality mask as much as possible.
    """

    original_observation = _batch_observation(observation)
    ablated_observation = _batch_observation(
        ablate_modality_observation(
            observation,
            modality=modality,
            prompt=prompt,
            prompt_tokenizer=prompt_tokenizer,
            state_in_prompt=state_in_prompt,
        )
    )
    reference_actions = _batch_actions(reference_actions).astype(jnp.float32)
    noise = jax.random.normal(rng, reference_actions.shape, dtype=jnp.float32)

    original_actions = sample_actions_fn(
        rng,
        original_observation,
        num_steps=num_steps,
        noise=noise,
    )
    ablated_actions = sample_actions_fn(
        rng,
        ablated_observation,
        num_steps=num_steps,
        noise=noise,
    )

    original_error = _prediction_error(original_actions, reference_actions)
    ablated_error = _prediction_error(ablated_actions, reference_actions)
    delta_mse = ablated_error.mse - original_error.mse
    return original_error, ablated_error, delta_mse


def save_error_curve(
    rows: Sequence[dict[str, float | int]],
    *,
    output_dir: pathlib.Path,
    modality: str,
    episode_index: str,
) -> tuple[pathlib.Path, pathlib.Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{modality}_action_error_episode_{episode_index}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "dataset_index",
                "original_mse",
                "ablated_mse",
                "delta_mse",
                "relative_delta_mse",
                "original_rmse",
                "ablated_rmse",
                "original_mae",
                "ablated_mae",
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

    plot_path = output_dir / f"{modality}_action_error_episode_{episode_index}.png"
    frames = [row["frame"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(frames, [row["original_mse"] for row in rows], marker="o", linewidth=1.5, label="original")
    axes[0].plot(frames, [row["ablated_mse"] for row in rows], marker="o", linewidth=1.5, label="ablated")
    axes[0].set_ylabel("action MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames, [row["delta_mse"] for row in rows], marker="o", linewidth=1.5)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[1].set_xlabel("Episode frame")
    axes[1].set_ylabel("delta MSE")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{modality} action prediction error change over episode {episode_index}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return csv_path, plot_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate modality contribution via action prediction error change.")
    parser.add_argument("--config-name", default="pi05_bi_vitac")
    parser.add_argument("--checkpoint-dir", default="/home/rvsa/codehub/ManiSkill-vitac/checkpoints/11999")
    parser.add_argument("--episode-index", default=10)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--sample-interval", type=int, default=3)
    parser.add_argument("--num-steps", "-k", type=int, default=10)
    parser.add_argument("--remove-modality", choices=("vision", "tactile", "state", "language_prompt"), default="vision")
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("eval_outputs/action_error"))
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
    sample_actions_fn = nnx_utils.module_jit(model.sample_actions)
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
    print("metric=action_prediction_error_change")
    print(f"num_steps={args.num_steps}")
    print(f"seed={args.seed}")

    rows = []
    base_rng = jax.random.key(args.seed)
    for i, (frame, dataset_index, observation, reference_actions, prompt) in enumerate(
        zip(episode.frames, episode.indices, episode.observations, episode.actions, episode.prompts, strict=True)
    ):
        frame_rng = jax.random.fold_in(base_rng, int(dataset_index))
        original_error, ablated_error, delta_mse = evaluate_modality_error_change(
            model,
            sample_actions_fn,
            observation,
            reference_actions,
            modality=args.remove_modality,
            num_steps=args.num_steps,
            rng=frame_rng,
            prompt=prompt,
            prompt_tokenizer=prompt_tokenizer,
            state_in_prompt=state_in_prompt,
        )

        original_mse = _scalar(original_error.mse)
        ablated_mse = _scalar(ablated_error.mse)
        delta_mse_scalar = _scalar(delta_mse)
        relative_delta_mse = delta_mse_scalar / original_mse if original_mse != 0 else float("nan")
        row = {
            "frame": int(frame),
            "dataset_index": int(dataset_index),
            "original_mse": original_mse,
            "ablated_mse": ablated_mse,
            "delta_mse": delta_mse_scalar,
            "relative_delta_mse": relative_delta_mse,
            "original_rmse": _scalar(original_error.rmse),
            "ablated_rmse": _scalar(ablated_error.rmse),
            "original_mae": _scalar(original_error.mae),
            "ablated_mae": _scalar(ablated_error.mae),
        }
        rows.append(row)
        print(
            f"frame={row['frame']} dataset_index={row['dataset_index']} "
            f"original_mse={row['original_mse']:.6f} "
            f"ablated_mse={row['ablated_mse']:.6f} "
            f"delta_mse={row['delta_mse']:.6f} "
            f"relative_delta_mse={row['relative_delta_mse']:.6f}"
        )

    if args.sample_interval is not None:
        csv_path, plot_path = save_error_curve(
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
