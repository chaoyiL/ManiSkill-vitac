from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from collections.abc import Sequence

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVAL_SCRIPTS = ROOT / "eval_scripts"
POLICY_SRC = ROOT / "policy" / "src"
for path in (POLICY_SRC, EVAL_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import numpy as np

from loglike_evaluate import (
    DEFAULT_HUTCHINSON_SAMPLES,
    DEFAULT_HUTCHINSON_SEED,
    LikelihoodIntegrationResult,
    _add_batch_dim,
    _eval_utils,
    _scalar,
    create_velocity_context,
    load_model,
    standard_normal_log_prob,
    velocity_and_hutchinson_trace,
)
from openpi.training import config as _config

RowValue = float | int | str


def _select_frames(
    *,
    frame_count: int,
    seed: int,
    num_random_frames: int,
    frames: Sequence[int] | None,
) -> tuple[int, ...]:
    if frames is not None:
        selected = tuple(int(frame) for frame in frames)
        if not selected:
            raise ValueError("--frames cannot be empty.")
        for frame in selected:
            if frame < 0 or frame >= frame_count:
                raise ValueError(f"Frame {frame} is out of range; available frames are 0..{frame_count - 1}.")
        return selected

    if frame_count <= 0:
        raise ValueError("Episode has no available frames.")
    if num_random_frames <= 0:
        raise ValueError(f"--num-random-frames must be positive, got {num_random_frames}.")
    if num_random_frames > frame_count:
        raise ValueError(
            f"Cannot sample {num_random_frames} frames without replacement; only {frame_count} frames are available."
        )

    rng = np.random.default_rng(seed)
    return tuple(sorted(int(frame) for frame in rng.choice(frame_count, size=num_random_frames, replace=False)))


def load_selected_episode_once(
    train_config,
    checkpoint_dir: str | pathlib.Path,
    episode_index,
    *,
    max_frames: int | None,
    seed: int,
    num_random_frames: int,
    frames: Sequence[int] | None,
) -> tuple[_eval_utils.EpisodeData, tuple[int, ...]]:
    """Create the transformed dataset once, select frames, and materialize only those samples."""

    _, raw_dataset, transformed_dataset = _eval_utils.create_transformed_dataset(train_config, checkpoint_dir)
    episode_indices = _eval_utils._indices_for_episode(raw_dataset, episode_index)
    if max_frames is not None:
        episode_indices = episode_indices[:max_frames]

    selected_frames = _select_frames(
        frame_count=len(episode_indices),
        seed=seed,
        num_random_frames=num_random_frames,
        frames=frames,
    )

    selected_dataset_indices = tuple(episode_indices[frame] for frame in selected_frames)
    raw_samples = []
    observations = []
    actions = []
    prompts = []
    for dataset_index in selected_dataset_indices:
        raw = raw_dataset[dataset_index]
        transformed = _eval_utils._copy_tree(transformed_dataset[dataset_index])
        transformed = _eval_utils._normalize_observation_dict(transformed)
        raw_samples.append(raw)
        observations.append(_eval_utils._model.Observation.from_dict(transformed))
        actions.append(jnp.asarray(transformed["actions"]))
        prompts.append(_eval_utils._prompt_from_raw(raw))

    episode = _eval_utils.EpisodeData(
        indices=selected_dataset_indices,
        frames=selected_frames,
        raw_samples=tuple(raw_samples),
        observations=tuple(observations),
        actions=tuple(actions),
        prompts=tuple(prompts),
    )
    return episode, selected_frames


def _parse_k_values(values: Sequence[str]) -> tuple[int, ...]:
    k_values = tuple(int(value) for value in values)
    if not k_values:
        raise ValueError("At least one k value is required.")
    if any(value <= 0 for value in k_values):
        raise ValueError(f"All k values must be positive, got {k_values}.")
    return k_values


def _l2_norm_per_batch(x: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=tuple(range(1, x.ndim))))


def integrate_to_base_with_trace_diagnostics(
    model,
    context,
    x,
    *,
    num_steps: int,
    hutchinson_samples: int,
    hutchinson_seed: int,
) -> tuple[LikelihoodIntegrationResult, dict[str, jax.Array]]:
    """Run the same Euler likelihood integral while retaining per-step diagnostics."""

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    batch_size = x.shape[0]
    dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)
    step_indices = jnp.arange(num_steps, dtype=jnp.int32)
    rng_key = jax.random.PRNGKey(hutchinson_seed)

    def scan_body(carry, step_index):
        x_t, r_tot, t = carry
        step_rng_key = jax.random.fold_in(rng_key, step_index)
        velocity, divergence = velocity_and_hutchinson_trace(
            model,
            context,
            x_t,
            t,
            rng_key=step_rng_key,
            num_samples=hutchinson_samples,
        )
        metrics = {
            "step": step_index,
            "t": t,
            "divergence": divergence,
            "divergence_dt": divergence * dt,
            "x_norm": _l2_norm_per_batch(x_t),
            "velocity_norm": _l2_norm_per_batch(velocity),
        }
        return (x_t + velocity * dt, r_tot + divergence * dt, t + dt), metrics

    @jax.jit
    def run_scan(x, step_indices):
        t0 = jnp.zeros((batch_size,), dtype=jnp.float32)
        r_tot0 = jnp.zeros((batch_size,), dtype=jnp.float32)
        (x_base, r_tot, _), metrics = jax.lax.scan(scan_body, (x, r_tot0, t0), step_indices)
        return x_base, r_tot, metrics

    x_base, r_tot, metrics = run_scan(x, step_indices)
    log_p_base = standard_normal_log_prob(x_base)
    log_likelihood = log_p_base + r_tot
    result = LikelihoodIntegrationResult(
        x_base=x_base,
        r_tot=r_tot,
        log_p_base=log_p_base,
        log_likelihood=log_likelihood,
    )
    return result, metrics


def integrate_to_base_fast(
    model,
    context,
    x,
    *,
    num_steps: int,
    hutchinson_samples: int,
    hutchinson_seed: int,
) -> LikelihoodIntegrationResult:
    """Run the likelihood integral without retaining per-step diagnostics."""

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    batch_size = x.shape[0]
    dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)
    step_indices = jnp.arange(num_steps, dtype=jnp.int32)
    rng_key = jax.random.PRNGKey(hutchinson_seed)

    def scan_body(carry, step_index):
        x_t, r_tot, t = carry
        step_rng_key = jax.random.fold_in(rng_key, step_index)
        velocity, divergence = velocity_and_hutchinson_trace(
            model,
            context,
            x_t,
            t,
            rng_key=step_rng_key,
            num_samples=hutchinson_samples,
        )
        return (x_t + velocity * dt, r_tot + divergence * dt, t + dt), None

    @jax.jit
    def run_scan(x, step_indices):
        t0 = jnp.zeros((batch_size,), dtype=jnp.float32)
        r_tot0 = jnp.zeros((batch_size,), dtype=jnp.float32)
        (x_base, r_tot, _), _ = jax.lax.scan(scan_body, (x, r_tot0, t0), step_indices)
        return x_base, r_tot

    x_base, r_tot = run_scan(x, step_indices)
    log_p_base = standard_normal_log_prob(x_base)
    log_likelihood = log_p_base + r_tot
    return LikelihoodIntegrationResult(
        x_base=x_base,
        r_tot=r_tot,
        log_p_base=log_p_base,
        log_likelihood=log_likelihood,
    )


def summarize_fast_k_result(
    *,
    result: LikelihoodIntegrationResult,
    k: int,
    episode_index,
    frame: int,
    dataset_index: int,
    hutchinson_samples: int,
    hutchinson_seed: int,
) -> dict[str, RowValue]:
    return {
        "episode": episode_index,
        "frame": int(frame),
        "dataset_index": int(dataset_index),
        "k": int(k),
        "dt": 1.0 / k,
        "trace_method": "hutchinson",
        "hutchinson_samples": int(hutchinson_samples),
        "hutchinson_seed": int(hutchinson_seed),
        "log_likelihood": _scalar(result.log_likelihood),
        "r_tot": _scalar(result.r_tot),
        "log_p_base": _scalar(result.log_p_base),
        "x_base_norm": _scalar(_l2_norm_per_batch(result.x_base)),
        "mean_divergence": float("nan"),
        "std_divergence": float("nan"),
        "min_divergence": float("nan"),
        "max_divergence": float("nan"),
        "sum_raw_divergence": float("nan"),
        "sum_divergence_dt": _scalar(result.r_tot),
        "mean_x_norm": float("nan"),
        "max_x_norm": float("nan"),
        "mean_velocity_norm": float("nan"),
        "max_velocity_norm": float("nan"),
    }


def summarize_k_result(
    *,
    result: LikelihoodIntegrationResult,
    metrics: dict[str, jax.Array],
    k: int,
    episode_index,
    frame: int,
    dataset_index: int,
    hutchinson_samples: int,
    hutchinson_seed: int,
) -> dict[str, RowValue]:
    divergence = np.asarray(metrics["divergence"], dtype=np.float64)
    divergence_dt = np.asarray(metrics["divergence_dt"], dtype=np.float64)
    x_norm = np.asarray(metrics["x_norm"], dtype=np.float64)
    velocity_norm = np.asarray(metrics["velocity_norm"], dtype=np.float64)

    return {
        "episode": episode_index,
        "frame": int(frame),
        "dataset_index": int(dataset_index),
        "k": int(k),
        "dt": 1.0 / k,
        "trace_method": "hutchinson",
        "hutchinson_samples": int(hutchinson_samples),
        "hutchinson_seed": int(hutchinson_seed),
        "log_likelihood": _scalar(result.log_likelihood),
        "r_tot": _scalar(result.r_tot),
        "log_p_base": _scalar(result.log_p_base),
        "x_base_norm": _scalar(_l2_norm_per_batch(result.x_base)),
        "mean_divergence": float(np.mean(divergence)),
        "std_divergence": float(np.std(divergence)),
        "min_divergence": float(np.min(divergence)),
        "max_divergence": float(np.max(divergence)),
        "sum_raw_divergence": float(np.sum(divergence)),
        "sum_divergence_dt": float(np.sum(divergence_dt)),
        "mean_x_norm": float(np.mean(x_norm)),
        "max_x_norm": float(np.max(x_norm)),
        "mean_velocity_norm": float(np.mean(velocity_norm)),
        "max_velocity_norm": float(np.max(velocity_norm)),
    }


def run_k_sweep(
    *,
    model,
    observation,
    reference_actions,
    k_values: Sequence[int],
    episode_index,
    frame: int,
    dataset_index: int,
    hutchinson_samples: int,
    hutchinson_seed: int,
    diagnostics: bool,
) -> list[dict[str, RowValue]]:
    print("=== K divergence-integral sweep ===")
    print(f"episode={episode_index} frame={frame} dataset_index={dataset_index}")
    print("divergence_method=hutchinson_rademacher_jvp")
    print(f"hutchinson_samples={hutchinson_samples}")
    print(f"hutchinson_seed={hutchinson_seed}")
    print("ode_solver=euler")
    if diagnostics:
        print(
            "k,dt,log_likelihood,r_tot,log_p_base,x_base_norm,"
            "mean_divergence,std_divergence,min_divergence,max_divergence,"
            "sum_raw_divergence,sum_divergence_dt,mean_velocity_norm,max_velocity_norm"
        )
    else:
        print("k,dt,log_likelihood,r_tot,log_p_base,x_base_norm")

    rows: list[dict[str, RowValue]] = []
    x = jnp.asarray(reference_actions, dtype=jnp.float32)
    if x.ndim == 2:
        x = x[None, ...]
    context = create_velocity_context(model, _add_batch_dim(observation))

    for k in k_values:
        if diagnostics:
            result, metrics = integrate_to_base_with_trace_diagnostics(
                model,
                context,
                x,
                num_steps=k,
                hutchinson_samples=hutchinson_samples,
                hutchinson_seed=hutchinson_seed,
            )
            row = summarize_k_result(
                result=result,
                metrics=metrics,
                k=k,
                episode_index=episode_index,
                frame=frame,
                dataset_index=dataset_index,
                hutchinson_samples=hutchinson_samples,
                hutchinson_seed=hutchinson_seed,
            )
        else:
            result = integrate_to_base_fast(
                model,
                context,
                x,
                num_steps=k,
                hutchinson_samples=hutchinson_samples,
                hutchinson_seed=hutchinson_seed,
            )
            row = summarize_fast_k_result(
                result=result,
                k=k,
                episode_index=episode_index,
                frame=frame,
                dataset_index=dataset_index,
                hutchinson_samples=hutchinson_samples,
                hutchinson_seed=hutchinson_seed,
            )
        rows.append(row)
        if diagnostics:
            print(
                f"{row['k']},{row['dt']:.9f},{row['log_likelihood']:.9f},"
                f"{row['r_tot']:.9f},{row['log_p_base']:.9f},{row['x_base_norm']:.9f},"
                f"{row['mean_divergence']:.9f},{row['std_divergence']:.9f},"
                f"{row['min_divergence']:.9f},{row['max_divergence']:.9f},"
                f"{row['sum_raw_divergence']:.9f},{row['sum_divergence_dt']:.9f},"
                f"{row['mean_velocity_norm']:.9f},{row['max_velocity_norm']:.9f}"
            )
        else:
            print(
                f"{row['k']},{row['dt']:.9f},{row['log_likelihood']:.9f},"
                f"{row['r_tot']:.9f},{row['log_p_base']:.9f},{row['x_base_norm']:.9f}"
            )
    return rows


def save_frame_plots(
    rows: Sequence[dict[str, RowValue]],
    *,
    output_dir: pathlib.Path,
    episode_index,
) -> list[pathlib.Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save k sweep plots.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted({int(row["frame"]) for row in rows})
    plot_paths: list[pathlib.Path] = []

    for frame in frames:
        frame_rows = sorted((row for row in rows if int(row["frame"]) == frame), key=lambda row: int(row["k"]))
        k_values = [int(row["k"]) for row in frame_rows]
        plot_path = output_dir / f"k_sweep_episode_{episode_index}_frame_{frame}.png"

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_values, [float(row["log_likelihood"]) for row in frame_rows], marker="o", label="log_likelihood")
        ax.plot(k_values, [float(row["log_p_base"]) for row in frame_rows], marker="o", label="log_p_base")
        ax.plot(k_values, [float(row["r_tot"]) for row in frame_rows], marker="o", label="divergence_integral")
        ax.set_title(f"Episode {episode_index}, frame {frame}")
        ax.set_xlabel("k value")
        ax.set_ylabel("value")
        ax.set_xticks(k_values)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        plot_paths.append(plot_path)

    return plot_paths


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Euler step count k for random frames and report divergence-integral diagnostics "
            "with Hutchinson trace estimates."
        )
    )
    parser.add_argument("--config-name", default="pi05_bi_vitac")
    parser.add_argument("--checkpoint-dir", default="/home/rvsa/codehub/ManiSkill-vitac/checkpoints/11999")
    parser.add_argument("--episode-index", default=10)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        help="Specific episode-relative frames to test. If omitted, random frames are sampled.",
    )
    parser.add_argument("--num-random-frames", type=int, default=3, help="Number of random frames to sample.")
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for frame sampling. Defaults to --hutchinson-seed so the same seed can drive both.",
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
    parser.add_argument(
        "--k-values",
        nargs="+",
        default=("10", "50", "100"),
        help="Euler step counts to test.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Retain and report per-step divergence diagnostics. Slower; plotting does not require it.",
    )
    parser.add_argument("--output-csv", type=pathlib.Path, help="CSV path for k sweep results.")
    parser.add_argument(
        "--plot-dir",
        type=pathlib.Path,
        default=pathlib.Path("eval_outputs/loglike/k_trace_sweep"),
        help="Directory for per-frame k sweep plots.",
    )
    args = parser.parse_args(argv)

    if args.hutchinson_samples <= 0:
        raise ValueError(f"--hutchinson-samples must be positive, got {args.hutchinson_samples}.")

    k_values = _parse_k_values(args.k_values)
    train_config = _config.get_config(args.config_name)
    seed = args.hutchinson_seed if args.seed is None else args.seed
    episode, frames = load_selected_episode_once(
        train_config,
        args.checkpoint_dir,
        args.episode_index,
        max_frames=args.max_frames,
        seed=seed,
        num_random_frames=args.num_random_frames,
        frames=args.frames,
    )
    model = load_model(train_config, args.checkpoint_dir)

    print(f"frame_sampling_seed={seed}")
    print(f"selected_frames={frames}")

    rows: list[dict[str, RowValue]] = []
    for frame, dataset_index, observation, reference_actions in zip(
        episode.frames,
        episode.indices,
        episode.observations,
        episode.actions,
        strict=True,
    ):
        rows.extend(
            run_k_sweep(
                model=model,
                observation=observation,
                reference_actions=reference_actions,
                k_values=k_values,
                episode_index=args.episode_index,
                frame=frame,
                dataset_index=dataset_index,
                hutchinson_samples=args.hutchinson_samples,
                hutchinson_seed=args.hutchinson_seed,
                diagnostics=args.diagnostics,
            )
        )

    plot_paths = save_frame_plots(
        rows,
        output_dir=args.plot_dir,
        episode_index=args.episode_index,
    )
    for plot_path in plot_paths:
        print(f"plot={plot_path}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
