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
    LikelihoodIntegrationResult,
    _add_batch_dim,
    _scalar,
    create_velocity_context,
    load_episode,
    load_model,
    predict_velocity_with_context,
    standard_normal_log_prob,
    velocity_and_fd_coordinate_trace,
)
from openpi.shared import nnx_utils
from openpi.training import config as _config


def _parse_eps_values(values: Sequence[str]) -> tuple[float, ...]:
    eps_values = tuple(float(value) for value in values)
    if not eps_values:
        raise ValueError("At least one fd eps value is required.")
    if any(value <= 0 for value in eps_values):
        raise ValueError(f"All fd eps values must be positive, got {eps_values}.")
    return eps_values


def _pick_best_eps(rows: Sequence[dict[str, float]], *, error_key: str) -> dict[str, float]:
    return min(rows, key=lambda row: row[error_key])


def exact_jacobian_trace_from_velocity(velocity_fn, x: jax.Array) -> jax.Array:
    """Autodiff baseline for validating finite-difference eps choices."""

    x = jnp.asarray(x, dtype=jnp.float32)
    batch_size = x.shape[0]
    x_shape = x.shape
    event_size = int(np.prod(x_shape[1:]))
    flat_shape = (batch_size, event_size)
    flat_x = x.reshape(flat_shape)

    def flat_velocity(x_flat: jax.Array) -> jax.Array:
        return velocity_fn(x_flat.reshape(x_shape)).reshape(flat_shape)

    def scan_body(trace: jax.Array, index: jax.Array) -> tuple[jax.Array, None]:
        direction = jax.nn.one_hot(index, event_size, dtype=jnp.float32)
        tangent = jnp.broadcast_to(direction[None, :], flat_shape)
        _, tangent_out = jax.jvp(flat_velocity, (flat_x,), (tangent,))
        return trace + tangent_out[:, index], None

    trace0 = jnp.zeros((batch_size,), dtype=jnp.float32)
    trace, _ = jax.lax.scan(scan_body, trace0, jnp.arange(event_size, dtype=jnp.int32))
    return trace


def velocity_and_autodiff_divergence(model, context, x: jax.Array, t: jax.Array) -> tuple[jax.Array, jax.Array]:
    x = jnp.asarray(x, dtype=jnp.float32)

    def velocity_fn(x_arg: jax.Array) -> jax.Array:
        return predict_velocity_with_context(model, context, x_arg, t).astype(jnp.float32)

    velocity = velocity_fn(x)
    divergence = exact_jacobian_trace_from_velocity(velocity_fn, x)
    return velocity, divergence


def integrate_to_base_log_likelihood_autodiff(
    loglike_fn,
    observation,
    reference_actions,
    *,
    num_steps: int,
) -> LikelihoodIntegrationResult:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    x = jnp.asarray(reference_actions, dtype=jnp.float32)
    if x.ndim == 2:
        x = x[None, ...]
    observation = _add_batch_dim(observation)
    step_indices = jnp.arange(num_steps, dtype=jnp.float32)
    x_base, r_tot, log_p_base, log_likelihood = loglike_fn(observation, x, step_indices)
    return LikelihoodIntegrationResult(
        x_base=x_base,
        r_tot=r_tot,
        log_p_base=log_p_base,
        log_likelihood=log_likelihood,
    )


def integrate_to_base_log_likelihood_fd(
    model,
    observation,
    reference_actions,
    *,
    num_steps: int,
    fd_eps: float,
) -> LikelihoodIntegrationResult:
    """Integrate to base using coordinate finite-difference divergence."""

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    x = jnp.asarray(reference_actions, dtype=jnp.float32)
    if x.ndim == 2:
        x = x[None, ...]
    observation = _add_batch_dim(observation)

    batch_size = x.shape[0]
    dt = jnp.asarray(1.0 / num_steps, dtype=jnp.float32)
    context = create_velocity_context(model, observation)
    step_indices = jnp.arange(num_steps, dtype=jnp.float32)

    def scan_body(carry, _):
        x_t, r_tot, t = carry
        velocity, divergence = velocity_and_fd_coordinate_trace(model, context, x_t, t, fd_eps)
        return (x_t + velocity * dt, r_tot + divergence * dt, t + dt), None

    t0 = jnp.zeros((batch_size,), dtype=jnp.float32)
    r_tot0 = jnp.zeros((batch_size,), dtype=jnp.float32)
    (x_base, r_tot, _), _ = jax.lax.scan(scan_body, (x, r_tot0, t0), step_indices)

    log_p_base = standard_normal_log_prob(x_base)
    log_likelihood = log_p_base + r_tot
    return LikelihoodIntegrationResult(
        x_base=x_base,
        r_tot=r_tot,
        log_p_base=log_p_base,
        log_likelihood=log_likelihood,
    )


def run_pointwise_sweep(
    *,
    model,
    context,
    x,
    t,
    eps_values: Sequence[float],
    episode_index,
    frame: int,
    dataset_index: int,
    time: float,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    exact_velocity, exact_divergence = velocity_and_autodiff_divergence(model, context, x, t)
    exact_divergence_scalar = _scalar(exact_divergence)

    print("=== Pointwise divergence sweep ===")
    print(f"episode={episode_index} frame={frame} dataset_index={dataset_index}")
    print(f"time={time:.6f}")
    print("autodiff_method=exact_jacobian_trace")
    print(f"autodiff_divergence={exact_divergence_scalar:.9f}")
    print("fd_eps,fd_coordinate_trace,abs_error,relative_error,max_velocity_delta")

    rows: list[dict[str, float]] = []
    for fd_eps in eps_values:
        fd_velocity, fd_divergence = velocity_and_fd_coordinate_trace(model, context, x, t, fd_eps)
        fd_divergence_scalar = _scalar(fd_divergence)
        abs_error = abs(fd_divergence_scalar - exact_divergence_scalar)
        relative_error = abs_error / abs(exact_divergence_scalar) if exact_divergence_scalar != 0 else float("nan")
        max_velocity_delta = _scalar(jnp.max(jnp.abs(fd_velocity - exact_velocity)))
        row = {
            "fd_eps": fd_eps,
            "fd_coordinate_trace": fd_divergence_scalar,
            "autodiff_divergence": exact_divergence_scalar,
            "abs_error": abs_error,
            "relative_error": relative_error,
            "max_velocity_delta": max_velocity_delta,
        }
        rows.append(row)
        print(
            f"{fd_eps:.9g},{fd_divergence_scalar:.9f},{abs_error:.9f},"
            f"{relative_error:.9f},{max_velocity_delta:.9f}"
        )

    best = _pick_best_eps(rows, error_key="abs_error")
    print(
        "best_eps_pointwise="
        f"{best['fd_eps']:.9g} abs_error={best['abs_error']:.9f} "
        f"relative_error={best['relative_error']:.9f}"
    )
    return rows, best


def run_integration_sweep(
    *,
    model,
    observation,
    reference_actions,
    loglike_fn,
    eps_values: Sequence[float],
    num_steps: int,
    episode_index,
    frame: int,
    dataset_index: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    ad_result = integrate_to_base_log_likelihood_autodiff(
        loglike_fn,
        observation,
        reference_actions,
        num_steps=num_steps,
    )
    ad_log_likelihood = _scalar(ad_result.log_likelihood)
    ad_r_tot = _scalar(ad_result.r_tot)
    ad_log_p_base = _scalar(ad_result.log_p_base)

    print()
    print("=== Full likelihood integration sweep ===")
    print(f"episode={episode_index} frame={frame} dataset_index={dataset_index}")
    print(f"num_steps={num_steps}")
    print("autodiff_method=exact_jacobian_trace")
    print(f"autodiff_log_likelihood={ad_log_likelihood:.9f}")
    print(f"autodiff_r_tot={ad_r_tot:.9f}")
    print(f"autodiff_log_p_base={ad_log_p_base:.9f}")
    print(
        "fd_eps,fd_log_likelihood,fd_r_tot,log_likelihood_abs_error,"
        "r_tot_abs_error,log_likelihood_rel_error"
    )

    rows: list[dict[str, float]] = []
    for fd_eps in eps_values:
        fd_result = integrate_to_base_log_likelihood_fd(
            model,
            observation,
            reference_actions,
            num_steps=num_steps,
            fd_eps=fd_eps,
        )
        fd_log_likelihood = _scalar(fd_result.log_likelihood)
        fd_r_tot = _scalar(fd_result.r_tot)
        log_likelihood_abs_error = abs(fd_log_likelihood - ad_log_likelihood)
        r_tot_abs_error = abs(fd_r_tot - ad_r_tot)
        log_likelihood_rel_error = (
            log_likelihood_abs_error / abs(ad_log_likelihood) if ad_log_likelihood != 0 else float("nan")
        )
        row = {
            "fd_eps": fd_eps,
            "fd_log_likelihood": fd_log_likelihood,
            "fd_r_tot": fd_r_tot,
            "autodiff_log_likelihood": ad_log_likelihood,
            "autodiff_r_tot": ad_r_tot,
            "log_likelihood_abs_error": log_likelihood_abs_error,
            "r_tot_abs_error": r_tot_abs_error,
            "log_likelihood_rel_error": log_likelihood_rel_error,
        }
        rows.append(row)
        print(
            f"{fd_eps:.9g},{fd_log_likelihood:.9f},{fd_r_tot:.9f},"
            f"{log_likelihood_abs_error:.9f},{r_tot_abs_error:.9f},{log_likelihood_rel_error:.9f}"
        )

    best = _pick_best_eps(rows, error_key="log_likelihood_abs_error")
    print(
        "best_eps_integration="
        f"{best['fd_eps']:.9g} log_likelihood_abs_error={best['log_likelihood_abs_error']:.9f} "
        f"r_tot_abs_error={best['r_tot_abs_error']:.9f} "
        f"log_likelihood_rel_error={best['log_likelihood_rel_error']:.9f}"
    )
    return rows, best


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare autodiff Jacobian traces against coordinate finite-difference traces, "
            "then run the same likelihood integration task with autodiff and find the fd_eps "
            "closest to the autodiff result."
        )
    )
    parser.add_argument("--config-name", default="pi05_bi_vitac")
    parser.add_argument("--checkpoint-dir", default="/home/rvsa/codehub/ManiSkill-vitac/checkpoints/11999")
    parser.add_argument("--episode-index", default=10)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--num-steps", "-k", type=int, default=10, help="Euler steps for likelihood integration.")
    parser.add_argument("--time", type=float, default=0.0, help="Flow time at which to compare pointwise divergence.")
    parser.add_argument(
        "--eps-values",
        nargs="+",
        default=("1e-1", "3e-2", "1e-2", "3e-3", "1e-3", "3e-4", "1e-4"),
        help="Finite-difference eps values to test.",
    )
    parser.add_argument("--output-csv", type=pathlib.Path, help="CSV path for pointwise sweep results.")
    parser.add_argument(
        "--integration-output-csv",
        type=pathlib.Path,
        help="CSV path for full likelihood integration sweep results.",
    )
    args = parser.parse_args(argv)

    eps_values = _parse_eps_values(args.eps_values)
    train_config = _config.get_config(args.config_name)
    episode = load_episode(
        train_config,
        args.checkpoint_dir,
        args.episode_index,
        max_frames=args.max_frames,
        frame_indices=(args.frame,),
    )
    model = load_model(train_config, args.checkpoint_dir)
    if not hasattr(model, "integrate_to_base_log_likelihood"):
        raise TypeError("Expected a Pi0/Pi05 model with integrate_to_base_log_likelihood.")
    loglike_fn = nnx_utils.module_jit(model.integrate_to_base_log_likelihood, static_argnums=(4,))

    observation = episode.observations[0]
    reference_actions = episode.actions[0]
    x = jnp.asarray(reference_actions, dtype=jnp.float32)[None, ...]
    t = jnp.full((x.shape[0],), args.time, dtype=jnp.float32)
    context = create_velocity_context(model, _add_batch_dim(observation))

    pointwise_rows, _ = run_pointwise_sweep(
        model=model,
        context=context,
        x=x,
        t=t,
        eps_values=eps_values,
        episode_index=args.episode_index,
        frame=args.frame,
        dataset_index=episode.indices[0],
        time=args.time,
    )
    integration_rows, best_integration = run_integration_sweep(
        model=model,
        observation=observation,
        reference_actions=reference_actions,
        loglike_fn=loglike_fn,
        eps_values=eps_values,
        num_steps=args.num_steps,
        episode_index=args.episode_index,
        frame=args.frame,
        dataset_index=episode.indices[0],
    )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "fd_eps",
                    "fd_coordinate_trace",
                    "autodiff_divergence",
                    "abs_error",
                    "relative_error",
                    "max_velocity_delta",
                ],
            )
            writer.writeheader()
            writer.writerows(pointwise_rows)
        print(f"pointwise_output_csv={args.output_csv}")

    if args.integration_output_csv is not None:
        args.integration_output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.integration_output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "fd_eps",
                    "fd_log_likelihood",
                    "fd_r_tot",
                    "autodiff_log_likelihood",
                    "autodiff_r_tot",
                    "log_likelihood_abs_error",
                    "r_tot_abs_error",
                    "log_likelihood_rel_error",
                ],
            )
            writer.writeheader()
            writer.writerows(integration_rows)
        print(f"integration_output_csv={args.integration_output_csv}")

    print()
    print(
        "summary: closest fd_eps to autodiff integration = "
        f"{best_integration['fd_eps']:.9g} "
        f"(log_likelihood_abs_error={best_integration['log_likelihood_abs_error']:.9f})"
    )


if __name__ == "__main__":
    main()
