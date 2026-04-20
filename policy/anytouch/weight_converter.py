"""Convert PyTorch AnyTouch (TactileVideoMAE) checkpoint to JAX/Flax params.

Usage
-----
>>> from policy.anytouch.weight_converter import load_params_from_pytorch
>>> from policy.anytouch.model.tactile_mae_jax import TactileMAEConfig
>>> config = TactileMAEConfig.from_clip_config("policy/anytouch/CLIP-B-16")
>>> params = load_params_from_pytorch("checkpoints/anytouch/checkpoint-2frames.pth", config)
>>> # params is a nested dict ready for module.apply({"params": params}, ...)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[2]
_POLICY_ROOT = _FILE.parents[1]
print(_POLICY_ROOT)
for _path in (str(_REPO_ROOT), str(_POLICY_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from anytouch.model.tactile_mae_jax import Module, TactileMAEConfig

# except ModuleNotFoundError:
#     from policy.anytouch.model.tactile_mae_jax import Module, TactileMAEConfig

logger = logging.getLogger("anytouch")

# Possible key prefixes in AnyTouch checkpoints (tried in order).
_KNOWN_PREFIXES = [
    "touch_mae_model.",  # Full AnyTouch training checkpoint
    "",                  # Standalone TactileVideoMAE state dict
]


@dataclass(frozen=True)
class ConversionCheckResult:
    """Summary of a PyTorch vs JAX forward consistency check."""

    max_abs_diff: float
    mean_abs_diff: float
    rel_abs_diff: float
    l2_diff: float
    torch_output_shape: tuple[int, ...]
    jax_output_shape: tuple[int, ...]


def _load_raw_state_dict(checkpoint_path: str | Path) -> dict:
    """Load a raw PyTorch checkpoint and unwrap common wrappers."""
    import torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        return raw["model"]
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return raw["state_dict"]
    return raw


def _to_numpy_state_dict(state_dict: dict) -> dict[str, np.ndarray]:
    """Convert a PyTorch state dict to numpy arrays."""
    np_state_dict = {}
    for k, v in state_dict.items():
        if hasattr(v, "detach"):
            np_state_dict[k] = v.detach().float().cpu().numpy()
        elif hasattr(v, "numpy"):
            np_state_dict[k] = v.float().numpy()
        else:
            np_state_dict[k] = np.asarray(v, dtype=np.float32)
    return np_state_dict


def _strip_prefix_from_state_dict(state_dict: dict, prefix: str) -> dict:
    """Strip checkpoint prefix so the dict matches the PyTorch module keys."""
    if not prefix:
        return state_dict
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def _detect_prefix(state_dict: dict[str, np.ndarray]) -> str:
    """Auto-detect the key prefix used in the checkpoint."""
    marker = "touch_model.embeddings.class_embedding"
    for prefix in _KNOWN_PREFIXES:
        if prefix + marker in state_dict:
            return prefix
    raise KeyError(
        f"Cannot detect key prefix. Looked for '*{marker}' with prefixes "
        f"{_KNOWN_PREFIXES}. Available keys (first 10): {list(state_dict)[:10]}"
    )


def convert_pytorch_state_dict(
    state_dict: dict[str, np.ndarray],
    config: TactileMAEConfig,
) -> dict:
    """Convert a PyTorch ``TactileVideoMAE`` state dict to a Flax params dict.

    Parameters
    ----------
    state_dict : dict[str, np.ndarray]
        PyTorch state dict with values already converted to numpy arrays.
    config : TactileMAEConfig
        Model configuration (used for layer count validation).

    Returns
    -------
    dict
        Nested Flax params dict (not wrapped in ``{"params": ...}``).
    """
    pfx = _detect_prefix(state_dict)
    logger.info("Detected checkpoint key prefix: %r", pfx)

    params: dict = {}

    def _get(key: str) -> np.ndarray:
        return np.asarray(state_dict[pfx + key])

    # --- Patch embedding (Conv3d) ---
    # PyTorch: (out_ch, in_ch, kD, kH, kW) → Flax: (kD, kH, kW, in_ch, out_ch)
    w = _get("touch_model.embeddings.patch_embedding.weight")
    ckpt_stride = w.shape[2]  # temporal kernel size == stride
    if ckpt_stride != config.stride:
        logger.warning(
            "Checkpoint Conv3d temporal kernel=%d but config.stride=%d. "
            "Make sure the config matches the checkpoint.",
            ckpt_stride,
            config.stride,
        )
    params["patch_embedding"] = {"kernel": np.transpose(w, (2, 3, 4, 1, 0))}

    # --- Class embedding (768,) ---
    params["class_embedding"] = _get("touch_model.embeddings.class_embedding")

    # --- Position embedding ---
    # PyTorch nn.Embedding .weight: (N+1, D)  →  Flax param: same shape
    params["position_embedding"] = _get("touch_model.embeddings.position_embedding.weight")

    # --- Sensor token (20, 5, 768) ---
    params["sensor_token"] = _get("sensor_token")

    # --- Pre-LayerNorm (note HuggingFace typo: "pre_layrnorm") ---
    params["pre_layernorm"] = {
        "scale": _get("touch_model.pre_layrnorm.weight"),
        "bias": _get("touch_model.pre_layrnorm.bias"),
    }

    # --- Post-LayerNorm ---
    params["post_layernorm"] = {
        "scale": _get("touch_model.post_layernorm.weight"),
        "bias": _get("touch_model.post_layernorm.bias"),
    }

    # --- Transformer encoder layers ---
    encoder: dict = {}
    for i in range(config.num_hidden_layers):
        p = f"touch_model.encoder.layers.{i}"
        layer: dict = {}

        # Layer norms
        layer["layer_norm1"] = {
            "scale": _get(f"{p}.layer_norm1.weight"),
            "bias": _get(f"{p}.layer_norm1.bias"),
        }
        layer["layer_norm2"] = {
            "scale": _get(f"{p}.layer_norm2.weight"),
            "bias": _get(f"{p}.layer_norm2.bias"),
        }

        # Self-attention: weights go into "base/" sub-dict (LoRADense layout).
        # PyTorch: (out, in) → Flax: (in, out)
        attn: dict = {}
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            attn[proj] = {
                "base": {
                    "kernel": _get(f"{p}.self_attn.{proj}.weight").T,
                    "bias": _get(f"{p}.self_attn.{proj}.bias"),
                }
            }
        layer["self_attn"] = attn

        # MLP: same LoRADense layout
        layer["mlp"] = {
            "fc1": {"base": {
                "kernel": _get(f"{p}.mlp.fc1.weight").T,
                "bias": _get(f"{p}.mlp.fc1.bias"),
            }},
            "fc2": {"base": {
                "kernel": _get(f"{p}.mlp.fc2.weight").T,
                "bias": _get(f"{p}.mlp.fc2.bias"),
            }},
        }

        encoder[f"layers_{i}"] = layer

    params["encoder"] = encoder

    # --- Touch projection (Dense, no bias) ---
    # PyTorch: (projection_dim, hidden_size) → Flax: (hidden_size, projection_dim)
    params["touch_projection"] = {
        "kernel": _get("touch_projection.weight").T,
    }

    return params


def load_params_from_pytorch(
    checkpoint_path: str | Path,
    config: TactileMAEConfig,
) -> dict:
    """Load a PyTorch AnyTouch checkpoint and convert to Flax params.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to ``.pth`` checkpoint file.
    config : TactileMAEConfig
        Model configuration.

    Returns
    -------
    dict
        Nested Flax params dict (use as ``module.apply({"params": params}, ...)``).
    """
    state_dict = _load_raw_state_dict(checkpoint_path)
    np_state_dict = _to_numpy_state_dict(state_dict)

    params = convert_pytorch_state_dict(np_state_dict, config)
    logger.info("Converted PyTorch checkpoint → JAX params from %s", checkpoint_path)
    return params


def infer_config_from_checkpoint(
    checkpoint_path: str | Path,
    clip_config_path: str | Path = "policy/anytouch/CLIP-B-16",
) -> TactileMAEConfig:
    """Infer ``TactileMAEConfig`` by inspecting a PyTorch checkpoint.

    Auto-detects ``stride`` from the Conv3d kernel shape and ``num_frames``
    from the position embedding size.
    """
    sd = _load_raw_state_dict(checkpoint_path)

    pfx = _detect_prefix({k: None for k in sd})  # type: ignore[arg-type]

    # Conv3d weight shape: (out, in, kD, kH, kW)
    conv_key = pfx + "touch_model.embeddings.patch_embedding.weight"
    conv_shape = sd[conv_key].shape
    stride = conv_shape[2]  # temporal kernel == stride
    patch_size = conv_shape[3]

    # Position embedding: (num_patches + 1, D)
    pos_key = pfx + "touch_model.embeddings.position_embedding.weight"
    num_pos = sd[pos_key].shape[0]  # num_patches + 1
    num_patches = num_pos - 1

    # Sensor token: (num_sensor_types, num_sensor_tokens, D)
    sensor_key = pfx + "sensor_token"
    sensor_shape = sd[sensor_key].shape
    num_sensor_types = sensor_shape[0]
    num_sensor_tokens = sensor_shape[1]

    # Infer image_size from num_patches and stride
    # num_patches = grid_size^2 * (num_frames // stride)
    # With a common CLIP config, read image_size from there
    base_config = TactileMAEConfig.from_clip_config(clip_config_path)
    grid_size = base_config.image_size // patch_size
    spatial_patches = grid_size ** 2
    temporal_patches = num_patches // spatial_patches
    num_frames = temporal_patches * stride

    config = TactileMAEConfig.from_clip_config(
        clip_config_path,
        num_frames=num_frames,
        stride=stride,
        num_sensor_types=num_sensor_types,
        num_sensor_tokens=num_sensor_tokens,
    )
    logger.info(
        "Inferred config from checkpoint: num_frames=%d, stride=%d, "
        "num_patches=%d, num_output_tokens=%d",
        num_frames,
        stride,
        config.num_patches,
        config.num_output_tokens,
    )
    return config


def _build_pytorch_anytouch_model(
    checkpoint_path: str | Path,
    config: TactileMAEConfig,
    clip_config_path: str | Path = "policy/anytouch/CLIP-B-16",
):
    """Instantiate the original PyTorch AnyTouch model and load checkpoint weights."""
    import torch
    from transformers import AutoConfig

    anytouch_root = Path(__file__).resolve().parent
    if str(anytouch_root) not in sys.path:
        sys.path.insert(0, str(anytouch_root))

    from model.tactile_mae import TactileVideoMAE

    class _Args:
        mask_ratio = 0.0
        stride = config.stride

    state_dict = _load_raw_state_dict(checkpoint_path)
    prefix = _detect_prefix({k: None for k in state_dict})  # type: ignore[arg-type]
    stripped_state_dict = _strip_prefix_from_state_dict(state_dict, prefix)

    torch_config = AutoConfig.from_pretrained(str(clip_config_path))
    model = TactileVideoMAE(
        _Args(),
        torch_config,
        num_frames=config.num_frames,
        tube_size=config.stride,
    )
    missing_keys, unexpected_keys = model.load_state_dict(stripped_state_dict, strict=False)
    if missing_keys:
        logger.warning("PyTorch verification model missing keys: %s", missing_keys)
    if unexpected_keys:
        logger.warning("PyTorch verification model unexpected keys: %s", unexpected_keys)
    model.eval()
    return model


def compare_jax_and_pytorch_outputs(
    checkpoint_path: str | Path,
    *,
    config: TactileMAEConfig | None = None,
    clip_config_path: str | Path = "policy/anytouch/CLIP-B-16",
    batch_size: int = 2,
    seed: int = 0,
) -> ConversionCheckResult:
    """Run a random-input forward pass in PyTorch and JAX and compare outputs."""
    import torch

    if config is None:
        config = infer_config_from_checkpoint(checkpoint_path, clip_config_path)

    torch_model = _build_pytorch_anytouch_model(
        checkpoint_path=checkpoint_path,
        config=config,
        clip_config_path=clip_config_path,
    )
    jax_params = load_params_from_pytorch(checkpoint_path, config)
    jax_module = Module(config)

    rng = np.random.default_rng(seed)
    pixel_values = rng.standard_normal(
        (
            batch_size,
            config.num_channels,
            config.num_frames,
            config.image_size,
            config.image_size,
        ),
        dtype=np.float32,
    )
    sensor_type = rng.integers(
        low=0,
        high=config.num_sensor_types,
        size=(batch_size,),
        dtype=np.int32,
    )

    with torch.no_grad():
        torch_output = torch_model(
            torch.from_numpy(pixel_values),
            sensor_type=torch.from_numpy(sensor_type).long(),
        )
        torch_output = torch_output.detach().cpu().numpy()

    jax_output, _ = jax_module.apply(
        {"params": jax_params},
        jnp.asarray(pixel_values),
        jnp.asarray(sensor_type),
    )
    jax_output = np.asarray(jax.device_get(jax_output))

    abs_diff = np.abs(torch_output - jax_output)

    result = ConversionCheckResult(
        max_abs_diff=float(abs_diff.max()),
        mean_abs_diff=float(abs_diff.mean()),
        rel_abs_diff=float(abs_diff.mean()/jax_output.mean()),
        l2_diff=float(np.linalg.norm(torch_output - jax_output)),
        torch_output_shape=tuple(torch_output.shape),
        jax_output_shape=tuple(jax_output.shape),
    )
    logger.info(
        "JAX/PyTorch forward diff: max=%.6e mean=%.6e l2=%.6e",
        result.max_abs_diff,
        result.mean_abs_diff,
        result.l2_diff,
    )
    return result


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AnyTouch PyTorch → JAX weight conversion utilities.")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the PyTorch checkpoint.")
    parser.add_argument(
        "--clip-config-path",
        type=str,
        default="policy/anytouch/CLIP-B-16",
        help="Path to the CLIP config directory or config.json.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Random test batch size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used to generate the test input.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a PyTorch/JAX random-input forward consistency check.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _build_argparser().parse_args()

    if args.check:
        result = compare_jax_and_pytorch_outputs(
            checkpoint_path=args.checkpoint_path,
            clip_config_path=args.clip_config_path,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print()
        print(
            "PyTorch/JAX forward check:",
            f"torch_shape={result.torch_output_shape}",
            f"jax_shape={result.jax_output_shape}",
            f"max_abs_diff={result.max_abs_diff:.6e}",
            f"mean_abs_diff={result.mean_abs_diff:.6e}",
            f"rel_diff={result.rel_abs_diff:.6e}"
        )
