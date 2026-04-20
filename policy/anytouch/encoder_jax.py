"""High-level JAX AnyTouch encoder interface.

This module provides a clean external API for the AnyTouch tactile encoder
implemented in pure JAX/Flax.  It replaces ``anytouch_bridge.py`` (which used
``jax.pure_callback`` to call a frozen PyTorch model) with a native JAX
implementation that supports full gradient flow and JIT compilation.

Quick start
-----------
>>> from policy.anytouch.encoder_jax import AnyTouchEncoderJax
>>> encoder = AnyTouchEncoderJax.from_pretrained(
...     config_path="policy/anytouch/CLIP-B-16",
...     checkpoint_path="checkpoints/anytouch/checkpoint-2frames.pth",
... )
>>> tokens = encoder.encode(pixel_values, sensor_type)  # (B, N, 512)

Integration with Pi0 (Flax NNX)
--------------------------------
>>> import flax.nnx.bridge as nnx_bridge
>>> from policy.anytouch.encoder_jax import create_anytouch_module, load_anytouch_params
>>> module = create_anytouch_module(config_path="policy/anytouch/CLIP-B-16")
>>> touch = nnx_bridge.ToNNX(module)
>>> touch.lazy_init(fake_pixels, fake_sensor_type, rngs=rngs)
>>> # Then load pretrained weights into the NNX module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from anytouch.model.tactile_mae_jax import (
    Module,
    TactileMAEConfig,
    _Module,
)

logger = logging.getLogger("anytouch")

# Re-export for convenience
__all__ = [
    "TactileMAEConfig",
    "AnyTouchEncoderJax",
    "create_anytouch_module",
    "load_anytouch_params",
]

# HuggingFace repo & checkpoint variants (same as anytouch_bridge.py)
ANYTOUCH_HF_REPO = "xxuan01/AnyTouch2-Model"
ANYTOUCH_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "checkpoints" / "anytouch"
ANYTOUCH_CHECKPOINTS = {
    "2frames": "checkpoint-2frames.pth",
    "4frames": "checkpoint-4frames.pth",
    "4frames-touchd-digit": "checkpoint-4frames-touchd-digit.pth",
    "4frames-touchd-gelsight": "checkpoint-4frames-touchd-gelsight.pth",
}
ANYTOUCH_DEFAULT_VARIANT = "2frames"


def ensure_checkpoint(
    variant: str = ANYTOUCH_DEFAULT_VARIANT,
    cache_dir: str | Path | None = None,
) -> str:
    """Return path to checkpoint, downloading from HuggingFace if needed."""
    if variant not in ANYTOUCH_CHECKPOINTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(ANYTOUCH_CHECKPOINTS)}")

    filename = ANYTOUCH_CHECKPOINTS[variant]
    cache_dir = Path(cache_dir) if cache_dir is not None else ANYTOUCH_DEFAULT_CACHE_DIR
    local_path = cache_dir / filename

    if local_path.exists():
        return str(local_path)

    logger.info("Downloading '%s' from %s ...", filename, ANYTOUCH_HF_REPO)
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=ANYTOUCH_HF_REPO, filename=filename, local_dir=str(cache_dir))
    return str(downloaded)


# ---------------------------------------------------------------------------
# Convenience functions (for pi0 integration)
# ---------------------------------------------------------------------------

def create_anytouch_module(
    config_path: str | Path = "policy/anytouch/CLIP-B-16",
    *,
    num_frames: int = 1,
    stride: int = 1,
    dtype_mm: str = "float32",
    **kw,
) -> _Module:
    """Create the Flax Linen module (un-initialized).

    Returns a ``_Module`` instance suitable for ``nnx_bridge.ToNNX(module)``
    or ``module.init(rng, pixel_values, sensor_type)``.
    """
    config = TactileMAEConfig.from_clip_config(
        config_path, num_frames=num_frames, stride=stride, **kw,
    )
    return Module(config, dtype_mm=dtype_mm)


def load_anytouch_params(
    config_path: str | Path = "policy/anytouch/CLIP-B-16",
    checkpoint_path: str | Path | None = None,
    variant: str = ANYTOUCH_DEFAULT_VARIANT,
    *,
    num_frames: int | None = None,
    stride: int | None = None,
) -> tuple[TactileMAEConfig, dict]:
    """Load config and pretrained JAX params from a PyTorch checkpoint.

    If ``num_frames`` or ``stride`` are not specified, they are auto-inferred
    from the checkpoint's Conv3d kernel shape and position embedding size.

    Returns
    -------
    config : TactileMAEConfig
    params : dict
        Nested dict usable as ``module.apply({"params": params}, ...)``.
    """
    from anytouch.weight_converter import (
        infer_config_from_checkpoint,
        load_params_from_pytorch,
    )

    if checkpoint_path is None:
        checkpoint_path = ensure_checkpoint(variant=variant)

    if num_frames is None or stride is None:
        config = infer_config_from_checkpoint(checkpoint_path, config_path)
    else:
        config = TactileMAEConfig.from_clip_config(
            config_path, num_frames=num_frames, stride=stride,
        )

    params = load_params_from_pytorch(checkpoint_path, config)
    return config, params


# ---------------------------------------------------------------------------
# Self-contained encoder (for standalone usage)
# ---------------------------------------------------------------------------

class AnyTouchEncoderJax:
    """Pure JAX AnyTouch encoder with loaded weights.

    This class bundles the Flax module, JAX params, and JIT-compiled forward
    function for convenient standalone usage.  For Pi0 integration, prefer
    using ``create_anytouch_module`` + ``load_anytouch_params`` separately.

    Attributes
    ----------
    config : TactileMAEConfig
    module : _Module
    params : dict
    output_dim : int
        Projection dimension (default 512).
    num_output_tokens : int
        Number of tokens per sample in the output sequence.
    """

    def __init__(
        self,
        config: TactileMAEConfig,
        module: _Module,
        params: dict,
    ):
        self.config = config
        self.module = module
        self.params = params
        self.output_dim = config.projection_dim
        self.num_output_tokens = config.num_output_tokens
        self._apply_fn = jax.jit(lambda p, x, s: module.apply({"params": p}, x, s)[0])

    @classmethod
    def from_pretrained(
        cls,
        config_path: str | Path = "policy/anytouch/CLIP-B-16",
        checkpoint_path: str | Path | None = None,
        variant: str = ANYTOUCH_DEFAULT_VARIANT,
        *,
        num_frames: int = 1,
        stride: int = 1,
        dtype_mm: str = "float32",
    ) -> AnyTouchEncoderJax:
        """Load pretrained weights and create the encoder."""
        config, params = load_anytouch_params(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            variant=variant,
            num_frames=num_frames,
            stride=stride,
        )
        module = Module(config, dtype_mm=dtype_mm)

        logger.info(
            "AnyTouchEncoderJax ready  (num_frames=%d, stride=%d, "
            "output_tokens=%d, output_dim=%d)",
            config.num_frames,
            config.stride,
            config.num_output_tokens,
            config.projection_dim,
        )
        return cls(config=config, module=module, params=params)

    def encode(
        self,
        pixel_values: jax.Array,
        sensor_type: jax.Array,
    ) -> jax.Array:
        """Encode tactile input.

        Parameters
        ----------
        pixel_values : (B, C, T, H, W) float32
        sensor_type  : (B,) int32

        Returns
        -------
        tokens : (B, num_output_tokens, output_dim) float32
        """
        return self._apply_fn(self.params, pixel_values, sensor_type)

    def __call__(self, pixel_values: jax.Array, sensor_type: jax.Array) -> jax.Array:
        """Alias for ``encode``."""
        return self.encode(pixel_values, sensor_type)
