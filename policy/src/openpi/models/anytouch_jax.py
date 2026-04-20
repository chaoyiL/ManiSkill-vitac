"""Native JAX AnyTouch encoder adapter for openpi.

This module provides the bridge between the ``policy.anytouch`` Flax Linen
implementation and the openpi Pi0 model.  It exposes:

- ``create_module()``: build a Flax Linen ``_Module`` ready for ``nnx_bridge.ToNNX``
- ``AnyTouchWeightLoader``: load pretrained PyTorch AnyTouch weights into the
  NNX parameter tree during training initialization.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import flax.traverse_util
import numpy as np

import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Ensure the anytouch package is importable.
_ANYTOUCH_ROOT = Path(__file__).resolve().parents[3] / "anytouch"
if str(_ANYTOUCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANYTOUCH_ROOT.parent))

from anytouch.model.tactile_mae_jax import (  # type: ignore[reportMissingImports]  # noqa: E402
    Module,
    TactileMAEConfig,
    _Module,
)

# HuggingFace repo for AnyTouch pretrained weights.
ANYTOUCH_HF_REPO = "xxuan01/AnyTouch2-Model"
ANYTOUCH_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[4] / "checkpoints" / "anytouch"
ANYTOUCH_CHECKPOINTS = {
    "2frames": "checkpoint-2frames.pth",
    "4frames": "checkpoint-4frames.pth",
    "4frames-touchd-digit": "checkpoint-4frames-touchd-digit.pth",
    "4frames-touchd-gelsight": "checkpoint-4frames-touchd-gelsight.pth",
}
ANYTOUCH_DEFAULT_VARIANT = "2frames"


def _ensure_checkpoint(
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
    logger.info("Downloading AnyTouch '%s' from %s ...", filename, ANYTOUCH_HF_REPO)
    from huggingface_hub import hf_hub_download
    return str(hf_hub_download(repo_id=ANYTOUCH_HF_REPO, filename=filename, local_dir=str(cache_dir)))


# ---------------------------------------------------------------------------
# Module factory
# ---------------------------------------------------------------------------

def create_module(
    config_path: str | Path = "policy/anytouch/CLIP-B-16",
    *,
    num_frames: int = 1,
    stride: int = 1,
    dtype_mm: str = "bfloat16",
    lora_rank: int = 0,
    lora_alpha: float = 1.0,
) -> _Module:
    """Create an un-initialized Flax Linen AnyTouch module.

    Args:
        lora_rank: LoRA rank for attention/MLP projections. 0 = no LoRA.
        lora_alpha: LoRA scaling factor (effective scale = lora_alpha / lora_rank).
    """
    config = TactileMAEConfig.from_clip_config(config_path, num_frames=num_frames, stride=stride)
    return Module(config, dtype_mm=dtype_mm, lora_rank=lora_rank, lora_alpha=lora_alpha)


# ---------------------------------------------------------------------------
# Weight loader
# ---------------------------------------------------------------------------

class AnyTouchWeightLoader:
    """Load pretrained AnyTouch weights from a PyTorch checkpoint into NNX params.

    Usage in ``TrainConfig``::

        weight_loader=weight_loaders.CompositeWeightLoader([
            weight_loaders.CheckpointWeightLoader("gs://..."),
            AnyTouchWeightLoader(
                config_path="policy/anytouch/CLIP-B-16",
                variant="2frames",
            ),
        ])

    The loader converts the PyTorch state dict and merges the AnyTouch
    sub-tree into the full model params under the ``anytouch/`` prefix.
    """

    def __init__(
        self,
        config_path: str | Path = "policy/anytouch/CLIP-B-16",
        checkpoint_path: str | Path | None = None,
        variant: str = ANYTOUCH_DEFAULT_VARIANT,
        *,
        num_frames: int | None = None,
        stride: int | None = None,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.variant = variant
        self.num_frames = num_frames
        self.stride = stride

    def load(self, params: at.Params) -> at.Params:
        """Merge pretrained AnyTouch weights into *params*."""
        from anytouch.weight_converter import (  # type: ignore[reportMissingImports]
            infer_config_from_checkpoint,
            load_params_from_pytorch,
        )

        ckpt_path = self.checkpoint_path
        if ckpt_path is None:
            ckpt_path = _ensure_checkpoint(variant=self.variant)

        if self.num_frames is None or self.stride is None:
            config = infer_config_from_checkpoint(ckpt_path, self.config_path)
        else:
            config = TactileMAEConfig.from_clip_config(
                self.config_path, num_frames=self.num_frames, stride=self.stride,
            )

        anytouch_params = load_params_from_pytorch(ckpt_path, config)

        # Flatten both trees, inject AnyTouch params under their matching paths.
        # In the Pi0 NNX param tree, the anytouch keys are "anytouch/<key>".
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_touch = flax.traverse_util.flatten_dict({"anytouch": anytouch_params}, sep="/")

        merged = 0
        added = 0
        for k, v in flat_touch.items():
            if k in flat_ref:
                flat_ref[k] = np.asarray(v, dtype=flat_ref[k].dtype)
                merged += 1
            else:
                # Composite loaders may pass a partial tree that doesn't include
                # anytouch keys yet. In that case, add the converted checkpoint
                # weights directly so they can be validated later.
                flat_ref[k] = np.asarray(v)
                added += 1

        logger.info(
            "AnyTouchWeightLoader: merged %d, added %d / %d parameters",
            merged,
            added,
            len(flat_touch),
        )

        return flax.traverse_util.unflatten_dict(flat_ref, sep="/")
