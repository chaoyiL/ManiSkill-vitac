"""AnyTouch Tactile Video MAE encoder — pure JAX / Flax Linen implementation.

Supports optional LoRA fine-tuning on all attention projections (Q/K/V/Out)
and MLP layers (fc1/fc2).  LoRA parameters are named with the ``lora_``
prefix so the existing ``.*lora.*`` freeze filter in Pi0Config works directly.

External interface
------------------
>>> from policy.anytouch.model.tactile_mae_jax import Module, TactileMAEConfig
>>> config = TactileMAEConfig.from_clip_config("policy/anytouch/CLIP-B-16")
>>> module = Module(config, lora_rank=8)   # lora_rank=0 → no LoRA (default)
>>> params = module.init(jax.random.key(0), fake_pixels, fake_sensor_type)
>>> tokens, aux = module.apply(params, pixel_values, sensor_type)
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TactileMAEConfig:
    """Configuration that mirrors the CLIP-B/16 vision config used by AnyTouch."""

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    projection_dim: int = 512
    layer_norm_eps: float = 1e-5
    num_frames: int = 1
    stride: int = 1
    num_sensor_types: int = 20
    num_sensor_tokens: int = 5

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size ** 2 * (self.num_frames // self.stride)

    @property
    def num_output_tokens(self) -> int:
        return 1 + self.num_sensor_tokens + self.num_patches

    @classmethod
    def from_clip_config(
        cls,
        config_path: str | Path,
        *,
        num_frames: int = 1,
        stride: int = 1,
        num_sensor_types: int = 20,
        num_sensor_tokens: int = 5,
    ) -> TactileMAEConfig:
        config_path = Path(config_path)
        if config_path.is_dir():
            config_path = config_path / "config.json"
        with open(config_path) as f:
            raw = json.load(f)
        vc = raw["vision_config"]
        return cls(
            hidden_size=vc["hidden_size"],
            intermediate_size=vc["intermediate_size"],
            num_attention_heads=vc["num_attention_heads"],
            num_hidden_layers=vc["num_hidden_layers"],
            image_size=vc["image_size"],
            patch_size=vc["patch_size"],
            num_channels=vc["num_channels"],
            projection_dim=raw["projection_dim"],
            layer_norm_eps=vc.get("layer_norm_eps", 1e-5),
            num_frames=num_frames,
            stride=stride,
            num_sensor_types=num_sensor_types,
            num_sensor_tokens=num_sensor_tokens,
        )


# ---------------------------------------------------------------------------
# LoRA-enabled Dense layer
# ---------------------------------------------------------------------------

class LoRADense(nn.Module):
    """nn.Dense with an optional LoRA side-path.

    When ``lora_rank > 0``, the output is:
        W·x + (lora_b · lora_a · x) * (alpha / rank)

    LoRA parameter names contain ``lora_`` so the existing freeze filter
    ``.*lora.*`` targets them automatically.

    Base weight ``kernel`` and ``bias`` are stored under the same names as a
    plain ``nn.Dense``, so pretrained weights load without any key remapping.
    """

    features: int
    lora_rank: int = 0
    lora_alpha: float = 1.0
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Base projection (identical param layout to nn.Dense)
        out = nn.Dense(
            self.features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            name="base",
        )(x)

        if self.lora_rank > 0:
            in_features = x.shape[-1]
            scale = self.lora_alpha / self.lora_rank

            # A: (in, rank)   B: (rank, out)
            lora_a = self.param(
                "lora_a",
                nn.initializers.normal(stddev=1 / math.sqrt(in_features)),
                (in_features, self.lora_rank),
            )
            lora_b = self.param(
                "lora_b",
                nn.initializers.zeros,
                (self.lora_rank, self.features),
            )
            # Compute in float32 for numerical stability, then cast
            x32 = x.astype(jnp.float32)
            lora_out = jnp.dot(jnp.dot(x32, lora_a), lora_b) * scale
            out = out + lora_out.astype(self.dtype)

        return out


# ---------------------------------------------------------------------------
# Transformer sub-modules (CLIP-style, pre-norm)
# ---------------------------------------------------------------------------

class CLIPAttention(nn.Module):
    """Multi-head self-attention with optional LoRA on Q/K/V/Out projections."""

    hidden_size: int
    num_heads: int
    lora_rank: int = 0
    lora_alpha: float = 1.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        head_dim = self.hidden_size // self.num_heads
        scale = head_dim ** -0.5
        B, N, _ = x.shape

        def proj(name: str) -> jax.Array:
            return LoRADense(
                self.hidden_size,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                dtype=self.dtype,
                name=name,
            )(x)

        q = proj("q_proj")
        k = proj("k_proj")
        v = proj("v_proj")

        q = q.reshape(B, N, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        attn = jnp.matmul(q * scale, jnp.swapaxes(k, -2, -1))
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.matmul(attn, v)

        out = jnp.swapaxes(out, 1, 2).reshape(B, N, self.hidden_size)
        out = LoRADense(
            self.hidden_size,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            dtype=self.dtype,
            name="out_proj",
        )(out)
        return out


class CLIPMLP(nn.Module):
    """Two-layer MLP with optional LoRA on fc1/fc2."""

    hidden_size: int
    intermediate_size: int
    lora_rank: int = 0
    lora_alpha: float = 1.0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = LoRADense(
            self.intermediate_size,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            dtype=self.dtype,
            name="fc1",
        )(x)
        x = jax.nn.gelu(x, approximate=False)
        x = LoRADense(
            self.hidden_size,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            dtype=self.dtype,
            name="fc2",
        )(x)
        return x


class CLIPEncoderLayer(nn.Module):
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    lora_rank: int = 0
    lora_alpha: float = 1.0
    layer_norm_eps: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = nn.LayerNorm(epsilon=self.layer_norm_eps, name="layer_norm1")(x)
        x = CLIPAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            dtype=self.dtype,
            name="self_attn",
        )(x)
        x = residual + x

        residual = x
        x = nn.LayerNorm(epsilon=self.layer_norm_eps, name="layer_norm2")(x)
        x = CLIPMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            dtype=self.dtype,
            name="mlp",
        )(x)
        x = residual + x
        return x


class CLIPEncoder(nn.Module):
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    lora_rank: int = 0
    lora_alpha: float = 1.0
    layer_norm_eps: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for i in range(self.num_layers):
            x = CLIPEncoderLayer(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_attention_heads=self.num_attention_heads,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                layer_norm_eps=self.layer_norm_eps,
                dtype=self.dtype,
                name=f"layers_{i}",
            )(x)
        return x


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class _Module(nn.Module):
    """AnyTouch Tactile Video MAE encoder (Flax Linen) with optional LoRA.

    Input
    -----
    pixel_values : (B, C, T, H, W) float32
    sensor_type  : (B,) int32

    Output
    ------
    tokens : (B, num_output_tokens, projection_dim)
    out    : dict with intermediate representations
    """

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    projection_dim: int = 512
    layer_norm_eps: float = 1e-5
    num_frames: int = 1
    stride: int = 1
    num_sensor_types: int = 20
    num_sensor_tokens: int = 5
    dtype_mm: str = "float32"
    # LoRA: 0 = disabled, >0 = rank of LoRA adapters on all attn+mlp projections
    lora_rank: int = 0
    lora_alpha: float = 1.0

    @nn.compact
    def __call__(self, pixel_values: jax.Array, sensor_type: jax.Array) -> tuple[jax.Array, dict]:
        dtype = jnp.dtype(self.dtype_mm)
        B = pixel_values.shape[0]

        if pixel_values.ndim == 4:
            pixel_values = pixel_values[:, :, None, :, :]
        # (B, C, T, H, W) → (B, T, H, W, C)
        x = jnp.transpose(pixel_values.astype(jnp.float32), (0, 2, 3, 4, 1))

        # 3D patch embedding (always float32)
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=(self.stride, self.patch_size, self.patch_size),
            strides=(self.stride, self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=False,
            dtype=jnp.float32,
            name="patch_embedding",
        )(x)  # (B, T', H', W', D)

        num_patches = x.shape[1] * x.shape[2] * x.shape[3]
        x = x.reshape(B, num_patches, self.hidden_size)

        # Keep position embedding parameter shape static across calls.
        # This avoids ScopeParamShapeError when runtime tactile length/resolution
        # differs from the lazy-init shape.
        expected_num_patches = (self.image_size // self.patch_size) ** 2 * (self.num_frames // self.stride)
        position_embedding = self.param(
            "position_embedding",
            nn.initializers.normal(stddev=0.02),
            (expected_num_patches + 1, self.hidden_size),
        )
        x = x + position_embedding[1 : num_patches + 1, :]

        class_embedding = self.param(
            "class_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.hidden_size,),
        )
        cls_token = class_embedding + position_embedding[0, :]
        cls_token = jnp.broadcast_to(cls_token[None, None, :], (B, 1, self.hidden_size))

        sensor_tokens = self.param(
            "sensor_token",
            nn.initializers.zeros,
            (self.num_sensor_types, self.num_sensor_tokens, self.hidden_size),
        )
        sensor_emb = sensor_tokens[sensor_type]

        x = jnp.concatenate([cls_token, sensor_emb, x], axis=1)
        x = x.astype(dtype)

        x = nn.LayerNorm(epsilon=self.layer_norm_eps, name="pre_layernorm")(x)

        x = CLIPEncoder(
            num_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            layer_norm_eps=self.layer_norm_eps,
            dtype=dtype,
            name="encoder",
        )(x)

        out = {"last_hidden_state": x, "cls_token": x[:, 0, :]}

        x = nn.LayerNorm(epsilon=self.layer_norm_eps, name="post_layernorm")(x)

        x = nn.Dense(
            self.projection_dim, use_bias=False, dtype=dtype, name="touch_projection"
        )(x)

        return x, out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def Module(config: TactileMAEConfig | None = None, **kw) -> _Module:
    """Create a TactileVideoMAE Flax Linen module.

    Pass ``lora_rank=8`` (or any rank > 0) to enable LoRA on all attention
    and MLP projections.  ``lora_rank=0`` (default) is a plain encoder.
    """
    if config is not None:
        return _Module(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            projection_dim=config.projection_dim,
            layer_norm_eps=config.layer_norm_eps,
            num_frames=config.num_frames,
            stride=config.stride,
            num_sensor_types=config.num_sensor_types,
            num_sensor_tokens=config.num_sensor_tokens,
            **kw,
        )
    return _Module(**kw)
