import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # If provided, this defines the exact image key order expected by the model.
    # This is used to keep training and inference image ordering consistent.
    # If None, the model will use the insertion order of `observation.images`.
    image_keys: tuple[str, ...] | None = None

    # Set the model specific defaults.
    action_dim: int = 32
    state_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # AnyTouch tactile encoder settings (set anytouch_config_path to enable).
    anytouch_config_path: str | None = None
    # Explicit path to checkpoint file. If None, auto-downloads the anytouch_variant from HuggingFace.
    anytouch_checkpoint_path: str | None = None
    # Which pretrained variant to download: "2frames", "4frames", "4frames-touchd-digit", "4frames-touchd-gelsight".
    anytouch_variant: str = "2frames"
    anytouch_num_frames: int = 1
    anytouch_stride: int = 1
    anytouch_pool_tokens: int | None = None
    # LoRA fine-tuning for AnyTouch: 0 = full fine-tune, >0 = LoRA rank.
    # When lora_rank > 0, base weights are frozen and only lora_a/lora_b are trained.
    anytouch_lora_rank: int = 0
    anytouch_lora_alpha: float = 1.0

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    def fake_tactile_input(self) -> jnp.ndarray:
        """Fake tactile input for lazy_init: (1, C, T, H, W)."""
        return jnp.zeros((1, 3, max(self.anytouch_num_frames, 1), 224, 224), dtype=jnp.float32)

    def fake_sensor_type_input(self) -> jnp.ndarray:
        """Fake sensor type for lazy_init: (1,)."""
        return jnp.zeros((1,), dtype=jnp.int32)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.state_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config.

        Frozen means: parameter exists in the model but receives no gradient.
        The logic is:
          - PaliGemma/action-expert base weights are frozen when LoRA variants
            are selected (identical to the original behaviour).
          - AnyTouch base weights (``anytouch/.*/base/.*``) are frozen when
            ``anytouch_lora_rank > 0``; only the LoRA adapters are trained.
          - When ``anytouch_lora_rank == 0`` the AnyTouch encoder participates
            fully in training (no freeze).
        """
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")

        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params_filter))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter)
            has_lora = True

        # AnyTouch LoRA: freeze base weights inside the encoder, train only adapters.
        if self.anytouch_lora_rank > 0:
            # Match all anytouch params, then exclude the lora_a/lora_b adapters.
            filters.append(nnx_utils.PathRegex(".*anytouch.*"))
            has_lora = True

        if has_lora:
            # Exclude ALL lora_* params from the freeze set so they are trained.
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora_.*")))

        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
