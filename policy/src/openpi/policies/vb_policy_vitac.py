import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


#
# Single source of truth for image ordering.
#
# IMPORTANT: This order must be identical for both training and inference.
# The model will embed images by iterating over `obs.images` in insertion order,
# and `preprocess_observation(..., image_keys=...)` will also rebuild the dict
# in the order of `image_keys`.
#

VIS_ONLY_IMAGE_KEYS: tuple[str, ...] = (
    "left_image",
    "right_image",
)


def make_vitac_example(image_shape=(224, 224, 3), state_dim=20) -> dict:
    """Creates a random input example for the VB policy."""
    return {
        "observation.images.camera0": np.random.randint(256, size=image_shape, dtype=np.uint8),
        "observation.images.camera1": np.random.randint(256, size=image_shape, dtype=np.uint8),
        "observation.images.tactile_left_0": np.random.randint(256, size=image_shape, dtype=np.uint8),
        "observation.images.tactile_right_0": np.random.randint(256, size=image_shape, dtype=np.uint8),
        "observation.images.tactile_left_1": np.random.randint(256, size=image_shape, dtype=np.uint8),
        "observation.images.tactile_right_1": np.random.randint(256, size=image_shape, dtype=np.uint8),
        "observation.state": np.random.rand(state_dim),
        "task": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class VBInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    def __call__(self, data: dict) -> dict:
        state = data["observation.state"]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        left_image = _parse_image(data["observation.images.camera0"])
        right_image = _parse_image(data["observation.images.camera1"])
        tactile_left_0 = _parse_image(data["observation.images.tactile_left_0"])
        tactile_right_0 = _parse_image(data["observation.images.tactile_right_0"])
        tactile_left_1 = _parse_image(data["observation.images.tactile_left_1"])
        tactile_right_1 = _parse_image(data["observation.images.tactile_right_1"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # Visual images only — tactile goes through AnyTouch.
                names = VIS_ONLY_IMAGE_KEYS
                images = (left_image, right_image)
                image_masks = (np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Stack tactile images into (4, C, H, W) then add T dim → (C, T=1, H, W).
        # AnyTouch expects (B, C, T, H, W); the batch dim is added by the data loader.
        tactile_imgs = np.stack([tactile_left_0, tactile_right_0, tactile_left_1, tactile_right_1], axis=0)
        # (4, H, W, C) → (4, C, H, W)
        tactile_imgs = einops.rearrange(tactile_imgs, "n h w c -> n c h w")
        # Normalize uint8 → [-1, 1] float32.
        tactile_imgs = tactile_imgs.astype(np.float32) / 255.0 * 2.0 - 1.0
        # Add temporal dim: (4, C, H, W) → (4, C, 1, H, W)
        tactile_imgs = tactile_imgs[:, :, None, :, :]
        inputs["tactile"] = tactile_imgs
        inputs["tactile_mask"] = np.True_
        inputs["tactile_sensor_type"] = np.zeros(tactile_imgs.shape[0], dtype=np.int32)

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "task" in data:
            if isinstance(data["task"], bytes):
                data["task"] = data["task"].decode("utf-8")
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class VBOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"])}