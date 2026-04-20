import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

VIS_IMAGE_KEYS: tuple[str, ...] = (
    "left_image",
    "right_image",
)

def make_vis_only_example(image_shape=(224, 224, 3), state_dim=20) -> dict:
    """Creates a random input example for the VB policy."""
    return {
        "observation.images.camera0": np.random.randint(256, size=image_shape, dtype=np.uint8),
        "observation.images.camera1": np.random.randint(256, size=image_shape, dtype=np.uint8),
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
        state = np.atleast_1d(np.asarray(data["observation.state"]))

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        left_image = _parse_image(data["observation.images.camera0"])
        right_image = _parse_image(data["observation.images.camera1"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = VIS_IMAGE_KEYS
                images = (left_image, right_image)
                image_masks = (np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

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
