"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_SRC = PROJECT_ROOT / "policy" / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(POLICY_SRC))

import numpy as np
import torch
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

DEFAULT_BATCH_SIZE = 256
DEFAULT_NUM_WORKERS = 4


class LimitedDataLoader:
    def __init__(self, data_loader: torch.utils.data.DataLoader, num_batches: int):
        self._data_loader = data_loader
        self._num_batches = num_batches

    def __iter__(self):
        for batch_index, batch in enumerate(self._data_loader):
            if batch_index >= self._num_batches:
                return
            yield batch


class ExtractStateAndActions(transforms.DataTransformFn):
    """Keep only the fields needed for norm stats."""

    state_keys = ("state", "observation.state", "observation/state")
    action_keys = ("actions", "action")

    def __call__(self, x: dict) -> dict:
        flat = transforms.flatten_dict(x)
        return {
            "state": np.asarray(self._get_first(flat, self.state_keys)),
            "actions": np.asarray(self._get_first(flat, self.action_keys)),
        }

    def _get_first(self, x: dict, keys: tuple[str, ...]):
        for key in keys:
            if key in x:
                return x[key]
        raise KeyError(f"None of {keys} found in dataset item. Available keys: {sorted(x)}")


class StateActionDataset(_data_loader.Dataset):
    """Read only state/action columns from a LeRobot dataset."""

    def __init__(self, dataset: _data_loader.Dataset):
        self._dataset = self._unwrap_lerobot_dataset(dataset)
        hf_dataset = getattr(self._dataset.hf_dataset, "_dataset", self._dataset.hf_dataset)
        self._hf_dataset = hf_dataset.select_columns(["observation.state", "episode_index", "actions"]).with_format(
            "torch"
        )

    def _unwrap_lerobot_dataset(self, dataset: _data_loader.Dataset):
        while not hasattr(dataset, "hf_dataset") and hasattr(dataset, "_dataset"):
            dataset = dataset._dataset
        if not hasattr(dataset, "hf_dataset"):
            raise TypeError("StateActionDataset requires a LeRobot dataset with an hf_dataset attribute.")
        return dataset

    def __getitem__(self, index) -> dict:
        item = self._hf_dataset[index]
        output = {"state": item["observation.state"]}

        if getattr(self._dataset, "delta_indices", None) is not None and "actions" in self._dataset.delta_indices:
            ep_idx = item["episode_index"].item()
            query_indices, _ = self._dataset._get_query_indices(index, ep_idx)
            output["actions"] = self._dataset._query_hf_dataset({"actions": query_indices["actions"]})["actions"]
        else:
            output["actions"] = item["actions"]

        return output

    def __len__(self) -> int:
        return len(self._dataset)


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[LimitedDataLoader, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    raw_dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    if data_config.repack_transforms.inputs:
        dataset = _data_loader.TransformedDataset(
            raw_dataset,
            [
                *data_config.repack_transforms.inputs,
                ExtractStateAndActions(),
            ],
        )
    elif isinstance(raw_dataset, torch.utils.data.ConcatDataset):
        dataset = torch.utils.data.ConcatDataset([StateActionDataset(dataset) for dataset in raw_dataset.datasets])
    else:
        dataset = StateActionDataset(raw_dataset)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided.")

    dataset_size = len(dataset)
    target_frames = min(max_frames, dataset_size) if max_frames is not None else dataset_size
    effective_batch_size = min(batch_size, target_frames)
    num_batches = target_frames // effective_batch_size
    if max_frames is not None and max_frames < dataset_size:
        shuffle = True
    else:
        shuffle = False
    print(
        f"Using batch_size={effective_batch_size}, num_workers={num_workers}, "
        f"num_batches={num_batches}, target_frames={target_frames}"
    )
    torch_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return LimitedDataLoader(torch_data_loader, num_batches), num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, batch_size, config.model, num_workers, max_frames
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    print("dir check: ", config.assets_dirs, data_config.asset_id)
    output_path = config.assets_dirs / data_config.asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
