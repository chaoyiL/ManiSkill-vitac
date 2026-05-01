# ManiSkill-vitac

User-side training and deployment workspace for visual-tactile robot policies. This repository builds on OpenPI/Pi0.5, adds bimanual visual and AnyTouch tactile policy configurations, trains on LeRobot datasets, and deploys trained checkpoints to a robot bridge over websocket.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `policy/src/openpi/training/config.py` | Training, dataset, model, checkpoint, and data-conversion configuration. |
| `policy/scripts/train.py` | Main JAX training entrypoint. |
| `policy/scripts/compute_norm_stats.py` | Computes dataset normalization statistics used by training and deployment. |
| `deploy_scripts/infer.py` | Loads a trained checkpoint and sends policy actions to the robot bridge. |
| `scripts/*.sh` | Convenience wrappers that run the Python entrypoints through `uv`. |
| `client/interface_client.py` | Persistent websocket client used during deployment. |
| `robot_visualization/` | Offline visualizer for recorded Zarr and LeRobot episodes. |

## Requirements

- Linux x86_64.
- Python `>=3.11,<3.12`.
- NVIDIA GPU with a CUDA 12-compatible driver. The project pins `jax[cuda12]` and PyTorch CUDA 12.8 wheels.
- `uv` for dependency management.
- FFmpeg and a recent `libstdc++`. The wrappers prefer libraries from the active conda environment to avoid `av`, `libavformat`, or `CXXABI` import errors.
- Network access on first setup or first training run, because dependencies and pretrained weights are pulled from GitHub, Hugging Face, Google Cloud Storage, and the PyTorch wheel index.

## Environment Setup

Create a Python 3.11 conda environment for system libraries, then let `uv` manage the project virtual environment:

```bash
conda create -n maniskill-vitac python=3.11 -y
conda activate maniskill-vitac
conda install -c conda-forge 'ffmpeg=7' libstdcxx-ng -y

python -m pip install -U uv
uv sync --locked
```

Keep the conda environment active when using the shell wrappers so PyAV and related libraries load the conda FFmpeg/libstdc++ builds.

If you use private Hugging Face datasets or checkpoints, log in before training:

```bash
hf auth login
```

## Download Hugging Face Datasets

To download a Hugging Face dataset manually before training, replace the placeholders below with the publisher namespace and dataset name:

```bash
hf download <publisher-huggingface-username>/<dataset-name> \
  --repo-type dataset \
  --cache-dir ~/.cache/huggingface/dataset
```

The dataset snapshot is downloaded to a cache path like:

```text
~/.cache/huggingface/dataset/datasets--<publisher-huggingface-username>--<dataset-name>/snapshots/<snapshot-hash>
```

Link that snapshot into the LeRobot cache path expected by the configured `repo_id`:

```bash
mkdir -p ~/.cache/huggingface/lerobot/<DATASET_REPO_NAMESPACE>
ln -s ~/.cache/huggingface/dataset/datasets--<publisher-huggingface-username>--<dataset-name>/snapshots/<snapshot-hash> \
  ~/.cache/huggingface/lerobot/<DATASET_REPO_NAMESPACE>/<DATASET_TRAIN_NAME>
```

For example, with `DATASET_REPO_NAMESPACE = "chaoyi"`, the link target should be under:

```text
~/.cache/huggingface/lerobot/chaoyi/<DATASET_TRAIN_NAME>
```

If Hugging Face cannot be reached from the machine, switch to the mirror endpoint before logging in or downloading:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=<your-token>
```

Some assets are downloaded automatically by the code on first use. Make sure the machine has network access before running the corresponding entrypoint:

| Downloaded asset | Source | Triggered by |
| --- | --- | --- |
| Pi0.5 base weights | `gs://openpi-assets/checkpoints/pi05_base/params` | `policy/scripts/train.py` through `./scripts/train.sh` when starting `pi05_bi` or `pi05_bi_vitac` training from scratch. |
| PaliGemma tokenizer | `gs://big_vision/paligemma_tokenizer.model` | `policy/scripts/compute_norm_stats.py`, `policy/scripts/train.py`, and `deploy_scripts/infer.py` when they create model/data transforms. |
| AnyTouch weights | `xxuan01/AnyTouch2-Model` | `policy/scripts/train.py` through `./scripts/train.sh pi05_bi_vitac` when loading the tactile encoder pretrained weights. |

## Configure Data and Policy

All dataset, policy, and training presets are defined in `policy/src/openpi/training/config.py`. For a new experiment, edit the constants near the bottom of that file first, then choose one of the entries in `_CONFIGS`.

### Dataset naming

The active LeRobot dataset is controlled here:

```python
DATASET_TRAIN_NAME = "0118_data_1smooth"
DATASET_RAW_TASK_NAME = "raw_0118_data"
DATASET_REPO_NAMESPACE = "chaoyi"
```

These constants are expanded into:

| Variable | Meaning | Where it is used |
| --- | --- | --- |
| `DATASET_TRAIN_NAME` | Processed LeRobot dataset name. Change this when training on a new converted dataset. | Becomes `data_name`, `asset_id`, and the default `exp_name` inside `config.py`. |
| `DATASET_RAW_TASK_NAME` | Raw data task name used by the data collection/conversion pipeline. | Keep it aligned with the raw data folder or conversion script for the dataset. |
| `DATASET_REPO_NAMESPACE` | Hugging Face/LeRobot namespace. | Combined with `DATASET_TRAIN_NAME` to form `repo_id`. |
| `repo_id` | Full LeRobot dataset id. | `SimpleDataConfig(repo_id=repo_id)` loads training data from this dataset. |
| `asset_id` | Normalization asset name. | `compute_norm_stats.py` writes `assets/<asset_id>/norm_stats.json`; training and deployment load the same id. |
| `assets_dir` | Asset root passed into `AssetsConfig`. | Defaults to `assets`, so normalization stats live under `assets/<asset_id>/`. |

For the default values, the derived paths are:

- LeRobot `repo_id`: `chaoyi/0118_data_1smooth`
- normalization asset id: `0118_data_1smooth`
- config-level default checkpoint experiment name: `0118_data_1smooth`

### Policy presets

The available policy configs are the `TrainConfig(...)` entries in `_CONFIGS`:

| Config           | Use case                         | Data transform                              | Model inputs                                                                                                       | Pretrained weights                                                                    |
| ---              | ---                              | ---                                         | ---                                                                                                                | ---                                                                                   |
| `pi05_bi`        | Bimanual vision-only policy.     | `vb_policy_vis.VBInputs` / `VBOutputs`      | Two RGB images, `state_dim=20`, `action_dim=20`, `action_horizon=50`.                                               | Pi0.5 base checkpoint from `gs://openpi-assets/checkpoints/pi05_base/params`.         |
| `pi05_bi_vitac` | Bimanual visual-tactile policy.  | `vb_policy_vitac.VBInputs` / `VBOutputs`    | Two RGB images plus four tactile images through AnyTouch, `state_dim=20`, `action_dim=20`, `action_horizon=50`.    | Pi0.5 base checkpoint plus AnyTouch weights from Hugging Face.                        |

To add a new preset, duplicate one `TrainConfig(...)` entry in `_CONFIGS`, give it a unique `name`, and update the model, data transform, freeze, and weight-loader fields together.

Policy-level constants above `_CONFIGS` apply to both default configs unless overridden inside a specific `TrainConfig`:

| Parameter | Current value | Meaning | Change when |
| --- | --- | --- | --- |
| `action_horizon` | `50` | Number of future action steps predicted by the policy. Deployment uses this to set `steps_per_inference`. | Your dataset/control loop uses a different action horizon. Recompute normalization stats and retrain after changing it. |
| `anytouch_pool_tokens` | `49` | Number of pooled AnyTouch patch tokens kept before projection. | You need a different tactile token budget. Keep it compatible with the 196-token AnyTouch patch grid. |
| `anytouch_lora_rank` | `8` | LoRA rank for AnyTouch fine-tuning. `0` means full AnyTouch fine-tuning. | You want to trade memory/trainable parameters against adaptation capacity. Update the `freeze_filter` config consistently. |
| `anytouch_lora_alpha` | `8.0` | LoRA scaling for AnyTouch adapters. | You tune AnyTouch LoRA strength. |

Inside each `pi0_config.Pi0Config(...)`, the most commonly edited model parameters are:

| Parameter | Meaning |
| --- | --- |
| `state_dim` / `action_dim` | Must match the robot state and action vector dimensions in the LeRobot dataset and robot bridge. The defaults are both `20`. |
| `paligemma_variant` / `action_expert_variant` | Base model variants. The defaults use LoRA variants: `gemma_2b_lora` and `gemma_300m_lora`. |
| `pi05` | Enables Pi0.5 behavior. Keep `True` for the provided configs. |
| `image_keys` | Canonical model image order. Keep this synchronized with the matching policy transform and inference preprocessing. |
| `anytouch_config_path` | Enables the AnyTouch tactile encoder for `pi05_bi_vitac`. Set to `None` for vision-only configs. |
| `anytouch_checkpoint_path` / `anytouch_variant` | Explicit AnyTouch checkpoint path, or the Hugging Face variant to auto-download when the path is `None`. |
| `anytouch_num_frames` / `anytouch_stride` | Temporal tactile-frame settings expected by the AnyTouch encoder and weight loader. |

### Data config

Each `TrainConfig` has a `data=SimpleDataConfig(...)` block:

| Parameter | Meaning |
| --- | --- |
| `repo_id=repo_id` | LeRobot dataset used for training and normalization stats. |
| `AssetsConfig(asset_id=asset_id, assets_dir=assets_dir)` | Tells training where to load normalization stats from. With the defaults, this is `assets/<DATASET_TRAIN_NAME>/norm_stats.json`. |
| `data_transforms` | Converts LeRobot fields into the model input/output format. Use `vb_policy_vis` for `pi05_bi` and `vb_policy_vitac` for `pi05_bi_vitac`. |
| `base_config=DataConfig(prompt_from_task=True)` | Uses the LeRobot `task` field as the language prompt. Set a different prompt transform only if your dataset does not provide `task`. |
| `action_sequence_keys` | Defaults to `("actions",)`. Set this in `DataConfig(...)` only if your dataset stores actions under a different key. |

If your dataset field names differ from the defaults, edit `policy/src/openpi/policies/vb_policy_vis.py` or `policy/src/openpi/policies/vb_policy_vitac.py`; those files map raw LeRobot fields into the common model input format.

The LeRobot dataset must match the selected config. For `pi05_bi_vitac`, expected fields include:

- `observation.images.camera0`
- `observation.images.camera1`
- `observation.images.tactile_left_0`
- `observation.images.tactile_right_0`
- `observation.images.tactile_left_1`
- `observation.images.tactile_right_1`
- `observation.state`
- `actions`
- `task`

For `pi05_bi`, the tactile fields are not required; it uses `camera0`, `camera1`, `state`, `actions`, and `task`.

You can inspect a dataset through the configured `repo_id`:

```bash
uv run python policy/scripts/check_dataset.py --config-name pi05_bi_vitac --num-samples 10
```

### Training config

Training behavior is controlled by the shared constants above `_CONFIGS` and by fields passed into each `TrainConfig`:

| Parameter | Current value | Meaning |
| --- | --- | --- |
| `fsdp_devices` | `2` | Number of devices used for FSDP sharding. Set to `1` to disable FSDP. |
| `batch_size` | `fsdp_devices * 64` | Global batch size. It must be divisible by the number of visible JAX devices. |
| `num_workers` | `2` by default | DataLoader worker count. Increase only if CPU/RAM and storage throughput can keep up. |
| `num_train_steps` | `100000` | Number of optimizer steps. |
| `seed` | `42` by default | Random seed used by training. |
| `exp_name` | `data_name` in `config.py` | Checkpoint experiment directory name under `checkpoints/<config>/<exp_name>/`. Note that `./scripts/train.sh` currently overrides this with `--exp-name my_experiment` unless you edit the wrapper or call `policy/scripts/train.py` directly. |
| `lr_schedule` | `CosineDecaySchedule(warmup_steps=1000, peak_lr=2e-4, decay_steps=100000, decay_lr=2e-4)` | Learning-rate schedule used by the optimizer. These values come from the shared `warmup_steps`, `peak_lr`, `decay_steps`, and `decay_lr` constants above `_CONFIGS`. |
| `optimizer` | Default `AdamW` | Optimizer config. AdamW includes gradient clipping through `clip_gradient_norm`. |
| `weight_loader` | Config-specific | Loads pretrained Pi0.5 weights, and for `pi05_bi_vitac` also loads AnyTouch weights. |
| `freeze_filter` | Config-specific | Freezes base pretrained weights and leaves LoRA parameters trainable. Update it if you change LoRA/full-finetuning behavior. |
| `ema_decay` | `None` in both provided configs | Disables EMA for LoRA fine-tuning. |
| `assets_base_dir` | `./assets` | Root directory for normalization stats. |
| `checkpoint_base_dir` | `./checkpoints` | Root directory for training checkpoints. |
| `log_interval` | `200` by default | Training metric logging frequency in steps. |
| `save_interval` / `keep_period` | `5000` / `5000` | Checkpoint save frequency and retention period. |
| `overwrite` / `resume` | `False` / `False` by default | Controls whether an existing checkpoint directory is overwritten or resumed. They cannot both be `True`. |
| `wandb_enabled` | `True` by default | Enables WandB logging unless disabled through config or CLI. |

Most training fields can also be overridden from the CLI for one run:

```bash
uv run python policy/scripts/train.py pi05_bi_vitac \
  --exp-name 0118_data_1smooth \
  --batch-size 128 \
  --num-train-steps 100000
```

## Compute Normalization Stats

Run this once before training a new dataset/config pair:

```bash
./scripts/compute_norm_stats.sh pi05_bi_vitac
```

The script writes:

```text
assets/<asset_id>/norm_stats.json
```

For the default constants, that is:

```text
assets/0118_data_1smooth/norm_stats.json
```

This file is important. Training loads it and saved checkpoints copy it into `assets/` inside each checkpoint step. Deployment then loads normalization stats from the checkpoint directory.

## Train Checkpoints

WandB logging is enabled by default. Log in before training:

```bash
wandb login
```

The simplest training command is:

```bash
./scripts/train.sh pi05_bi_vitac
```

The wrapper runs:

```bash
uv run python policy/scripts/train.py "$CONFIG" --exp-name my_experiment --overwrite
```

so checkpoints are written under:

```text
checkpoints/pi05_bi_vitac/my_experiment/<step>/
```

Each saved step contains inference parameters, training state, and copied normalization assets:

```text
checkpoints/pi05_bi_vitac/my_experiment/35000/
  assets/
  params/
  train_state/
```

For more control, call the training entrypoint directly:

```bash
uv run python policy/scripts/train.py pi05_bi_vitac \
  --exp-name 0118_data_1smooth \
  --num-train-steps 100000 \
  --save-interval 5000 \
  --fsdp-devices 2 \
  --batch-size 128 \
  --overwrite
```

Resume an existing run with:

```bash
uv run python policy/scripts/train.py pi05_bi_vitac \
  --exp-name 0118_data_1smooth \
  --resume
```

Notes:

- `batch_size` must be divisible by the number of JAX devices.
- `fsdp_devices` is set to `2` in `config.py`; adjust it to match the target machine.
- WandB is enabled by default. Run `wandb login`, set `WANDB_MODE=offline`, or disable WandB through the training CLI if needed.
- Always pass the config name explicitly to the shell wrappers. Their historical default may not exist in the current `_CONFIGS` list.

## Deploy a Trained Checkpoint

Deployment loads a saved checkpoint, connects to a robot bridge websocket, receives observations, runs policy inference, and sends actions back to the robot.

Start the robot bridge/server first, then run:

```bash
./scripts/infer.sh \
  --config pi05_bi_vitac \
  --ckpt-dir checkpoints/pi05_bi_vitac/my_experiment/35000 \
  --data_type vitac \
  --language_prompt "Open the red pot, pick up the blue cylinder on the table and place it into the pot." \
  --ip 127.0.0.1 \
  --port 26421 \
  --token <your given token> \
  --control_frequency 5 \
  --controller_frequency 80
```

Important deployment options:

| Option | Meaning |
| --- | --- |
| `--config` | Training config used to build the policy. Must match the checkpoint. |
| `--ckpt-dir` | Checkpoint step directory containing `params/` and `assets/`. Relative paths are resolved from the repo root. |
| `--data_type` | Observation mode sent to the robot bridge, for example `vitac` or `vision`. |
| `--language_prompt` | Task prompt injected into observations. |
| `--ip`, `--port`, `--token` | Websocket bridge address and bearer token. |
| `--save_obs` | Whether to save received observations under `eval_obs/<timestamp>/`. Defaults to `True`. |
| `--single_arm_mode` | Set to `True` only for a checkpoint and robot bridge configured for single-arm control. |
| `--no_state_obs_mode` | Set to `True` only if the robot bridge sends no-state observations expected by the policy. |

After warmup, the script waits for Enter, sends `"start"` to the bridge, and begins the control loop. Stop with `Ctrl-C`; the script sends `"stop"` and closes the websocket.

## Common Issues

### Missing `norm_stats.json` during deployment

Run normalization before training, then train again so the checkpoint contains `assets/<asset_id>/norm_stats.json`:

```bash
./scripts/compute_norm_stats.sh pi05_bi_vitac
./scripts/train.sh pi05_bi_vitac
```

### Checkpoint path mismatch

Training writes to `checkpoints/<config>/<exp_name>/<step>/`. Deployment expects the step directory itself, for example:

```bash
--ckpt-dir checkpoints/pi05_bi_vitac/my_experiment/35000
```
