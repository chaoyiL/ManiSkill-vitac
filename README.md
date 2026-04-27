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
conda install -c conda-forge ffmpeg libstdcxx-ng -y

python -m pip install -U uv
uv sync --locked
```

Keep the conda environment active when using the shell wrappers. They add `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` so PyAV and related libraries load the conda FFmpeg/libstdc++ builds.

If you use private Hugging Face datasets or checkpoints, log in before training:

```bash
hf auth login
```

Weights used by the default configs are downloaded automatically:

- Pi0.5 base weights from `gs://openpi-assets/checkpoints/pi05_base/params`.
- PaliGemma tokenizer from `gs://big_vision/paligemma_tokenizer.model`.
- AnyTouch weights from `xxuan01/AnyTouch2-Model` when using `pi05_bi_vitac`.

## Configure Data and Policy

The active dataset is controlled near the bottom of `policy/src/openpi/training/config.py`:

```python
DATASET_TRAIN_NAME = "0118_data_1smooth"
DATASET_RAW_TASK_NAME = "raw_0118_data"
DATASET_REPO_NAMESPACE = "chaoyi"
```

These values produce:

- LeRobot `repo_id`: `chaoyi/0118_data_1smooth`
- normalization asset id: `0118_data_1smooth`
- default checkpoint experiment name in the config: `0118_data_1smooth`

Available training configs in this repository:

| Config | Use case | Notes |
| --- | --- | --- |
| `pi05_bi` | Bimanual vision-only policy. | Uses two RGB camera streams, `state_dim=20`, `action_dim=20`, `action_horizon=8`. |
| `pi05_bi_vitac` | Bimanual visual-tactile policy. | Uses two RGB streams plus four tactile streams through AnyTouch, `state_dim=20`, `action_dim=20`, `action_horizon=20`. |

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

You can inspect a dataset through the configured `repo_id`:

```bash
uv run python policy/scripts/check_dataset.py --config-name pi05_bi_vitac --num-samples 10
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
  --token 111 \
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

### `CXXABI_1.3.15` or `libavformat.so` import errors

Install FFmpeg and `libstdcxx-ng` in conda, keep the conda environment active, and export the library path if needed:

```bash
conda install -c conda-forge 'ffmpeg=7' libstdcxx-ng -y
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

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
