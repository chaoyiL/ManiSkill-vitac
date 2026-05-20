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

- **Most important:** Ubuntu version must be `>22.04`.
- Linux x86_64.
- Python `>=3.11,<3.12`.
- NVIDIA GPU with a CUDA 12-compatible driver. The project pins `jax[cuda12]` and PyTorch CUDA 12.8 wheels.
- `uv` for dependency management.
- Network access on first setup or first training run, because dependencies and pretrained weights are pulled from GitHub, Hugging Face, Google Cloud Storage, and the PyTorch wheel index.

## Environment Setup

Create a Python 3.11 conda environment for system libraries, then let `uv` manage the project virtual environment:

```bash
pip install uv
uv sync --locked
```

If you want to change the source, run:
```bash
mkdir -p ~/.config/uv
cat > ~/.config/uv/uv.toml << 'EOF'
[[index]]
url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/"
default = true
EOF
```
We use TUNA in the example, but you can also use other sources.

If you use private Hugging Face datasets or checkpoints, log in before training:

```bash
uv run hf auth login
```

## Download Hugging Face Datasets

To download a Hugging Face dataset manually before training, replace the placeholders below with the publisher namespace and dataset name:

```bash
uv run hf download <publisher-huggingface-username>/<dataset-name> \
  --repo-type dataset \
  --cache-dir ~/.cache/huggingface/dataset
```

The dataset snapshot is downloaded to a cache path like:

```text
~/.cache/huggingface/dataset/datasets--<publisher-huggingface-username>--<dataset-name>/snapshots/<snapshot-hash>
```

Link that snapshot into the LeRobot cache path expected by the configured `repo_id`:

```bash
ln -s ~/.cache/huggingface/dataset/datasets--<publisher-huggingface-username>--<dataset-name>/snapshots/<snapshot-hash> \
~/.cache/huggingface/lerobot/<DATASET_REPO_NAMESPACE>/<DATASET_TRAIN_NAME>
```

Our datasets are released under the Hugging Face account: https://huggingface.co/EricChen06. Here, `<publisher-huggingface-username>` should be set to `EricChen06`, and `<dataset-name>` should be replaced with the corresponding dataset name, such as `blue_clean_01`, `white_smash_01`, etc. After downloading a dataset, you can find `<snapshot-hash>` in the downloaded directory.

You can choose any namespace and training dataset name as you like, but please be sure the directory is corresponding with your `config.py`. For example, if `DATASET_REPO_NAMESPACE = "chaoyi"` and `DATASET_TRAIN_NAME = "0118_data_1smooth"`, the target directory should be:

```text
~/.cache/huggingface/lerobot/chaoyi/0118_data_1smooth
```

If Hugging Face cannot be reached from the machine, switch to the mirror endpoint before logging in or downloading:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=<your-token>
```

Some assets are downloaded automatically by the code on first use. Make sure the machine has network access before running the corresponding entrypoint:

| Downloaded asset | Source | Triggered by |
| --- | --- | --- |
| Pi0.5 base weights | `gs://openpi-assets/checkpoint/pi05_base/params` | `policy/scripts/train.py` through `./scripts/train.sh` when starting `pi05_bi` or `pi05_bi_vitac` training from scratch. |
| PaliGemma tokenizer | `gs://big_vision/paligemma_tokenizer.model` | `policy/scripts/compute_norm_stats.py`, `policy/scripts/train.py`, and `deploy_scripts/infer.py` when they create model/data transforms. |
| AnyTouch weights | `xxuan01/AnyTouch2-Model` | `policy/scripts/train.py` through `./scripts/train.sh pi05_bi_vitac` when loading the tactile encoder pretrained weights. |

## Configure Data and Policy

The main dataset, policy, and training knobs are grouped near the bottom of `policy/src/openpi/training/config.py`, before `_CONFIGS`. For a new run, edit these explicit parameters first.

### Dataset Parameters

| Parameter | Current value | Meaning |
| --- | --- | --- |
| `DATASET_TRAIN_NAME` | `"blue_clean_01"` | Training dataset name. It is reused as `data_name`, `asset_id`, and the default experiment name. |
| `DATASET_REPO_NAMESPACE` | `"chaoyi"` | LeRobot repository namespace. |
| `data_name` | `DATASET_TRAIN_NAME` | Local alias for the active training dataset name. |
| `repo_id` | `f"{DATASET_REPO_NAMESPACE}/{data_name}"` | Full LeRobot dataset id used by the data config. |
| `asset_id` | `data_name` | Asset id for normalization statistics. |
| `assets_dir` | `"assets"` | Root directory for dataset assets such as normalization statistics. |

With the default values, `repo_id` becomes `chaoyi/blue_clean_01`, and normalization statistics are expected under:

```text
assets/blue_clean_01/norm_stats.json
```

### Policy Parameters

| Parameter | Current value | Meaning |
| --- | --- | --- |
| `action_horizon` | `50` | Number of future action steps predicted by the policy. |
| `anytouch_pool_tokens` | `49` | Number of pooled AnyTouch tokens. This value must evenly divide `196`. |
| `anytouch_lora_rank` | `8` | LoRA rank used for AnyTouch fine-tuning. |
| `anytouch_lora_alpha` | `8.0` | LoRA scaling value for AnyTouch adapters. |

### Training Parameters

| Parameter | Current value | Meaning |
| --- | --- | --- |
| `fsdp_devices` | `1` | Number of devices used for FSDP sharding. |
| `batch_size` | `fsdp_devices * 64` | Global training batch size. |
| `num_train_steps` | `100000` | Number of optimizer steps. |
| `warmup_steps` | `1000` | Learning-rate warmup steps. |
| `peak_lr` | `2e-4` | Peak learning rate. |
| `decay_steps` | `100000` | Learning-rate decay steps. |
| `decay_lr` | `2e-4` | Final learning rate after decay. |

## Compute Normalization Stats

Run this once before training a new dataset/config pair:

```bash
./scripts/compute_norm_stats.sh <config_name>
```
`<config_name>` should be `pi05_bi` or `pi05_bi_vitac`.

The script writes:

```text
assets/<asset_id>/norm_stats.json
```

For the default constants, that is:

```text
assets/blue_clean_01/norm_stats.json
```

This file is important. Training loads it and saved checkpoints copy it into `assets/` inside each checkpoint step. Deployment then loads normalization stats from the checkpoint directory.

## Train Checkpoints

WandB logging is enabled by default. Log in before training:

```bash
wandb login
```

The simplest training command is:

```bash
./scripts/train.sh <config_name>
```

The wrapper runs:

```bash
uv run python policy/scripts/train.py "$CONFIG" --exp-name my_experiment --overwrite
```

so checkpoints are written under:

```text
checkpoint/<config_name>/<exp-name>/<step>/
```

Each saved step contains inference parameters, training state, and copied normalization assets:

```text
checkpoint/<config_name>/<exp-name>/<step>/
  assets/
  params/
  train_state/
```

For more control, call the training entrypoint directly:

```bash
uv run python policy/scripts/train.py <config_name> \
  --exp-name <any name you want> \
  --num-train-steps 100000 \
  --save-interval 5000 \
  --fsdp-devices 2 \
  --batch-size 128 \
  --overwrite
```

Resume an existing run with:

```bash
uv run python policy/scripts/train.py <config_name> \
  --exp-name <exp-name> \
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
  --config <config_name> \
  --ckpt-dir checkpoint/<config_name>/<exp_name>/<step> \
  --data_type vitac \
  --language_prompt "Anything you want to tell the robot." \
  --ip <ip we gave you> \
  --port 26421 \
  --token <your given token> \
  --save_obs True \
  --control_frequency 5 \
  --controller_frequency 80
```

If you want to deploy remotely, please unset all the proxies:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
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

After warmup, the script waits for Enter, sends `"start"` to the bridge, and begins the control loop. Stop with `Ctrl-C`; the script sends `"stop"` and closes the websocket.W

Speed Test:
```bash
# client
python deploy_scripts/network_speed_test.py inference \
  --ip 101.6.57.21 \
  --port 14214 \
  -n 100 \
  --jsonl infer_net.jsonl

# server
python deploy_scripts/network_speed_test.py robot \
  --host 0.0.0.0 \
  --port 26421 \
  -n 100 \
  --jsonl robot_net.jsonl
```