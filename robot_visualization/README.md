# robot_visualization

Offline 3D visualizer for recorded bimanual teleoperation and rollout episodes. Reads `.zarr.zip` archives produced by [robot_server/](../../robot_server/) and renders an interactive or headless view of end-effector trajectories, gripper geometry, wrist/scene camera streams, and tactile point clouds.

## Scope

The module covers two data paths:

1. **Native `.zarr.zip` logs** — episodes captured by the on-robot data collection pipeline, consumed directly via [replay_buffer.py](src/replay_buffer.py).
2. **LeRobot datasets** — HuggingFace `datasets`-backed folders, consumed through an adapter that bypasses the LeRobotDataset loader; see [src/visualize_lerobot_data.py](src/visualize_lerobot_data.py).

Both paths feed the same rendering core in [src/viz_vb_data.py](src/viz_vb_data.py) and its 3D extension [src/viz_3d_enhanced.py](src/viz_3d_enhanced.py).

## Layout

| Path | Role |
| --- | --- |
| [src/viz_3d_enhanced.py](src/viz_3d_enhanced.py) | Primary entrypoint. Renders a floor grid, coordinate axes, both arms, grippers (from STL), wrist cameras, tactile sensors, and RGB panels. Supports interactive playback and headless video export. |
| [src/viz_vb_data.py](src/viz_vb_data.py) | `CombinedVisualizer` base class: scene setup, camera pose, image composition, sensor key resolution. |
| [src/replay_buffer.py](src/replay_buffer.py) | Minimal replay-buffer reader for `.zarr.zip` archives. |
| [src/visualize_lerobot_data.py](src/visualize_lerobot_data.py) | Adapter that loads a LeRobot dataset folder via HuggingFace `datasets` and exposes it through the `ReplayBuffer` interface. |
| [src/imagecodecs_numcodecs.py](src/imagecodecs_numcodecs.py) | Codec registration for JPEG-XL / JPEG-2000 frames stored inside Zarr. |
| [src/meshes/](src/meshes/) | STL assets for the gripper geometry. |
| [scripts/record_all_episodes.ps1](scripts/record_all_episodes.ps1) | PowerShell batch driver that renders every episode of a given archive to `demo###.mp4`. |

## Installation

The visualizer uses a standard virtual environment (not `uv`):

```bash
cd user_client/robot_visualization
python3 -m venv venv && source venv/bin/activate
pip install numpy scipy opencv-python zarr trimesh pyrender imagecodecs
# Optional: LeRobot path
pip install datasets
```

On headless hosts, `pyrender` requires an EGL- or OSMesa-capable backend; set `PYOPENGL_PLATFORM=egl` (or `osmesa`) before launching.

## Usage

All commands below assume the working directory is the repository root, so that the `user_client.robot_visualization.src.*` package imports resolve.

### Interactive playback (`.zarr.zip`)

```bash
python user_client/robot_visualization/src/viz_3d_enhanced.py \
    path/to/your_data.zarr.zip
```

### Headless video export

```bash
python user_client/robot_visualization/src/viz_3d_enhanced.py \
    path/to/your_data.zarr.zip -r \
    --record_episode 1 --output_video demo.mp4 --fps 30
```

CLI flags (see [viz_3d_enhanced.py:703-714](src/viz_3d_enhanced.py#L703-L714)):

| Flag | Description |
| --- | --- |
| `zarr_path` (positional) | Path to the `.zarr.zip` archive. |
| `-r`, `--record` | Enable headless video export. |
| `-e`, `--record_episode` | Zero-based episode index to export. |
| `-o`, `--output_video` | Output MP4 path. |
| `--fps` | Output video frame rate (default `30`). |
| `-c`, `--continue_after_record` | After exporting, fall through to interactive mode. |

### LeRobot dataset

```bash
python user_client/robot_visualization/src/visualize_lerobot_data.py \
    --repo_id /path/to/lerobot_dataset_folder
```

The adapter expects `meta/info.json` and `meta/episodes.jsonl` to be present under the supplied folder.

### Batch rendering

On Windows, [scripts/record_all_episodes.ps1](scripts/record_all_episodes.ps1) iterates through every episode of an archive and writes `demo001.mp4` … `demoNNN.mp4` into a target directory. Adjust `-ZarrPath`, `-OutputDir`, `-EpisodeCount`, and `-Python` to match the local environment.

## Interactive Keybindings

| Key | Action |
| --- | --- |
| `A` / `D` | Step one frame backward / forward. |
| `W` / `S` | Previous / next episode. |
| `P` | Toggle auto-play. |
| `1` – `5` | Set playback speed to 0.25×, 0.5×, 1×, 2×, 5×. |
| `Q` | Quit. |

## Expected Data Keys

For each arm `robot{0,1}`, the visualizer looks up (with fallbacks, see [viz_vb_data.py:20-43](src/viz_vb_data.py#L20-L43)):

| Key pattern | Contents |
| --- | --- |
| `robot{i}_eef_pos` | End-effector pose (position + rotation). |
| `robot{i}_gripper_width` | Gripper aperture, used to animate the STL fingers. |
| `robot{i}_visual` / `camera{i}_rgb` | RGB frames for the RGB panel. |
| `robot{i}_{left,right}_tactile` | Tactile image streams. |
| `robot{i}_{left,right}_tactile_points` | Tactile point clouds, rendered in the 3D scene. |

Missing optional keys degrade gracefully: the affected panel or overlay is simply skipped.
