#!/usr/bin/env python3
"""
将采集的原始数据（Raw Data）直接转换为 LeRobot 格式
省去中间生成 .zarr.zip 的步骤，一步到位完成转换

流程: Raw Data (demos + dataset_plan.pkl) → LeRobot 格式
替代原流程: Raw Data → 08_generate_replay_buffer → .zarr.zip → convert_zarr_to_lerobot_smooth → LeRobot
"""

import argparse
import sys
import os
import re
import csv
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# 路径设置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from utils.pose_util import pose_to_mat, mat_to_pose
from utils.config_utils import get_mandatory_config

# 尝试导入 training config（用于获取 state_dim, action_dim）
try:
    from openpi.training import config as training_config
    HAS_TRAINING_CONFIG = True
except ImportError:
    HAS_TRAINING_CONFIG = False

from utils.cv_util import (
    get_fisheye_image_transform,
    get_tactile_image_transform,
    inpaint_tag,
    draw_fisheye_mask,
)


# ============================================================================
# LeRobot 输出格式映射
# ============================================================================
IMAGE_KEYS = [
    ("camera0_rgb", "observation.images.camera0"),
    ("camera1_rgb", "observation.images.camera1"),
    ("camera0_left_tactile", "observation.images.tactile_left_0"),
    ("camera0_right_tactile", "observation.images.tactile_right_0"),
    ("camera1_left_tactile", "observation.images.tactile_left_1"),
    ("camera1_right_tactile", "observation.images.tactile_right_1"),
]


def _process_image(img, target_h=224, target_w=224):
    """处理图像为标准格式"""
    if img is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.dtype != np.uint8:
        if img.dtype in (np.float32, np.float64):
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    try:
        return cv2.resize(img, (target_w, target_h))
    except Exception:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)


def _smooth_pose_matrices(mats, sigma=2.0):
    """对 SE(3) 位姿矩阵序列进行高斯平滑"""
    if sigma <= 0 or len(mats) < 3:
        return mats
    n = len(mats)
    mat_arr = np.stack(mats)
    trans = mat_arr[:, :3, 3]
    trans_smooth = gaussian_filter1d(trans, sigma=sigma, axis=0)
    rot = mat_arr[:, :3, :3]
    rot_smooth = gaussian_filter1d(rot, sigma=sigma, axis=0)
    smoothed = []
    for t in range(n):
        U, _, Vt = np.linalg.svd(rot_smooth[t])
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = trans_smooth[t]
        smoothed.append(mat)
    return smoothed


def _load_image_files_from_folder(video_path: Path, hand: str, demo_dir: Path) -> list:
    """从图像文件夹加载有序图像文件列表"""
    csv_file = demo_dir / f"{hand}_hand_timestamps.csv"
    img_files = []
    csv_has_filename = False

    if csv_file.exists():
        with csv_file.open("r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            csv_has_filename = "filename" in fieldnames
            if csv_has_filename:
                for row in reader:
                    filename = row.get("filename", "")
                    if filename:
                        img_path = video_path / filename
                        if img_path.exists():
                            img_files.append(img_path)
                        else:
                            img_path_alt = video_path / Path(filename).name
                            if img_path_alt.exists():
                                img_files.append(img_path_alt)

    if not img_files:
        all_img_files = list(video_path.glob("*.jpg"))
        img_files = sorted(
            all_img_files,
            key=lambda p: int(re.search(r"(\d+)(?=\.jpg$)", p.name).group(1))
            if re.search(r"(\d+)(?=\.jpg$)", p.name)
            else p.name,
        )
    return img_files


def _get_hand_from_usage_name(usage_name: str) -> str:
    if usage_name.startswith("left_hand") or (usage_name.startswith("left_") and "visual" in usage_name):
        return "left"
    if usage_name.startswith("right_hand") or (usage_name.startswith("right_") and "visual" in usage_name):
        return "right"
    if "left" in usage_name.split("_")[:2]:
        return "left"
    if "right" in usage_name.split("_")[:2]:
        return "right"
    return "left"


def _get_hand_position_idx(usage_name: str, task: dict) -> int:
    if "left_hand" in usage_name:
        return 0
    if "right_hand" in usage_name:
        return 1
    if "visual" in usage_name:
        if usage_name in ("visual", "left_visual"):
            return 0
        if usage_name in ("right_hand_visual", "right_visual"):
            return 1
        return task.get("hand_position_idx", 0)
    return task.get("hand_position_idx", 0)


# ============================================================================
# 从原始数据加载单个 Episode
# ============================================================================
def load_episode_from_raw(
    plan_episode: dict,
    demos_path: Path,
    visual_out_res: tuple,
    tactile_out_res: tuple,
    visual_input_res: tuple,
    tactile_input_res: Optional[tuple],
    use_mask: bool,
    use_inpaint_tag: bool,
    use_tactile_img: bool,
    tag_scale: float,
    fisheye_mask_params: Optional[dict],
) -> dict:
    """
    从原始数据加载一个 episode 的所有数据（机器人状态 + 图像）
    返回格式与 zarr 中单 episode 切片一致，供后续 LeRobot 帧构建使用
    """
    grippers = plan_episode["grippers"]
    cameras = plan_episode["cameras"]

    video_start, video_end = cameras[0]["video_start_end"]
    n_frames = video_end - video_start

    episode_data = {}

    # ----- 机器人状态 -----
    for gripper_id, gripper in enumerate(grippers):
        eef_pose = gripper["quest_pose"]
        eef_pos = eef_pose[..., :3].astype(np.float32)
        eef_rot = eef_pose[..., 3:].astype(np.float32)
        gripper_widths = np.expand_dims(gripper["gripper_width"], axis=-1).astype(np.float32)

        robot_name = f"robot{gripper_id}"
        episode_data[f"{robot_name}_eef_pos"] = eef_pos[video_start:video_end]
        episode_data[f"{robot_name}_eef_rot_axis_angle"] = eef_rot[video_start:video_end]
        episode_data[f"{robot_name}_gripper_width"] = gripper_widths[video_start:video_end]

    iw, ih = visual_input_res
    visual_resize_tf = get_fisheye_image_transform(in_res=(iw, ih), out_res=visual_out_res)

    # ----- 图像 -----
    for cam_id, camera in enumerate(cameras):
        video_path_rel = camera["image_folder"]
        video_path = demos_path / video_path_rel
        if not video_path.is_dir():
            raise FileNotFoundError(f"Image folder not found: {video_path}")

        usage_name = camera.get("usage_name", f"camera{cam_id}")
        hand = _get_hand_from_usage_name(usage_name)

        img_files = _load_image_files_from_folder(video_path, hand, video_path.parent)
        if not img_files:
            raise FileNotFoundError(f"No images in {video_path}")

        # 只取本 episode 的帧范围
        frame_files = img_files[video_start:video_end]
        if len(frame_files) != n_frames:
            raise ValueError(
                f"Frame count mismatch: expected {n_frames}, got {len(frame_files)} "
                f"for {usage_name} in {video_path}"
            )

        img0 = cv2.imread(str(frame_files[0]))
        if img0 is None:
            raise RuntimeError(f"Failed to read {frame_files[0]}")
        actual_iw, actual_ih = img0.shape[1], img0.shape[0]

        if "visual" in usage_name:
            resize_tf = visual_resize_tf
        else:
            if use_tactile_img and tactile_input_res:
                resize_tf = get_tactile_image_transform(
                    in_res=(actual_iw, actual_ih), out_res=tactile_out_res
                )
            else:
                resize_tf = get_fisheye_image_transform(
                    in_res=(actual_iw, actual_ih), out_res=tactile_out_res
                )

        hand_position_idx = _get_hand_position_idx(usage_name, camera)

        if "visual" in usage_name:
            dataset_name = f"camera{hand_position_idx}_rgb"
        elif "tactile" in usage_name:
            if not use_tactile_img:
                continue
            parts = usage_name.split("_")
            if len(parts) >= 4 and parts[1] == "hand" and parts[3] == "tactile":
                sensor_side = parts[2]
                dataset_name = f"camera{hand_position_idx}_{sensor_side}_tactile"
            else:
                continue
        else:
            continue

        demo_dir = video_path.parent
        aruco_pkl_path = (
            demo_dir / "tag_detection_right.pkl"
            if ("right" in usage_name and "visual" in usage_name)
            else demo_dir / "tag_detection_left.pkl"
        )
        tag_detection_results = None
        if aruco_pkl_path.exists():
            try:
                tag_detection_results = pickle.load(open(aruco_pkl_path, "rb"))
            except Exception:
                pass

        last_detected_corners = {}
        images = []
        for local_idx, img_path in enumerate(frame_files):
            frame_idx = video_start + local_idx
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                images.append(np.zeros(visual_out_res + (3,), dtype=np.uint8))
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if "visual" in usage_name:
                if use_inpaint_tag and tag_detection_results is not None and frame_idx < len(tag_detection_results):
                    this_det = tag_detection_results[frame_idx]
                    current_frame_corners = []
                    if "tag_dict" in this_det and this_det["tag_dict"]:
                        for tag_id, tag_info in this_det["tag_dict"].items():
                            if "corners" in tag_info:
                                corners = tag_info["corners"]
                                current_frame_corners.append(corners)
                                last_detected_corners[tag_id] = corners
                    for tag_id, cached_corners in last_detected_corners.items():
                        tag_detected_in_current = tag_id in this_det.get("tag_dict", {})
                        if not tag_detected_in_current:
                            import random
                            jittered_corners = [
                                [c[0] + random.uniform(-2, 2), c[1] + random.uniform(-2, 2)]
                                for c in cached_corners
                            ]
                            current_frame_corners.append(jittered_corners)
                    for corners in current_frame_corners:
                        img = inpaint_tag(img, corners=corners, tag_scale=tag_scale)

                if use_mask and fisheye_mask_params:
                    radius = get_mandatory_config(fisheye_mask_params, "radius", "convert_raw_to_lerobot")
                    center = fisheye_mask_params.get("center")
                    fill_color = get_mandatory_config(fisheye_mask_params, "fill_color", "convert_raw_to_lerobot")
                    img = draw_fisheye_mask(img, radius=radius, center=center, fill_color=fill_color)

            img = resize_tf(img)
            images.append(img)

        episode_data[dataset_name] = np.stack(images, axis=0)

    return episode_data


# ============================================================================
# 构建 LeRobot 帧（从 episode_data）
# ============================================================================
def build_lerobot_frames(
    episode_data: dict,
    num_robots: int,
    state_dim: int,
    action_dim: int,
    language_instruction: list,
    smooth_sigma: float,
    no_state: bool,
    img_size: tuple,
) -> list:
    """从 episode_data 构建 LeRobot 帧列表"""
    ep_len = len(episode_data["robot0_eef_pos"])

    init_mats = []
    curr_mats_all = []
    for i in range(num_robots):
        pos = episode_data.get(f"robot{i}_eef_pos")
        rot = episode_data.get(f"robot{i}_eef_rot_axis_angle")
        if pos is not None and rot is not None:
            poses = np.concatenate([pos, rot], axis=-1)
            init_mats.append(pose_to_mat(poses[0]))
            curr_mats_all.append([pose_to_mat(poses[t]) for t in range(ep_len)])
        else:
            init_mats.append(np.eye(4))
            curr_mats_all.append([np.eye(4)] * ep_len)

    for i in range(num_robots):
        curr_mats_all[i] = _smooth_pose_matrices(curr_mats_all[i], sigma=smooth_sigma)

    frames = []
    for t in range(ep_len):
        f = {}
        f["task"] = language_instruction[min(t, len(language_instruction) - 1)]

        for src_key, feat_key in IMAGE_KEYS:
            if src_key in episode_data:
                img = _process_image(episode_data[src_key][t], img_size[0], img_size[1])
                f[feat_key] = img
            else:
                f[feat_key] = np.zeros(img_size, dtype=np.uint8)

        state = []
        if no_state:
            for i in range(num_robots):
                gk = f"robot{i}_gripper_width"
                g = episode_data.get(gk)
                state.append(float(g[t][0]) if g is not None else 0.0)
        else:
            for i in range(num_robots):
                c2w = curr_mats_all[i][t]
                c2i = np.linalg.inv(init_mats[i]) @ c2w
                state.extend(mat_to_pose(c2i))
                gk = f"robot{i}_gripper_width"
                g = episode_data.get(gk)
                state.append(float(g[t][0]) if g is not None else 0.0)
            if num_robots >= 2:
                state.extend(mat_to_pose(np.linalg.inv(curr_mats_all[1][t]) @ curr_mats_all[0][t]))

        assert len(state) == state_dim, f"state dim {len(state)} != {state_dim}"
        f["observation.state"] = np.asarray(state, dtype=np.float32)

        if t < ep_len - 1:
            act = []
            for i in range(num_robots):
                c2w = curr_mats_all[i][t]
                n2w = curr_mats_all[i][t + 1]
                n2c = np.linalg.inv(c2w) @ n2w
                pos3 = mat_to_pose(n2c)[:3]
                r1, r2 = n2c[:3, 0], n2c[:3, 1]
                act.extend(np.concatenate([pos3, r1, r2]))
                gk = f"robot{i}_gripper_width"
                g = episode_data.get(gk)
                act.extend(g[t] if g is not None else [0.0])
            f["actions"] = np.asarray(act[:action_dim], dtype=np.float32)
        else:
            f["actions"] = np.zeros(action_dim, dtype=np.float32)

        frames.append(f)

    return frames


def _compute_state_action_dims(num_robots: int, no_state: bool) -> tuple:
    """根据 num_robots 和 no_state 计算 state_dim 和 action_dim"""
    if no_state:
        state_dim = num_robots  # 每个机器人一个夹爪宽度
    else:
        # 每个机器人: 6(pose) + 1(gripper); 双臂时额外 +6(相对位姿)
        state_dim = num_robots * 7 + (6 if num_robots >= 2 else 0)
    # 每个机器人: 3(pos) + 3(r1) + 3(r2) + 1(gripper) = 10
    action_dim = num_robots * 10
    return state_dim, action_dim


# ============================================================================
# 主转换器
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="原始数据 → LeRobot 格式（跳过 zarr.zip 中间步骤）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- Task / output ----
    parser.add_argument("--task_name", type=str, required=True, help="任务名称（对应 data/{task_name}/）")
    parser.add_argument("--output_repo_id", type=str, default=None, help="输出 repo_id，默认 chaoyi/{task_name}")
    # ---- Conversion settings ----
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--language_instruction", type=str, nargs="+", default=None)
    parser.add_argument("--single_arm", action="store_true", help="单臂模式")
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--no_state", action="store_true", default=True, help="no_state 模式")
    parser.add_argument("--state_dim", type=int, default=None, help="覆盖 state_dim")
    parser.add_argument("--action_dim", type=int, default=None, help="覆盖 action_dim")
    # ---- Image processing (masking / inpainting) ----
    parser.add_argument("--use_tactile_img", action="store_true", default=False)
    parser.add_argument("--use_mask", action="store_true", default=False)
    parser.add_argument("--fisheye_radius", type=int, default=390)
    parser.add_argument("--fisheye_center", type=int, nargs=2, default=None, metavar=("X", "Y"))
    parser.add_argument("--fisheye_fill_color", type=int, nargs=3, default=[0, 0, 0], metavar=("R", "G", "B"))
    parser.add_argument("--use_inpaint_tag", action="store_true", default=False)
    parser.add_argument("--tag_scale", type=float, default=1.3)
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data"
    task_name = args.task_name
    input_path = data_dir / task_name
    demos_path = input_path / "demos"
    plan_path = input_path / "dataset_plan.pkl"

    if not plan_path.exists():
        print(f"错误: 未找到 {plan_path}，请先运行 07_generate_dataset_plan.py")
        sys.exit(1)

    plan = pickle.load(plan_path.open("rb"))
    print(f"加载 dataset_plan: {len(plan)} 个 episodes")

    # 图像分辨率：01_crop_img.py 已将图像 resize 到目标尺寸，此处直接使用标准尺寸
    visual_out_res = (224, 224)
    tactile_out_res = (224, 224)
    use_mask = args.use_mask
    use_inpaint_tag = args.use_inpaint_tag
    use_tactile_img = args.use_tactile_img
    tag_scale = args.tag_scale
    fisheye_mask_params = None
    if use_mask:
        fisheye_mask_params = {
            "radius": args.fisheye_radius,
            "center": args.fisheye_center,
            "fill_color": args.fisheye_fill_color,
        }

    # 01_crop_img.py 已将所有图像 resize 到 224x224，直接使用标准尺寸
    first_episode = plan[0]
    visual_input_res = visual_out_res
    tactile_input_res = tactile_out_res

    output_repo_id = args.output_repo_id or f"chaoyi/{task_name}"
    language_instruction = args.language_instruction or ["perform manipulation task"]

    n_grippers = len(first_episode["grippers"])
    n_cameras = len(first_episode["cameras"])
    if args.single_arm:
        num_robots, num_cameras = 1, 1
    else:
        num_robots, num_cameras = n_grippers, n_cameras

    # state/action 维度：优先根据实际数据自动计算，用户可用 --state_dim/--action_dim 覆盖
    auto_state_dim, auto_action_dim = _compute_state_action_dims(num_robots, args.no_state)
    if args.state_dim is not None and args.action_dim is not None:
        state_dim, action_dim = args.state_dim, args.action_dim
    else:
        state_dim, action_dim = auto_state_dim, auto_action_dim
        if args.state_dim is not None:
            state_dim = args.state_dim
        if args.action_dim is not None:
            action_dim = args.action_dim

    img_size = (224, 224, 3)  # LeRobot 标准

    print(f"\n{'='*60}")
    print(f"原始数据 → LeRobot 直接转换")
    print(f"输入: {input_path}")
    print(f"输出: {output_repo_id}")
    print(f"Episodes: {len(plan)}, Robots: {num_robots}, Cameras: {num_cameras}")
    print(f"state_dim={state_dim}, action_dim={action_dim}")
    print(f"{'='*60}\n")

    dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=args.fps,
        robot_type="single_arm" if args.single_arm else "bimanual",
        features={
            **{feat_key: {"dtype": "image", "shape": img_size, "names": ["height", "width", "channel"]}
               for _, feat_key in IMAGE_KEYS},
            "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": ["observation.state"]},
            "actions": {"dtype": "float32", "shape": (action_dim,), "names": ["actions"]},
        },
        use_videos=False,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_frames = 0
    for ep_idx, plan_episode in enumerate(tqdm(plan, desc="Converting")):
        try:
            episode_data = load_episode_from_raw(
                plan_episode=plan_episode,
                demos_path=demos_path,
                visual_out_res=visual_out_res,
                tactile_out_res=tactile_out_res,
                visual_input_res=visual_input_res,
                tactile_input_res=tactile_input_res,
                use_mask=use_mask,
                use_inpaint_tag=use_inpaint_tag,
                use_tactile_img=use_tactile_img,
                tag_scale=tag_scale,
                fisheye_mask_params=fisheye_mask_params,
            )
        except Exception as e:
            print(f"\n警告: 跳过 episode {ep_idx}: {e}")
            continue

        frames = build_lerobot_frames(
            episode_data=episode_data,
            num_robots=num_robots,
            state_dim=state_dim,
            action_dim=action_dim,
            language_instruction=language_instruction,
            smooth_sigma=args.smooth_sigma,
            no_state=args.no_state,
            img_size=img_size,
        )

        for f in frames:
            dataset.add_frame(f)
        dataset.save_episode()
        total_frames += len(frames)

    print(f"\n{'='*60}")
    print(f"转换完成: {len(plan)} episodes, {total_frames} frames")
    print(f"保存位置: {dataset.root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
