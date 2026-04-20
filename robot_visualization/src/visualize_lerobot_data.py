#!/usr/bin/env python3
"""
LeRobot数据可视化适配器（修复版）

直接使用 HuggingFace datasets 加载，绕过 LeRobotDataset 的 bug

使用方法:
    python src/visualize_lerobot_data.py --repo_id ~/Downloads/_test_127
"""

import sys
import os
import numpy as np
import argparse
import json

# 直接使用 HuggingFace datasets
try:
    from datasets import load_dataset
except ImportError:
    print("错误: 请先安装 datasets")
    print("安装命令: pip install datasets")
    sys.exit(1)

from user_client.robot_visualization.src.replay_buffer import ReplayBuffer
from user_client.robot_visualization.src.viz_3d_enhanced import Enhanced3DVisualizer


class LeRobotReplayBufferAdapter:
    """LeRobot到ReplayBuffer的适配器（修复版）"""
    
    def __init__(self, repo_id: str):
        print(f"\n{'='*70}")
        print(f"加载LeRobot数据集: {repo_id}")
        print(f"{'='*70}\n")
        
        # 处理本地路径
        repo_id = self._resolve_path(repo_id)
        
        try:
            # 直接用 datasets 库加载，绕过 LeRobotDataset 的 bug
            print("  使用 HuggingFace datasets 加载...")
            dataset_dict = load_dataset(repo_id)
            
            if 'train' in dataset_dict:
                self.dataset = dataset_dict['train']
            else:
                # 如果没有 train split，使用第一个可用的
                self.dataset = dataset_dict[list(dataset_dict.keys())[0]]
            
            self.repo_id = repo_id
            
            # 加载 meta 信息
            meta_path = os.path.join(repo_id, 'meta', 'info.json')
            with open(meta_path, 'r') as f:
                self.meta_info = json.load(f)
            
            # 加载 episodes 信息
            episodes_path = os.path.join(repo_id, 'meta', 'episodes.jsonl')
            self.episodes_info = []
            with open(episodes_path, 'r') as f:
                for line in f:
                    self.episodes_info.append(json.loads(line))
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            raise
        
        print(f"✓ 数据集信息:")
        print(f"  - Episodes: {self.meta_info['total_episodes']}")
        print(f"  - 总帧数: {self.meta_info['total_frames']}")
        print(f"  - FPS: {self.meta_info['fps']}")
        print(f"  - Features: {len(self.meta_info['features'])}")
        
        self._create_buffer_structure()
    
    def _resolve_path(self, path: str) -> str:
        """解析路径"""
        if path.startswith('~'):
            path = os.path.expanduser(path)
        
        if os.path.exists(path):
            path = os.path.abspath(path)
            print(f"  使用本地路径: {path}")
        
        return path
    
    def _create_buffer_structure(self):
        """创建ReplayBuffer兼容的数据结构"""
        self.n_episodes = self.meta_info['total_episodes']
        self.n_steps = self.meta_info['total_frames']
        
        self._data = {}
        self._data['robot0_eef_pos'] = []
        self._data['robot0_eef_rot_axis_angle'] = []
        self._data['robot0_gripper_width'] = []
        self._data['robot1_eef_pos'] = []
        self._data['robot1_eef_rot_axis_angle'] = []
        self._data['robot1_gripper_width'] = []
        self._data['camera0_rgb'] = []
        self._data['camera1_rgb'] = []
        self._data['camera0_left_tactile'] = []
        self._data['camera0_right_tactile'] = []
        self._data['camera1_left_tactile'] = []
        self._data['camera1_right_tactile'] = []
        self._episode_ends = []
        
        print("\n正在转换数据格式...")
        self._load_all_data()
        print("✓ 数据转换完成\n")
    
    def _load_all_data(self):
        """加载所有数据到内存"""
        # 按 episode 组织数据
        episodes_dict = {}
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            ep_idx = int(item['episode_index'])
            
            if ep_idx not in episodes_dict:
                episodes_dict[ep_idx] = []
            episodes_dict[ep_idx].append(item)
        
        # 按 episode 顺序处理
        total_frames = 0
        for ep_idx in sorted(episodes_dict.keys()):
            episode_data = episodes_dict[ep_idx]
            print(f"  转换 Episode {ep_idx}: {len(episode_data)} 帧")
            
            for item in episode_data:
                try:
                    state = np.array(item['observation.state'], dtype=np.float32)
                    
                    # Robot 0 (左臂)
                    self._data['robot0_eef_pos'].append(state[0:3].copy())
                    self._data['robot0_eef_rot_axis_angle'].append(state[3:6].copy())
                    self._data['robot0_gripper_width'].append(np.array([state[6]]))
                    
                    # Robot 1 (右臂)
                    self._data['robot1_eef_pos'].append(state[7:10].copy())
                    self._data['robot1_eef_rot_axis_angle'].append(state[10:13].copy())
                    self._data['robot1_gripper_width'].append(np.array([state[13]]))
                    
                    # 图像
                    for key, storage_key in [
                        ('observation.images.camera0', 'camera0_rgb'),
                        ('observation.images.camera1', 'camera1_rgb'),
                    ]:
                        if key in item:
                            img = np.array(item[key], dtype=np.uint8)
                            self._data[storage_key].append(img)
                        else:
                            self._data[storage_key].append(np.zeros((224, 224, 3), dtype=np.uint8))
                    
                    # 触觉
                    for robot_id in [0, 1]:
                        for side in ['left', 'right']:
                            key = f'observation.images.tactile_{side}_{robot_id}'
                            storage_key = f'camera{robot_id}_{side}_tactile'
                            if key in item:
                                img = np.array(item[key], dtype=np.uint8)
                                self._data[storage_key].append(img)
                            else:
                                self._data[storage_key].append(np.zeros((224, 224, 3), dtype=np.uint8))
                    
                    total_frames += 1
                    
                except Exception as e:
                    print(f"    警告: 帧加载失败: {e}")
                    continue
            
            self._episode_ends.append(total_frames)
        
        # 转换为numpy数组
        print("\n  转换为numpy数组...")
        for key in self._data:
            if len(self._data[key]) > 0:
                self._data[key] = np.array(self._data[key])
                print(f"    {key}: {self._data[key].shape}")
    
    def get_episode_slice(self, episode_idx: int):
        if episode_idx == 0:
            start = 0
        else:
            start = self._episode_ends[episode_idx - 1]
        end = self._episode_ends[episode_idx]
        return slice(start, end)
    
    def keys(self):
        return self._data.keys()
    
    def __getitem__(self, key):
        if key not in self._data:
            raise KeyError(f"Key '{key}' not found")
        return self._data[key]


def main():
    parser = argparse.ArgumentParser(
        description='LeRobot数据可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 可视化数据集
  python src/visualize_lerobot_data.py --repo_id ~/Downloads/_test_127
  
  # 录制视频
  python src/visualize_lerobot_data.py --repo_id ~/Downloads/_test_127 \\
      --record --record_episode 0 --output demo.mp4
        """
    )
    
    parser.add_argument('--repo_id', type=str, required=True, 
                       help='数据集路径或repo_id')
    parser.add_argument('--episode', type=int, default=0, 
                       help='起始episode')
    parser.add_argument('--record', '-r', action='store_true', 
                       help='录制视频')
    parser.add_argument('--record_episode', '-e', type=int, default=0, 
                       help='录制的episode')
    parser.add_argument('--output_video', '-o', type=str, default=None, 
                       help='输出视频文件名')
    parser.add_argument('--fps', type=int, default=30, 
                       help='视频FPS')
    parser.add_argument('--continue_after_record', '-c', action='store_true', 
                       help='录制后继续')
    
    args = parser.parse_args()
    
    try:
        adapter = LeRobotReplayBufferAdapter(args.repo_id)
        
        if args.record and args.output_video is None:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_repo_id = os.path.basename(args.repo_id)
            args.output_video = f"lerobot_ep{args.record_episode}_{safe_repo_id}_{ts}.mp4"
        
        print(f"{'='*70}")
        print("启动可视化器...")
        print(f"{'='*70}\n")
        
        episodes = np.arange(adapter.n_episodes)
        
        visualizer = Enhanced3DVisualizer(
            replay_buffer=adapter,
            episodes=episodes,
            record_mode=args.record,
            record_episode=args.record_episode,
            output_video=args.output_video,
            record_fps=args.fps,
            continue_after_record=args.continue_after_record
        )
        
        print("\n✓ 完成")
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()