from math import ceil
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from datetime import datetime
import threading
from queue import Queue
import json

import click
import cv2
import jax
import numpy as np


import plotly.graph_objects as go
from openpi.training import config as _config
from openpi.policies import policy_config
from utils.precise_sleep import precise_wait
from client.interface_client import InterfaceClient

class ObsSaver:
    """异步保存observation数据，不影响eval过程"""

    def __init__(self, save_dir: str, data_type: str):
        """
        Args:
            save_dir: 保存目录
            data_type: 数据类型 ('vision' 或 'vitac')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / f"eval_obs_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data_type = data_type

        # 使用队列进行异步保存
        self.save_queue = Queue(maxsize=100)  # 限制队列大小，避免内存溢出
        self.save_thread = None
        self.running = False
        self.step_count = 0

        print(f"[ObsSaver] Initialized. Save directory: {self.save_dir}")

    def start(self):
        """启动保存线程"""
        self.running = True
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        print(f"[ObsSaver] Started saving thread")

    def stop(self):
        """停止保存线程"""
        self.running = False
        if self.save_thread:
            self.save_thread.join(timeout=5.0)
        print(f"[ObsSaver] Stopped. Total steps saved: {self.step_count}")

    def save_obs(self, obs: dict, step_idx: int = None):
        """
        将obs添加到保存队列（非阻塞）

        Args:
            obs: observation字典
            step_idx: 步骤索引（如果为None，使用内部计数器）
        """
        if not self.running:
            return

        if step_idx is None:
            step_idx = self.step_count
            self.step_count += 1

        try:
            # 非阻塞添加，如果队列满了就跳过
            self.save_queue.put_nowait((step_idx, obs))
        except:
            # 队列满了，跳过这次保存
            pass

    def _save_worker(self):
        """后台保存线程"""
        while self.running:
            try:
                # 从队列获取数据，超时1秒
                step_idx, obs = self.save_queue.get(timeout=1.0)
                self._save_single_obs(step_idx, obs)
                self.save_queue.task_done()
            except:
                continue

    def _numpy_to_json_serializable(self, obj):
        """将numpy数组转换为JSON可序列化的格式"""
        if isinstance(obj, np.ndarray):
            # 转换为列表
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            # numpy标量转换为Python原生类型
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._numpy_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._numpy_to_json_serializable(item) for item in obj]
        else:
            return obj

    def _save_single_obs(self, step_idx: int, obs: dict):
        """保存单个observation - 保存所有obs数据"""
        step_dir = self.save_dir / f"step_{step_idx:06d}"
        step_dir.mkdir(exist_ok=True)

        # 保存时间戳为JSON
        if 'timestamp' in obs:
            timestamp_data = self._numpy_to_json_serializable(obs['timestamp'])
            with open(step_dir / "timestamp.json", 'w') as f:
                json.dump(timestamp_data, f, indent=2)

        # 遍历所有obs数据并保存
        for key, value in obs.items():
            if key == 'timestamp':
                continue

            if isinstance(value, np.ndarray) and len(value.shape) >= 3:
                # 检查是否是图像数据（camera, rgb, tactile相关）
                if 'camera' in key or 'rgb' in key or 'tactile' in key:
                    # 保存为图像文件（取最后一帧）
                    if len(value.shape) == 4:  # (T, H, W, C)
                        img = value[-1]  # 取最后一帧
                    elif len(value.shape) == 3:  # (H, W, C)
                        img = value
                    else:
                        # 不是标准图像格式，保存为JSON
                        json_data = self._numpy_to_json_serializable(value)
                        with open(step_dir / f"{key}.json", 'w') as f:
                            json.dump(json_data, f, indent=2)
                        continue

                    # 转换数据类型和格式
                    if img.dtype == np.float32:
                        img = (img * 255).astype(np.uint8)
                    elif img.max() <= 1.0 and img.dtype in [np.float32, np.float64]:
                        img = (img * 255).astype(np.uint8)

                    # RGB转BGR用于cv2保存
                    # if len(img.shape) == 3 and img.shape[-1] == 3:
                    #     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    # else:
                    img_path = step_dir / f"{key}.jpg"
                    cv2.imwrite(str(img_path), img)
                else:
                    # 非图像数据，保存为JSON（包括robot pose, gripper width等）
                    json_data = self._numpy_to_json_serializable(value)
                    with open(step_dir / f"{key}.json", 'w') as f:
                        json.dump(json_data, f, indent=2)
            elif isinstance(value, np.ndarray):
                # 低维数据（robot pose, gripper width等），保存为JSON
                json_data = self._numpy_to_json_serializable(value)
                with open(step_dir / f"{key}.json", 'w') as f:
                    json.dump(json_data, f, indent=2)
            else:
                # 其他类型数据，保存为JSON
                json_data = self._numpy_to_json_serializable(value)
                with open(step_dir / f"{key}.json", 'w') as f:
                    json.dump(json_data, f, indent=2)


@click.command()
@click.option('--config', '-c', default=f'pi05_bi_vitac', help='Config name for policy.')
@click.option('--ckpt-dir', '-i', default='/home/rvsa/codehub/ManiSkill/user_client/checkpoint/pi05_bi_vitac/my_experiment/35000', help='Path to checkpoint directory')
@click.option('--data_type', '-dt', default='vitac',help='vision, vitac, vitacpc')
@click.option('--language_prompt', '-lp', default='Open the red pot, pick up the blue cylinder on the table and place it into the pot.', help='Language prompt')

@click.option('--save_obs', '-so', default=False, help='Save observation data for verification (saves every step)')
@click.option('--control_frequency', '-f', default=5, type=float, help="Control frequency in Hz.")
@click.option('--controller_frequency', '-cf', default=80, type=float, help="Controller frequency in Hz.")

@click.option('--single_arm_mode', default=False, help='single arm mode')
@click.option('--no_state_obs_mode', default=False, help='no state obs mode')

@click.option('--ip', default='127.0.0.1', help='which ip the messages are sent to')
@click.option('--port', default='8000', help='port')
@click.option('--token', default='111', help='your test token')

def main(config,
    ckpt_dir,
    data_type,
    language_prompt,
    save_obs,
    control_frequency,
    controller_frequency,
    single_arm_mode,
    no_state_obs_mode,
    ip,
    port,
    token
    ):
    
    train_config = _config.get_config(config)
    checkpoint_dir = Path(ckpt_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (Path(ROOT_DIR) / checkpoint_dir).resolve()
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir)
    # steps_per_inference = 1 #使用temporal ensembling后，steps_per_inference = 1
    action_horizon = int(train_config.model.action_horizon)
    steps_per_inference = action_horizon

    client = InterfaceClient(ip, port, token=token)

    config_dict = {
        "data_type" : data_type,
        "language_prompt" : language_prompt,
        "control_frequency" : control_frequency,
        "controller_frequency" : controller_frequency,
        "single_arm_mode" : single_arm_mode,
        "no_state_obs_mode" : no_state_obs_mode,
        "steps_per_inference" : steps_per_inference,
        "action_horizon" : action_horizon,
    }

    client.send_config(config_dict)

    arm_num = None
    if single_arm_mode:
        arm_num = 1
    else:
        arm_num = 2

    print("steps_per_inference:", steps_per_inference)
    print("jax backend:", jax.default_backend())
    print("jax devices:", jax.devices())

    with SharedMemoryManager() as shm_manager:

        cv2.setNumThreads(2)

        print("Warming up policy inference")

        policy.reset()
        obs_seq, obs_dict = client.recv_obs()

        # calculate raw action and send it to robot
        result = policy.infer(obs_dict)
        raw_action = result['actions']
        assert raw_action.shape[-1] == 10 * arm_num

        print('################################## Ready! ##################################')
        input("press enter to start...")
        client.send_state("start")

        try:
            policy.reset()
            last_status_log_time = time.monotonic()
            iter_idx = 0

            while True:
                obs_seq, obs_dict = client.recv_obs()

                infer_start = time.monotonic()
                result = policy.infer(obs_dict)
                infer_elapsed = time.monotonic() - infer_start

                client.send_action(result['actions'], obs_seq=obs_seq)

                now = time.monotonic()
                if now - last_status_log_time >= 2.0:
                    print(
                        f"[main] iter={iter_idx} obs_seq={obs_seq} "
                        f"infer_time_ms={infer_elapsed * 1000.0:.1f}"
                    )
                    last_status_log_time = now

                iter_idx += 1

        except KeyboardInterrupt:
            print("Interrupted!")
            client.send_state("stop")
        finally:
            client.close()

if __name__ == '__main__':
    main()
