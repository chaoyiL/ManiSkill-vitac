"""检查训练数据集的脚本。

此脚本用于检查训练数据集的基本信息，包括：
- 数据集基本信息（episode数量、总帧数等）
- 数据格式和字段
- 数据统计信息（状态、动作的范围和分布）
- 可视化样本数据

用法:
    # 检查LeRobot数据集
    uv run scripts/check_dataset.py --config-name your_config_name

    # 检查LeRobot数据集（指定数据集repo_id）
    uv run scripts/check_dataset.py --repo-id your_username/your_dataset_name

    # 检查RLDS数据集（DROID）
    uv run scripts/check_dataset.py --config-name pi05_full_droid_finetune
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tyro

try:
    from matplotlib import pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_lerobot_dataset(repo_id: str, num_samples: int = 10):
    """检查LeRobot数据集的基本信息。

    Args:
        repo_id: LeRobot数据集ID
        num_samples: 要检查的样本数量
    """
    logger.info(f"检查LeRobot数据集: {repo_id}")

    try:
        # 加载数据集元数据
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        logger.info(f"数据集元数据:")
        logger.info(f"  - FPS: {dataset_meta.fps}")
        logger.info(f"  - 机器人类型: {dataset_meta.robot_type}")
        logger.info(f"  - 任务列表: {dataset_meta.tasks[:10] if len(dataset_meta.tasks) > 10 else dataset_meta.tasks}")

        # 加载数据集
        dataset = lerobot_dataset.LeRobotDataset(repo_id)
        logger.info(f"\n数据集基本信息:")
        logger.info(f"  - 总episode数: {len(dataset)}")
        logger.info(f"  - 总帧数: {dataset.num_frames}")

        # 检查第一个episode
        if len(dataset) > 0:
            first_episode = dataset[0]
            logger.info(f"\n第一个episode的字段:")
            for key, value in first_episode.items():
                if isinstance(value, np.ndarray):
                    logger.info(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, (str, int, float)):
                    logger.info(f"  - {key}: {type(value).__name__} = {value}")
                else:
                    logger.info(f"  - {key}: {type(value).__name__}")

            # 检查多个样本
            logger.info(f"\n检查前{num_samples}个样本:")
            state_values = []
            action_values = []
            image_shapes = []

            for i in range(min(num_samples, len(dataset))):
                sample = dataset[i]
                if "state" in sample:
                    state_values.append(sample["state"])
                if "actions" in sample:
                    action_values.append(sample["actions"])
                # 检查图像字段
                for key in sample.keys():
                    if "image" in key.lower() and isinstance(sample[key], np.ndarray):
                        image_shapes.append((key, sample[key].shape))

            # 统计信息
            if state_values:
                states = np.array(state_values)
                logger.info(f"\n状态统计:")
                logger.info(f"  - 形状: {states.shape}")
                logger.info(f"  - 均值: {np.mean(states, axis=0)}")
                logger.info(f"  - 标准差: {np.std(states, axis=0)}")
                logger.info(f"  - 最小值: {np.min(states, axis=0)}")
                logger.info(f"  - 最大值: {np.max(states, axis=0)}")

            if action_values:
                actions = np.array(action_values)
                logger.info(f"\n动作统计:")
                logger.info(f"  - 形状: {actions.shape}")
                logger.info(f"  - 均值: {np.mean(actions, axis=0)}")
                logger.info(f"  - 标准差: {np.std(actions, axis=0)}")
                logger.info(f"  - 最小值: {np.min(actions, axis=0)}")
                logger.info(f"  - 最大值: {np.max(actions, axis=0)}")

            if image_shapes:
                logger.info(f"\n图像字段:")
                for key, shape in set(image_shapes):
                    logger.info(f"  - {key}: {shape}")

        # 检查数据集目录结构
        dataset_path = Path(lerobot_dataset.HF_LEROBOT_HOME) / repo_id
        if dataset_path.exists():
            logger.info(f"\n数据集文件结构:")
            logger.info(f"  - 路径: {dataset_path}")
            if (dataset_path / "info.json").exists():
                with open(dataset_path / "info.json") as f:
                    info = json.load(f)
                    logger.info(f"  - info.json内容:")
                    logger.info(f"    {json.dumps(info, indent=2, ensure_ascii=False)}")
            episodes_dir = dataset_path / "episodes"
            if episodes_dir.exists():
                episode_files = list(episodes_dir.glob("*.h5"))
                logger.info(f"  - Episode文件数: {len(episode_files)}")

    except Exception as e:
        logger.error(f"检查数据集时出错: {e}", exc_info=True)
        raise


def check_dataset_with_config(config_name: str, num_samples: int = 10):
    """使用训练配置检查数据集。

    Args:
        config_name: 训练配置名称
        num_samples: 要检查的样本数量
    """
    logger.info(f"使用配置 '{config_name}' 检查数据集")

    try:
        config = _config.get_config(config_name)
        data_config = config.data.create(config.assets_dirs, config.model)

        logger.info(f"\n数据配置:")
        logger.info(f"  - repo_id: {data_config.repo_id}")
        logger.info(f"  - asset_id: {data_config.asset_id}")
        logger.info(f"  - rlds_data_dir: {data_config.rlds_data_dir}")
        logger.info(f"  - prompt_from_task: {data_config.prompt_from_task}")

        if data_config.rlds_data_dir is not None:
            logger.info("\n检测到RLDS数据集（DROID）")
            logger.info("RLDS数据集检查需要实际加载数据，这可能需要一些时间...")
            # 创建数据加载器来检查数据
            data_loader = _data_loader.create_data_loader(
                config,
                shuffle=False,
                num_batches=min(5, num_samples // config.batch_size + 1),
                skip_norm_stats=True,
            )
            logger.info("成功创建RLDS数据加载器")
            # 检查第一个batch
            batch_iter = iter(data_loader)
            first_batch = next(batch_iter)
            observation, actions = first_batch
            logger.info(f"\n第一个batch的信息:")
            logger.info(f"  - Observation类型: {type(observation)}")
            logger.info(f"  - Actions形状: {actions.shape if hasattr(actions, 'shape') else type(actions)}")
            if hasattr(observation, "images"):
                logger.info(f"  - 图像字段: {list(observation.images.keys())}")
            if hasattr(observation, "state"):
                logger.info(f"  - 状态形状: {observation.state.shape if hasattr(observation.state, 'shape') else type(observation.state)}")

        elif data_config.repo_id is not None:
            check_lerobot_dataset(data_config.repo_id, num_samples)
        else:
            logger.warning("配置中没有指定数据集（repo_id或rlds_data_dir）")

    except Exception as e:
        logger.error(f"检查数据集时出错: {e}", exc_info=True)
        raise


def visualize_samples(repo_id: str, num_samples: int = 5, output_dir: Path | None = None):
    """可视化数据集样本。

    Args:
        repo_id: LeRobot数据集ID
        num_samples: 要可视化的样本数量
        output_dir: 输出目录（如果为None，则只显示不保存）
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib未安装，无法可视化样本。请运行: uv pip install matplotlib")
        return

    try:
        dataset = lerobot_dataset.LeRobotDataset(repo_id)
        num_samples = min(num_samples, len(dataset))

        for i in range(num_samples):
            sample = dataset[i]
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # 显示图像（如果有）
            image_keys = [k for k in sample.keys() if "image" in k.lower()]
            if image_keys:
                img = sample[image_keys[0]]
                if isinstance(img, np.ndarray) and len(img.shape) == 3:
                    axes[0].imshow(img)
                    axes[0].set_title(f"样本 {i}: {image_keys[0]}")
                    axes[0].axis("off")

            # 显示状态/动作（如果有）
            if "state" in sample:
                state = sample["state"]
                if isinstance(state, np.ndarray):
                    axes[1].bar(range(len(state)), state)
                    axes[1].set_title(f"状态 (样本 {i})")
                    axes[1].set_xlabel("维度")
                    axes[1].set_ylabel("值")

            plt.tight_layout()
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_dir / f"sample_{i}.png")
                plt.close()
            else:
                plt.show()

    except Exception as e:
        logger.error(f"可视化样本时出错: {e}", exc_info=True)


def main(
    config_name: str | None = None,
    repo_id: str | None = None,
    num_samples: int = 10,
    visualize: bool = False,
    output_dir: str | None = None,
):
    """主函数。

    Args:
        config_name: 训练配置名称（如果提供，将使用配置中的数据设置）
        repo_id: LeRobot数据集ID（如果提供且config_name为None，将直接检查此数据集）
        num_samples: 要检查的样本数量
        visualize: 是否可视化样本
        output_dir: 可视化输出目录（如果提供，图像将保存到此目录）
    """
    if config_name:
        check_dataset_with_config(config_name, num_samples)
        if visualize and repo_id:
            visualize_samples(repo_id, num_samples, Path(output_dir) if output_dir else None)
    elif repo_id:
        check_lerobot_dataset(repo_id, num_samples)
        if visualize:
            visualize_samples(repo_id, num_samples, Path(output_dir) if output_dir else None)
    else:
        raise ValueError("必须提供 config_name 或 repo_id")


if __name__ == "__main__":
    tyro.cli(main)
