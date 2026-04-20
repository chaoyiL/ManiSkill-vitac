"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.vb_single_arm as vb_single_arm
import openpi.policies.vb_single_vitac as vb_single_vitac
import openpi.policies.vb_policy_vis as vb_policy_vis
import openpi.policies.vb_policy_vitac as vb_policy_vitac
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 200
    # How often (in steps) to save checkpoints.
    save_interval: int = 5000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        # return (pathlib.Path(self.assets_base_dir) / self.name).resolve()
        # compute_norm_stats.py and train.py must stay consistent with the assets_dir
        return pathlib.Path(self.assets_base_dir).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# ============================================================================
# Data conversion pipeline config (replaces VB_task_config.yaml)
# ============================================================================
@dataclasses.dataclass
class DataConvertConfig:
    """Unified configuration for the data collection → LeRobot conversion pipeline.

    This replaces VB_task_config.yaml as the single source of truth.
    The pipeline (run_data_collection_pipeline_le.py) distributes these
    parameters to the appropriate steps:
      - Image processing params  → 01_crop_img.py
      - ArUco params             → 04/05 (via a generated temp yaml)
      - Dataset plan params      → 07 (via a generated temp yaml)
      - Conversion params        → convert_raw_to_lerobot_smooth.py
    """
    # ---- Task identity ----
    task_name: str = "default"
    task_type: str = "bimanual"       # "single" or "bimanual"
    single_hand_side: str = "left"    # only used when task_type == "single"

    # ---- Image processing (01_crop_img.py) ----
    # Target resolution after cropping; images are resized here so later steps
    # receive correctly-sized images without needing to know the source resolution.
    visual_out_res: tuple = (224, 224)
    tactile_out_res: tuple = (224, 224)
    # Apply a circular fisheye mask to visual images
    use_mask: bool = False
    fisheye_mask_radius: int = 390
    fisheye_mask_center: tuple | None = None   # None → auto-detect center
    fisheye_mask_fill_color: tuple = (0, 0, 0)

    # ---- ArUco detection (04_get_aruco_pos.py, 05_get_width.py) ----
    cam_intrinsic_json_path: str = "../../assets/intri_result/gopro_intrinsics_2_7k.json"
    aruco_dict: str = "DICT_4X4_50"
    marker_size_map: dict = dataclasses.field(
        default_factory=lambda: {0: 0.02, 1: 0.02, 2: 0.02, 3: 0.02}
    )
    left_aruco_left_id: int = 0
    left_aruco_right_id: int = 1
    right_aruco_left_id: int = 2
    right_aruco_right_id: int = 3
    aruco_max_workers: int = 4

    # ---- Dataset plan (07_generate_dataset_plan.py) ----
    min_episode_length: int = 10
    visual_cam_latency: float = 0.101
    pose_latency: float = 0.002
    use_tactile_img: bool = True

    # ---- Conversion (convert_raw_to_lerobot_smooth.py) ----
    output_repo_id: str | None = None  # defaults to "chaoyi/{task_name}"
    fps: int = 30
    language_instruction: list = dataclasses.field(
        default_factory=lambda: ["perform manipulation task"]
    )
    single_arm: bool = False
    smooth_sigma: float = 1.0
    # When True, state vector contains only gripper widths (no EEF pose deltas).
    no_state: bool = True
    # Inpaint ArUco tag regions (requires 04's aruco detection results).
    use_inpaint_tag: bool = True
    tag_scale: float = 1.3

# ============================================================================
# 数据集命名 - 在此修改可统一切换数据集
# ============================================================================
# 训练数据名称（用于 repo_id、asset_id、exp_name）
DATASET_TRAIN_NAME: str = "0118_data_1smooth"
# 原始数据任务名（数据采集管线：data/{task_name}/）
DATASET_RAW_TASK_NAME: str = "raw_0118_data"
# LeRobot 仓库命名空间（如 chaoyi）
DATASET_REPO_NAMESPACE: str = "chaoyi"

# Edit this instance to configure your experiment.
DATA_CONVERT_CONFIG = DataConvertConfig(
    task_name=DATASET_RAW_TASK_NAME,
    task_type="bimanual",
    single_hand_side="left",
    # Image processing
    visual_out_res=(224, 224),
    tactile_out_res=(224, 224),
    use_mask=False,
    fisheye_mask_radius=390,
    fisheye_mask_center=None,
    fisheye_mask_fill_color=(0, 0, 0),
    # ArUco
    cam_intrinsic_json_path="./assets/intri_result/gopro_intrinsics_2_7k.json",
    aruco_dict="DICT_4X4_50",
    marker_size_map={0: 0.02, 1: 0.02, 2: 0.02, 3: 0.02},
    left_aruco_left_id=0,
    left_aruco_right_id=1,
    right_aruco_left_id=2,
    right_aruco_right_id=3,
    aruco_max_workers=4,
    # Dataset plan
    min_episode_length=10,
    visual_cam_latency=0.101,
    pose_latency=0.002,
    use_tactile_img=True,
    # Conversion
    output_repo_id=None,
    fps=30,
    language_instruction=["perform manipulation task"],
    single_arm=False,
    smooth_sigma=1.0,
    no_state=False,
    use_inpaint_tag=True,
    tag_scale=1.3,
)


# Use `get_config` if you need to get a config by name in your code.
'''data config - 由顶部 DATASET_* 变量派生'''
data_name = DATASET_TRAIN_NAME
repo_id = f"{DATASET_REPO_NAMESPACE}/{data_name}"  # 需与 convert_zarr_to_lerobot.py 中 repo_id 一致
asset_id = data_name
assets_dir = "assets"

'''training config'''
fsdp_devices = 2
batch_size = fsdp_devices * 64
num_train_steps = 100000
# lr
warmup_steps = 1000
peak_lr = 2e-4
decay_steps = 100000
decay_lr = 2e-4

_CONFIGS = [
    TrainConfig(
        name="pi05_single",
        model=pi0_config.Pi0Config(
            state_dim=7,
            action_dim=10,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            pi05=True,
            image_keys=vb_single_arm.VIS_IMAGE_KEYS,
            ),
        data=SimpleDataConfig(
            repo_id=repo_id,
            assets=AssetsConfig(
                asset_id=asset_id,
                assets_dir=assets_dir,
            ),
            data_transforms=lambda model: _transforms.Group(
                inputs=[vb_single_arm.VBInputs(model_type=ModelType.PI05)],
                outputs=[vb_single_arm.VBOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,       
            decay_steps=decay_steps,
            decay_lr=decay_lr,      
        ),

        # Freeze filter for LoRA fine-tuning (freeze pre-trained weights, train LoRA adapters)
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            pi05=True
        ).get_freeze_filter(),
        # Disable EMA for LoRA fine-tuning
        ema_decay=None,
        # Can use larger batch size with LoRA (lower memory footprint)
        fsdp_devices=fsdp_devices,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        exp_name=data_name,
        # Load pre-trained weights for PaliGemma and action_expert, skip tactile components
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),

    TrainConfig(
        name="pi05_single_vitac",
        model=pi0_config.Pi0Config(
            state_dim=7,
            action_dim=10,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            pi05=True,
            image_keys=vb_single_vitac.VIS_IMAGE_KEYS,
            ),
        data=SimpleDataConfig(
            repo_id=repo_id,
            assets=AssetsConfig(
                asset_id=asset_id,
                assets_dir=assets_dir,
            ),
            data_transforms=lambda model: _transforms.Group(
                inputs=[vb_single_vitac.VBInputs(model_type=ModelType.PI05)],
                outputs=[vb_single_vitac.VBOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,       
            decay_steps=decay_steps,
            decay_lr=decay_lr,      
        ),

        # Freeze filter for LoRA fine-tuning (freeze pre-trained weights, train LoRA adapters)
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            pi05=True
        ).get_freeze_filter(),
        # Disable EMA for LoRA fine-tuning
        ema_decay=None,
        # Can use larger batch size with LoRA (lower memory footprint)
        fsdp_devices=fsdp_devices,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        exp_name=data_name,
        # Load pre-trained weights for PaliGemma and action_expert, skip tactile components
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),

    TrainConfig(
        name="pi05_bi",
        model=pi0_config.Pi0Config(
            state_dim=20,
            action_dim=20,
            action_horizon=8,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            pi05=True,
            image_keys=vb_policy_vis.VIS_IMAGE_KEYS,
            ),
        data=SimpleDataConfig(
            repo_id=repo_id,
            assets=AssetsConfig(
                asset_id = asset_id,
                assets_dir=assets_dir,
            ),
            data_transforms=lambda model: _transforms.Group(
                inputs=[vb_policy_vis.VBInputs(model_type=ModelType.PI05)],
                outputs=[vb_policy_vis.VBOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),

        # Freeze filter for LoRA fine-tuning (freeze pre-trained weights, train LoRA adapters)
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            pi05=True
        ).get_freeze_filter(),
        # Disable EMA for LoRA fine-tuning
        ema_decay=None,
        # Can use larger batch size with LoRA (lower memory footprint)
        fsdp_devices=fsdp_devices,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        exp_name=data_name,
        # Load pre-trained weights for PaliGemma and action_expert, skip tactile components
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),

    TrainConfig(
        name="pi05_bi_no_state",
        model=pi0_config.Pi0Config(
            state_dim=2, # state只有一个夹爪宽度
            action_dim=20,
            action_horizon=50,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            pi05=True,
            image_keys=vb_policy_vis.VIS_IMAGE_KEYS,
            ),
        data=SimpleDataConfig(
            repo_id=repo_id,
            assets=AssetsConfig(
                asset_id=asset_id,
                assets_dir=assets_dir,
            ),
            data_transforms=lambda model: _transforms.Group(
                inputs=[vb_policy_vis.VBInputs(model_type=ModelType.PI05)],
                outputs=[vb_policy_vis.VBOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),

        # Freeze filter for LoRA fine-tuning (freeze pre-trained weights, train LoRA adapters)
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            pi05=True
        ).get_freeze_filter(),
        # Disable EMA for LoRA fine-tuning
        ema_decay=None,
        # Can use larger batch size with LoRA (lower memory footprint)
        fsdp_devices=fsdp_devices,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        exp_name=data_name,
        # Load pre-trained weights for PaliGemma and action_expert, skip tactile components
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    ),

    TrainConfig(
        name="pi05_bi_vitac",
        model=pi0_config.Pi0Config(
            state_dim=20,
            action_dim=20,
            action_horizon=20,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            pi05=True,
            # Only visual images go through SigLIP; tactile goes through AnyTouch.
            image_keys=vb_policy_vitac.VIS_ONLY_IMAGE_KEYS,
            # AnyTouch tactile encoder (native JAX, LoRA fine-tuned).
            anytouch_config_path="policy/anytouch/CLIP-B-16",
            anytouch_checkpoint_path=None,   # None = auto-download from HuggingFace
            anytouch_variant="2frames",      # "2frames", "4frames", ...
            anytouch_num_frames=2,
            anytouch_stride=2,
            anytouch_pool_tokens=49, # None = no pool
            # LoRA fine-tuning: base weights frozen, only lora_a/lora_b trained.
            # Set to 0 to fully fine-tune the AnyTouch encoder instead.
            anytouch_lora_rank=8,
            anytouch_lora_alpha=8.0,
        ),
        data=SimpleDataConfig(
            repo_id=repo_id,
            assets=AssetsConfig(
                asset_id = asset_id,
                assets_dir=assets_dir,
            ),
            data_transforms=lambda model: _transforms.Group(
                inputs=[vb_policy_vitac.VBInputs(model_type=ModelType.PI05)],
                outputs=[vb_policy_vitac.VBOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),

        # Freeze filter: PaliGemma base + AnyTouch base frozen; LoRA adapters trained.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            pi05=True,
            anytouch_lora_rank=8,
        ).get_freeze_filter(),
        # Disable EMA for LoRA fine-tuning
        ema_decay=None,
        fsdp_devices=fsdp_devices,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        exp_name=data_name,
        # 1) Load PaliGemma/action-expert pretrained weights from gs://
        # 2) Load AnyTouch pretrained weights from local checkpoint (auto-downloaded)
        weight_loader=weight_loaders.CompositeWeightLoader(loaders=(
            weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            weight_loaders.AnyTouchWeightLoader(
                config_path="policy/anytouch/CLIP-B-16",
                variant="2frames",
                num_frames=2,
                stride=2,
            ),
        )),
    ),

]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
