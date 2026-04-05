#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("ctrlflow")
@dataclass
class CtrlFlowConfig(PreTrainedConfig):

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    drop_n_last_frames: int = 7

    # Architecture / modeling.
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # SDE/ODE 类型选择.
    sde_type: str = "Mixed-W2KL"
    num_train_timesteps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # DS-SDE 专属超参（仅 sde_type="DS-SDE" 时生效）
    ds_gamma_min: float = 0.5
    ds_gamma_max: float = 3.0
    ds_g_min: float = 0.01
    ds_g_max: float = 1.0

    # W2-ODE 专属超参（仅 sde_type="W2-ODE" 时生效）
    w2_path_type: str = "linear"           # linear / cosine / sqrt / poly
    w2_path_poly_order: float = 2.0        # poly 路径阶数 p
    w2_lyapunov_reg_weight: float = 0.0   # §7.1 Lyapunov 正则项权重（0 → 纯速度场监督）
    w2_inference_noise_scale: float = 0.0  # §5.2.1 推理扩散项强度 β（0 → 纯 ODE）
    w2_sampling_method: str = "euler"      # euler / rk4

    # Mixed-W2KL 专属超参（仅 sde_type="Mixed-W2KL" 时生效）
    # ─────────────────────────────────────────────────────────────────
    # 前向路径复用 DS-SDE（共用 ds_gamma_min/max、ds_g_min/max），
    # 以下为混合 loss 和推理的额外控制参数。
    #
    # t_switch：W₂/KL 切换点（归一化时间 [0,1]）
    #   t > t_switch → W₂ 主导（高噪声，全局搬运）
    #   t ≤ t_switch → KL 主导（低噪声，精细对齐）
    mix_t_switch: float = 0.5
    #
    # mix_switch_sharpness：sigmoid 切换陡峭度 γ
    #   γ 大 → 接近硬切换，γ 小 → 平滑过渡
    #   推荐范围：4.0（平滑）~ 12.0（近似硬切）
    mix_switch_sharpness: float = 6.0
    #
    # mix_kl_sde_noise_scale：KL 阶段推理随机项强度（相对于 g²(t)）
    #   0.0 → KL 阶段也用纯 ODE（无随机扩散）
    #   1.0 → 使用完整 DS-SDE 扩散项（对应 Schrödinger Bridge 极限）
    mix_kl_sde_noise_scale: float = 1.0
    #
    # mix_kl_as_velocity_weight：KL 分量中 velocity 监督的比例 λ
    #   1.0 → KL 分量退化为纯 velocity 监督（两分量等价，用于对照实验）
    #   0.0 → KL 分量退化为纯 epsilon 监督（最大异构，强调 score 学习）
    #   0.5 → 默认：velocity 和 epsilon 各占一半
    mix_kl_as_velocity_weight: float = 0.5
    #
    # mix_lyapunov_reg_weight：§7.1 Lyapunov 正则项权重
    #   0.0 → 关闭正则项
    mix_lyapunov_reg_weight: float = 0.0

    # Inference
    num_inference_steps: int | None = None

    # Optimization
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample", "score"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )

        supported_sde_types = ["VP-SDE", "DS-SDE", "W2-ODE", "Mixed-W2KL"]
        if self.sde_type not in supported_sde_types:
            raise ValueError(
                f"`sde_type` must be one of {supported_sde_types}. Got {self.sde_type}."
            )

        # W2-ODE 专属参数校验
        if self.sde_type == "W2-ODE":
            supported_path_types = ["linear", "cosine", "sqrt", "poly"]
            if self.w2_path_type not in supported_path_types:
                raise ValueError(
                    f"`w2_path_type` must be one of {supported_path_types}. Got {self.w2_path_type}."
                )
            supported_sampling_methods = ["euler", "rk4"]
            if self.w2_sampling_method not in supported_sampling_methods:
                raise ValueError(
                    f"`w2_sampling_method` must be one of {supported_sampling_methods}. "
                    f"Got {self.w2_sampling_method}."
                )
            if self.w2_path_poly_order <= 0:
                raise ValueError(
                    f"`w2_path_poly_order` must be positive. Got {self.w2_path_poly_order}."
                )
            if self.w2_lyapunov_reg_weight < 0:
                raise ValueError(
                    f"`w2_lyapunov_reg_weight` must be >= 0. Got {self.w2_lyapunov_reg_weight}."
                )
            if self.w2_inference_noise_scale < 0:
                raise ValueError(
                    f"`w2_inference_noise_scale` must be >= 0. Got {self.w2_inference_noise_scale}."
                )

        # Mixed-W2KL 专属参数校验
        if self.sde_type == "Mixed-W2KL":
            if not (0.0 < self.mix_t_switch < 1.0):
                raise ValueError(
                    f"`mix_t_switch` must be in (0, 1). Got {self.mix_t_switch}."
                )
            if self.mix_switch_sharpness <= 0:
                raise ValueError(
                    f"`mix_switch_sharpness` must be positive. Got {self.mix_switch_sharpness}."
                )
            if self.mix_kl_sde_noise_scale < 0:
                raise ValueError(
                    f"`mix_kl_sde_noise_scale` must be >= 0. Got {self.mix_kl_sde_noise_scale}."
                )
            if not (0.0 <= self.mix_kl_as_velocity_weight <= 1.0):
                raise ValueError(
                    f"`mix_kl_as_velocity_weight` must be in [0, 1]. "
                    f"Got {self.mix_kl_as_velocity_weight}."
                )
            if self.mix_lyapunov_reg_weight < 0:
                raise ValueError(
                    f"`mix_lyapunov_reg_weight` must be >= 0. Got {self.mix_lyapunov_reg_weight}."
                )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(
                f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}."
            )
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                self.crop_shape = None
        if self.crop_shape is not None and (
            self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0
        ):
            raise ValueError(
                f"`crop_shape` must have positive dimensions. Got {self.crop_shape}."
            )

        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if (
                    self.crop_shape[0] > image_ft.shape[1]
                    or self.crop_shape[1] > image_ft.shape[2]
                ):
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
