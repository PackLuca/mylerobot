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


@PreTrainedConfig.register_subclass("ctrlflow1")
@dataclass
class CTRLFlow1Config(PreTrainedConfig):
    """Configuration class for CTRLFlow1Policy.

    Extends DiffusionConfig with Wasserstein (WGAN-GP) stability regularization
    based on the CTRL-Flow framework (Lyapunov-stable generative modeling).

    The UNet backbone and all diffusion inference parameters are identical to
    DiffusionPolicy. The only additions are the Kantorovich potential network φ
    and the associated training hyperparameters.

    New args vs DiffusionConfig:
        phi_hidden_dim: Hidden dimension of the Kantorovich potential network φ.
            φ is a lightweight MLP that operates on flattened action trajectories.
            Larger values improve the quality of the Wasserstein estimate at the
            cost of slightly more memory during training (φ is discarded at inference).
        phi_n_layers: Number of hidden layers in the φ network.
        phi_lr: Learning rate for the φ network optimizer (typically ~10x the UNet lr).
        phi_n_inner_steps: How many gradient steps to take on φ per UNet step.
            Following the WGAN convention, 5 is a safe default.
        phi_gp_lambda: Gradient penalty coefficient for the 1-Lipschitz constraint on φ.
        wasserstein_lambda: Weight of the Wasserstein regularization term relative to
            the base diffusion MSE loss  (loss = loss_diffusion + λ * loss_wasserstein).
        wasserstein_warmup_steps: Number of training steps over which the Wasserstein
            weight is linearly ramped from 0 to `wasserstein_lambda`. Setting this to
            ~30% of total training steps gives a stable two-phase schedule as described
            in CTRL-Flow Section 5.3.
    """

    # ── Inputs / output structure (unchanged from DiffusionConfig) ──────────
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

    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # ── Vision backbone (unchanged) ──────────────────────────────────────────
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # ── UNet (unchanged) ─────────────────────────────────────────────────────
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # ── Noise scheduler (unchanged) ──────────────────────────────────────────
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # ── Inference (unchanged) ────────────────────────────────────────────────
    num_inference_steps: int | None = None

    # ── Compilation (unchanged) ──────────────────────────────────────────────
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # ── Loss computation (unchanged) ─────────────────────────────────────────
    do_mask_loss_for_padding: bool = False

    # ── UNet optimizer (unchanged) ───────────────────────────────────────────
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    # ── Kantorovich potential network φ (NEW) ────────────────────────────────
    phi_hidden_dim: int = 256
    phi_n_layers: int = 3
    phi_lr: float = 5e-4           # ~5x unet lr；φ需要更快收敛但不能过快震荡
    phi_n_inner_steps: int = 3     # WGAN-style critic步数，200K步用3足够
    phi_gp_lambda: float = 15.0   # 比默认10稍大，200K步有足够时间学好Lipschitz

    # ── Wasserstein regularisation schedule (NEW) ────────────────────────────
    # 200K步两阶段策略：
    #   0~60K  (30%): w从0线性爬升到wasserstein_lambda，扩散先收敛
    #   60K~200K(70%): w恒定，PCGrad冲突检测保护UNet不被干扰
    wasserstein_lambda: float = 0.02        # 梯度量级已由conflict_scale自动控制
    wasserstein_warmup_steps: int = 60000   # 30% of 200K

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. "
                f"Got {self.prediction_type}."
            )

        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
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
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(
                f"`crop_shape` must have positive dimensions. Got {self.crop_shape}."
            )

        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor "
                f"(which is determined by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

        if self.phi_n_layers < 1:
            raise ValueError(f"`phi_n_layers` must be ≥ 1. Got {self.phi_n_layers}.")
        if self.wasserstein_lambda < 0:
            raise ValueError(
                f"`wasserstein_lambda` must be non-negative. Got {self.wasserstein_lambda}."
            )
        if self.wasserstein_warmup_steps < 0:
            raise ValueError(
                f"`wasserstein_warmup_steps` must be non-negative. "
                f"Got {self.wasserstein_warmup_steps}."
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
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, "
                        "but we expect all image shapes to match."
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
