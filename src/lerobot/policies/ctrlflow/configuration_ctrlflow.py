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
import math
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("ctrlflow")
@dataclass
class CTRLFlowConfig(PreTrainedConfig):
    """Configuration class for CTRLFlowPolicy.

    Faithful implementation of the CTRL-Flow framework (Ma, 2024) Theorem 1.

    ── SDE specification (paper §4.3.1) ────────────────────────────────────

        f(x, t) = -∇_x log( p_t(x) / p_data(x) )
        g = sqrt(2) · I

    ── Probability path ────────────────────────────────────────────────────

    Straight-line interpolation:
        x_t = (1 - t) · x_0  +  t · ε,    t ∈ [0, t_max],  t_max < 1
        x_0 ~ p_data,   ε ~ N(0, I)

    ── Training target ─────────────────────────────────────────────────────

    The score difference along the straight path evaluates to:

        -∇_x log( p_t(x_t) / p_data(x_t) )  ≈  (x_0 - ε) / (1 - t)

    This is dimensionally correct: as t→0, target → x_0 - ε (O(1));
    as t→t_max, target grows as 1/t_min_gap (bounded by construction).

        target(t) = (x_0 - ε) / (1 - t)

    ── Inference ───────────────────────────────────────────────────────────

    Reverse SDE (Anderson 1982, consistent with Theorem 1):
        dx = f_θ(x, t) dt  +  sqrt(2) dW̄_t

    Euler-Maruyama (t decreases from t_max to 0, step size Δt):
        x_{k+1} = x_k  +  Δt · f_θ(x_k, t_k)  +  sqrt(2·Δt) · ε_k

    Initial sample: x ~ N(0, I)  [marginal of x_t at t=t_max ≈ N(0,I)
    when t_max is close to 1 and x_0 is normalised].

    ── Lyapunov regulariser (Theorem 1, Condition 1) ───────────────────────

        R(θ) = ||f_θ(x_t, t)||²  -  ||target(t)||²
        L_lyapunov = E[ max(0, R(θ)) ]

    Both terms carry the same 1/(1-t) scaling, so R is dimensionally
    consistent and the hinge correctly penalises magnitude overshoot.

    Args:
        n_obs_steps: Observation history length.
        horizon: Action prediction horizon (must be divisible by
            2**len(down_dims)).
        n_action_steps: Actions executed per policy invocation.
        normalization_mapping: FeatureType → NormalizationMode.
        drop_n_last_frames: Frames dropped at trajectory end.
        vision_backbone: torchvision ResNet variant name.
        resize_shape: (H, W) image resize before backbone.
        crop_ratio: Crop size as fraction of resize_shape.
        crop_shape: Explicit (H, W) crop (overrides crop_ratio).
        crop_is_random: Random crop (train) / center crop (eval).
        pretrained_backbone_weights: torchvision weight identifier.
        use_group_norm: GroupNorm instead of BatchNorm in backbone.
        spatial_softmax_num_keypoints: SpatialSoftmax keypoint count.
        use_separate_rgb_encoder_per_camera: One encoder per camera.
        down_dims: UNet downsampling feature dims per stage.
        kernel_size: UNet 1-D conv kernel size.
        n_groups: GroupNorm groups in UNet residual blocks.
        diffusion_step_embed_dim: Timestep embedding output dim.
        use_film_scale_modulation: FiLM scale modulation in UNet.
        t_min_gap: Gap from t=1; t_max = 1 - t_min_gap.
            Prevents 1/(1-t) from diverging. Default 0.01 bounds
            target magnitude at 1/0.01 = 100.
        num_sde_steps: Euler-Maruyama steps at inference.
        sde_noise_scale: Diffusion multiplier s; g_eff = s·sqrt(2)·I.
            s=1 → paper design. s=0 → deterministic ODE ablation.
        score_loss_weight: Weight for L_score.
        lyapunov_lambda: Weight λ ≥ 0 for L_lyapunov.
        do_mask_loss_for_padding: Mask padded action loss entries.
        compile_model: torch.compile the UNet.
        compile_mode: torch.compile mode string.
        optimizer_lr / betas / eps / weight_decay: Adam hyperparams.
        scheduler_name: LR scheduler identifier.
        scheduler_warmup_steps: LR warmup steps.
    """

    # ------------------------------------------------------------------ #
    # Inputs / outputs
    # ------------------------------------------------------------------ #
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE":  NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    drop_n_last_frames: int = 7

    # ------------------------------------------------------------------ #
    # Vision encoder
    # ------------------------------------------------------------------ #
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # ------------------------------------------------------------------ #
    # UNet
    # ------------------------------------------------------------------ #
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # ------------------------------------------------------------------ #
    # CTRL-Flow time schedule
    # t ∈ [0, t_max],  t_max = 1 - t_min_gap
    # Δt = t_max / num_sde_steps
    # ------------------------------------------------------------------ #
    t_min_gap: float = 0.05     # t_max = 1 - t_min_gap; bounds 1/(1-t)
    num_sde_steps: int = 50     # Euler-Maruyama steps at inference

    # ------------------------------------------------------------------ #
    # CTRL-Flow SDE diffusion  (paper: g = sqrt(2)·I)
    # g_eff = sde_noise_scale * sqrt(2) · I
    # ------------------------------------------------------------------ #
    sde_noise_scale: float = 1.0

    # ------------------------------------------------------------------ #
    # Training loss
    # ------------------------------------------------------------------ #
    score_loss_weight: float = 1.0
    lyapunov_lambda: float = 0.01

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    do_mask_loss_for_padding: bool = False
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # ------------------------------------------------------------------ #
    # Optimisation
    # ------------------------------------------------------------------ #
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    # ------------------------------------------------------------------ #
    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be a ResNet variant. Got {self.vision_backbone}."
            )
        if not (0.0 < self.t_min_gap < 1.0):
            raise ValueError(f"`t_min_gap` must be in (0, 1). Got {self.t_min_gap}.")
        if self.num_sde_steps < 1:
            raise ValueError(f"`num_sde_steps` must be >= 1. Got {self.num_sde_steps}.")
        if self.sde_noise_scale < 0.0:
            raise ValueError(f"`sde_noise_scale` must be >= 0. Got {self.sde_noise_scale}.")
        if self.lyapunov_lambda < 0.0:
            raise ValueError(f"`lyapunov_lambda` must be >= 0. Got {self.lyapunov_lambda}.")
        if self.score_loss_weight <= 0.0:
            raise ValueError(f"`score_loss_weight` must be > 0. Got {self.score_loss_weight}.")

        if self.resize_shape is not None:
            if len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape):
                raise ValueError(
                    f"`resize_shape` must be two positive ints. Got {self.resize_shape}."
                )
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )

        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dims. Got {self.crop_shape}.")

        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                f"horizon must be divisible by 2**len(down_dims)={downsampling_factor}. "
                f"Got {self.horizon=} and {self.down_dims=}."
            )

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #

    @property
    def t_max(self) -> float:
        """Upper integration bound: t_max = 1 - t_min_gap < 1.
        Prevents target = (x0 - eps)/(1-t) from diverging.
        """
        return 1.0 - self.t_min_gap

    @property
    def dt(self) -> float:
        """Euler-Maruyama step size: Δt = t_max / num_sde_steps."""
        return self.t_max / self.num_sde_steps

    @property
    def g_coeff(self) -> float:
        """Effective diffusion magnitude: sde_noise_scale * sqrt(2).
        Paper design: sde_noise_scale=1 → g_coeff = sqrt(2).
        Diffusion std per EM step: g_coeff * sqrt(Δt).
        """
        return self.sde_noise_scale * math.sqrt(2.0)

    # ------------------------------------------------------------------ #
    # LeRobot hooks
    # ------------------------------------------------------------------ #

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
                "Provide at least one image or the environment state as input."
            )
        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if (
                    self.crop_shape[0] > image_ft.shape[1]
                    or self.crop_shape[1] > image_ft.shape[2]
                ):
                    raise ValueError(
                        f"`crop_shape` {self.crop_shape} exceeds image shape "
                        f"{image_ft.shape} for key `{key}`."
                    )
        if len(self.image_features) > 0:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, ft in self.image_features.items():
                if ft.shape != first_ft.shape:
                    raise ValueError(
                        f"Image shape mismatch: `{key}` {ft.shape} != "
                        f"`{first_key}` {first_ft.shape}."
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
