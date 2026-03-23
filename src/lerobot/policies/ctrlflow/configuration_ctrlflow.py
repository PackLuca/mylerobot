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

    Strictly implements the CTRL-Flow framework (Ma, 2024) §4.3.1
    Gradient Flow Construction with the exact drift and diffusion from the paper:

        f(x, t) = -∇_x u_t(x) = -∇_x log( p_t(x) / p_data(x) )
        g = sqrt(2) * I

    Theorem 1 verification (α = β = 1, Lyapunov functional L = KL divergence):

        Condition 1 — Drift-Dissipation Matching:
            ∫ <f, ∇δL/δρ> p dx
            = -∫ ||∇_x log(p_t/p_data)||² p dx
            = -Φ  ✓  (with α = 1)

        Condition 2 — Diffusion-Enhanced Dissipation  (g = sqrt(2)*I):
            ½ ∫ tr[g g^T ∇²_x δL/δρ] p dx
            = ∫ Δ_x log(p_t/p_data) p dx
            = -∫ ||∇_x log p_t||² p dx
            ≈ -Φ  ✓  (with β = 1, by integration by parts)

    Inference uses the Euler-Maruyama discretisation of the SDE:

        x_{t-Δt} = x_t  +  Δt · f_θ(x_t, t)  +  sqrt(2Δt) · ε,   ε ~ N(0, I)
                   └─── drift term ──────────┘   └─ diffusion term ─┘
                         (deterministic)              (stochastic)

    where Δt = 1 / num_sde_steps.

    Training minimises:

        L = score_loss_weight · L_score  +  lyapunov_lambda · L_lyapunov

        L_score    = MSE( f_θ(x_t, t),  -∇_x log(p_t/p_data) )
                   ≈ MSE( f_θ(x_t, t),  action - noise )        [straight path]

        L_lyapunov = E[ max(0,  <f_θ, ∇δL/δρ>  +  Φ) ]
                                 ↑ Condition 1 residual

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass.
        horizon: Action prediction horizon. Must be divisible by
            2**len(down_dims) due to UNet downsampling.
        n_action_steps: Number of actions to execute per policy invocation.
        normalization_mapping: Maps FeatureType strings to NormalizationMode.
        drop_n_last_frames: Frames dropped at trajectory end to avoid padding.
        vision_backbone: torchvision ResNet backbone name.
        resize_shape: (H, W) to resize images before backbone.
        crop_ratio: Fraction of resize_shape used as crop size (in (0, 1]).
        crop_shape: (H, W) explicit crop shape (overrides crop_ratio path).
        crop_is_random: Random crop at train time; center crop at eval.
        pretrained_backbone_weights: torchvision pretrained weight identifier.
        use_group_norm: Replace BatchNorm with GroupNorm in the backbone.
        spatial_softmax_num_keypoints: Number of SpatialSoftmax keypoints.
        use_separate_rgb_encoder_per_camera: One RGB encoder per camera view.
        down_dims: UNet temporal downsampling feature dimensions per stage.
        kernel_size: UNet 1-D convolutional kernel size.
        n_groups: GroupNorm group count in UNet residual blocks.
        diffusion_step_embed_dim: Output dim of the timestep embedding MLP.
        use_film_scale_modulation: Enable FiLM scale modulation in UNet.
        num_sde_steps: Number of Euler-Maruyama steps at inference.
            Δt = 1 / num_sde_steps.  Larger values → lower discretisation
            error but more UNet evaluations.
        sde_noise_scale: Multiplier s on the diffusion term at inference:
            g_eff = s * sqrt(2) * I.
            s = 1.0  →  exact paper design  g = sqrt(2)*I  (default).
            s = 0.0  →  ablation: pure deterministic ODE (no diffusion).
            Intermediate values allow studying the effect of diffusion strength.
        score_loss_weight: Weight for the primary score-matching MSE loss.
        lyapunov_lambda: Weight λ ≥ 0 for the Lyapunov regulariser.
            λ = 0.0  →  plain score matching, no stability constraint.
            λ > 0.0  →  penalise violations of Drift-Dissipation Matching.
        do_mask_loss_for_padding: Mask loss on copy-padded action entries.
        compile_model: Apply torch.compile to the UNet for faster inference.
        compile_mode: torch.compile mode string (e.g. "reduce-overhead").
        optimizer_lr: Adam learning rate.
        optimizer_betas: Adam (β₁, β₂).
        optimizer_eps: Adam ε for numerical stability.
        optimizer_weight_decay: Adam weight decay.
        scheduler_name: LR scheduler name passed to DiffuserSchedulerConfig.
        scheduler_warmup_steps: Number of LR warmup steps.
    """

    # ------------------------------------------------------------------ #
    # Inputs / output structure
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Vision encoder — identical to DiffusionConfig
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
    # UNet architecture — identical to DiffusionConfig
    # ------------------------------------------------------------------ #
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # ------------------------------------------------------------------ #
    # CTRL-Flow SDE parameters
    # Paper: f(x,t) = -∇_x log(p_t/p_data),  g = sqrt(2)*I
    # Euler-Maruyama: x_{t-Δt} = x_t + Δt·f_θ + sqrt(2Δt)·ε
    # ------------------------------------------------------------------ #
    num_sde_steps: int = 10         # number of integration steps (Δt = 1/N)
    sde_noise_scale: float = 1.0    # s: g_eff = s·sqrt(2)·I; 1.0 = paper design

    # ------------------------------------------------------------------ #
    # Training loss
    # ------------------------------------------------------------------ #
    score_loss_weight: float = 1.0  # weight for L_score (MSE)
    lyapunov_lambda: float = 0.1    # weight λ for L_lyapunov (Drift-Dissipation)

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    do_mask_loss_for_padding: bool = False
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # ------------------------------------------------------------------ #
    # Optimisation presets — same defaults as DiffusionConfig
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
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.num_sde_steps < 1:
            raise ValueError(f"`num_sde_steps` must be >= 1. Got {self.num_sde_steps}.")
        if self.sde_noise_scale < 0.0:
            raise ValueError(f"`sde_noise_scale` must be >= 0. Got {self.sde_noise_scale}.")
        if self.lyapunov_lambda < 0.0:
            raise ValueError(f"`lyapunov_lambda` must be >= 0. Got {self.lyapunov_lambda}.")
        if self.score_loss_weight <= 0.0:
            raise ValueError(f"`score_loss_weight` must be > 0. Got {self.score_loss_weight}.")

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(
                f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}."
            )
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            self.crop_shape = (
                (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
                if self.crop_ratio < 1.0
                else None
            )

        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon must be divisible by the UNet downsampling factor "
                f"2**len(down_dims)={downsampling_factor}. "
                f"Got {self.horizon=} and {self.down_dims=}."
            )

    # ------------------------------------------------------------------ #
    # Derived quantities used by CTRLFlowModel
    # ------------------------------------------------------------------ #

    @property
    def dt(self) -> float:
        """Euler-Maruyama step size  Δt = 1 / num_sde_steps."""
        return 1.0 / self.num_sde_steps

    @property
    def g_coeff(self) -> float:
        """Effective diffusion coefficient magnitude at inference.

        Paper design: g = sqrt(2) * I  →  g_coeff = sqrt(2).
        With sde_noise_scale s: g_coeff = s * sqrt(2).

        The Euler-Maruyama diffusion increment per step is:
            sqrt(g² · Δt) · ε = sqrt(2 · sde_noise_scale² · Δt) · ε
        """
        return self.sde_noise_scale * math.sqrt(2.0)

    # ------------------------------------------------------------------ #
    # LeRobot framework hooks
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
                "You must provide at least one image or the environment state among the inputs."
            )
        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` {self.crop_shape} exceeds image shape {image_ft.shape} "
                        f"for key `{key}`."
                    )
        if len(self.image_features) > 0:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, ft in self.image_features.items():
                if ft.shape != first_ft.shape:
                    raise ValueError(
                        f"Image shape mismatch: `{key}` {ft.shape} != `{first_key}` {first_ft.shape}."
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
