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
    """Configuration class for CtrlFlowPolicy.

    CtrlFlow is a χ²-CTRL-Flow generative policy built on top of Diffusion Policy.
    It replaces the standard KL-based score matching loss with a Pearson χ²
    density-ratio loss derived from CTRL-Flow's Lyapunov stability framework,
    yielding mass-covering dynamics that improve multi-modal action coverage.

    The framework guarantees global asymptotic stability via:
        - Drift term:      f*(x,t) = -2·∇_x r_hat(x,t)
        - Lyapunov functional: D_χ²(p_t ‖ p_data)
        - Dissipation functional: Φ = 2·E_{p_data}[‖∇r‖²]

    Defaults are configured for training with PushT providing proprioceptive
    and single camera observations.

    The parameters you will most likely need to change are the ones which depend
    on the environment / sensors. Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated
          as multiple camera views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        horizon: Diffusion model action prediction size.
        n_action_steps: The number of action steps to run in the environment for one invocation.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy.
        normalization_mapping: A dictionary that maps FeatureType to NormalizationMode.
        vision_backbone: Name of the torchvision resnet backbone for encoding images.
        resize_shape: (H, W) shape to resize images to as a preprocessing step.
        crop_ratio: Ratio in (0, 1] used to derive the crop size from resize_shape.
        crop_shape: (H, W) shape to crop images to.
        crop_is_random: Whether the crop should be random at training time.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize backbone.
        use_group_norm: Whether to replace batch normalization with group normalization.
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        use_separate_rgb_encoder_per_camera: Whether to use a separate RGB encoder per camera.
        down_dims: Feature dimension for each stage of temporal downsampling in the UNet.
        kernel_size: The convolutional kernel size of the UNet.
        n_groups: Number of groups used in the group norm of the UNet's convolutional blocks.
        diffusion_step_embed_dim: Output dimension of the diffusion timestep embedding network.
        use_film_scale_modulation: Whether to use FiLM scale modulation in addition to bias.
        noise_scheduler_type: Name of the noise scheduler. Supported: ["DDPM", "DDIM"].
        num_train_timesteps: Number of diffusion steps for the forward diffusion schedule.
        beta_schedule: Name of the diffusion beta schedule.
        beta_start: Beta value for the first forward-diffusion step.
        beta_end: Beta value for the last forward-diffusion step.
        prediction_type: UNet prediction type. Must be "epsilon" for CtrlFlow.
        clip_sample: Whether to clip the sample during denoising.
        clip_sample_range: Magnitude of the clipping range.
        num_inference_steps: Number of reverse diffusion steps at inference time.
        do_mask_loss_for_padding: Whether to mask the loss for copy-padded actions.

        # ── χ²-CTRL-Flow specific ──────────────────────────────────────────
        chi2_warmup_steps: Number of initial training steps using MSE loss (warm-start)
            before switching to χ² density-ratio loss. Helps the UNet backbone converge
            to a reasonable initialization before ratio estimation begins.
        chi2_lyapunov_lambda: Weight λ for the Lyapunov stability regularizer.
            This term penalizes collapse of the dissipation functional Φ = E[‖∇r‖²]
            toward zero, preventing the density-ratio network from degenerating to a
            constant. Set to 0.0 to disable.
        chi2_phi_min: Minimum threshold for the dissipation functional Φ = E[‖∇r‖²].
            The Lyapunov regularizer penalizes Φ < chi2_phi_min. This value depends
            on the action-space dimensionality and normalization; tune accordingly.
            Default 0.1 is appropriate for normalized action spaces with action_dim ≤ 7.
        chi2_diffusion_beta: Diffusion strength β in the CTRL-Flow SDE
            dX = f*(X,t)dt + √(2β)·dW. Controls the stochastic exploration strength
            during inference. Larger values increase coverage but add noise.
        chi2_ratio_net_hidden_dim: Hidden layer dimension for the 3-layer MLP density-ratio
            network r_hat(x, t, cond) → ℝ₊. Larger values increase capacity but slow training.
        chi2_num_inference_steps: Number of Euler integration steps for χ²-CTRL-Flow ODE
            sampling in Phase 2. Independent of the DDPM/DDIM num_inference_steps used
            during warm-start. Typically 10–50 steps suffice for the learned ODE.
    """

    # ── Inputs / output structure ──────────────────────────────────────────
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

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # ── Architecture / modeling ────────────────────────────────────────────
    # Vision backbone.
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
    # Noise scheduler (used during warm-start Phase 1 only).
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # ── Inference ──────────────────────────────────────────────────────────
    # num_inference_steps is used for DDPM/DDIM in Phase 1 warm-start sampling.
    num_inference_steps: int | None = None

    # ── Optimization ──────────────────────────────────────────────────────
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # ── Loss computation ───────────────────────────────────────────────────
    do_mask_loss_for_padding: bool = False

    # ── Training presets ───────────────────────────────────────────────────
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    # ── χ²-CTRL-Flow specific parameters ──────────────────────────────────
    chi2_warmup_steps: int = 5000
    chi2_lyapunov_lambda: float = 0.01
    chi2_phi_min: float = 0.1
    chi2_diffusion_beta: float = 0.1
    chi2_ratio_net_hidden_dim: int = 256
    # Number of Euler ODE steps used in Phase 2 inference.
    # Separate from num_inference_steps which controls DDPM/DDIM in Phase 1.
    chi2_num_inference_steps: int = 20

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        # CtrlFlow only supports epsilon prediction type because the drift
        # f*(x,t) = -2·∇r replaces the epsilon-prediction role of the UNet.
        if self.prediction_type != "epsilon":
            raise ValueError(
                f"CtrlFlowPolicy requires `prediction_type='epsilon'`. "
                f"Got '{self.prediction_type}'."
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
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
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
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        # Check UNet downsampling compatibility.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor "
                f"(determined by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

        if self.chi2_warmup_steps < 0:
            raise ValueError(f"`chi2_warmup_steps` must be >= 0. Got {self.chi2_warmup_steps}.")
        if self.chi2_lyapunov_lambda < 0:
            raise ValueError(f"`chi2_lyapunov_lambda` must be >= 0. Got {self.chi2_lyapunov_lambda}.")
        if self.chi2_phi_min < 0:
            raise ValueError(f"`chi2_phi_min` must be >= 0. Got {self.chi2_phi_min}.")
        if self.chi2_diffusion_beta <= 0:
            raise ValueError(f"`chi2_diffusion_beta` must be > 0. Got {self.chi2_diffusion_beta}.")
        if self.chi2_ratio_net_hidden_dim <= 0:
            raise ValueError(
                f"`chi2_ratio_net_hidden_dim` must be > 0. Got {self.chi2_ratio_net_hidden_dim}."
            )
        if self.chi2_num_inference_steps <= 0:
            raise ValueError(
                f"`chi2_num_inference_steps` must be > 0. Got {self.chi2_num_inference_steps}."
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
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

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
                        f"but we expect all image shapes to match."
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
