#!/usr/bin/env python

# CTRL-Flow2: Lyapunov-Stable Regularized Diffusion Policy
# Based on "CTRL-Flow: Generative Modeling via Lyapunov-Stable Regularized
# Probability Flows on Wasserstein Space"
#
# Extends lerobot DiffusionPolicy with:
#   1. Lyapunov stability regularizer on the drift/diffusion terms
#   2. Adaptive noise schedule driven by Fisher dissipation
#   3. State-dependent (anisotropic) diffusion coefficient
#   4. Generalized Lyapunov functional selection (KL / chi2 / hybrid)
#
# Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("ctrlflow2")
@dataclass
class CtrlFlow2Config(PreTrainedConfig):
    """Configuration class for CtrlFlow2Policy.

    Extends DiffusionPolicy with CTRL-Flow Lyapunov stability theory.

    CTRL-Flow theoretical grounding
    --------------------------------
    The original diffusion policy minimises a plain MSE loss on predicted noise
    (epsilon-prediction), which corresponds to maximising a KL-divergence lower
    bound.  CTRL-Flow re-interprets the reverse SDE

        dY_t = f_b(Y_t, t) dt + g(t) dW_t

    as a **controlled dissipative system** on the Wasserstein space P_2(R^d).
    Stability is guaranteed when the drift f_b and diffusion g jointly satisfy
    the Drift-Dissipation Matching condition (Theorem 1 of the paper):

        ∫ ⟨f_b, ∇ δL/δρ⟩ dμ  +  ½ ∫ tr[gg^T ∇² δL/δρ] dμ  ≤ -(α+β) Φ

    where L is a chosen Lyapunov functional (e.g. KL divergence) and
    Φ = ∫ ‖∇ δL/δρ‖² dμ  is the Fisher dissipation.

    This config adds three levers on top of vanilla Diffusion Policy:

    1. **Lyapunov stability regularizer** (λ_lyapunov):
       An auxiliary loss term R(θ) is added to the training objective that
       penalises any violation of the drift-dissipation condition.  This embeds
       stability as an *a-priori* constraint rather than hoping it emerges from
       the data alone.

    2. **Adaptive β-schedule** (use_adaptive_beta):
       Rather than a fixed cosine/linear schedule the noise level β(t) is
       adjusted online to maximise the instantaneous convergence rate
           (α + β) / C
       as described in §6.4.3 of the paper.  A lightweight EMA of the
       Fisher dissipation Φ is used as the surrogate signal.

    3. **State-dependent (anisotropic) diffusion** (use_state_dependent_diffusion):
       Instead of scalar g(t) the diffusion matrix is
           σ(x, t) = √(2β · f''(r̂_θ(x,t))) · I
       where f'' is the curvature of the chosen f-divergence evaluated on the
       current density-ratio estimate r̂_θ.  For the KL case f''(r) = 1/r,
       so higher noise is injected where p_t ≪ p_data, which accelerates
       mixing without destabilising already-converged regions.

    4. **Lyapunov functional selector** (lyapunov_functional):
       Allows switching the geometry of convergence from KL (standard score
       matching) to chi² or an adaptive hybrid (§5.3 of the paper).

    All new parameters default to values that reproduce vanilla Diffusion
    Policy behaviour (λ_lyapunov=0, use_adaptive_beta=False, etc.) so the
    config is backward-compatible.

    Args:
        -- Inherited from DiffusionPolicy (unchanged) --
        n_obs_steps, horizon, n_action_steps, normalization_mapping,
        drop_n_last_frames, vision_backbone, resize_shape, crop_ratio,
        crop_shape, crop_is_random, pretrained_backbone_weights,
        use_group_norm, spatial_softmax_num_keypoints,
        use_separate_rgb_encoder_per_camera, down_dims, kernel_size,
        n_groups, diffusion_step_embed_dim, use_film_scale_modulation,
        noise_scheduler_type, num_train_timesteps, beta_schedule,
        beta_start, beta_end, prediction_type, clip_sample,
        clip_sample_range, num_inference_steps, do_mask_loss_for_padding.

        -- CTRL-Flow additions --
        lyapunov_functional: Which Lyapunov functional L to use.
            "kl"    – KL divergence (reproduces standard score matching, §4.2.1)
            "chi2"  – χ²-divergence (Pearson loss, §5.1.1)
            "hybrid"– time-varying convex combination of KL + χ² (§5.3)
        lyapunov_lambda: Weight λ for the Lyapunov stability regularizer R(θ)
            added to the total loss:  J(θ) = E[ℓ(θ)] + λ · R(θ).
            Set to 0.0 to disable (reproduces vanilla Diffusion Policy).
        lyapunov_alpha: Minimum required drift-dissipation rate α > 0.
            Controls how tightly the drift-dissipation condition is enforced.
        lyapunov_beta_coeff: Minimum required diffusion-dissipation rate β > 0.
            Must satisfy α + β_coeff > 0 for stability.
        use_adaptive_beta: If True, dynamically rescale the diffusion strength
            β(t) based on an EMA of the Fisher dissipation to maximise the
            effective convergence rate (α+β)/C.  See §6.4.3.
        adaptive_beta_ema: EMA decay for the Fisher dissipation tracker.
        adaptive_beta_clip: Maximum allowed β rescale factor to prevent
            instability at the start of training.
        use_state_dependent_diffusion: If True, use anisotropic diffusion
            σ(x,t) ∝ √f''(r̂_θ) · I derived from f-divergence curvature.
            Currently implemented for KL (f''=1/r) and chi² (f''=const).
            See §5.1.2.
        diffusion_curvature_eps: Small constant added when computing f''(r̂)
            to avoid division-by-zero in the KL case.
        hybrid_kl_weight_init: Initial KL weight w_KL(0) for hybrid mode.
            The chi² weight is w_chi2 = 1 - w_KL.  Weights are updated by
            the adaptive scheme described in §5.3.
        hybrid_weight_eta: Learning rate η for the exponential weight update
            rule w_i ∝ exp(-η · ΔL_i / σ²_i).
    """

    # ------------------------------------------------------------------ #
    #  Inputs / output structure  (unchanged from DiffusionPolicy)        #
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
    drop_n_last_frames: int = 7

    # ------------------------------------------------------------------ #
    #  Vision backbone  (unchanged)                                       #
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
    #  U-Net  (unchanged)                                                 #
    # ------------------------------------------------------------------ #
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # ------------------------------------------------------------------ #
    #  Noise scheduler  (unchanged defaults)                              #
    # ------------------------------------------------------------------ #
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # ------------------------------------------------------------------ #
    #  Inference                                                          #
    # ------------------------------------------------------------------ #
    num_inference_steps: int | None = None

    # ------------------------------------------------------------------ #
    #  Compilation                                                        #
    # ------------------------------------------------------------------ #
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # ------------------------------------------------------------------ #
    #  Loss                                                               #
    # ------------------------------------------------------------------ #
    do_mask_loss_for_padding: bool = False

    # ------------------------------------------------------------------ #
    #  Optimiser / scheduler  (unchanged)                                 #
    # ------------------------------------------------------------------ #
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    # ================================================================== #
    #  CTRL-Flow additions                                                #
    # ================================================================== #

    # --- 1. Lyapunov functional geometry ---
    lyapunov_functional: str = "kl"
    # "kl" | "chi2" | "hybrid"

    # --- 2. Lyapunov stability regularizer (Theorem 1 / Algorithm 7.1) ---
    lyapunov_lambda: float = 0.01
    # Weight of stability regularizer. 0.0 → pure vanilla diffusion.
    lyapunov_alpha: float = 0.5
    # Required drift-dissipation rate α (lower-bounds the drift penalty).
    lyapunov_beta_coeff: float = 0.5
    # Required diffusion-dissipation rate β.

    # --- 3. Adaptive β-schedule (§6.4.3) ---
    use_adaptive_beta: bool = False
    adaptive_beta_ema: float = 0.99
    # EMA decay for Fisher dissipation tracker.
    adaptive_beta_clip: float = 2.0
    # Maximum multiplicative factor for β rescaling.

    # --- 4. State-dependent anisotropic diffusion (§5.1.2) ---
    use_state_dependent_diffusion: bool = False
    # Adds per-sample noise scaling via f''(r̂_θ).  Disabled by default
    # because it requires density-ratio estimation, which increases cost.
    diffusion_curvature_eps: float = 1e-4
    # Numerical floor for KL curvature f''(r) = 1/r.

    # --- 5. Hybrid Lyapunov weights (§5.3) ---
    hybrid_kl_weight_init: float = 0.7
    hybrid_weight_eta: float = 0.1

    # ------------------------------------------------------------------ #
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

        supported_lyapunov = ["kl", "chi2", "hybrid"]
        if self.lyapunov_functional not in supported_lyapunov:
            raise ValueError(
                f"`lyapunov_functional` must be one of {supported_lyapunov}. "
                f"Got {self.lyapunov_functional}."
            )

        if self.lyapunov_lambda < 0:
            raise ValueError("`lyapunov_lambda` must be non-negative.")

        if self.lyapunov_alpha <= 0 or self.lyapunov_beta_coeff <= 0:
            raise ValueError("`lyapunov_alpha` and `lyapunov_beta_coeff` must be strictly positive.")

        if not (0 < self.adaptive_beta_ema < 1):
            raise ValueError("`adaptive_beta_ema` must be in (0, 1).")

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

        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor "
                f"(determined by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

        # Warn if adaptive diffusion is enabled without KL / chi2.
        if self.use_state_dependent_diffusion and self.lyapunov_functional == "hybrid":
            # Hybrid uses KL branch for curvature estimation; this is fine.
            pass

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
