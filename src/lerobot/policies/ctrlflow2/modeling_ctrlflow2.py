#!/usr/bin/env python

# CTRL-Flow2: Lyapunov-Stable Regularized Diffusion Policy
# Based on "CTRL-Flow: Generative Modeling via Lyapunov-Stable Regularized
# Probability Flows on Wasserstein Space"
#
# This file implements the CtrlFlow2Policy by extending lerobot's DiffusionPolicy
# with three CTRL-Flow mechanisms derived from Theorem 1 of the paper:
#
#   (A) Lyapunov Stability Regularizer  (Algorithm 7.1)
#       Adds a penalty R(θ) to the training loss that enforces the
#       Drift-Dissipation Matching condition:
#         ∫⟨f_b, ∇δL/δρ⟩dμ + ½∫tr[gg^T ∇²δL/δρ]dμ ≤ -(α+β)Φ
#
#   (B) Adaptive β-Schedule  (§6.4.3)
#       Dynamically adjusts the effective noise magnitude to maximise the
#       instantaneous Lyapunov convergence rate (α+β)/C, using an EMA of
#       the observed Fisher dissipation Φ as a proxy for C.
#
#   (C) State-Dependent Anisotropic Diffusion  (§5.1.2)
#       Scales the noise injection by σ(x,t) = √(2β·f''(r̂_θ(x,t)))·I,
#       where f'' is the curvature of the chosen f-divergence.
#       KL: f''(r) = 1/r  →  more noise where p_t ≪ p_data.
#       χ²: f''(r) = 2    →  uniform curvature-weighted noise.
#
# All three mechanisms default to reproducing vanilla DiffusionPolicy when
# their respective switches are off (lyapunov_lambda=0, use_adaptive_beta=False,
# use_state_dependent_diffusion=False).
#
# Licensed under the Apache License, Version 2.0.

import math
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.policies.ctrlflow2.configuration_ctrlflow2 import CtrlFlow2Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ---------------------------------------------------------------------------
# Top-level policy
# ---------------------------------------------------------------------------

class CtrlFlow2Policy(PreTrainedPolicy):
    """
    CTRL-Flow2 Policy – Lyapunov-Stable Diffusion for Robot Action Generation.

    Inherits the architecture of DiffusionPolicy (1-D UNet + ResNet image encoder)
    and augments the training objective with CTRL-Flow stability constraints
    derived from Lyapunov theory on Wasserstein space.

    Key differences from DiffusionPolicy.forward():
        - compute_loss() now returns a dict with the individual loss components
          so that they can be logged separately.
        - The CtrlFlow2Model tracks a running EMA of the Fisher dissipation for
          the adaptive β-schedule.

    Usage:
        Identical to DiffusionPolicy.  Pass a CtrlFlow2Config and a dataset_stats
        dict during construction.
    """

    config_class = CtrlFlow2Config
    name = "ctrlflow2"

    def __init__(self, config: CtrlFlow2Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None
        self.diffusion = CtrlFlow2Model(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Call on env.reset()."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        return self.diffusion.generate_actions(batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        Run the batch through the model and return the total loss plus a dict
        of named sub-losses for logging.

        Returns:
            loss:        scalar total training loss
            output_dict: {"loss_diffusion": ..., "loss_lyapunov": ...,
                          "fisher_dissipation": ..., "lyapunov_reg": ...}
        """
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        loss, output_dict = self.diffusion.compute_loss(batch)
        return loss, output_dict


# ---------------------------------------------------------------------------
# Noise-scheduler factory (unchanged from DiffusionPolicy)
# ---------------------------------------------------------------------------

def _make_noise_scheduler(name: str, **kwargs) -> DDPMScheduler | DDIMScheduler:
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


# ---------------------------------------------------------------------------
# CTRL-Flow2 Model
# ---------------------------------------------------------------------------

class CtrlFlow2Model(nn.Module):
    """
    Core model for CtrlFlow2Policy.

    Architecture is identical to DiffusionModel (UNet + optional RGB encoder).
    The key changes live entirely in compute_loss():

        total_loss = loss_diffusion + λ · R(θ)

    where R(θ) is the Lyapunov stability regularizer and λ = lyapunov_lambda.

    Additionally:
        - _compute_fisher_dissipation() estimates Φ = E[‖∇ log p_t‖²] for use
          in the adaptive β-schedule and the regularizer.
        - _get_diffusion_scale() optionally returns the state-dependent σ(x,t).
        - _adaptive_beta_scale carries a persistent EMA of Φ for §6.4.3.
    """

    def __init__(self, config: CtrlFlow2Config):
        super().__init__()
        self.config = config

        # ---- Observation encoders ----
        global_cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [CtrlFlow2RgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = CtrlFlow2RgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        self.unet = CtrlFlow2ConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

        # ---- CTRL-Flow state ----
        # EMA of Fisher dissipation Φ for adaptive β-schedule (§6.4.3).
        # Stored as a non-parameter buffer so it survives save/load.
        self.register_buffer(
            "_fisher_ema",
            torch.tensor(1.0),  # initialise to 1 (neutral)
        )

        # Hybrid Lyapunov weights w_KL, w_chi2 (§5.3).
        # Stored as buffers (not learnable parameters).
        self.register_buffer(
            "_hybrid_w_kl",
            torch.tensor(config.hybrid_kl_weight_init),
        )
        self.register_buffer(
            "_hybrid_w_chi2",
            torch.tensor(1.0 - config.hybrid_kl_weight_init),
        )

    # ======================================================================
    # Inference
    # ======================================================================

    @torch.no_grad()
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Reverse diffusion with optional CTRL-Flow adaptive β-rescaling."""
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )

            # State-dependent diffusion: rescale noise at inference time.
            if self.config.use_state_dependent_diffusion:
                model_output = self._apply_diffusion_curvature_scale(model_output, sample)

            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and state into a single conditioning vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    # ======================================================================
    # Training loss
    # ======================================================================

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        Compute total CTRL-Flow2 loss.

        Total loss = loss_diffusion + λ · R(θ)

        where:
            loss_diffusion  = standard MSE on noise (or sample) prediction
            R(θ)            = Lyapunov stability regularizer (Theorem 1)
                            = max(0, drift_violation) + max(0, diffusion_violation)
            λ               = config.lyapunov_lambda

        Returns
        -------
        loss : scalar Tensor
        output_dict : dict with keys
            "loss_diffusion"    – base denoising MSE
            "loss_lyapunov"     – λ · R(θ) contribution
            "fisher_dissipation"– estimated Φ (for monitoring / adaptive β)
            "lyapunov_reg"      – raw R(θ) before λ scaling
            "beta_scale"        – current adaptive β scale factor
        """
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch

        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        # ---- Forward diffusion ----
        trajectory = batch[ACTION]
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()

        # Optionally rescale the noise using the adaptive β-schedule (§6.4.3).
        beta_scale = self._get_adaptive_beta_scale()
        scaled_eps = eps * beta_scale if self.config.use_adaptive_beta else eps

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, scaled_eps, timesteps)

        # ---- UNet forward pass ----
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # ---- Base diffusion loss ----
        if self.config.prediction_type == "epsilon":
            target = scaled_eps if self.config.use_adaptive_beta else eps
        elif self.config.prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss_diffusion = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError("'action_is_pad' missing from batch with do_mask_loss_for_padding=True.")
            in_episode_bound = ~batch["action_is_pad"]
            loss_diffusion = loss_diffusion * in_episode_bound.unsqueeze(-1)

        loss_diffusion = loss_diffusion.mean()

        # ---- CTRL-Flow Lyapunov stability regularizer ----
        # Compute and (optionally) apply R(θ) and the adaptive β update.
        lyapunov_reg = torch.tensor(0.0, device=trajectory.device)
        fisher_phi = torch.tensor(0.0, device=trajectory.device)

        if self.config.lyapunov_lambda > 0 or self.config.use_adaptive_beta:
            lyapunov_reg, fisher_phi = self._compute_lyapunov_regularizer(
                pred=pred,
                target=target,
                noisy_trajectory=noisy_trajectory,
                timesteps=timesteps,
            )
            # Update EMA of Fisher dissipation for adaptive β-schedule.
            if self.config.use_adaptive_beta and self.training:
                self._update_fisher_ema(fisher_phi.detach())

        loss_lyapunov = self.config.lyapunov_lambda * lyapunov_reg

        # ---- Hybrid Lyapunov weight update (§5.3) ----
        if self.config.lyapunov_functional == "hybrid" and self.training:
            self._update_hybrid_weights(loss_diffusion.detach())

        total_loss = loss_diffusion + loss_lyapunov

        output_dict = {
            "loss_diffusion": loss_diffusion.detach(),
            "loss_lyapunov": loss_lyapunov.detach(),
            "fisher_dissipation": fisher_phi.detach(),
            "lyapunov_reg": lyapunov_reg.detach(),
            "beta_scale": torch.tensor(beta_scale),
        }
        return total_loss, output_dict

    # ======================================================================
    # CTRL-Flow core helpers
    # ======================================================================

    def _compute_lyapunov_regularizer(
        self,
        pred: Tensor,
        target: Tensor,
        noisy_trajectory: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the Lyapunov stability regularizer R(θ) from Theorem 1 /
        Algorithm 7.1 of the CTRL-Flow paper.

        The regularizer enforces the Drift-Dissipation Matching condition:

            ∫⟨f_b, ∇δL/δρ⟩dμ  ≤  -α · Φ          (drift term)
            ½∫tr[gg^T ∇²δL/δρ]dμ  ≤  -β · Φ       (diffusion term)

        In the epsilon-prediction setting the score ∇ log p_t(x) is
        approximated by -eps_pred / sqrt(1 - ᾱ_t).  The Fisher dissipation is
        then Φ ≈ E[‖eps_pred‖²] (unnormalised, up to schedule constants).

        For the KL Lyapunov functional (Proposition 1):
            δL/δρ = log p_t / p_data  →  ∇δL/δρ ≈ score ∝ eps_pred
        
        Drift-dissipation condition in discrete / ε-prediction form:
            ⟨eps_pred, eps_pred⟩  ≥  α · Φ        (always satisfied with α=1)

        The *violation* arises when the model predicts a drift that reduces the
        Lyapunov functional too slowly (small alignment between pred and target).
        We measure this as:

            drift_violation    = max(0,  α · Φ - ⟨pred, target⟩)
            diffusion_violation = max(0,  β · Φ - ½‖pred‖²)

        where Φ = mean(‖target‖²) is the empirical Fisher dissipation.

        For chi²-divergence the curvature f''(r)=2 modulates the target norm;
        for hybrid the KL and chi² violations are combined with current weights.

        Returns
        -------
        reg  : scalar regularizer R(θ) ≥ 0
        phi  : scalar Fisher dissipation Φ (for logging / adaptive β)
        """
        functional = self.config.lyapunov_functional

        # Fisher dissipation  Φ = E[‖∇δL/δρ‖²]  ≈ E[‖target‖²]
        phi_kl = (target ** 2).mean()

        if functional == "kl":
            phi = phi_kl
            reg = self._violation_kl(pred, target, phi)

        elif functional == "chi2":
            # chi² curvature f''(r)=2 → scale target norms by 2
            phi = 2.0 * phi_kl
            reg = self._violation_chi2(pred, target, phi)

        elif functional == "hybrid":
            phi = self._hybrid_w_kl * phi_kl + self._hybrid_w_chi2 * 2.0 * phi_kl
            reg_kl = self._violation_kl(pred, target, phi_kl)
            reg_chi2 = self._violation_chi2(pred, target, 2.0 * phi_kl)
            reg = self._hybrid_w_kl * reg_kl + self._hybrid_w_chi2 * reg_chi2

        else:
            raise ValueError(f"Unknown lyapunov_functional: {functional}")

        return reg, phi

    def _violation_kl(self, pred: Tensor, target: Tensor, phi: Tensor) -> Tensor:
        """
        Lyapunov regularizer for KL-divergence functional (Proposition 1).

        Drift-dissipation:  ∫⟨f_b, score⟩dμ ≤ -α·Φ
          In ε-prediction: alignment = ⟨pred, target⟩ should be ≥ α·Φ.
          Violation = max(0,  α·Φ - alignment)

        Diffusion-dissipation: ½‖pred‖² should be ≥ β·Φ.
          Violation = max(0,  β·Φ - ½‖pred‖²)
        """
        alpha = self.config.lyapunov_alpha
        beta = self.config.lyapunov_beta_coeff

        # Drift term: alignment between prediction and target (score direction)
        # Shape: (B, T, D) → scalar
        alignment = (pred * target).mean()
        drift_violation = F.relu(alpha * phi - alignment)

        # Diffusion term: energy of the predicted score
        half_pred_sq = 0.5 * (pred ** 2).mean()
        diffusion_violation = F.relu(beta * phi - half_pred_sq)

        return drift_violation + diffusion_violation

    def _violation_chi2(self, pred: Tensor, target: Tensor, phi: Tensor) -> Tensor:
        """
        Lyapunov regularizer for χ²-divergence functional.

        For f(t)=(t-1)², f'(t)=2(t-1), f''(t)=2.
        The Pearson loss modulates the target by the density ratio estimate.
        We approximate r̂ ≈ 1 (unit ratio at start of training) and use the
        same violation structure as KL but with the chi²-scaled φ.
        """
        alpha = self.config.lyapunov_alpha
        beta = self.config.lyapunov_beta_coeff

        # chi² curvature scales the required dissipation by f''=2
        alignment = (pred * target).mean()
        drift_violation = F.relu(alpha * phi - alignment)

        half_pred_sq = 0.5 * (pred ** 2).mean()
        diffusion_violation = F.relu(beta * phi - half_pred_sq)

        return drift_violation + diffusion_violation

    # ---- Adaptive β-schedule helpers (§6.4.3) ----------------------------

    def _get_adaptive_beta_scale(self) -> float:
        """
        Return the current β rescaling factor.

        The factor is clipped to [1/clip, clip] to prevent instability.
        When use_adaptive_beta is False returns 1.0 (no rescaling).
        """
        if not self.config.use_adaptive_beta:
            return 1.0
        # Maximise (α+β)/C:  increase β when Fisher dissipation is large
        # (distribution is far from target) and decrease it when small.
        # We use the EMA of Φ as a proxy for C; higher Φ → larger β scale.
        phi_ema = self._fisher_ema.item()
        # Simple proportional control: scale ∝ log(1 + Φ_ema)
        raw_scale = 1.0 + 0.5 * math.log1p(phi_ema)
        return float(min(raw_scale, self.config.adaptive_beta_clip))

    def _update_fisher_ema(self, phi: Tensor) -> None:
        """Update the EMA of the Fisher dissipation."""
        decay = self.config.adaptive_beta_ema
        self._fisher_ema.copy_(decay * self._fisher_ema + (1.0 - decay) * phi)

    # ---- State-dependent diffusion helpers (§5.1.2) ----------------------

    def _apply_diffusion_curvature_scale(self, model_output: Tensor, x: Tensor) -> Tensor:
        """
        Scale model output by the f-divergence curvature σ(x,t) = √f''(r̂).

        For KL:   f''(r) = 1/r.  We approximate r̂ ≈ exp(-‖x‖²/2d) (a
                  rough Gaussian density ratio) as a cheap proxy.
        For chi²: f''(r) = 2  →  constant scale √2.
        For hybrid: weighted combination.
        """
        eps = self.config.diffusion_curvature_eps
        functional = self.config.lyapunov_functional
        action_dim = x.shape[-1]

        if functional in ("kl", "hybrid"):
            # Approximate log r̂ ≈ -0.5 * ‖x‖²/d  (Gaussian proxy)
            log_r = -0.5 * (x ** 2).sum(dim=-1, keepdim=True) / action_dim
            r_hat = torch.exp(log_r).clamp(min=eps)
            # f''(r) = 1/r  for KL
            curvature_kl = 1.0 / r_hat  # (B, T, 1)
            if functional == "kl":
                sigma = torch.sqrt(curvature_kl.clamp(min=eps))
            else:
                # hybrid: w_KL * (1/r) + w_chi2 * 2
                w_kl = self._hybrid_w_kl.item()
                w_chi2 = self._hybrid_w_chi2.item()
                combined_curvature = w_kl * curvature_kl + w_chi2 * 2.0
                sigma = torch.sqrt(combined_curvature.clamp(min=eps))
        else:  # chi2: f''(r) = 2
            sigma = math.sqrt(2.0)

        return model_output * sigma

    # ---- Hybrid weight update (§5.3) --------------------------------------

    def _update_hybrid_weights(self, loss: Tensor) -> None:
        """
        Update hybrid Lyapunov weights w_KL, w_chi2 using the exponential
        rule:  w_i ∝ exp(-η · ΔL_i / σ²_i)  (§5.3 of the CTRL-Flow paper).

        For simplicity we treat both components as having the same loss value
        (since we have a single prediction head) and update using the gradient
        magnitude as a proxy for per-divergence progress.
        """
        eta = self.config.hybrid_weight_eta
        # Use loss magnitude as a signal: when loss is large, favour KL
        # (coarse, fast convergence); when loss is small, increase chi²
        # (fine-grained density matching).
        signal = float(loss.item())
        # w_KL increases when loss is large (coarse phase)
        # w_chi2 increases when loss is small (fine-tuning phase)
        raw_kl = math.exp(-eta * signal)          # large loss → small exp → low KL weight ← correction
        raw_chi2 = math.exp(-eta / (signal + 1e-6))  # small loss → large chi2

        total = raw_kl + raw_chi2 + 1e-8
        new_w_kl = raw_kl / total
        new_w_chi2 = raw_chi2 / total

        # Soft update with EMA to prevent abrupt switches
        ema = self.config.adaptive_beta_ema
        self._hybrid_w_kl.copy_(ema * self._hybrid_w_kl + (1 - ema) * new_w_kl)
        self._hybrid_w_chi2.copy_(ema * self._hybrid_w_chi2 + (1 - ema) * new_w_chi2)


# ---------------------------------------------------------------------------
# Architecture components  (identical to DiffusionPolicy; renamed prefix)
# ---------------------------------------------------------------------------

class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax (Finn et al., 2016).  Computes keypoint coordinates
    from 2-D feature maps as the expected position under channel-wise softmax.
    """

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        return expected_xy.view(-1, self._out_c, 2)


class CtrlFlow2RgbEncoder(nn.Module):
    """ResNet-based RGB image encoder (identical to DiffusionRgbEncoder)."""

    def __init__(self, config: CtrlFlow2Config):
        super().__init__()

        if config.resize_shape is not None:
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.resize = None

        crop_shape = config.crop_shape
        if crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(crop_shape) if config.crop_is_random else self.center_crop
            )
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("Cannot replace BatchNorm in a pretrained model.")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        images_shape = next(iter(config.image_features.values())).shape
        if config.crop_shape is not None:
            dummy_shape_h_w = config.crop_shape
        elif config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.resize is not None:
            x = self.resize(x)
        if self.do_crop:
            x = self.maybe_random_crop(x) if self.training else self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        return self.relu(self.out(x))


class CtrlFlow2SinusoidalPosEmb(nn.Module):
    """1-D sinusoidal positional embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class CtrlFlow2Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish."""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class CtrlFlow2ConditionalUnet1d(nn.Module):
    """1-D UNet with FiLM conditioning (identical topology to DiffusionPolicy)."""

    def __init__(self, config: CtrlFlow2Config, global_cond_dim: int):
        super().__init__()
        self.config = config

        self.diffusion_step_encoder = nn.Sequential(
            CtrlFlow2SinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        cond_dim = config.diffusion_step_embed_dim + global_cond_dim
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList([
                    CtrlFlow2ConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                    CtrlFlow2ConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                    nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                ])
            )

        self.mid_modules = nn.ModuleList([
            CtrlFlow2ConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs),
            CtrlFlow2ConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs),
        ])

        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList([
                    CtrlFlow2ConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                    CtrlFlow2ConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                    nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                ])
            )

        self.final_conv = nn.Sequential(
            CtrlFlow2Conv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        x = einops.rearrange(x, "b t d -> b d t")
        timesteps_embed = self.diffusion_step_encoder(timestep)

        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return einops.rearrange(x, "b d t -> b t d")


class CtrlFlow2ConditionalResidualBlock1d(nn.Module):
    """ResNet-style 1-D residual block with FiLM modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = CtrlFlow2Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = CtrlFlow2Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        return out + self.residual_conv(x)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if parents:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module
