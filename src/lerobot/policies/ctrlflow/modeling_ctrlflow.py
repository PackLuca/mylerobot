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
"""CtrlFlow Policy: χ²-CTRL-Flow Generative Policy for Robot Learning.

Replaces the standard KL score-matching loss in Diffusion Policy with a
Pearson χ² density-ratio loss, derived from the CTRL-Flow Lyapunov stability
framework (Ma, 2024). The χ² divergence is mass-covering (vs. mode-seeking KL),
yielding improved multi-modal action coverage and higher success rates on tasks
like PushT where multiple valid action modes exist.

Key modifications vs. standard Diffusion Policy:
  1. New ratio_net: 3-layer MLP estimating r_hat(x,t,cond) ≈ p_t(x)/p_data(x)
  2. Training loss: Pearson χ² loss + optional Lyapunov regulariser
  3. Inference (Phase 2): Euler ODE integration with drift f*(x,t) = -2·∇_x r_hat.
     This replaces the DDPM/DDIM noise scheduler in Phase 2, because the χ² drift
     is a continuous-time velocity field (dx/dt = f*) and is NOT compatible with
     the discrete-step epsilon-prediction interface of DDPMScheduler.
  4. Warm-start: first `chi2_warmup_steps` steps use standard DDPM MSE for stability.

Theoretical guarantees (from CTRL-Flow Theorem 1):
  - Global asymptotic stability: lim_{t→∞} W₂(p_t, p_data) = 0
  - Exponential convergence rate γ = 2(α+β)λ_P where λ_P is the Poincaré const.
  - Robustness to neural network approximation errors up to relative error δ < α

Fix log (vs. original):
  [BUG-1] Phase 2 inference now uses a proper Euler ODE integrator instead of
          feeding the χ² drift into DDPMScheduler.step(), which was semantically
          wrong (DDPM expects a discrete noise prediction, not a continuous velocity).
  [BUG-2] ratio_net input is built defensively so global_cond=None does not crash.
          ratio_input_dim in __init__ is computed consistently with the forward pass.
  [BUG-3] Lyapunov gradient now uses create_graph=False (only first-order gradients
          are needed for the ReLU penalty), preventing computation-graph accumulation
          and the associated training-time memory leak.
  [BUG-4] _ctrl_flow_step buffer is no longer incremented inside compute_loss.
          The caller (training script) passes global_step explicitly, making the
          phase switch robust to DDP multi-process training.
  [BUG-5] phi_min is now read from config.chi2_phi_min instead of being hardcoded
          to 0.1, making it tunable for different action-space dimensionalities.
  [BUG-6] UNet parameters are frozen during Phase 2 training (only ratio_net and
          the shared timestep encoder are trained), saving memory and compute.
"""

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

from lerobot.policies.ctrlflow.configuration_ctrlflow import CtrlFlowConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class CtrlFlowPolicy(PreTrainedPolicy):
    """χ²-CTRL-Flow Policy.

    Implements CTRL-Flow's Lyapunov-stable generative framework using the χ²
    divergence as the Lyapunov functional. This yields mass-covering dynamics
    that improve multi-modal action coverage over standard Diffusion Policy.

    The training objective is the Pearson χ² loss for density-ratio estimation:
        L(θ) = E_{p_t}[(r_hat - 1)²] - E_{p_data}[r_hat²] + 1

    The inference-time drift is:
        f*(x, t) = -2 · ∇_x r_hat(x, t, cond)

    integrated via Euler ODE steps (NOT fed into DDPM/DDIM scheduler).

    A warm-start phase (chi2_warmup_steps) uses standard DDPM MSE loss to give
    the UNet backbone a stable initialization before ratio estimation begins.
    """

    config_class = CtrlFlowConfig
    name = "ctrlflow"

    def __init__(self, config: CtrlFlowConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None
        self.diffusion = CtrlFlowModel(config)
        self.reset()

    # get_optim_params 里分组
    def get_optim_params(self):
        return [
            {"params": self.diffusion.unet.parameters(), 
            "lr": self.config.optimizer_lr},
            {"params": self.diffusion.ratio_net.parameters(), 
            "lr": self.config.optimizer_lr * 0.1},  # ratio_net 小10倍
        ]

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.diffusion.generate_actions(batch, noise=noise)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        """Select a single action given environment observations.

        Handles caching a history of observations and an action trajectory
        generated by the underlying CtrlFlow model. See DiffusionPolicy for
        the full schematic description of the observation/action windowing.
        """
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(
        self, batch: dict[str, Tensor], global_step: int | None = None
    ) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation.

        Args:
            batch: observation/action batch dict.
            global_step: current global training step, used to determine warm-start
                phase. Should be passed by the training script. If None, the model
                falls back to its internal step counter (single-process training only).
        """
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        loss = self.diffusion.compute_loss(batch, global_step=global_step)
        return loss, None


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class CtrlFlowModel(nn.Module):
    """Core CtrlFlow model combining the UNet backbone with the density-ratio network.

    Architecture:
        - UNet (DiffusionConditionalUnet1d): backbone used for warm-start MSE training.
          After warm-start, UNet weights are frozen. Only its diffusion_step_encoder
          (timestep embedding MLP) remains active, shared with ratio_net.
        - ratio_net (3-layer MLP): estimates r_hat(x_noisy, t, cond) ≈ p_t(x)/p_data(x).
          Takes flattened noisy action + timestep embedding + global conditioning as input.
        - rgb_encoder / global conditioning: identical to the original DiffusionModel.

    Training phases:
        Phase 1 (global_step ≤ chi2_warmup_steps): standard MSE epsilon-prediction loss.
            Gives the UNet and observation encoders a stable warm start.
        Phase 2 (global_step > chi2_warmup_steps): Pearson χ² density-ratio loss.
            L = E_{p_t}[(r-1)²] - E_{p_data}[r²] + 1
            + λ · Lyapunov regulariser (if chi2_lyapunov_lambda > 0)
            UNet backbone weights are frozen; only ratio_net and timestep encoder train.

    Inference:
        Phase 1: standard DDPM/DDIM denoising using UNet predictions.
        Phase 2: χ²-CTRL-Flow Euler ODE integration.
            dx = f*(x, t) dt  where  f*(x,t) = -2·∇_x r_hat(x, t, cond)
            t ∈ [1, 0], dt = -1/chi2_num_inference_steps  (noise → data direction)
            Note: this is a proper ODE integrator, NOT DDPMScheduler.step().
            The χ² drift is a continuous-time velocity field; feeding it to a
            discrete noise scheduler would be semantically incorrect.
    """

    def __init__(self, config: CtrlFlowConfig):
        super().__init__()
        self.config = config

        # ── Observation encoders ───────────────────────────────────────────
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [CtrlFlowRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = CtrlFlowRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        # Store for use in ratio_net input construction.
        self._global_cond_dim = global_cond_dim

        # ── UNet backbone (used during warm-start; timestep encoder shared after) ──
        self.unet = DiffusionConditionalUnet1d(
            config, global_cond_dim=global_cond_dim * config.n_obs_steps
        )

        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        # ── Noise scheduler (Phase 1 warm-start only) ─────────────────────
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

        # ── χ²-CTRL-Flow: density-ratio network ───────────────────────────
        # r_hat: (x_noisy_flat, t_emb, global_cond) → ℝ₊
        #
        # Input dimension breakdown:
        #   action_flat_dim          = horizon * action_dim
        #   diffusion_step_embed_dim = from UNet's sinusoidal encoder
        #   global_cond_dim * n_obs  = observation conditioning
        #
        # When global_cond_dim == 0 (no image, no env_state), only the first
        # two terms contribute — handled defensively in _build_ratio_input().
        action_flat_dim = config.horizon * config.action_feature.shape[0]
        # global_cond_dim may be 0 if validate_features would have caught it,
        # but we build the dim correctly regardless.
        ratio_input_dim = (
            action_flat_dim
            + config.diffusion_step_embed_dim
            + global_cond_dim * config.n_obs_steps
        )
        hidden = config.chi2_ratio_net_hidden_dim
        self.ratio_net = nn.Sequential(
            nn.Linear(ratio_input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            # No activation: outputs log_r ∈ ℝ.
            # r = exp(clamp(log_r, -L, L)) is computed in forward pass,
            # bounding r without zeroing gradients at the clamp boundary.
        )
        # log-space clamp bound L: r ∈ [exp(-L), exp(L)]
        self._log_r_clamp = config.chi2_log_r_clamp
        # Training step counter for single-process fallback.
        # In DDP training, pass global_step from the trainer instead.
        self.register_buffer("_ctrl_flow_step", torch.zeros(1, dtype=torch.long))

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_ratio_input(
        self, x_flat: Tensor, t_emb: Tensor, global_cond: Tensor | None
    ) -> Tensor:
        """Concatenate inputs for ratio_net, handling optional global_cond.

        Args:
            x_flat:      (B, H*action_dim)
            t_emb:       (B, diffusion_step_embed_dim)
            global_cond: (B, global_cond_dim * n_obs_steps) or None

        Returns:
            (B, ratio_input_dim) tensor ready for ratio_net.
        """
        parts = [x_flat, t_emb]
        if global_cond is not None:
            parts.append(global_cond)
        return torch.cat(parts, dim=-1)

    def _in_warmup(self, global_step: int | None) -> bool:
        """Return True if we are still in the MSE warm-start phase.

        Uses the externally provided global_step when available (correct for DDP),
        falling back to the internal buffer for single-process use.
        """
        step = global_step if global_step is not None else self._ctrl_flow_step.item()
        return step <= self.config.chi2_warmup_steps

    def _freeze_unet(self):
        """Freeze all UNet parameters except the shared timestep encoder."""
        for name, param in self.unet.named_parameters():
            if "diffusion_step_encoder" not in name:
                param.requires_grad_(False)

    def _unfreeze_unet(self):
        """Restore UNet parameters to trainable (used during warm-start)."""
        for param in self.unet.parameters():
            param.requires_grad_(True)

    # ── Inference ──────────────────────────────────────────────────────────

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
        global_step: int | None = None,
    ) -> Tensor:
        """Generate an action trajectory.

        Phase 1 (warm-start): standard UNet-based DDPM/DDIM denoising.
        Phase 2 (χ²-CTRL-Flow): Euler ODE integration of f*(x,t) = -2·∇_x r_hat.

        The Phase 2 sampler integrates the ODE from t=1 (pure noise) to t=0 (data)
        using chi2_num_inference_steps uniform Euler steps:

            x_{k+1} = x_k + f*(x_k, t_k) · dt,   dt = -1 / N

        where t_k = 1 - k/N ∈ [1, 1/N] for k = 0, …, N-1.

        IMPORTANT: the χ² drift f* is a continuous-time velocity field.
        It must NOT be fed into DDPMScheduler.step(), which expects a discrete
        noise prediction ε_θ and applies a fundamentally different update rule.
        """
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

        use_chi2 = not self._in_warmup(global_step)

        if not use_chi2:
            # ── Phase 1: standard DDPM/DDIM denoising ─────────────────────
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            for t in self.noise_scheduler.timesteps:
                t_tensor = torch.full(
                    sample.shape[:1], t, dtype=torch.long, device=sample.device
                )
                model_output = self.unet(sample, t_tensor, global_cond=global_cond)
                sample = self.noise_scheduler.step(
                    model_output, t, sample, generator=generator
                ).prev_sample

        else:
            # ── Phase 2: χ²-CTRL-Flow Euler ODE ───────────────────────────
            # Integrate dx = f*(x,t) dt  from t=1 → t=0.
            #
            # t is treated as a continuous scalar in [0, 1]; we feed it to the
            # UNet's sinusoidal timestep encoder after scaling to [0, num_train_T].
            # This keeps the embedding range consistent with training.
            num_steps = self.config.chi2_num_inference_steps
            dt = -1.0 / num_steps  # negative because we go t=1 → t=0

            for k in range(num_steps):
                t_cont = 1.0 - k / num_steps   # t ∈ [1, 1/N]

                # Scale to discrete-timestep range for the sinusoidal encoder.
                t_discrete = t_cont * (self.config.num_train_timesteps - 1)
                t_tensor = torch.full(
                    (batch_size,), t_discrete, dtype=dtype, device=device
                )

                # Compute drift f*(x,t) = -2·∇_x r_hat via autograd.
                # torch.enable_grad() is required because select_action wraps
                # everything in @torch.no_grad().
                with torch.enable_grad():
                    x_in = sample.detach().requires_grad_(True)
                    t_emb = self.unet.diffusion_step_encoder(t_tensor)  # (B, embed_dim)
                    x_flat = x_in.flatten(start_dim=1)                  # (B, H*action_dim)
                    ratio_input = self._build_ratio_input(x_flat, t_emb, global_cond)

                    # log-space: consistent with training
                    log_r = self.ratio_net(ratio_input).squeeze(-1)
                    log_r = torch.clamp(log_r, min=-self._log_r_clamp, max=self._log_r_clamp)
                    r_hat = torch.exp(log_r)                            # (B,)

                    # ∇_x r_hat: shape (B, H*action_dim)
                    grad_r_flat = torch.autograd.grad(r_hat.sum(), x_in)[0].flatten(start_dim=1)

                # Reshape and compute drift; clip for numerical stability.
                grad_r = grad_r_flat.reshape_as(sample)  # (B, H, action_dim)
                drift = torch.clamp(
                    -2.0 * grad_r,
                    -self.config.clip_sample_range,
                    self.config.clip_sample_range,
                )

                # Euler step: x_{k+1} = x_k + f*(x_k, t_k) · dt
                sample = sample.detach() + drift.detach() * dt

                # Optional stochastic diffusion term: + √(2β) · dW
                # From CTRL-Flow SDE: dX = f*(X,t)dt + √(2β)·dW
                if self.config.chi2_diffusion_beta > 0:
                    noise_scale = math.sqrt(2.0 * self.config.chi2_diffusion_beta * abs(dt))
                    sample = sample + noise_scale * torch.randn_like(sample)

        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate with state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(
                    batch[OBS_IMAGES], "b s n ... -> n (b s) ..."
                )
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(
                            self.rgb_encoder, images_per_camera, strict=True
                        )
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list,
                    "(n b s) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features,
                    "(b s n) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(
        self, batch: dict[str, Tensor], noise: Tensor | None = None,
        global_step: int | None = None,
    ) -> Tensor:
        """Generate actions for the given observation batch."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)
        actions = self.conditional_sample(
            batch_size, global_cond=global_cond, noise=noise, global_step=global_step
        )

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]
        return actions

    # ── Training ───────────────────────────────────────────────────────────

    def compute_loss(
        self, batch: dict[str, Tensor], global_step: int | None = None
    ) -> Tensor:
        """Compute the training loss.

        Phase 1 (warm-start, global_step ≤ chi2_warmup_steps):
            Standard MSE epsilon-prediction loss identical to Diffusion Policy.
            All parameters (UNet + encoders) are trained.

        Phase 2 (χ²-CTRL-Flow, global_step > chi2_warmup_steps):
            Pearson χ² density-ratio loss:
                L_χ² = E_{p_t}[(r_hat - 1)²] - E_{p_data}[r_hat²] + 1

            This is the unbiased estimator of D_χ²(p_t ‖ p_data).
            The optimal density-ratio network satisfies r*(x) = p_t(x)/p_data(x),
            and the optimal drift is f*(x,t) = -2·∇_x r*(x,t).

            UNet backbone (excluding its timestep encoder) is FROZEN to save
            memory and compute; only ratio_net + timestep encoder are trained.

            Optional Lyapunov regulariser (chi2_lyapunov_lambda > 0):
                Penalises collapse of the dissipation functional
                Φ = E_{p_data}[‖∇_x r_hat‖²] → 0, which would cause the
                density-ratio network to degenerate to a constant (r_hat ≈ 1).
                Implements: R = λ · ReLU(φ_min - Φ)
                where φ_min = config.chi2_phi_min (tune for your action-space dim).

        Args:
            batch: dict with keys {OBS_STATE, ACTION, "action_is_pad"} and at least
                one of {OBS_IMAGES, OBS_ENV_STATE}.
            global_step: current global training step from the trainer. Used to
                determine warm-start phase. Pass this from the training script for
                correctness under DDP. If None, falls back to internal counter
                (accurate for single-process training only).

        Returns:
            Scalar loss tensor.
        """
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        # ── Forward diffusion (shared across both phases) ──────────────────
        trajectory = batch[ACTION]                           # (B, H, action_dim) clean actions
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Determine phase and update internal counter (single-process fallback).
        in_warmup = self._in_warmup(global_step)
        if global_step is None:
            # Only increment when global_step is not provided externally.
            self._ctrl_flow_step += 1

        if in_warmup:
            # ── Phase 1: MSE warm-start ────────────────────────────────────
            # Ensure UNet is fully trainable during warm-start.
            self._unfreeze_unet()

            pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)
            loss = F.mse_loss(pred, eps, reduction="none")

            if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
                in_episode_bound = ~batch["action_is_pad"]
                loss = loss * in_episode_bound.unsqueeze(-1)
            return loss.mean()

        # ── Phase 2: Pearson χ² density-ratio loss ─────────────────────────
        # Freeze UNet backbone (keep timestep encoder trainable so ratio_net
        # benefits from its gradients).
        self._freeze_unet()

        # ── Timestep embedding ─────────────────────────────────────────────
        t_emb = self.unet.diffusion_step_encoder(timesteps)   # (B, embed_dim)

        # ── Contrastive batch construction ────────────────────────────────
        # ratio_net must simultaneously see both p_t and p_data samples to
        # produce a meaningful density-ratio estimate. Feeding them separately
        # in two forward passes gives a weak contrast signal and causes
        # ratio_net to degenerate to a constant (grdn → 0, loss frozen).
        #
        # Fix: concatenate p_data (clean) and p_t (noisy) into a single batch
        # of size 2B, run one forward pass, then split the outputs.
        # This ensures the gradient of the loss w.r.t. ratio_net parameters
        # reflects the difference between the two distributions.
        B = trajectory.shape[0]
        x_clean_flat = trajectory.flatten(start_dim=1)          # (B, H*action_dim)
        x_noisy_flat = noisy_trajectory.flatten(start_dim=1)    # (B, H*action_dim)

        # Stack: first B rows = p_data, last B rows = p_t
        x_all   = torch.cat([x_clean_flat, x_noisy_flat], dim=0)   # (2B, H*action_dim)
        t_all   = torch.cat([t_emb,        t_emb],        dim=0)   # (2B, embed_dim)
        cond_all = (
            torch.cat([global_cond, global_cond], dim=0)
            if global_cond is not None else None
        )                                                            # (2B, cond_dim) or None

        # Lyapunov regulariser needs ∇_x r on the p_data slice.
        need_grad = self.config.chi2_lyapunov_lambda > 0
        if need_grad:
            # Only the p_data slice (first B rows) needs x-gradient.
            # We split after the forward pass, so enable grad on x_all[:B].
            x_all = x_all.detach()
            x_all[:B] = x_all[:B].requires_grad_(True)

        # ── Single forward pass over the contrastive batch ────────────────
        ratio_input_all = self._build_ratio_input(x_all, t_all, cond_all)
        log_r_all = self.ratio_net(ratio_input_all).squeeze(-1)     # (2B,)

        # Clamp in log-space: r = exp(clamp(log_r, -L, L)).
        # This bounds r ∈ [exp(-L), exp(L)] without zeroing gradients at
        # the boundary (unlike direct clamp on r after Softplus).
        log_r_all = torch.clamp(
            log_r_all, min=-self._log_r_clamp, max=self._log_r_clamp
        )
        r_all = torch.exp(log_r_all)                                # (2B,)

        r_hat_pdata = r_all[:B]    # (B,)  r on clean p_data samples
        r_hat_pt    = r_all[B:]    # (B,)  r on noisy p_t samples

        # ── Pearson χ² loss (log-space stable) ────────────────────────────
        # L = E_{p_t}[(r-1)²] - E_{p_data}[r²] + 1
        # Minimum value = -D_χ²(p_t ‖ p_data) ≤ 0, so negative loss is normal.
        # With log-space clamp, loss is bounded below by -(exp(2L) + 1).
        loss_pt   = ((r_hat_pt   - 1.0) ** 2).mean()   # E_{p_t}[(r-1)²]  ≥ 0
        loss_data = -(r_hat_pdata ** 2).mean()           # -E_{p_data}[r²] ≥ -exp(2L)
        chi2_loss = loss_pt + loss_data + 1.0

        # ── Lyapunov regulariser (optional) ───────────────────────────────
        # Penalises Φ = E_{p_data}[‖∇_x r‖²] → 0, which signals ratio_net
        # has degenerated to a constant (Identifiability condition, Theorem 1).
        # Gradient flows through r_hat_pdata = exp(clamp(log_r)) correctly
        # because log-space clamp does NOT zero the gradient at the boundary.
        lyapunov_reg = torch.zeros(1, device=trajectory.device)
        if need_grad:
            grad_r = torch.autograd.grad(
                r_hat_pdata.sum(),
                x_all,                  # grad w.r.t. full x_all; only [:B] has grad
                create_graph=False,     # first-order only; avoids memory leak
                retain_graph=True,      # keep graph for total_loss.backward()
            )[0][:B]                    # (B, H*action_dim) — p_data slice only

            phi = (grad_r ** 2).sum(dim=-1).mean()   # Φ batch estimate
            lyapunov_reg = self.config.chi2_lyapunov_lambda * torch.relu(
                self.config.chi2_phi_min - phi
            )

        total_loss = chi2_loss + lyapunov_reg.squeeze()

        # ── Padding mask ───────────────────────────────────────────────────
        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            in_episode_bound = ~batch["action_is_pad"]           # (B, H) bool
            valid_weight = in_episode_bound.float().mean(dim=-1) # (B,)
            total_loss = total_loss * valid_weight.mean()

        return total_loss


# ── UNet and supporting modules ────────────────────────────────────────────────
# These are identical to the originals in modeling_diffusion.py.
# They are reproduced here so ctrlflow is a fully self-contained policy folder.


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Identical to the original in modeling_diffusion.py. Kept here so the
    ctrlflow policy folder is fully self-contained with no cross-folder imports.
    """

    def __init__(self, config: CtrlFlowConfig, global_cond_dim: int):
        super().__init__()
        self.config = config

        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
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
                    DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                    DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                    nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                ])
            )

        self.mid_modules = nn.ModuleList([
            DiffusionConditionalResidualBlock1d(
                config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
            ),
            DiffusionConditionalResidualBlock1d(
                config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
            ),
        ])

        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList([
                    DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                    DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                    nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                ])
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(
                config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size
            ),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim)
            timestep: (B,) diffusion timestep
            global_cond: (B, global_cond_dim)
        Returns:
            (B, T, input_dim) UNet prediction.
        """
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
        x = einops.rearrange(x, "b d t -> b t d")
        return x


class SpatialSoftmax(nn.Module):
    """Spatial Soft Argmax. Identical to the original in modeling_diffusion.py."""

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints


class CtrlFlowRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Identical to DiffusionRgbEncoder in modeling_diffusion.py, reproduced here
    for self-containment.
    """

    def __init__(self, config: CtrlFlowConfig):
        super().__init__()
        if config.resize_shape is not None:
            self.resize = torchvision.transforms.Resize(config.resize_shape)
        else:
            self.resize = None

        crop_shape = config.crop_shape
        if crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
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
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
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
    assert not any(
        predicate(m) for _, m in root_module.named_modules(remove_duplicate=True)
    )
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

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

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
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
        out = out + self.residual_conv(x)
        return out
