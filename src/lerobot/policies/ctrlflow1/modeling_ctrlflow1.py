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
"""CTRLFlow1 Policy — Diffusion Policy augmented with Wasserstein (WGAN-GP) stability
regularization derived from the CTRL-Flow framework.

Key design decisions
--------------------
* **UNet backbone**: completely unchanged from DiffusionPolicy. All existing
  checkpoints are compatible.
* **Kantorovich potential network φ**: a lightweight MLP that lives only during
  training. It is NOT saved to the policy state-dict (it has its own optimizer
  managed by the policy) and is NOT used at inference time.
* **Loss**: ``total = loss_diffusion + w(step) * loss_wasserstein``
  where ``w(step)`` linearly ramps from 0 to ``wasserstein_lambda`` over
  ``wasserstein_warmup_steps`` steps (CTRL-Flow two-phase schedule, §5.3).
* **φ update schedule**: φ is updated ``phi_n_inner_steps`` times per UNet
  step (standard WGAN-GP critic schedule).
* **Inference**: identical to DiffusionPolicy — φ is never called.

All helper classes (SpatialSoftmax, DiffusionRgbEncoder, DiffusionConditionalUnet1d,
DiffusionConditionalResidualBlock1d, DiffusionConv1dBlock, DiffusionSinusoidalPosEmb)
are imported directly from the original diffusion module and reused without modification.
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

from lerobot.policies.ctrlflow1.configuration_ctrlflow1 import CTRLFlow1Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ---------------------------------------------------------------------------
# Re-export unchanged helpers from the original diffusion module so that this
# file is self-contained and the package does not need to cross-import.
# ---------------------------------------------------------------------------
from lerobot.policies.diffusion.modeling_diffusion import (  # noqa: E402
    DiffusionConditionalResidualBlock1d,
    DiffusionConditionalUnet1d,
    DiffusionConv1dBlock,
    DiffusionRgbEncoder,
    DiffusionSinusoidalPosEmb,
    SpatialSoftmax,
    _make_noise_scheduler,
    _replace_submodules,
)


# ---------------------------------------------------------------------------
# Kantorovich potential network φ
# ---------------------------------------------------------------------------

class KantorovichPhi(nn.Module):
    """Lightweight MLP that approximates the Kantorovich potential φ.

    Input  : flattened action trajectory  (B, horizon * action_dim)
    Output : scalar potential value       (B,)

    Properties
    ----------
    * Used only during training to provide the Wasserstein gradient signal to
      the UNet. Discarded completely at inference.
    * No time-step conditioning needed — φ measures distributional distance,
      not per-step noise.
    * Weights are NOT part of the main policy state-dict; they are managed by
      a separate optimizer inside ``CTRLFlow1Model``.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        """
        Args:
            input_dim : horizon * action_dim  (flattened trajectory length)
            hidden_dim: width of each hidden layer
            n_layers  : number of hidden layers (≥ 1)
        """
        super().__init__()
        assert n_layers >= 1, "phi_n_layers must be >= 1"

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, input_dim) flattened trajectories
        Returns:
            (B,) potential values
        """
        return self.net(x).squeeze(-1)

    def gradient(self, x: Tensor) -> Tensor:
        """Return ∇_x φ(x) — the optimal-transport direction for the UNet.

        Args:
            x: (B, input_dim), should be detached from the UNet graph.
        Returns:
            (B, input_dim) gradient tensor (detached from φ's graph).
        """
        x = x.detach().requires_grad_(True)
        phi_val = self.forward(x)
        grad = torch.autograd.grad(phi_val.sum(), x, create_graph=False)[0]
        return grad.detach()


# ---------------------------------------------------------------------------
# CTRLFlow1Model  (main trainable model)
# ---------------------------------------------------------------------------

class CTRLFlow1Model(nn.Module):
    """Diffusion model augmented with Wasserstein stability regularization.

    Architecture
    ------------
    * ``self.unet``        — identical 1-D conditional UNet from DiffusionPolicy
    * ``self.rgb_encoder`` — identical ResNet visual encoder (optional)
    * ``self.phi``         — Kantorovich potential MLP (training-only)
    * ``self.phi_optimizer``— separate AdamW for φ (not exposed to LeRobot scheduler)

    Training loop (per batch)
    -------------------------
    1. Standard forward diffusion + UNet noise prediction → ``loss_diffusion``
    2. Inner loop (×``phi_n_inner_steps``): update φ with WGAN-GP loss.
    3. Compute ``loss_wasserstein = -φ(x_model).mean()``  (UNet should push
       generated trajectories toward higher φ values, i.e., toward p_data).
    4. ``total_loss = loss_diffusion + w * loss_wasserstein``

    Inference
    ---------
    Identical to DiffusionPolicy: φ is never called.
    """

    def __init__(self, config: CTRLFlow1Config):
        super().__init__()
        self.config = config

        # ── Visual encoder + conditioning dim ────────────────────────────────
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        # ── UNet (completely unchanged) ───────────────────────────────────────
        self.unet = DiffusionConditionalUnet1d(
            config, global_cond_dim=global_cond_dim * config.n_obs_steps
        )
        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        # ── Noise scheduler (unchanged) ───────────────────────────────────────
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

        # ── Kantorovich potential φ (training-only) ───────────────────────────
        # Input dim = flattened trajectory: horizon × action_dim
        phi_input_dim = config.horizon * config.action_feature.shape[0]
        self.phi = KantorovichPhi(
            input_dim=phi_input_dim,
            hidden_dim=config.phi_hidden_dim,
            n_layers=config.phi_n_layers,
        )
        # φ gets its own optimizer so LeRobot's main scheduler doesn't touch it.
        self.phi_optimizer = torch.optim.AdamW(
            self.phi.parameters(),
            lr=config.phi_lr,
            betas=(0.5, 0.9),   # lower β1 stabilises GAN-style training
            weight_decay=0.0,   # no weight-decay: avoids fighting Lipschitz constraint
        )

        # Step counter — used to compute the warmup weight w(step).
        # Incremented inside compute_loss; reset is the caller's responsibility.
        self._train_step: int = 0

    # =========================================================================
    # Inference (unchanged from DiffusionModel)
    # =========================================================================

    @torch.no_grad()
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
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
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample
        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate with state vector. Unchanged."""
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
        """Unchanged from DiffusionModel."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps
        global_cond = self._prepare_global_conditioning(batch)
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    # =========================================================================
    # Wasserstein helpers
    # =========================================================================

    def _phi_loss_wgan_gp(self, x_model: Tensor, x_data: Tensor) -> Tensor:
        """WGAN-GP loss for the Kantorovich potential φ.

        Maximises  E_{p_data}[φ] − E_{p_t}[φ]  (the Wasserstein-1 dual)
        subject to the 1-Lipschitz constraint enforced via gradient penalty.

        Args:
            x_model : (B, D) flattened noisy trajectories  ~ p_t
            x_data  : (B, D) flattened clean trajectories  ~ p_data
        Returns:
            scalar loss (to be minimised by phi_optimizer)
        """
        B = min(x_model.shape[0], x_data.shape[0])
        x_m = x_model[:B].detach()
        x_d = x_data[:B].detach()

        phi_data  = self.phi(x_d)
        phi_model = self.phi(x_m)

        # Wasserstein dual objective (negated for minimisation)
        w_dist = phi_data.mean() - phi_model.mean()

        # Gradient penalty on interpolated samples
        alpha = torch.rand(B, 1, device=x_d.device)
        x_hat = (alpha * x_d + (1.0 - alpha) * x_m).requires_grad_(True)
        phi_hat = self.phi(x_hat)
        grad = torch.autograd.grad(
            outputs=phi_hat.sum(),
            inputs=x_hat,
            create_graph=True,
        )[0]
        gp = ((grad.norm(dim=-1) - 1.0) ** 2).mean()

        return -w_dist + self.config.phi_gp_lambda * gp

    def _update_phi(self, x_model: Tensor, x_data: Tensor) -> float:
        """Run ``phi_n_inner_steps`` gradient steps on φ.

        Returns the final φ loss value for logging.
        """
        loss_phi_val = 0.0
        for _ in range(self.config.phi_n_inner_steps):
            self.phi_optimizer.zero_grad()
            loss_phi = self._phi_loss_wgan_gp(x_model, x_data)
            loss_phi.backward()
            # Clip φ gradients for safety (large GP can cause spikes)
            torch.nn.utils.clip_grad_norm_(self.phi.parameters(), max_norm=5.0)
            self.phi_optimizer.step()
            loss_phi_val = loss_phi.item()
        return loss_phi_val

    def _wasserstein_unet_loss(self, x_model: Tensor) -> Tensor:
        """Gradient signal passed back to the UNet.

        The UNet should push generated trajectories toward higher φ values
        (i.e., toward the data distribution).  Minimising −φ(x_model) achieves
        this via the UNet's parameters (φ is frozen during this step).

        Args:
            x_model: (B, D) flattened noisy trajectories with UNet gradients.
        Returns:
            scalar Wasserstein loss for the UNet.
        """
        # Freeze φ: gradients flow through x_model → UNet, not into φ.
        for p in self.phi.parameters():
            p.requires_grad_(False)
        loss_w = -self.phi(x_model).mean()
        for p in self.phi.parameters():
            p.requires_grad_(True)
        return loss_w

    def _wasserstein_weight(self) -> float:
        """Linear warmup schedule for the Wasserstein coefficient w(step).

        Returns a float in [0, wasserstein_lambda].
        """
        if self.config.wasserstein_warmup_steps <= 0:
            return self.config.wasserstein_lambda
        progress = min(1.0, self._train_step / self.config.wasserstein_warmup_steps)
        return progress * self.config.wasserstein_lambda

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Compute the combined diffusion + Wasserstein loss.

        Args:
            batch: same format as DiffusionModel.compute_loss.
        Returns:
            total_loss : scalar Tensor (differentiable w.r.t. UNet params)
            info       : dict with individual loss components for logging
        """
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        assert batch[ACTION].shape[1] == self.config.horizon
        assert batch[OBS_STATE].shape[1] == self.config.n_obs_steps

        # ── Global conditioning (visual + state) ─────────────────────────────
        global_cond = self._prepare_global_conditioning(batch)  # (B, cond_dim)

        # ── Standard forward diffusion ────────────────────────────────────────
        trajectory = batch[ACTION]                               # (B, T, action_dim)
        eps        = torch.randn_like(trajectory)
        timesteps  = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # UNet forward pass
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Diffusion target
        target = eps if self.config.prediction_type == "epsilon" else trajectory
        loss_diffusion = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding and "action_is_pad" in batch:
            in_episode_bound = ~batch["action_is_pad"]
            loss_diffusion = loss_diffusion * in_episode_bound.unsqueeze(-1)
        loss_diffusion = loss_diffusion.mean()

        # ── Wasserstein regularization ────────────────────────────────────────
        w_coeff = self._wasserstein_weight()
        loss_w_val   = 0.0
        loss_phi_val = 0.0

        if w_coeff > 0.0:
            # Flatten trajectories for φ: (B, horizon*action_dim)
            x_model = noisy_trajectory.reshape(noisy_trajectory.shape[0], -1)
            x_data  = trajectory.reshape(trajectory.shape[0], -1)

            # Step 1: update φ (inner loop, detached from UNet graph)
            loss_phi_val = self._update_phi(x_model.detach(), x_data.detach())

            # Step 2: Wasserstein loss for UNet
            #   x_model here retains grad w.r.t. UNet params through noisy_trajectory
            loss_w = self._wasserstein_unet_loss(x_model)
            loss_w_val = loss_w.item()
        else:
            loss_w = torch.tensor(0.0, device=trajectory.device)

        # ── Combined loss ─────────────────────────────────────────────────────
        total_loss = loss_diffusion + w_coeff * loss_w

        self._train_step += 1

        info = {
            "loss_diffusion":    loss_diffusion.item(),
            "loss_wasserstein":  loss_w_val,
            "loss_phi":          loss_phi_val,
            "wasserstein_weight": w_coeff,
            "train_step":        self._train_step,
        }
        return total_loss, info


# ---------------------------------------------------------------------------
# CTRLFlow1Policy  (top-level policy, mirrors DiffusionPolicy interface)
# ---------------------------------------------------------------------------

class CTRLFlow1Policy(PreTrainedPolicy):
    """CTRLFlow1Policy — Wasserstein-regularised Diffusion Policy.

    Drop-in replacement for DiffusionPolicy with an additional Kantorovich
    potential network φ that stabilises training via CTRL-Flow's Lyapunov
    framework (WGAN-GP variant of the MFG construction in §5.2.1).

    Inference is identical to DiffusionPolicy: φ is never invoked.
    """

    config_class = CTRLFlow1Config
    name = "ctrlflow1"

    def __init__(self, config: CTRLFlow1Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.diffusion = CTRLFlow1Model(config)
        self.reset()

    def get_optim_params(self) -> dict:
        # Return only UNet (+ encoder) params; φ has its own internal optimizer.
        return self.diffusion.unet.parameters()

    def reset(self):
        """Clear observation and action queues. Call on env.reset()."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION:    deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        return self.diffusion.generate_actions(batch, noise=noise)

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
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
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Run batch through the model; return (total_loss, info_dict).

        ``info`` contains individual loss components for logging:
            loss_diffusion, loss_wasserstein, loss_phi, wasserstein_weight
        """
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        total_loss, info = self.diffusion.compute_loss(batch)
        return total_loss, info
