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
"""CTRL-Flow Policy — faithful implementation of Ma (2024) Theorem 1.

══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL SPECIFICATION
══════════════════════════════════════════════════════════════════════════════

Paper SDE (§4.3.1):
    f(x, t) = -∇_x log( p_t(x) / p_data(x) )
    g = sqrt(2) · I

Probability path (straight-line interpolation):
    x_t = (1 - t) · x_0  +  t · ε,    t ∈ [0, t_max]
    x_0 ~ p_data,   ε ~ N(0, I),   t_max = 1 - t_min_gap < 1

Training target — score difference along the straight path:

    -∇_{x_t} log p_t(x_t | x_0)  =  -(x_t - x_0) / (1-t)²  · (1-t)
                                   =  (x_0 - x_t) / (1-t)
                                   =  (x_0 - ε) · (1-t) / (1-t)   ... wait

    More carefully: x_t = (1-t)x_0 + tε, so
        x_t - x_0 = t(ε - x_0)
    The conditional "drift" pointing back to x_0 from x_t is:
        (x_0 - x_t) / (1-t)  =  -t(ε - x_0)/(1-t)  =  t(x_0 - ε)/(1-t)

    But we want -∇ log(p_t/p_data), which for the straight-path marginal
    is approximated (in expectation over x_0|x_t) as the vector field
    that generates the path, rescaled to unit-time parametrisation:

        f(x_t, t)  ≈  E[x_0 - ε | x_t] / (1 - t)

    The regression target for a single (x_0, ε) sample is therefore:

        target(x_0, ε, t)  =  (x_0 - ε) / (1 - t)              ... (★)

    Note: (x_0 - ε) is the unnormalised tangent; dividing by (1-t) converts
    it to the correct score-difference units (action / time), matching the
    physical dimensions of f in the Fokker-Planck equation.

Inference — Euler-Maruyama reverse SDE (t_max → 0):
    x_{k+1} = x_k  +  Δt · f_θ(x_k, t_k)  +  sqrt(2·Δt) · ε_k
    ε_k ~ N(0, I),   Δt = t_max / N,   t_k = t_max - k·Δt

    g = sqrt(2)·I → diffusion std per step = sqrt(2·Δt)            ... (★★)

Initial condition:
    x_0 ~ N(0, I)    [marginal of x_{t_max} when t_max ≈ 1]

Lyapunov regulariser (Theorem 1, Condition 1):
    R(θ) = ||f_θ(x_t, t)||²  -  ||target(t)||²
    L_lyapunov = E[ max(0, R(θ)) ]

    Both f_θ and target carry the 1/(1-t) factor → R is dimensionless
    and the hinge penalises magnitude overshoot of the predicted drift.

Total loss:
    L = score_loss_weight · MSE(f_θ(x_t, t), target)
      + lyapunov_lambda   · E[max(0, R(θ))]
"""

import math
from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionRgbEncoder,
)
from lerobot.policies.ctrlflow.configuration_ctrlflow import CTRLFlowConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ================================================================== #
#  Top-level policy wrapper                                            #
# ================================================================== #

class CTRLFlowPolicy(PreTrainedPolicy):
    """CTRL-Flow Policy (Ma, 2024).

    Trains f_θ ≈ -∇_x log(p_t/p_data) via score-difference regression
    on a straight-line probability path, then samples via the Lyapunov-
    stable reverse SDE:

        dX = f_θ(X, t) dt  +  sqrt(2) dW̄_t,    g = sqrt(2)·I
    """

    config_class = CTRLFlowConfig
    name = "ctrlflow"

    def __init__(self, config: CTRLFlowConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.ctrlflow = CTRLFlowModel(config)
        self.reset()

    def get_optim_params(self):
        return self.ctrlflow.parameters()

    def reset(self):
        """Clear observation / action queues. Call on env.reset()."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION:    deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor],
                             noise: Tensor | None = None) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1)
                 for k in batch if k in self._queues}
        return self.ctrlflow.generate_actions(batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor],
                      noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack(
                [batch[k] for k in self.config.image_features], dim=-4
            )
        self._queues = populate_queues(self._queues, batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack(
                [batch[k] for k in self.config.image_features], dim=-4
            )
        return self.ctrlflow.compute_loss(batch), None


# ================================================================== #
#  Core model                                                          #
# ================================================================== #

class CTRLFlowModel(nn.Module):
    """Core CTRL-Flow model.

    The UNet f_θ directly approximates the drift:
        f_θ(x_t, t) ≈ -∇_x log( p_t(x_t) / p_data(x_t) )
                     ≈ (x_0 - ε) / (1 - t)            [along straight path]

    No σ-rescaling is needed outside the network because the network is
    trained to output the physically-dimensioned score difference directly.
    """

    def __init__(self, config: CTRLFlowConfig):
        super().__init__()
        self.config = config

        # ---- Observation encoders ------------------------------------ #
        global_cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        # ---- UNet: f_θ(x_t, t) ≈ (x_0 - ε)/(1-t) ------------------- #
        # The UNet receives t ∈ [0, t_max] as a float "timestep".
        # The sinusoidal / MLP embedding in DiffusionConditionalUnet1d
        # handles continuous-valued inputs directly.
        self.unet = DiffusionConditionalUnet1d(
            config, global_cond_dim=global_cond_dim * config.n_obs_steps
        )
        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

    # ------------------------------------------------------------------ #
    #  Observation encoding                                               #
    # ------------------------------------------------------------------ #

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode all observations into a single (B, C) vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_feats = torch.cat(
                    [enc(im) for enc, im in zip(self.rgb_encoder, imgs, strict=True)]
                )
                img_feats = einops.rearrange(
                    img_feats, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_feats = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_feats = einops.rearrange(
                    img_feats, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            feats.append(img_feats)

        if self.config.env_state_feature:
            feats.append(batch[OBS_ENV_STATE])

        return torch.cat(feats, dim=-1).flatten(start_dim=1)  # (B, C)

    # ------------------------------------------------------------------ #
    #  Inference: Euler-Maruyama reverse SDE                              #
    # ------------------------------------------------------------------ #

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Sample action trajectories via the CTRL-Flow reverse SDE.

        Reverse SDE (paper Theorem 1, g = sqrt(2)·I):
            dX = f_θ(X, t) dt  +  sqrt(2) dW̄_t

        Euler-Maruyama update at step k (t decreasing from t_max to 0):
            t_k     = t_max - k · Δt
            x_{k+1} = x_k  +  Δt · f_θ(x_k, t_k)       ← drift
                            +  sqrt(2·Δt) · ε_k           ← diffusion (g=√2)
            ε_k ~ N(0, I)

        Diffusion is skipped on the final step (k = N-1, t_k → 0) to
        avoid injecting noise when the sample is near the data manifold.

        Initial sample:
            x ~ N(0, I)   [matches marginal of x_{t_max} when t_max ≈ 1
                           and actions are normalised to zero mean, unit std]

        Args:
            batch_size: Number of action trajectories to generate.
            global_cond: (B, C) observation conditioning vector.
            generator: Optional torch.Generator for reproducibility.
            noise: Optional pre-sampled initial x ~ N(0, I).

        Returns:
            (B, horizon, action_dim) action trajectories.
        """
        cfg    = self.config
        device = get_device_from_parameters(self)
        dtype  = get_dtype_from_parameters(self)

        action_dim = cfg.action_feature.shape[0]

        # ── Initial sample: x ~ N(0, I) ────────────────────────────────
        # Matches the marginal of x_{t_max} = (1-t_max)·x_0 + t_max·ε
        # ≈ ε when t_max ≈ 1 and x_0 is O(1).
        if noise is not None:
            x = noise
        else:
            x = torch.randn(
                (batch_size, cfg.horizon, action_dim),
                dtype=dtype, device=device, generator=generator,
            )

        dt       = cfg.dt                               # Δt = t_max / N
        diff_std = cfg.g_coeff * math.sqrt(dt)          # sqrt(2·Δt)

        # ── Euler-Maruyama loop: t_max → 0 ─────────────────────────────
        for k in range(cfg.num_sde_steps):
            # Current pseudo-time (decreasing)
            t_k = cfg.t_max - k * dt                    # scalar

            # Batch timestep tensor for UNet conditioning
            t_batch = torch.full(
                (batch_size,), t_k, dtype=dtype, device=device
            )                                           # (B,)

            # f_θ(x_k, t_k) ≈ (x_0 - ε) / (1 - t_k)
            drift = self.unet(x, t_batch, global_cond=global_cond)  # (B,T,D)

            # ── Drift step: Δt · f_θ ───────────────────────────────────
            x = x + dt * drift

            # ── Diffusion step: sqrt(2·Δt) · ε  (g = sqrt(2)·I) ───────
            # Omit on the final step (trajectory has reached data manifold).
            is_last_step = (k == cfg.num_sde_steps - 1)
            if not is_last_step and diff_std > 0.0:
                eps = torch.randn_like(x, generator=generator)
                x = x + diff_std * eps

        return x

    def generate_actions(self, batch: dict[str, Tensor],
                         noise: Tensor | None = None) -> Tensor:
        """Encode observations and run CTRL-Flow sampler.

        Returns:
            (B, n_action_steps, action_dim) actions for execution.
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)
        actions     = self.conditional_sample(
            batch_size, global_cond=global_cond, noise=noise
        )
        start = n_obs_steps - 1
        end   = start + self.config.n_action_steps
        return actions[:, start:end]

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute the CTRL-Flow training loss.

        ── Step-by-step ────────────────────────────────────────────────────

        1. Sample time:
               t ~ Uniform(0, t_max),    t_max = 1 - t_min_gap

        2. Forward process (straight-line interpolation):
               ε  ~ N(0, I)
               x_t = (1 - t) · x_0  +  t · ε

        3. Training target — score difference (★):
               target = (x_0 - ε) / (1 - t)

           Derivation:
               x_t = (1-t)·x_0 + t·ε
               dx_t/dt = ε - x_0             [tangent of straight path]
               f(x_t,t) = -∇ log(p_t/p_data) ≈ (x_0 - ε) / (1 - t)
                          ↑ rescale tangent to score-difference units

        4. UNet prediction:
               pred = f_θ(x_t, t)

        5. Score-matching loss:
               L_score = MSE(pred, target)

        6. Lyapunov regulariser (Theorem 1, Condition 1) (★★):
               R(θ) = ||pred||² - ||target||²
               L_lyapunov = E[max(0, R(θ))]

           Both pred and target carry the same 1/(1-t) factor, so R
           is dimensionally consistent. The hinge penalises overshoot.

        7. Total loss:
               L = score_loss_weight · L_score
                 + lyapunov_lambda   · L_lyapunov

        Args:
            batch: Must contain OBS_STATE, ACTION, "action_is_pad",
                   and at least one of OBS_IMAGES / OBS_ENV_STATE.

        Returns:
            Scalar loss tensor.
        """
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch

        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon     = batch[ACTION].shape[1]
        assert horizon     == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        cfg         = self.config
        x0          = batch[ACTION]                     # (B, T, D) clean actions
        B           = x0.shape[0]
        device      = x0.device
        dtype       = x0.dtype

        global_cond = self._prepare_global_conditioning(batch)

        # ── 1. Sample time t ~ Uniform(0, t_max) ───────────────────────
        t = torch.rand(B, device=device, dtype=dtype) * cfg.t_max  # (B,)
        # Reshape for broadcasting over (T, D)
        t_view = t.view(B, 1, 1)                        # (B, 1, 1)

        # ── 2. Forward process: x_t = (1-t)·x_0 + t·ε ─────────────────
        eps = torch.randn_like(x0)                      # (B, T, D) ~ N(0,I)
        x_t = (1.0 - t_view) * x0 + t_view * eps       # (B, T, D)

        # ── 3. Training target: (x_0 - ε) / (1 - t) ───────────────────
        # (1 - t) is bounded away from 0 by t_max = 1 - t_min_gap,
        # so (1 - t) >= t_min_gap > 0 always.
        one_minus_t = (1.0 - t_view)                    # (B, 1, 1) >= t_min_gap
        target = (x0 - eps) / one_minus_t               # (B, T, D)

        # ── 4. UNet prediction: f_θ(x_t, t) ────────────────────────────
        pred = self.unet(x_t, t, global_cond=global_cond)           # (B, T, D)

        # ── 5. Score-matching loss ──────────────────────────────────────
        score_loss = F.mse_loss(pred, target, reduction="none")      # (B, T, D)

        # ── 6. Lyapunov regulariser (Theorem 1, Condition 1) ───────────
        # R(θ) = ||f_θ||² - ||target||²  per time step
        pred_sq   = (pred   ** 2).sum(dim=-1)           # (B, T)
        target_sq = (target ** 2).sum(dim=-1)           # (B, T)

        lyapunov_residual = pred_sq - target_sq         # (B, T)
        lyapunov_reg = F.relu(lyapunov_residual).unsqueeze(-1)       # (B, T, 1)

        # ── 7. Optional padding mask ────────────────────────────────────
        if cfg.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "Batch must contain 'action_is_pad' when "
                    f"{cfg.do_mask_loss_for_padding=}."
                )
            mask = (~batch["action_is_pad"]).unsqueeze(-1)  # (B, T, 1)
            score_loss   = score_loss   * mask
            lyapunov_reg = lyapunov_reg * mask

        # ── 8. Combine losses ───────────────────────────────────────────
        total_loss = (
            cfg.score_loss_weight * score_loss.mean()
            + cfg.lyapunov_lambda  * lyapunov_reg.mean()
        )
        return total_loss
