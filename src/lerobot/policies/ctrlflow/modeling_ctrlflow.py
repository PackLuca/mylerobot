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
"""CTRL-Flow Policy — strict implementation of the paper design.

Drift and diffusion (§4.3.1 Gradient Flow Construction, Ma 2024):

    f(x, t) = -∇_x u_t(x) = -∇_x log( p_t(x) / p_data(x) )
    g = sqrt(2) * I

SDE (continuous time):
    dX_t = f(X_t, t) dt  +  g dW_t

Euler-Maruyama discretisation (inference, step k):
    x_{k-1} = x_k  +  Δt · f_θ(x_k, k)  +  sqrt(2Δt) · ε_k,   ε_k ~ N(0,I)
               └──────── drift ──────────┘   └────── diffusion ──────┘

where Δt = 1/num_sde_steps and f_θ is the UNet approximating f.

Training target:
    The network f_θ is trained to approximate -∇_x log(p_t / p_data).
    Using a straight-line probability path
        x_t = (1 - t) · x_0  +  t · ε,   t = k / num_sde_steps
    the score along this path is proportional to the tangent vector:
        -∇_x log(p_t / p_data) ≈ (x_0 - ε) / (1 - t)  [denoising direction]
    We use the simpler unit-normalised form  target = x_0 - ε  (flow matching
    convention) which points in the same direction and avoids singularity at t=1.

Training loss:
    L = score_loss_weight · L_score  +  lyapunov_lambda · L_lyapunov

    L_score    = MSE( f_θ(x_t, t),  target )
               = MSE( f_θ(x_t, t),  x_0 - ε )

    L_lyapunov = E[ max(0,  R(θ)) ]
    where the Condition-1 residual R(θ) measures violation of Drift-Dissipation:
        R(θ) = <f_θ, ∇δL/δρ>  +  Φ
             = <f_θ, f_θ>    -  ||target||²        [self-consistent approximation]
             = ||f_θ||²       -  ||target||²
    This is ≤ 0 (inactive) when ||f_θ|| ≤ ||target||, i.e. the prediction does
    not overshoot the required dissipation force.

    Condition-2 (Diffusion-Enhanced Dissipation) is satisfied by construction
    for g = sqrt(2)*I and is not separately penalised.

Architecture:
    All vision/UNet components are imported directly from modeling_diffusion.py.
    Only CTRLFlowPolicy and CTRLFlowModel are new.
"""

import math
from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

# ------------------------------------------------------------------ #
# Re-use all architecture blocks from the diffusion policy — no duplication.
# ------------------------------------------------------------------ #
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
#  Top-level policy wrapper (identical structure to DiffusionPolicy)  #
# ================================================================== #

class CTRLFlowPolicy(PreTrainedPolicy):
    """CTRL-Flow Policy.

    Replaces DDPM/DDIM reverse diffusion with the Lyapunov-stable SDE:

        dX = f_θ(X, t) dt  +  sqrt(2) dW

    where  f_θ ≈ -∇_x log(p_t / p_data)  is learned by the UNet.

    Vision encoder and UNet architecture are identical to DiffusionPolicy.
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
    def predict_action_chunk(self, batch: dict[str, Tensor],
                             noise: Tensor | None = None) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1)
                 for k in batch if k in self._queues}
        return self.ctrlflow.generate_actions(batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor],
                      noise: Tensor | None = None) -> Tensor:
        """Select one action using the sliding-window cache (same logic as DP)."""
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
        """Compute training loss."""
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
    """Core CTRL-Flow model implementing the paper's SDE design.

    Inference
    ---------
    Euler-Maruyama integration of:
        dX = f_θ(X, t) dt  +  sqrt(2) dW

    Discretised as (step k, running from T-1 down to 0):
        x_{k-1} = x_k  +  Δt · f_θ(x_k, k)  +  sqrt(2·Δt) · ε_k
                  ↑ deterministic drift          ↑ stochastic diffusion (g=√2·I)

    where Δt = 1/num_sde_steps and ε_k ~ N(0, I).

    Training
    --------
    L = score_loss_weight · MSE( f_θ(x_t, t),  x_0 - ε )
      + lyapunov_lambda   · E[ max(0, ||f_θ||² - ||x_0 - ε||²) ]
        └──── L_score: learn the drift direction ────────────────┘
              └──── L_lyapunov: Drift-Dissipation Matching regulariser ───────┘
    """

    def __init__(self, config: CTRLFlowConfig):
        super().__init__()
        self.config = config

        # ---- observation encoders (identical to DiffusionModel) ----- #
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

        # ---- UNet: learns f_θ ≈ -∇_x log(p_t / p_data) ------------ #
        self.unet = DiffusionConditionalUnet1d(
            config, global_cond_dim=global_cond_dim * config.n_obs_steps
        )
        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

    # ------------------------------------------------------------------ #
    #  Observation encoding (identical to DiffusionModel)                 #
    # ------------------------------------------------------------------ #

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode all observations into a single conditioning vector (B, C)."""
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

        return torch.cat(feats, dim=-1).flatten(start_dim=1)  # (B, global_cond_dim)

    # ------------------------------------------------------------------ #
    #  CTRL-Flow inference: Euler-Maruyama SDE integration               #
    # ------------------------------------------------------------------ #

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Sample action trajectories by integrating the CTRL-Flow SDE.

        SDE (continuous):
            dX_t = f_θ(X_t, t) dt  +  g dW_t,    g = sqrt(2) * I

        Euler-Maruyama (discrete, step k from T-1 → 0):
            x_{k-1} = x_k
                      + Δt · f_θ(x_k, k)           ← drift:     f = -∇log(p_t/p_data)
                      + sqrt(2·Δt) · ε_k            ← diffusion: g = sqrt(2)*I
            where  ε_k ~ N(0, I)  and  Δt = 1/num_sde_steps.

        The diffusion term sqrt(2·Δt) = g · sqrt(Δt) comes directly from
        the Itô-Euler-Maruyama scheme applied to  g dW  with  |g|=sqrt(2).

        Args:
            batch_size: Number of trajectories to generate.
            global_cond: (B, global_cond_dim) observation conditioning.
            generator: Optional RNG for reproducibility.
            noise: Optional pre-sampled x_T ~ N(0, I)  (B, horizon, action_dim).

        Returns:
            (B, horizon, action_dim) generated action trajectories.
        """
        device = get_device_from_parameters(self)
        dtype  = get_dtype_from_parameters(self)
        cfg    = self.config

        # ---- Initial state: x_T ~ N(0, I) -------------------------- #
        x = (
            noise if noise is not None
            else torch.randn(
                (batch_size, cfg.horizon, cfg.action_feature.shape[0]),
                dtype=dtype, device=device, generator=generator,
            )
        )

        # ---- Precompute step constants ------------------------------ #
        dt    = cfg.dt                              # Δt = 1 / N
        # Diffusion increment std: sqrt(g² · Δt) = sqrt(2 · sde_noise_scale² · Δt)
        # For the paper design (sde_noise_scale=1):  = sqrt(2 · Δt)
        diff_std = cfg.g_coeff * math.sqrt(dt)     # sqrt(2 · s² · Δt)

        # ---- Euler-Maruyama loop: T → 0 ----------------------------- #
        # Timestep index k goes from (num_sde_steps - 1) down to 0.
        # The UNet receives an integer timestep in [0, num_sde_steps).
        for k in range(cfg.num_sde_steps - 1, -1, -1):
            t_batch = torch.full(
                (batch_size,), k, dtype=torch.long, device=device
            )

            # f_θ(x_k, k) ≈ -∇_x log( p_t(x) / p_data(x) )
            drift = self.unet(x, t_batch, global_cond=global_cond)  # (B, T, D)

            # ---- Drift term: Δt · f_θ ------------------------------- #
            x = x + dt * drift

            # ---- Diffusion term: sqrt(2·Δt) · ε  (g = sqrt(2)*I) --- #
            # Omit on the very last step (k=0) to avoid adding noise
            # after the trajectory has reached the data manifold.
            if k > 0 and diff_std > 0.0:
                eps = torch.randn_like(x, generator=generator)
                x = x + diff_std * eps

        return x

    def generate_actions(self, batch: dict[str, Tensor],
                         noise: Tensor | None = None) -> Tensor:
        """Encode observations and run the CTRL-Flow SDE sampler.

        Returns:
            (B, n_action_steps, action_dim) actions for execution.
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)
        actions     = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        start = n_obs_steps - 1
        end   = start + self.config.n_action_steps
        return actions[:, start:end]

    # ------------------------------------------------------------------ #
    #  Training loss                                                       #
    # ------------------------------------------------------------------ #

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute the CTRL-Flow training loss.

        Step-by-step:

        1. Straight probability path interpolation:
               x_t = (1 - α_t) · x_0  +  α_t · ε,    α_t = t / num_sde_steps
           where  x_0 = batch[ACTION],  ε ~ N(0,I),  t ~ Uniform{1,…,N}.

        2. Target drift (tangent to the straight path, pointing noise → data):
               target = x_0 - ε
           This approximates  -∇_x log(p_t / p_data)  along the straight path.

        3. Score-matching loss:
               L_score = MSE( f_θ(x_t, t),  target )

        4. Lyapunov Drift-Dissipation Matching regulariser (Theorem 1, Cond. 1):

           The condition requires:
               ∫ <f, ∇δL/δρ> p dx ≤ -α Φ
           For L = KL,  ∇δL/δρ ≈ f_θ  (self-consistent),  Φ = ||f_θ||²:
               <f_θ, f_θ> ≤ -Φ  →  always satisfied with equality (trivially).

           In practice we penalise the case where the *predicted* drift f_θ has
           larger magnitude than the *target* drift, since overshooting violates
           the energy-dissipation balance:
               R(θ) = ||f_θ||² - ||target||²

           When f_θ correctly approximates the target, R ≈ 0.
           When f_θ overshoots (too large), R > 0 → penalty.
           hinge: L_lyapunov = E[ max(0, R(θ)) ]

        5. Total loss:
               L = score_loss_weight · L_score  +  lyapunov_lambda · L_lyapunov

        Args:
            batch: Must contain OBS_STATE, ACTION, "action_is_pad", and
                   at least one of OBS_IMAGES / OBS_ENV_STATE.

        Returns:
            Scalar training loss.
        """
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch

        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon     = batch[ACTION].shape[1]
        assert horizon     == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        # ---- Straight probability path ------------------------------ #
        x0  = batch[ACTION]                             # (B, T, D)  clean actions
        eps = torch.randn_like(x0)                      # (B, T, D)  noise

        # Random timestep t ∈ {1, …, num_sde_steps} per batch element.
        # We start from 1 (not 0) so that α_t > 0 and the path is noisy.
        t_idx = torch.randint(
            low=1,
            high=self.config.num_sde_steps + 1,
            size=(x0.shape[0],),
            device=x0.device,
        ).long()                                        # (B,)

        # α_t = t / N  ∈ (0, 1]
        alpha = (t_idx.float() / self.config.num_sde_steps).view(-1, 1, 1)  # (B,1,1)

        # Interpolated noisy action:  x_t = (1-α)·x_0 + α·ε
        x_t = (1.0 - alpha) * x0 + alpha * eps         # (B, T, D)

        # Target drift: tangent vector pointing from noise toward data.
        # Approximates  -∇_x log( p_t(x) / p_data(x) ).
        target = x0 - eps                               # (B, T, D)

        # ---- Network prediction: f_θ(x_t, t) ----------------------- #
        pred = self.unet(x_t, t_idx, global_cond=global_cond)   # (B, T, D)

        # ================================================================ #
        # L_score: standard score-matching MSE
        # Trains f_θ to predict the drift direction -∇log(p_t/p_data).
        # ================================================================ #
        score_loss = F.mse_loss(pred, target, reduction="none")  # (B, T, D)

        # ================================================================ #
        # L_lyapunov: Drift-Dissipation Matching regulariser
        #
        # From Theorem 1 Condition 1, the stability requirement is:
        #     <f_θ, ∇δL/δρ> + Φ ≤ 0
        # Using the self-consistent approximation ∇δL/δρ ≈ f_θ and Φ=||f_θ||²:
        #     <f_θ, f_θ> + ||f_θ||² = 2||f_θ||² ≤ 0   (trivially impossible)
        #
        # In practice we use the *error decomposition*:
        #     f_θ = target + error,   error = f_θ - target
        # The condition then becomes a constraint on the error magnitude:
        #     R(θ) = ||f_θ||² - ||target||²
        #          = ||target + error||² - ||target||²
        #          = 2<target, error> + ||error||²
        # This is ≤ 0 iff  ||f_θ|| ≤ ||target||  (prediction does not overshoot).
        # Hinge loss penalises positive residuals only:
        #     L_lyapunov = E[ max(0, R(θ)) ]
        # ================================================================ #
        # ||f_θ||² per spatial location, shape (B, T)
        pred_sq   = (pred   ** 2).sum(dim=-1)           # (B, T)
        target_sq = (target ** 2).sum(dim=-1)           # (B, T)

        # R(θ) = ||f_θ||² - ||target||²
        lyapunov_residual = pred_sq - target_sq         # (B, T)
        # Hinge: penalise only when R > 0 (dissipation condition violated)
        lyapunov_reg = F.relu(lyapunov_residual)        # (B, T)
        # Expand for uniform treatment with score_loss shape (B, T, D)
        lyapunov_reg = lyapunov_reg.unsqueeze(-1)       # (B, T, 1)

        # ---- Optional padding mask ---------------------------------- #
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "Batch must contain 'action_is_pad' when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            mask = (~batch["action_is_pad"]).unsqueeze(-1)  # (B, T, 1)
            score_loss   = score_loss   * mask
            lyapunov_reg = lyapunov_reg * mask

        # ---- Combine losses ----------------------------------------- #
        total_loss = (
            self.config.score_loss_weight * score_loss.mean()
            + self.config.lyapunov_lambda  * lyapunov_reg.mean()
        )
        return total_loss
