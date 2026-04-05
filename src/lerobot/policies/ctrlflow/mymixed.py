"""Mixed W₂→KL 调度器 v3（双路径设计）

===========================================================================
【版本历史】
===========================================================================

v1：初始实现，epsilon 逆变换在 σ'(t)≈0 时数值爆炸（loss=23855）
v2：引入 batch-level 归一化，修复了爆炸，但归一化把 loss 压平在 0.88，
    模型无法收敛（kl_as_velocity_weight 实际无效）
v3（本版本）：双路径设计，两个阶段用不同前向路径，velocity target 真正异构

===========================================================================
【核心设计：双路径前向】
===========================================================================

v1/v2 的根本问题：在同一条 DS-SDE 路径上，
    v*_W2(t) = α'_DS(t)·x₀ + σ'_DS(t)·ε
    v*_KL(t) = α'_DS(t)·x₀ + σ'_DS(t)·ε
两者数学上完全相同，kl_as_velocity_weight 控制的只是 epsilon 逆变换的占比，
而不是两种几何距离的真实差异。

v3 的解决方案：让两个阶段走不同的前向路径：

  W₂ 阶段（t > t_switch）：线性插值路径（Benamou-Brenier 最优传输直线）
    x_t = (1-t)·x₀ + t·ε
    v*_W2 = ε - x₀                         ← 常数，不依赖 t
    物理含义：最短路径搬运，对应 W₂ 测地线

  KL 阶段（t ≤ t_switch）：DS-SDE 非线性路径
    x_t = α_DS(t)·x₀ + σ_DS(t)·ε
    v*_KL = α'_DS(t)·x₀ + σ'_DS(t)·ε      ← 随 t 变化，曲率由 γ(t)/g(t) 决定
    物理含义：score-matching 对齐，对应 KL 散度耗散

两个 target 在几何上真正不同：
  - v*_W2 = ε - x₀ 是常数向量，指向"最短搬运方向"
  - v*_KL 随 t 连续变化，反映 DS-SDE 路径的局部切向量

===========================================================================
【add_noise 的分流设计】
===========================================================================

训练时每个样本按其 timestep 决定走哪条路：

  t > t_switch → 用线性插值加噪：x_t = (1-t)·x₀ + t·ε
  t ≤ t_switch → 用 DS-SDE 加噪：x_t = α_DS(t)·x₀ + σ_DS(t)·ε

推理时不使用 add_noise，只用 step，无需分流。

===========================================================================
【loss 设计】
===========================================================================

    L = w₁(t)·‖v̂ - v*_W2‖²  +  w₂(t)·‖v̂ - v*_KL‖²

其中：
    w₁(t) = sigmoid(γ·(t - t_switch))     t 大时 → 1（W₂ 主导）
    w₂(t) = 1 - w₁(t)                     t 小时 → 1（KL 主导）

两项 target 不同，权重真正有意义。

关键性质：
  - 两项都是 MSE，量纲一致，不需要归一化
  - 不涉及 epsilon 逆变换，无数值稳定性问题
  - kl_as_velocity_weight 参数已废弃（两个 target 天然不同）

可选 Lyapunov 正则项（§7.1）：
    L_lyap = ‖x_t + v̂·dt - x₀‖²           （dt < 0，推理方向）

===========================================================================
【推理过程】（与 v2 相同）
===========================================================================

  t > t_switch → 纯 ODE：x_{t+dt} = x_t + v̂·dt
  t ≤ t_switch → SDE：   x_{t+dt} = x_t + v̂·dt + √(2·g²·scale·|dt|)·z

===========================================================================
【接口约定】（与 v1/v2 完全兼容）
===========================================================================

  - prediction_type = "velocity"
  - add_noise(x0, noise, timesteps) → x_t
  - set_timesteps(num_inference_steps)
  - step(model_output, t, sample)   → SDEResult
  - compute_loss(pred, x0, noise, x_t, timesteps) → scalar
  - config.num_train_timesteps
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.ctrlflow.mysde import DSSDE, SDEResult


# ---------------------------------------------------------------------------
# 内部 config 容器
# ---------------------------------------------------------------------------

class _MixedConfig:
    def __init__(self, num_train_timesteps: int):
        self.num_train_timesteps = num_train_timesteps


# ---------------------------------------------------------------------------
# Mixed W₂→KL Scheduler v3（双路径）
# ---------------------------------------------------------------------------

class MixedW2KLScheduler(nn.Module):
    """DS-SDE / Linear 双路径 W₂→KL 混合调度器（v3）。

    W₂ 阶段（t > t_switch）：线性插值路径，v* = ε - x₀（常数目标）
    KL 阶段（t ≤ t_switch）：DS-SDE 非线性路径，v* = α'(t)x₀ + σ'(t)ε

    参数:
        num_train_timesteps: 训练离散步数
        gamma_min, gamma_max: DS-SDE γ(t) 调度参数（KL 阶段路径）
        g_min, g_max:         DS-SDE g(t) 调度参数（KL 阶段路径）
        clip_sample:          推理时是否裁剪输出
        clip_sample_range:    裁剪范围
        t_switch:             W₂/KL 切换点，t ∈ [0,1]，默认 0.5
        switch_sharpness:     sigmoid 陡峭度，默认 6.0
        kl_sde_noise_scale:   KL 阶段推理随机项强度，默认 1.0
        lyapunov_reg_weight:  §7.1 Lyapunov 正则项权重，默认 0.0
        num_precompute_steps: DS-SDE lookup table 精度，默认 1000
    """

    prediction_type: str = "velocity"

    def __init__(
        self,
        num_train_timesteps: int = 100,
        gamma_min: float = 0.5,
        gamma_max: float = 3.0,
        g_min: float = 0.01,
        g_max: float = 1.0,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        t_switch: float = 0.5,
        switch_sharpness: float = 6.0,
        kl_sde_noise_scale: float = 1.0,
        lyapunov_reg_weight: float = 0.0,
        num_precompute_steps: int = 1000,
    ):
        super().__init__()

        # ── 参数校验 ──────────────────────────────────────────────────
        assert 0.0 < t_switch < 1.0, \
            f"t_switch 必须在 (0,1) 内，得到 {t_switch}"
        assert switch_sharpness > 0, \
            f"switch_sharpness 必须为正数，得到 {switch_sharpness}"
        assert kl_sde_noise_scale >= 0.0, \
            f"kl_sde_noise_scale 必须 >= 0，得到 {kl_sde_noise_scale}"
        assert lyapunov_reg_weight >= 0.0, \
            f"lyapunov_reg_weight 必须 >= 0，得到 {lyapunov_reg_weight}"

        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.kl_sde_noise_scale = kl_sde_noise_scale
        self.lyapunov_reg_weight = lyapunov_reg_weight

        # register_buffer：自动跟随 .to(device) / .to(dtype)
        self.register_buffer('_t_switch', torch.tensor(t_switch))
        self.register_buffer('_sharpness', torch.tensor(switch_sharpness))

        # DS-SDE：负责 KL 阶段的路径系数及其导数
        self._dssde = DSSDE(
            num_train_timesteps=num_train_timesteps,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            g_min=g_min,
            g_max=g_max,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            prediction_type="epsilon",
            num_precompute_steps=num_precompute_steps,
        )

        self.timesteps: Tensor | None = None
        self.num_inference_steps: int | None = None
        self.config = _MixedConfig(num_train_timesteps=num_train_timesteps)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _w1(self, t: Tensor) -> Tensor:
        """W₂ 权重 w₁(t) = sigmoid(γ·(t - t_switch))

        t > t_switch → w₁ → 1（W₂/线性路径主导）
        t < t_switch → w₁ → 0（KL/DS-SDE 路径主导）
        """
        t_sw = self._t_switch.to(t.device, t.dtype)
        sh   = self._sharpness.to(t.device, t.dtype)
        return torch.sigmoid(sh * (t - t_sw))

    def _linear_alpha(self, t: Tensor) -> Tensor:
        """线性路径：α(t) = 1 - t"""
        return 1.0 - t

    def _linear_sigma(self, t: Tensor) -> Tensor:
        """线性路径：σ(t) = t"""
        return t

    # ------------------------------------------------------------------
    # 前向路径（双路径 add_noise）
    # ------------------------------------------------------------------

    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """按 t 值分流的双路径前向加噪。

        t > t_switch → 线性插值：x_t = (1-t)·x₀ + t·ε
        t ≤ t_switch → DS-SDE：  x_t = α_DS(t)·x₀ + σ_DS(t)·ε

        两条路产生的 x_t 在 t_switch 附近是连续的（两条路在该点的 α/σ 值
        不一定相等，但各自在自己的阶段内保持一致性，UNet 通过时间步嵌入
        感知当前 t，可以学到两段路径的拼接）。

        参数:
            x0:        干净数据 (B, T, D)
            noise:     标准高斯噪声 (B, T, D)
            timesteps: 整数时间步索引 (B,)，范围 [0, num_train_timesteps)
        返回:
            x_t: (B, T, D)
        """
        t_cont = timesteps.float() / self.num_train_timesteps  # (B,)

        # 广播到 (B, 1, 1)
        t_b = t_cont
        while t_b.dim() < x0.dim():
            t_b = t_b.unsqueeze(-1)

        t_sw = self._t_switch.item()

        # ── 线性路径（W₂ 阶段，t > t_switch）────────────────────────
        alpha_lin = self._linear_alpha(t_b)   # (B,1,1)
        sigma_lin = self._linear_sigma(t_b)   # (B,1,1)
        x_t_lin   = alpha_lin * x0 + sigma_lin * noise

        # ── DS-SDE 路径（KL 阶段，t ≤ t_switch）─────────────────────
        alpha_ds  = self._dssde._alpha(t_b)   # (B,1,1)
        sigma_ds  = self._dssde._sigma(t_b)   # (B,1,1)
        x_t_ds    = alpha_ds * x0 + sigma_ds * noise

        # ── 按样本分流（硬切换，训练时 t_switch 是边界）──────────────
        # mask: True 表示该样本走线性路径（W₂ 阶段）
        w2_mask = (t_cont > t_sw)              # (B,)
        while w2_mask.dim() < x0.dim():
            w2_mask = w2_mask.unsqueeze(-1)    # (B,1,1)

        x_t = torch.where(w2_mask, x_t_lin, x_t_ds)
        return x_t

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int) -> None:
        """设置推理时间节点（t 从 1 降到 1/N）。"""
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(
            1.0, 1.0 / num_inference_steps, num_inference_steps
        )

    def step(
        self,
        model_output: Tensor,
        t: Tensor | float,
        sample: Tensor,
        generator: torch.Generator | None = None,
    ) -> SDEResult:
        """一步推理积分，按 t 动态选择 ODE 或 SDE 步进。

        t > t_switch → 纯 ODE：  x_{t+dt} = x_t + v̂·dt
        t ≤ t_switch → SDE：     x_{t+dt} = x_t + v̂·dt + √(2·g²·scale·|dt|)·z

        model_output 是 UNet 预测的 velocity v̂，形状 (B, T, D)。
        """
        if not isinstance(t, Tensor):
            t_val = torch.tensor(t, dtype=torch.float32, device=sample.device)
        else:
            t_val = t.float().to(sample.device)

        dt = -1.0 / self.num_inference_steps

        prev_sample = sample + model_output * dt

        if t_val.item() <= self._t_switch.item() and self.kl_sde_noise_scale > 0.0:
            g_sq        = self._dssde._g_sq(t_val)
            noise_coeff = torch.sqrt(
                (2.0 * g_sq * self.kl_sde_noise_scale * abs(dt)).clamp(min=0.0)
            )
            z = torch.randn(
                sample.shape,
                device=sample.device,
                dtype=sample.dtype,
                generator=generator,
            )
            prev_sample = prev_sample + noise_coeff * z

        if self.clip_sample:
            prev_sample = prev_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        return SDEResult(prev_sample=prev_sample)

    # ------------------------------------------------------------------
    # 训练 loss（双路径 v3）
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        pred_velocity: Tensor,
        x0: Tensor,
        noise: Tensor,
        x_t: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """双路径 W₂→KL 混合训练 loss。

        参数:
            pred_velocity: v̂ = UNet 输出 (B, T, D)
            x0:            干净数据 (B, T, D)
            noise:         前向噪声 (B, T, D)
            x_t:           加噪状态 (B, T, D)（由双路径 add_noise 产生）
            timesteps:     整数时间步索引 (B,)

        ──────────────────────────────────────────────────────────────
        loss 结构：

            L = w₁(t)·‖v̂ - v*_W2‖²  +  w₂(t)·‖v̂ - v*_KL‖²

            v*_W2 = ε - x₀                           （线性路径导数，常数）
            v*_KL = α'_DS(t)·x₀ + σ'_DS(t)·ε        （DS-SDE 路径导数）

        两个 target 真正不同：
          - v*_W2 不依赖 t，对应全局搬运方向（Benamou-Brenier）
          - v*_KL 随 t 变化，对应 KL 散度耗散方向（score matching）

        两项都是 velocity MSE，量纲完全一致，无需归一化。
        ──────────────────────────────────────────────────────────────
        """
        # ── t 的连续值，广播到 (B,1,1) ────────────────────────────────
        t_cont = timesteps.float() / self.num_train_timesteps  # (B,)
        t_b = t_cont
        while t_b.dim() < x0.dim():
            t_b = t_b.unsqueeze(-1)  # (B,1,1)

        # ── W₂ target：线性路径导数 v*_W2 = ε - x₀ ──────────────────
        # 线性路径 x_t = (1-t)x₀ + tε，对 t 求导：dx_t/dt = -x₀ + ε
        v_target_w2 = noise - x0                               # (B,T,D)，不依赖 t

        # ── KL target：DS-SDE 路径导数 v*_KL = α'(t)x₀ + σ'(t)ε ─────
        alpha_p = self._dssde._alpha_prime(t_b)                # (B,1,1)
        sigma_p = self._dssde._sigma_prime(t_b)                # (B,1,1)
        v_target_kl = alpha_p * x0 + sigma_p * noise          # (B,T,D)

        # ── 软权重 w₁(t) / w₂(t)，形状 (B,1,1) ──────────────────────
        w1 = self._w1(t_cont)
        while w1.dim() < x0.dim():
            w1 = w1.unsqueeze(-1)
        w2 = 1.0 - w1

        # ── element-wise loss（保留 reduction="none" 以支持 padding mask）
        loss_w2 = F.mse_loss(pred_velocity, v_target_w2.detach(), reduction="none")
        loss_kl = F.mse_loss(pred_velocity, v_target_kl.detach(), reduction="none")

        # ── 加权混合 ──────────────────────────────────────────────────
        # 注意：两项 target 不同，loss 值量级可能略有差异。
        # 由于都是 velocity MSE，量纲一致，直接加权即可，无需归一化。
        loss = w1 * loss_w2 + w2 * loss_kl

        # ── 可选：Lyapunov 正则项（§7.1）─────────────────────────────
        if self.lyapunov_reg_weight > 0.0:
            # 推理方向走一步：x_{t-dt} = x_t + v̂·(-dt) = x_t - v̂/N
            dt_val = 1.0 / self.num_train_timesteps
            x_next = x_t - pred_velocity * dt_val
            loss_lyap = F.mse_loss(x_next, x0.detach(), reduction="none")
            loss = loss + self.lyapunov_reg_weight * loss_lyap

        return loss.mean()
