"""Mixed W₂→KL 调度器（CTRL-Flow §5.3 分层混合构造）- 修复版

===========================================================================
【修复说明】
===========================================================================

本版本修复了以下问题：
1. ✅ 量纲混合：velocity 和 epsilon loss 归一化到同一量纲
2. ✅ 数值稳定性：增强 fp16 训练和端点处的稳定性
3. ✅ 设备匹配：使用 register_buffer 缓存常量
4. ✅ Lyapunov 符号：修正推理方向的 dt 符号
5. ✅ 文档一致性：更新文档说明

===========================================================================
【理论基础】（保持不变）
===========================================================================

基于 CTRL-Flow 论文 §5.3（Hierarchical Hybrid Construction），
将训练过程分为两个阶段：

  Phase 1（t > t_switch，高噪声端）：W₂ 距离主导
    - Lyapunov 泛函：L_W2 = W₂²(p_t, p_data)
    - 监督信号：velocity target  v*(t) = α'(t)·x₀ + σ'(t)·ε
    - 直觉：分布尚未重叠，OT 梯度提供全局搬运方向

  Phase 2（t < t_switch，低噪声端）：KL 散度主导
    - Lyapunov 泛函：L_KL = D_KL(p_t ‖ p_data)
    - 监督信号：epsilon target（通过 velocity 参数化间接学习）
    - 直觉：分布已接近，score matching 做精细密度对齐

混合 loss（加权叠加，满足论文定理 1 的稳定条件）：

    L_mix = w₁(t) · L_vel^W2
          + w₂(t) · [λ · L_vel^KL + (1-λ) · L_eps^KL]

其中：
    w₁(t) = sigmoid(γ · (t - t_switch))     t 大时 → 1（W₂ 主导）
    w₂(t) = 1 - w₁(t)                       t 小时 → 1（KL 主导）

    L_vel   = ‖v̂ - v*(t)‖²     velocity 监督（两阶段共享目标）
    L_eps   = ‖ε̂ - ε‖²         epsilon 监督（从 v̂ 反推 ε̂）
    ε̂       = (v̂ - α'·x₀_est) / σ'
    x₀_est  = (x_t - σ·ε) / α

【关键修复】
为防止 L_vel 和 L_eps 量纲不匹配导致梯度失衡，
本版本使用 running mean 归一化两项到同一尺度后再混合。

===========================================================================
【推理过程】（保持不变）
===========================================================================

推理时 UNet 输出 velocity v̂，步进公式为：

  t > t_switch（W₂ 主导阶段）：
    纯 ODE 积分，x_{t+dt} = x_t + v̂ · dt        （轨迹直，步数少）

  t ≤ t_switch（KL 主导阶段）：
    加入 DS-SDE 的随机扩散项，近似 Schrödinger Bridge：
    x_{t+dt} = x_t + v̂ · dt + √(2·g²(t)·scale·|dt|) · z

===========================================================================
【接口约定】（保持不变）
===========================================================================

对外暴露与 DSSDE / W2FlowODE 完全一致的接口：
  - prediction_type = "velocity"
  - add_noise(x0, noise, timesteps) → x_t
  - set_timesteps(num_inference_steps)
  - step(model_output, t, sample) → SDEResult
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
# Mixed W₂→KL Scheduler（修复版）
# ---------------------------------------------------------------------------

class MixedW2KLScheduler(nn.Module):  # 改为继承 nn.Module 以使用 register_buffer
    """DS-SDE 路径上的 W₂→KL 分层混合调度器（修复版）。

    前向路径（add_noise）：使用 DS-SDE 的 α(t)、σ(t)，与纯 DS-SDE 完全一致。
    训练 loss：velocity 坐标下的加权混合，见模块文档。
    推理步进：t > t_switch 时纯 ODE，t ≤ t_switch 时带随机扩散项。

    参数:
        num_train_timesteps: 训练离散步数
        gamma_min, gamma_max: DS-SDE γ(t) 调度参数
        g_min, g_max:         DS-SDE g(t) 调度参数
        clip_sample:          推理时是否裁剪输出
        clip_sample_range:    裁剪范围
        t_switch:             W₂/KL 切换点，t ∈ [0,1]，默认 0.5
        switch_sharpness:     sigmoid 陡峭度 γ，越大切换越硬，默认 6.0
        kl_sde_noise_scale:   KL 阶段推理随机项强度（0 → 纯 ODE），默认 1.0
        kl_as_velocity_weight: KL 分量中 velocity 监督的占比 λ，默认 0.5
        lyapunov_reg_weight:  §7.1 Lyapunov 正则项权重（0 → 关闭），默认 0.0
        num_precompute_steps: DSSDE lookup table 精度，默认 1000
        eps_loss_scale:       epsilon loss 的手动缩放因子（可选，默认 1.0）
    """

    # modeling_ctrlflow.py 通过此属性判断参数化类型
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
        kl_as_velocity_weight: float = 0.5,
        lyapunov_reg_weight: float = 0.0,
        num_precompute_steps: int = 1000,
        eps_loss_scale: float = 1.0,  # 新增：手动缩放因子
    ):
        super().__init__()  # 调用 nn.Module 的初始化
        
        # ── 参数校验 ──────────────────────────────────────────────────
        assert 0.0 < t_switch < 1.0, \
            f"t_switch 必须在 (0,1) 内，得到 {t_switch}"
        assert switch_sharpness > 0, \
            f"switch_sharpness 必须为正数，得到 {switch_sharpness}"
        assert 0.0 <= kl_sde_noise_scale, \
            f"kl_sde_noise_scale 必须 >= 0，得到 {kl_sde_noise_scale}"
        assert 0.0 <= kl_as_velocity_weight <= 1.0, \
            f"kl_as_velocity_weight 必须在 [0,1]，得到 {kl_as_velocity_weight}"
        assert lyapunov_reg_weight >= 0.0, \
            f"lyapunov_reg_weight 必须 >= 0，得到 {lyapunov_reg_weight}"
        assert eps_loss_scale > 0.0, \
            f"eps_loss_scale 必须为正数，得到 {eps_loss_scale}"

        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.kl_sde_noise_scale = kl_sde_noise_scale
        self.kl_as_velocity_weight = kl_as_velocity_weight
        self.lyapunov_reg_weight = lyapunov_reg_weight
        self.eps_loss_scale = eps_loss_scale

        # ✅ 修复 3：使用 register_buffer 缓存常量（自动处理设备）
        self.register_buffer('_t_switch_tensor', torch.tensor(t_switch))
        self.register_buffer('_sharpness_tensor', torch.tensor(switch_sharpness))

        # 内部 DS-SDE：负责 α/σ/α'/σ'/g² 的所有计算
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

        # 供 modeling_ctrlflow.py 的 sde.config 查询
        self.config = _MixedConfig(num_train_timesteps=num_train_timesteps)

    # ------------------------------------------------------------------
    # 权重调度
    # ------------------------------------------------------------------

    def _w1(self, t: Tensor) -> Tensor:
        """W₂ 权重：w₁(t) = sigmoid(γ·(t - t_switch))

        t > t_switch → w₁ → 1（W₂ 主导，高噪声端）
        t < t_switch → w₁ → 0（KL 主导，低噪声端）
        """
        # ✅ 修复 3：使用缓存的 tensor（自动匹配设备和类型）
        t_switch = self._t_switch_tensor.to(t.device, t.dtype)
        sharpness = self._sharpness_tensor.to(t.device, t.dtype)
        return torch.sigmoid(sharpness * (t - t_switch))

    # ------------------------------------------------------------------
    # 前向路径：委托给 DSSDE（与纯 DS-SDE 训练完全一致）
    # ------------------------------------------------------------------

    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """前向插值：x_t = α_DS(t)·x₀ + σ_DS(t)·ε"""
        return self._dssde.add_noise(x0, noise, timesteps)

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
        """一步推理积分，按当前 t 动态选择 ODE 或 SDE 步进。"""
        if not isinstance(t, Tensor):
            t_val = torch.tensor(t, dtype=torch.float32, device=sample.device)
        else:
            t_val = t.float().to(sample.device)

        dt = -1.0 / self.num_inference_steps

        # Euler 步（ODE 部分，两个阶段都有）
        prev_sample = sample + model_output * dt

        # t ≤ t_switch：加入 DS-SDE 随机扩散项
        t_switch_val = self._t_switch_tensor.item()
        if t_val.item() <= t_switch_val and self.kl_sde_noise_scale > 0.0:
            g_sq = self._dssde._g_sq(t_val)
            noise_var = 2.0 * g_sq * self.kl_sde_noise_scale * abs(dt)
            noise_coeff = torch.sqrt(noise_var.clamp(min=0.0))
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
    # 训练 loss（修复版）
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        pred_velocity: Tensor,
        x0: Tensor,
        noise: Tensor,
        x_t: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """W₂→KL 混合训练 loss（修复版）。

        参数:
            pred_velocity: v̂ = UNet 输出，形状 (B, T, D)
            x0:            干净数据，形状 (B, T, D)
            noise:         前向扩散时的高斯噪声，形状 (B, T, D)
            x_t:           加噪后的状态，形状 (B, T, D)
            timesteps:     整数时间步索引，形状 (B,)，范围 [0, num_train_timesteps)

        返回:
            标量 loss

        ──────────────────────────────────────────────────────────────
        修复内容：
        1. ✅ 量纲归一化：velocity 和 epsilon loss 归一化到同一尺度
        2. ✅ 数值稳定性：使用更安全的 clamp 值和条件分支
        3. ✅ Lyapunov 符号：修正 dt 的符号（推理方向）
        ──────────────────────────────────────────────────────────────
        """
        # ✅ 修复 2：更安全的数值常量
        SAFE_MIN = 1e-3  # 对 fp16 更友好

        # ── t 的连续值（归一化到 [0,1]），广播到 (B,1,1) ──────────────
        t_cont = timesteps.float() / self.num_train_timesteps  # (B,)
        t_b = t_cont
        while t_b.dim() < x0.dim():
            t_b = t_b.unsqueeze(-1)  # (B,1,1)

        # ── 从 lookup table 取路径系数及其导数 ────────────────────────
        alpha = self._dssde._alpha(t_b)        # (B,1,1)
        sigma = self._dssde._sigma(t_b)        # (B,1,1)
        alpha_p = self._dssde._alpha_prime(t_b)  # (B,1,1)
        sigma_p = self._dssde._sigma_prime(t_b)  # (B,1,1)

        # ✅ 修复 2：安全系数（避免除零和下溢）
        alpha_safe = alpha.abs().clamp(min=SAFE_MIN)
        sigma_safe = sigma.abs().clamp(min=SAFE_MIN)
        sigma_p_safe = sigma_p.abs().clamp(min=SAFE_MIN)

        # ── velocity target v*(t) = α'·x₀ + σ'·ε ─────────────────────
        v_target = alpha_p * x0 + sigma_p * noise  # (B,T,D)

        # ── 逐样本权重 w₁(t)，形状 (B,1,1) ───────────────────────────
        w1 = self._w1(t_cont)
        while w1.dim() < x0.dim():
            w1 = w1.unsqueeze(-1)
        w2 = 1.0 - w1

        # ── W₂ 分量：velocity MSE ─────────────────────────────────────
        loss_vel = F.mse_loss(pred_velocity, v_target.detach(), reduction="none")

        # ── KL 分量：epsilon MSE（修复版）──────────────────────────────
        # ✅ 修复 2：稳定的 x0_est（处理 sigma 很小的情况）
        x0_est = torch.where(
            sigma.abs() > SAFE_MIN,
            (x_t - sigma * noise) / alpha_safe,
            x_t / alpha_safe  # 退化情况：σ≈0 时，x_t ≈ α·x0
        )

        # ✅ 修复 2：从 velocity 反推 epsilon，并裁剪异常值
        pred_eps = (pred_velocity - alpha_p * x0_est.detach()) / sigma_p_safe
        pred_eps = pred_eps.clamp(-10.0, 10.0)  # 防止梯度爆炸

        loss_eps = F.mse_loss(pred_eps, noise.detach(), reduction="none")

        # ── ✅ 修复 1：量纲归一化混合 ────────────────────────────────
        # 使用 detach() 的 running mean 归一化（不影响梯度流）
        vel_scale = loss_vel.detach().mean() + 1e-8
        eps_scale = loss_eps.detach().mean() + 1e-8

        # 归一化到同一尺度
        loss_vel_norm = loss_vel / vel_scale
        loss_eps_norm = loss_eps / eps_scale * self.eps_loss_scale

        # 加权混合
        lam = self.kl_as_velocity_weight
        loss = (
            w1 * loss_vel_norm  # W₂ 阶段：纯 velocity
            + w2 * (lam * loss_vel_norm + (1.0 - lam) * loss_eps_norm)  # KL 阶段：混合
        )

        # 可选：恢复原始 scale（用于日志可读性）
        # avg_scale = (vel_scale + eps_scale * self.eps_loss_scale) / 2.0
        # loss = loss * avg_scale

        # ── ✅ 修复 4：Lyapunov 正则项（修正 dt 符号）─────────────────
        if self.lyapunov_reg_weight > 0.0:
            # 推理时 t 从 1→0，dt < 0
            dt_infer = torch.full_like(t_cont, -1.0 / self.num_train_timesteps)
            while dt_infer.dim() < pred_velocity.dim():
                dt_infer = dt_infer.unsqueeze(-1)

            # 正确的步进公式：x_{t-Δt} = x_t + v̂ · (-Δt)
            x_next_infer = x_t + pred_velocity * dt_infer

            # Lyapunov 递减条件：‖x_{t-Δt} - x₀‖² 应该减小
            loss_lyap = F.mse_loss(
                x_next_infer, x0.detach(), reduction="none"
            )

            # 可选：只在 W₂ 阶段施加（高噪声端更重要）
            # t_switch_val = self._t_switch_tensor.item()
            # lyap_mask = (t_cont > t_switch_val).float()
            # while lyap_mask.dim() < loss_lyap.dim():
            #     lyap_mask = lyap_mask.unsqueeze(-1)
            # loss_lyap = loss_lyap * lyap_mask

            loss = loss + self.lyapunov_reg_weight * loss_lyap

        return loss.mean()
