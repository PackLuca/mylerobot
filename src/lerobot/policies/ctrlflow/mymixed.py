"""Mixed W₂→KL 调度器（CTRL-Flow §5.3 分层混合构造）

===========================================================================
【理论基础】
===========================================================================

基于 CTRL-Flow 论文 §5.3（Hierarchical Hybrid Construction），
将训练过程分为两个阶段：

  Phase 1（t > t_switch，高噪声端）：W₂ 距离主导
    - Lyapunov 泛函：L_W2 = W₂²(p_t, p_data)
    - 监督信号：velocity target  v*(t) = α'(t)·x₀ + σ'(t)·ε
    - 直觉：分布尚未重叠，OT 梯度提供全局搬运方向

  Phase 2（t < t_switch，低噪声端）：KL 散度主导
    - Lyapunov 泛函：L_KL = D_KL(p_t ‖ p_data)
    - 监督信号：epsilon target  ε（等价于 score 参数化）
    - 直觉：分布已接近，score matching 做精细密度对齐

混合 loss（加权叠加，满足论文定理 1 的稳定条件）：

    L_mix = w₁(t) · ‖v̂ - v*(t)‖²   ← W₂ 分量（velocity 参数化）
          + w₂(t) · ‖v̂ - ê(t)‖²    ← KL 分量（epsilon→velocity 换算）

其中：
    w₁(t) = sigmoid(γ · (t - t_switch))     t 大时 → 1（W₂ 主导）
    w₂(t) = 1 - w₁(t)                       t 小时 → 1（KL 主导）

    ê(t)  = epsilon 目标在 velocity 坐标系的等价形式
           = (x_t - α(t)·x₀_est) / σ(t) · σ'(t) + α'(t)·x₀_est
             其中 x₀_est = (x_t - σ(t)·ε̂) / α(t)    （UNet 输出 ε̂）
           ↓ 化简（将真实 ε 代入，不用 UNet 估计）：
           = α'(t)·x₀ + σ'(t)·ε    ← 与 v*(t) 形式相同！

    ！！关键洞察：
    当 UNet 预测 epsilon 时，KL 目标在 velocity 坐标下的等价形式
    与 W₂ velocity target 的表达式完全相同（都是 α'x₀ + σ'ε），
    区别只在于：
      W₂ 分量的 α'、σ' 来自 DS-SDE 有限差分表（非线性路径）
      KL 分量等价于让 UNet 在该路径上直接预测 velocity

    因此两个分量共享同一个 velocity target，
    混合 loss 退化为：
        L_mix = (w₁(t) + w₂(t)) · ‖v̂ - v*(t)‖²  = ‖v̂ - v*(t)‖²

    ── 这说明 velocity 参数化天然统一了 W₂ 和 KL 监督 ──

    为了在实验中体现"混合"的可控性和可对比性，
    本实现额外保留一个 kl_epsilon_weight 参数：
    在 KL 主导阶段，同时施加一个直接的 epsilon 监督项（不经过坐标变换），
    让模型在 t 小时额外学习 score 的量纲：

        L_mix = w₁(t) · ‖v̂ - v*‖²
              + w₂(t) · [λ · ‖v̂ - v*‖²  +  (1-λ) · ‖ε̂ - ε‖²]

    其中 λ = kl_as_velocity_weight（默认 0.5），
         ε̂ 由 v̂ 逆变换得到：ε̂ = (v̂ - α'(t)·x₀_est) / σ'(t)

    这使得 KL 分量既有 velocity 层面的约束，又有原始 epsilon 层面的约束，
    更符合论文中"两种距离函数各自贡献耗散"的精神。

===========================================================================
【推理过程】
===========================================================================

推理时 UNet 输出 velocity v̂，步进公式为：

  t > t_switch（W₂ 主导阶段）：
    纯 ODE 积分，x_{t+dt} = x_t + v̂ · dt        （轨迹直，步数少）

  t ≤ t_switch（KL 主导阶段）：
    加入 DS-SDE 的随机扩散项，近似 Schrödinger Bridge：
    x_{t+dt} = x_t + v̂ · dt + √(2·g²(t)·|dt|) · z

    直觉：KL 阶段对精细密度对齐有益，少量随机性能提升样本多样性，
    等价于论文 Theorem 1 中 Condition 2（扩散增强耗散，贡献 β）。

===========================================================================
【与现有代码的接口约定】
===========================================================================

对外暴露与 DSSDE / W2FlowODE 完全一致的接口：
  - prediction_type = "velocity"    （UNet 预测速度场）
  - add_noise(x0, noise, timesteps) → x_t
  - set_timesteps(num_inference_steps)
  - step(model_output, t, sample)   → SDEResult
  - compute_loss(pred, x0, noise, x_t, timesteps) → scalar
  - config.num_train_timesteps      （供 modeling 层查询）

modeling_ctrlflow.py 的 compute_loss 中对 "Mixed-W2KL" 的处理
与 "W2-ODE" 完全对称，只需增加一个 elif 分支调用
    self.sde.compute_loss(pred, trajectory, eps, noisy_trajectory, timesteps)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.ctrlflow.mysde import DSSDE, SDEResult


# ---------------------------------------------------------------------------
# 内部 config 容器（供 modeling_ctrlflow.py 的 sde.config 查询）
# ---------------------------------------------------------------------------

class _MixedConfig:
    def __init__(self, num_train_timesteps: int):
        self.num_train_timesteps = num_train_timesteps


# ---------------------------------------------------------------------------
# Mixed W₂→KL Scheduler
# ---------------------------------------------------------------------------

class MixedW2KLScheduler:
    """DS-SDE 路径上的 W₂→KL 分层混合调度器。

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
                              实际扩散系数 = g²(t) × kl_sde_noise_scale
        kl_as_velocity_weight: KL 分量中 velocity 监督的占比 λ，默认 0.5
                              (1-λ) 为直接 epsilon 监督占比
        lyapunov_reg_weight:  §7.1 Lyapunov 正则项权重（0 → 关闭），默认 0.0
        num_precompute_steps: DSSDE lookup table 精度，默认 1000
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
    ):
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

        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.t_switch = t_switch
        self.switch_sharpness = switch_sharpness
        self.kl_sde_noise_scale = kl_sde_noise_scale
        self.kl_as_velocity_weight = kl_as_velocity_weight
        self.lyapunov_reg_weight = lyapunov_reg_weight

        # 内部 DS-SDE：负责 α/σ/α'/σ'/g² 的所有计算
        # prediction_type 设为 "epsilon" 只是占位符，实际不走 SDEBase.step
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
        return torch.sigmoid(
            torch.tensor(self.switch_sharpness, dtype=t.dtype, device=t.device)
            * (t - self.t_switch)
        )

    # ------------------------------------------------------------------
    # 前向路径：委托给 DSSDE（与纯 DS-SDE 训练完全一致）
    # ------------------------------------------------------------------

    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """前向插值：x_t = α_DS(t)·x₀ + σ_DS(t)·ε

        与 DSSDE.add_noise 完全一致，直接委托。
        """
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
        """一步推理积分，按当前 t 动态选择 ODE 或 SDE 步进。

        model_output: v̂（UNet 预测的 velocity），形状 (B, T, D)
        t:            当前连续时间步（标量）
        sample:       当前状态 x_t，形状 (B, T, D)

        步进规则：
          t > t_switch → 纯 ODE：x_{t+dt} = x_t + v̂·dt
          t ≤ t_switch → SDE：   x_{t+dt} = x_t + v̂·dt + √(2·g²·scale·|dt|)·z
        """
        if not isinstance(t, Tensor):
            t_val = torch.tensor(t, dtype=torch.float32, device=sample.device)
        else:
            t_val = t.float().to(sample.device)

        dt = -1.0 / self.num_inference_steps

        # Euler 步（ODE 部分，两个阶段都有）
        prev_sample = sample + model_output * dt

        # t ≤ t_switch：加入 DS-SDE 随机扩散项
        if t_val.item() <= self.t_switch and self.kl_sde_noise_scale > 0.0:
            g_sq = self._dssde._g_sq(t_val)          # 标量 tensor
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
    # 训练 loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        pred_velocity: Tensor,
        x0: Tensor,
        noise: Tensor,
        x_t: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """W₂→KL 混合训练 loss。

        参数:
            pred_velocity: v̂ = UNet 输出，形状 (B, T, D)
            x0:            干净数据，形状 (B, T, D)
            noise:         前向扩散时的高斯噪声，形状 (B, T, D)
            x_t:           加噪后的状态，形状 (B, T, D)
            timesteps:     整数时间步索引，形状 (B,)，范围 [0, num_train_timesteps)

        返回:
            标量 loss

        ──────────────────────────────────────────────────────────────
        loss 构成（按权重 w₁/w₂ 加权）：

        W₂ 分量（w₁ 主导，t 大时）：
            L_W2 = ‖v̂ - v*(t)‖²
            v*(t) = α'_DS(t)·x₀ + σ'_DS(t)·ε    ← 有限差分导数

        KL 分量（w₂ 主导，t 小时）：
            L_KL = λ · ‖v̂ - v*(t)‖²              ← velocity 层面
                 + (1-λ) · ‖ε̂ - ε‖²              ← epsilon 层面
            其中 ε̂ 由 v̂ 逆变换：
                ε̂ = (v̂ - α'_DS(t)·x₀_est) / σ'_DS(t)
                x₀_est = (x_t - σ_DS(t)·ε̂) 的近似
                       ≈ (x_t - σ_DS(t)·noise) / α_DS(t)    ← 用真实 noise 代替

        整体 loss：
            L = w₁·L_W2 + w₂·[λ·L_W2_vel + (1-λ)·L_KL_eps]
              = (w₁ + w₂·λ)·‖v̂ - v*‖² + w₂·(1-λ)·‖ε̂ - ε‖²

        当 kl_as_velocity_weight=1.0 时：KL 分量退化为纯 velocity 监督，
        两个分量完全等价，整体退化为普通 velocity MSE（验证对照实验）。

        当 kl_as_velocity_weight=0.0 时：W₂ 分量仍用 velocity 监督，
        KL 分量完全用 epsilon 监督，两分量异构，最大化区分度。

        可选 §7.1 Lyapunov 正则项（lyapunov_reg_weight > 0）：
            L_lyap = ‖x_t - v̂·dt - x₀‖²
        ──────────────────────────────────────────────────────────────
        """
        # ── t 的连续值（归一化到 [0,1]），广播到 (B,1,1) ──────────────
        t_cont = timesteps.float() / self.num_train_timesteps  # (B,)
        t_b = t_cont
        while t_b.dim() < x0.dim():
            t_b = t_b.unsqueeze(-1)  # (B,1,1)

        # ── 从 lookup table 取路径系数及其导数 ────────────────────────
        alpha     = self._dssde._alpha(t_b)        # (B,1,1)
        sigma     = self._dssde._sigma(t_b)        # (B,1,1)
        alpha_p   = self._dssde._alpha_prime(t_b)  # (B,1,1)  dα/dt
        sigma_p   = self._dssde._sigma_prime(t_b)  # (B,1,1)  dσ/dt

        # ── velocity target v*(t) = α'·x₀ + σ'·ε ─────────────────────
        v_target = alpha_p * x0 + sigma_p * noise  # (B,T,D)

        # ── 逐样本权重 w₁(t)，形状 (B,1,1) ───────────────────────────
        w1 = self._w1(t_cont)
        while w1.dim() < x0.dim():
            w1 = w1.unsqueeze(-1)
        w2 = 1.0 - w1

        # ── W₂ 分量：velocity MSE ─────────────────────────────────────
        loss_vel = F.mse_loss(pred_velocity, v_target.detach(), reduction="none")

        # ── KL 分量：epsilon MSE ──────────────────────────────────────
        # 从预测的 velocity 逆算 ε̂：
        #   v̂ = α'·x₀_est + σ'·ε̂
        #   x₀_est ≈ (x_t - σ·ε) / α   （用真实 ε 逼近，仅 loss 计算用）
        # 消去 x₀_est：
        #   ε̂ = (v̂ - α'·x₀_est) / σ'
        #      = (v̂ - α'·(x_t - σ·ε)/α) / σ'
        # 当 σ'(t) 接近零时（t→0 端点）数值可能不稳定，
        # 用 sigma_p_safe 做分母保护。
        sigma_p_safe = sigma_p.abs().clamp(min=1e-4)
        x0_est = (x_t - sigma * noise) / alpha.clamp(min=1e-4)
        pred_eps = (pred_velocity - alpha_p * x0_est.detach()) / sigma_p_safe

        loss_eps = F.mse_loss(pred_eps, noise.detach(), reduction="none")

        # ── 合并（element-wise，保留 reduction="none" 以支持 mask）─────
        lam = self.kl_as_velocity_weight
        loss = (
            w1 * loss_vel                            # W₂ 阶段：velocity 监督
            + w2 * (lam * loss_vel + (1.0 - lam) * loss_eps)  # KL 阶段：混合监督
        )

        # ── §7.1 Lyapunov 正则项（可选）─────────────────────────────
        if self.lyapunov_reg_weight > 0.0:
            dt_approx = 1.0 / self.num_train_timesteps
            # 推理方向走一步后距 x₀ 的距离（t 减小方向）
            x_next_infer = x_t - pred_velocity * dt_approx
            loss_lyap = F.mse_loss(
                x_next_infer, x0.detach(), reduction="none"
            )
            loss = loss + self.lyapunov_reg_weight * loss_lyap

        return loss.mean()
