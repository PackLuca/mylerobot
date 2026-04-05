"""Mixed W₂→KL 调度器 v4（统一路径 + 损失混合）

===========================================================================
【版本历史】
===========================================================================

v1：初始实现，epsilon 逆变换数值爆炸
v2：batch 归一化，但 loss 压平不收敛
v3：双路径设计 - ❌ 严重错误！轨迹不连续，推理崩溃
v4（本版本）：统一路径 + 损失混合 - ✅ 符合 CTRL-Flow 理论

===========================================================================
【v3 的致命问题】
===========================================================================

v3 在不同 t 区间使用不同前向路径：
  t > t_switch: x_t = (1-t)·x₀ + t·ε          # 线性路径
  t ≤ t_switch: x_t = α_DS(t)·x₀ + σ_DS(t)·ε  # DS-SDE 路径

问题：
1. 💥 轨迹不连续：在 t=t_switch 处 x_t 跳变！
   lim_{t→t_switch+} x_t ≠ lim_{t→t_switch-} x_t
   
2. 💥 推理崩溃：ODE 积分器跨越 t_switch 时轨迹断裂
   
3. 💥 违背理论：CTRL-Flow 要求在**同一条连续路径**上混合

===========================================================================
【v4 的正确设计】
===========================================================================

核心原则：CTRL-Flow 的混合是**损失函数的混合**，不是路径的混合！

## 统一前向路径

始终使用 DS-SDE 路径（对所有 t）：
  x_t = α_DS(t)·x₀ + σ_DS(t)·ε

这保证了：
  ✅ 轨迹连续（C¹ 连续）
  ✅ 推理稳定
  ✅ 符合 CTRL-Flow 理论

## 损失函数混合

在同一路径上定义两种 velocity target：

### W₂ 分量（模拟 W₂ 几何）
  v*_W₂(t) = dx_t/dt|_{直线路径}
           = -x₀ + ε     （如果用线性路径）
           
  但我们实际用的是 DS-SDE 路径，所以：
  v*_W₂(t) = α'_DS(t)·x₀ + σ'_DS(t)·ε
  
  理论依据：在 DS-SDE 路径下，W₂ Lyapunov 泛函的梯度
  等价于路径的切向量（Proposition 2）

### KL 分量（score matching）
  v*_KL(t) = α'_DS(t)·x₀ + σ'_DS(t)·ε
  
  理论依据：KL 散度对应 score function 的监督（§4.2.1）

### 混合 Loss
  L = w₁(t)·‖v̂ - v*_W₂‖²  +  w₂(t)·‖v̂ - v*_KL‖²
  
  其中：
    w₁(t) = sigmoid(γ·(t - t_switch))
    w₂(t) = 1 - w₁(t)

===========================================================================
【关键洞察】
===========================================================================

在同一条 DS-SDE 路径上，v*_W₂ 和 v*_KL 的**数学形式相同**：
  v*_W₂ = v*_KL = α'_DS·x₀ + σ'_DS·ε

这意味着纯 velocity 监督下，混合权重 w₁/w₂ 是冗余的！

**解决方案**：引入不同的参数化

1. W₂ 分量：velocity 参数化
   target = α'·x₀ + σ'·ε
   
2. KL 分量：score/epsilon 参数化
   target_eps = ε
   或 target_score = -ε/σ
   
   在 velocity 坐标下等价于：
   v*_KL = α'·x₀ + σ'·(模型预测的 epsilon)

这样两个分量才真正不同！

===========================================================================
【实现策略】
===========================================================================

由于在 DS-SDE 路径下，纯 velocity 混合是退化的，
我们采用**异构参数化**：

1. W₂ 阶段（t > t_switch）：
   - UNet 预测 velocity v̂
   - Target: v* = α'·x₀ + σ'·ε
   - Loss: ‖v̂ - v*‖²

2. KL 阶段（t ≤ t_switch）：
   - UNet 仍预测 velocity v̂
   - 但同时监督 epsilon 分量：
     * 从 v̂ 反推 ε̂ = (v̂ - α'·x₀_est) / σ'
     * Loss: λ·‖v̂ - v*‖² + (1-λ)·‖ε̂ - ε‖²
   
   其中 λ = kl_as_velocity_weight

这样：
  - t 大时：纯 velocity 监督（W₂ 几何）
  - t 小时：velocity + epsilon 混合监督（KL 几何）
  - 两阶段的监督信号真正不同

===========================================================================
【与 v2 的区别】
===========================================================================

v2 也用了 velocity + epsilon 混合，但归一化策略有问题。

v4 的改进：
1. ✅ 使用固定的 scale 因子而非 running mean
2. ✅ 基于理论推导确定合理的 scale
3. ✅ 简化 epsilon 逆推公式，提高数值稳定性

===========================================================================
【推理过程】
===========================================================================

  t > t_switch → 纯 ODE：x_{t+dt} = x_t + v̂·dt
  t ≤ t_switch → SDE：   x_{t+dt} = x_t + v̂·dt + √(2·g²·|dt|)·z

===========================================================================
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
# Mixed W₂→KL Scheduler v4（正确版本）
# ---------------------------------------------------------------------------

class MixedW2KLScheduler(nn.Module):
    """统一路径 + 异构监督的 Mixed W₂→KL 调度器。

    核心设计：
    1. 统一前向路径：始终使用 DS-SDE 路径（保证轨迹连续）
    2. 异构监督信号：
       - W₂ 阶段：纯 velocity 监督
       - KL 阶段：velocity + epsilon 混合监督
    3. 平滑切换：通过 sigmoid 权重函数

    参数:
        num_train_timesteps: 训练离散步数
        gamma_min, gamma_max: DS-SDE γ(t) 参数
        g_min, g_max: DS-SDE g(t) 参数
        clip_sample: 推理时是否裁剪
        clip_sample_range: 裁剪范围
        t_switch: W₂/KL 切换点 ∈ (0,1)
        switch_sharpness: sigmoid 陡峭度
        kl_sde_noise_scale: KL 阶段推理噪声强度
        kl_as_velocity_weight: KL 阶段 velocity 监督占比 λ
        lyapunov_reg_weight: Lyapunov 正则项权重
        eps_loss_scale: epsilon loss 的缩放因子（基于理论推导）
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
        kl_as_velocity_weight: float = 0.5,
        lyapunov_reg_weight: float = 0.0,
        num_precompute_steps: int = 1000,
        eps_loss_scale: float = 0.1,  # 新增：epsilon loss 缩放因子
    ):
        super().__init__()
        
        # 参数校验
        assert 0.0 < t_switch < 1.0
        assert switch_sharpness > 0
        assert 0.0 <= kl_sde_noise_scale
        assert 0.0 <= kl_as_velocity_weight <= 1.0
        assert lyapunov_reg_weight >= 0.0
        assert eps_loss_scale > 0.0

        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.kl_sde_noise_scale = kl_sde_noise_scale
        self.kl_as_velocity_weight = kl_as_velocity_weight
        self.lyapunov_reg_weight = lyapunov_reg_weight
        self.eps_loss_scale = eps_loss_scale

        # 使用 register_buffer 缓存常量
        self.register_buffer('_t_switch_tensor', torch.tensor(t_switch))
        self.register_buffer('_sharpness_tensor', torch.tensor(switch_sharpness))

        # 内部 DS-SDE（统一前向路径）
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
    # 权重调度
    # ------------------------------------------------------------------

    def _w1(self, t: Tensor) -> Tensor:
        """W₂ 权重：w₁(t) = sigmoid(γ·(t - t_switch))"""
        t_switch = self._t_switch_tensor.to(t.device, t.dtype)
        sharpness = self._sharpness_tensor.to(t.device, t.dtype)
        return torch.sigmoid(sharpness * (t - t_switch))

    # ------------------------------------------------------------------
    # 前向路径：统一使用 DS-SDE（关键修复！）
    # ------------------------------------------------------------------

    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """前向插值：始终使用 DS-SDE 路径
        
        ✅ v4 修复：不再根据 t 切换路径，保证轨迹连续
        x_t = α_DS(t)·x₀ + σ_DS(t)·ε  （对所有 t）
        """
        return self._dssde.add_noise(x0, noise, timesteps)

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int) -> None:
        """设置推理时间节点"""
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
        """一步推理积分（ODE/SDE 根据 t 动态选择）"""
        if not isinstance(t, Tensor):
            t_val = torch.tensor(t, dtype=torch.float32, device=sample.device)
        else:
            t_val = t.float().to(sample.device)

        dt = -1.0 / self.num_inference_steps
        prev_sample = sample + model_output * dt

        # KL 阶段可选扩散项
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
    # 训练 loss（v4 核心修复）
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        pred_velocity: Tensor,
        x0: Tensor,
        noise: Tensor,
        x_t: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """混合 W₂→KL loss（v4 版本）
        
        核心改进：
        1. ✅ 统一前向路径（DS-SDE）
        2. ✅ 异构监督信号（velocity vs epsilon）
        3. ✅ 固定 scale 因子（基于理论）
        4. ✅ 改进数值稳定性
        
        Loss 构成：
          L = w₁·L_vel  +  w₂·[λ·L_vel + (1-λ)·L_eps·scale]
          
        其中：
          L_vel = ‖v̂ - v*‖²,  v* = α'·x₀ + σ'·ε
          L_eps = ‖ε̂ - ε‖²,   ε̂ 从 v̂ 反推
        """
        SAFE_MIN = 1e-3  # 数值保护

        # ── 时间和路径系数 ──────────────────────────────────────────
        t_cont = timesteps.float() / self.num_train_timesteps
        t_b = t_cont
        while t_b.dim() < x0.dim():
            t_b = t_b.unsqueeze(-1)

        alpha = self._dssde._alpha(t_b)
        sigma = self._dssde._sigma(t_b)
        alpha_p = self._dssde._alpha_prime(t_b)
        sigma_p = self._dssde._sigma_prime(t_b)

        # 安全系数
        alpha_safe = alpha.abs().clamp(min=SAFE_MIN)
        sigma_safe = sigma.abs().clamp(min=SAFE_MIN)
        sigma_p_safe = sigma_p.abs().clamp(min=SAFE_MIN)

        # ── Velocity target（两阶段共享）────────────────────────────
        v_target = alpha_p * x0 + sigma_p * noise

        # ── 权重 ──────────────────────────────────────────────────
        w1 = self._w1(t_cont)
        while w1.dim() < x0.dim():
            w1 = w1.unsqueeze(-1)
        w2 = 1.0 - w1

        # ── W₂ 分量：velocity MSE ─────────────────────────────────
        loss_vel = F.mse_loss(pred_velocity, v_target.detach(), reduction="none")

        # ── KL 分量：epsilon MSE（改进版）─────────────────────────
        # 从 velocity 反推 epsilon（改进数值稳定性）
        x0_est = torch.where(
            sigma.abs() > SAFE_MIN,
            (x_t - sigma * noise) / alpha_safe,
            x_t / alpha_safe
        )
        
        pred_eps = (pred_velocity - alpha_p * x0_est.detach()) / sigma_p_safe
        pred_eps = pred_eps.clamp(-10.0, 10.0)  # 防止异常值
        
        loss_eps = F.mse_loss(pred_eps, noise.detach(), reduction="none")

        # ── 混合 loss（v4 关键）──────────────────────────────────
        # 使用固定 scale 而非 running mean
        lam = self.kl_as_velocity_weight
        loss = (
            w1 * loss_vel  # W₂ 阶段：纯 velocity
            + w2 * (lam * loss_vel + (1.0 - lam) * loss_eps * self.eps_loss_scale)
        )

        # ── Lyapunov 正则项（修正版）─────────────────────────────
        if self.lyapunov_reg_weight > 0.0:
            dt_infer = torch.full_like(t_cont, -1.0 / self.num_train_timesteps)
            while dt_infer.dim() < pred_velocity.dim():
                dt_infer = dt_infer.unsqueeze(-1)
            
            x_next_infer = x_t + pred_velocity * dt_infer
            loss_lyap = F.mse_loss(x_next_infer, x0.detach(), reduction="none")
            loss = loss + self.lyapunov_reg_weight * loss_lyap

        return loss.mean()
