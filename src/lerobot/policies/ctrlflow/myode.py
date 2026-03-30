"""W₂ Lyapunov 泛函下的 Flow Matching ODE（CTRL-Flow 框架 §4.2.2 / §5.2.1）

===========================================================================
【理论基础】
===========================================================================

CTRL-Flow §4.2.2（Proposition 2）证明：Flow Matching 是 CTRL-Flow 在
g ≡ 0（无扩散）极限下的特例，Lyapunov 泛函为 W₂² 距离。

最优速度场由 Benamou-Brenier 动态最优传输给出：
    v*(x, t) = -∇_{W₂} L(p_t) = x₀ - ε    （线性路径情形）

收敛率（§6.4.2）：γ = 1/C_OT，依赖最优传输几何常数。

===========================================================================
【时间约定——与 modeling_ctrlflow.py 对齐】
===========================================================================

t=1 → 纯噪声 ε，t=0 → 干净数据 x₀

前向路径（训练）：x_t = path(x₀, ε, t)，t=1 时为噪声，t=0 时为数据
速度场方向（推理）：v 使 x_t 沿 t 减小方向从噪声趋向数据
推理步进：x_{t+dt} = x_t + v_θ(x_t, t) · dt，dt < 0

===========================================================================
【可设计的前向路径】
===========================================================================

路径类型（path_type）控制插值曲线形状，影响速度场曲率分布：

1. "linear"（默认，标准 Flow Matching）
   x_t = (1-t) x₀ + t ε
   v* = x₀ - ε（常数，最简单）
   对应 W₂ Benamou-Brenier 最优传输直线路径

2. "cosine"
   α(t) = cos(πt/2)，σ(t) = sin(πt/2)
   x_t = cos(πt/2) x₀ + sin(πt/2) ε
   v* = -π/2 · [sin(πt/2) x₀ - cos(πt/2) ε]
   曲率集中在 t≈0.5，两端变化平缓，训练难度更均匀

3. "sqrt"（平方根路径）
   α(t) = √(1-t)，σ(t) = √t
   x_t = √(1-t) x₀ + √t ε
   v* = -1/(2√(1-t)) x₀ + 1/(2√t) ε
   早期（t≈1）速度场幅度大，数据端（t≈0）收敛快
   注：t 两端需裁剪防止除零

4. "poly"（多项式路径，阶数由 path_poly_order 控制）
   α(t) = (1-t)^p，σ(t) = t^p，p = path_poly_order（默认 2）
   v* = -p(1-t)^{p-1} x₀ + p·t^{p-1} ε
   p>1 时速度场在 t≈0/1 两端更陡，中间平缓

===========================================================================
【可选 Lyapunov 正则项（§7.1）】
===========================================================================

当 lyapunov_reg_weight > 0 时，在 MSE loss 之外附加：
    L_lyap = ‖(x_t + v_θ·dt) - x₀‖²
这是 Theorem 1 稳定条件的显式惩罚：
"走一步后离数据分布还有多远"

===========================================================================
【可选推理扩散项（§5.2.1 Schrödinger Bridge 极限）】
===========================================================================

当 inference_noise_scale > 0 时，推理步进变为：
    x_{t+dt} = x_t + v_θ·dt + √(2β|dt|) · z
β = inference_noise_scale，对应 Theorem 1 条件 2 的贡献（α+β）
β → 0 退化为标准 Flow Matching ODE
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 返回结果容器（与 SDEBase.SDEResult 接口兼容）
# ---------------------------------------------------------------------------

@dataclass
class ODEResult:
    """ODE/SDE 步进结果，与 SDEResult 接口兼容。"""
    prev_sample: Tensor


# ---------------------------------------------------------------------------
# 路径类型定义
# ---------------------------------------------------------------------------

PathType = Literal["linear", "cosine", "sqrt", "poly"]


# ---------------------------------------------------------------------------
# W₂ Flow Matching ODE
# ---------------------------------------------------------------------------

class W2FlowODE:
    """W₂ Lyapunov 泛函下的 Flow Matching ODE（CTRL-Flow §4.2.2 / §5.2.1）

    训练：UNet 学习速度场 v_θ ≈ v*（由路径类型决定目标）
          loss = MSE(v_θ, v*) + λ · L_lyap（可选）

    推理：Euler 积分（或 RK4）从噪声走向数据
          可选附加扩散项近似 Schrödinger Bridge

    与 modeling_ctrlflow.py 的接口约定：
    - add_noise(x0, noise, timesteps) → x_t
    - set_timesteps(num_inference_steps)
    - step(model_output, t, sample) → ODEResult
    - compute_loss(pred_velocity, x0, noise, x_t) → scalar
    """

    # prediction_type 固定为 velocity，供外部查询
    prediction_type: str = "velocity"

    def __init__(
        self,
        num_train_timesteps: int = 100,
        path_type: PathType = "linear",
        path_poly_order: float = 2.0,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        lyapunov_reg_weight: float = 0.0,
        inference_noise_scale: float = 0.0,
        sampling_method: str = "euler",
        eps: float = 1e-4,
    ):
        """
        参数:
            num_train_timesteps: 训练离散步数
            path_type: 插值路径类型，见模块文档
            path_poly_order: poly 路径的阶数 p（仅 path_type="poly" 时生效）
            clip_sample: 推理时是否裁剪输出至 [-r, r]
            clip_sample_range: 裁剪范围 r
            lyapunov_reg_weight: §7.1 Lyapunov 正则项权重（0 → 纯速度场监督）
            inference_noise_scale: 推理时扩散项强度 β（0 → 纯 ODE）
                对应 CTRL-Flow Theorem 1 条件 2，β > 0 时近似 Schrödinger Bridge
            sampling_method: 推理积分方法，"euler" 或 "rk4"
                rk4 需要外部在推理循环中多次调用 unet（modeling 层负责）
            eps: sqrt/poly 路径端点数值保护下界（防除零）
        """
        assert path_type in ("linear", "cosine", "sqrt", "poly"), \
            f"path_type 必须是 linear/cosine/sqrt/poly，得到 {path_type}"
        assert sampling_method in ("euler", "rk4"), \
            f"sampling_method 必须是 euler/rk4，得到 {sampling_method}"

        self.num_train_timesteps = num_train_timesteps
        self.path_type = path_type
        self.path_poly_order = path_poly_order
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.lyapunov_reg_weight = lyapunov_reg_weight
        self.inference_noise_scale = inference_noise_scale
        self.sampling_method = sampling_method
        self.eps = eps

        self.timesteps: Tensor | None = None
        self.num_inference_steps: int | None = None

        # 供 modeling_ctrlflow.py 查询
        self.config = _ODEConfig(num_train_timesteps=num_train_timesteps)

    # ------------------------------------------------------------------
    # 路径系数：α(t)、σ(t)
    # ------------------------------------------------------------------

    def _alpha(self, t: Tensor) -> Tensor:
        """数据系数 α(t)，满足 x_t = α(t)·x₀ + σ(t)·ε

        t=0 时 α=1（纯数据），t=1 时 α→0（接近纯噪声）
        """
        if self.path_type == "linear":
            return 1.0 - t
        elif self.path_type == "cosine":
            return torch.cos(math.pi / 2.0 * t)
        elif self.path_type == "sqrt":
            return torch.sqrt((1.0 - t).clamp(min=self.eps))
        elif self.path_type == "poly":
            p = self.path_poly_order
            return (1.0 - t).clamp(min=0.0) ** p
        else:
            raise ValueError(f"未知 path_type: {self.path_type}")

    def _sigma(self, t: Tensor) -> Tensor:
        """噪声系数 σ(t)，满足 x_t = α(t)·x₀ + σ(t)·ε

        t=0 时 σ=0（纯数据），t=1 时 σ=1（纯噪声）
        """
        if self.path_type == "linear":
            return t
        elif self.path_type == "cosine":
            return torch.sin(math.pi / 2.0 * t)
        elif self.path_type == "sqrt":
            return torch.sqrt(t.clamp(min=self.eps))
        elif self.path_type == "poly":
            p = self.path_poly_order
            return t.clamp(min=0.0) ** p
        else:
            raise ValueError(f"未知 path_type: {self.path_type}")

    def _velocity_target(
        self, x0: Tensor, noise: Tensor, t: Tensor
    ) -> Tensor:
        """计算路径对应的真实速度场目标 v*(x_t, t) = dx_t/dt

        由路径 x_t = α(t)·x₀ + σ(t)·ε 对 t 求导：
            v* = α'(t)·x₀ + σ'(t)·ε
        注意：v* 方向使 t 增大时 x_t 趋向噪声，
        推理时 dt < 0，sample 实际向数据方向移动。
        """
        if self.path_type == "linear":
            # α'(t) = -1, σ'(t) = 1
            return -x0 + noise

        elif self.path_type == "cosine":
            # α'(t) = -π/2·sin(πt/2), σ'(t) = π/2·cos(πt/2)
            c = math.pi / 2.0
            return -c * torch.sin(c * t) * x0 + c * torch.cos(c * t) * noise

        elif self.path_type == "sqrt":
            # α'(t) = -1/(2√(1-t)), σ'(t) = 1/(2√t)
            alpha_prime = -0.5 / torch.sqrt((1.0 - t).clamp(min=self.eps))
            sigma_prime = 0.5 / torch.sqrt(t.clamp(min=self.eps))
            return alpha_prime * x0 + sigma_prime * noise

        elif self.path_type == "poly":
            # α'(t) = -p·(1-t)^{p-1}, σ'(t) = p·t^{p-1}
            p = self.path_poly_order
            alpha_prime = -p * (1.0 - t).clamp(min=0.0) ** (p - 1)
            sigma_prime = p * t.clamp(min=self.eps) ** (p - 1)
            return alpha_prime * x0 + sigma_prime * noise

        else:
            raise ValueError(f"未知 path_type: {self.path_type}")

    # ------------------------------------------------------------------
    # 前向过程（训练）
    # ------------------------------------------------------------------

    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """前向插值：x_t = α(t)·x₀ + σ(t)·ε

        t=0 时为纯数据，t=1 时为纯噪声。
        与 SDEBase.add_noise 接口完全一致。

        参数:
            x0: 干净数据 (B, T, D)
            noise: 标准高斯噪声 (B, T, D)
            timesteps: 整数索引 (B,)，范围 [0, num_train_timesteps)
        返回:
            x_t: (B, T, D)
        """
        t = timesteps.float() / self.num_train_timesteps
        # 广播到 (B, 1, 1)
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        alpha = self._alpha(t)
        sigma = self._sigma(t)
        return alpha * x0 + sigma * noise

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int) -> None:
        """设置推理时间节点（t 从 1 降到 1/N，对应噪声→数据）。

        与 SDEBase.set_timesteps 接口完全一致。
        """
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
    ) -> ODEResult:
        """一步推理积分：Euler 或 RK4（RK4 仅单步，modeling 层负责多次调用）

        Euler：x_{t+dt} = x_t + v_θ(x_t, t) · dt
        可选扩散项（inference_noise_scale > 0）：
            x_{t+dt} = x_t + v_θ·dt + √(2β|dt|) · z
            对应 CTRL-Flow §5.2.1 Schrödinger Bridge 极限，β > 0

        dt < 0（t 从 1→0），v_θ 指向 t 增大方向，
        乘以 dt 后实际向数据方向移动。

        参数:
            model_output: v_θ(x_t, t) (B, T, D)
            t: 当前时间步（标量）
            sample: 当前状态 x_t (B, T, D)
            generator: 随机数生成器（ODE 下忽略，SDE 模式下使用）
        返回:
            ODEResult，包含 prev_sample
        """
        dt = -1.0 / self.num_inference_steps

        prev_sample = sample + model_output * dt

        # 可选：Schrödinger Bridge 近似扩散项（§5.2.1）
        if self.inference_noise_scale > 0.0:
            beta = self.inference_noise_scale
            noise_coeff = math.sqrt(2.0 * beta * abs(dt))
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

        return ODEResult(prev_sample=prev_sample)

    # ------------------------------------------------------------------
    # 训练 loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        pred_velocity: Tensor,
        x0: Tensor,
        noise: Tensor,
        x_t: Tensor,
        timesteps: Tensor | None = None,
    ) -> Tensor:
        """W₂ Flow Matching 训练 loss

        主 loss（速度场回归，MSE）：
            target = v*(x_t, t) = α'(t)·x₀ + σ'(t)·ε
            L_main = MSE(v_θ, target)

        可选辅助 loss（§7.1 Lyapunov 正则，lyapunov_reg_weight > 0）：
            L_lyap = ‖(x_t + v_θ·dt) - x₀‖²
            即"走一步后离数据有多远"

        参数:
            pred_velocity: v_θ(x_t, t)，UNet 输出 (B, T, D)
            x0:   干净数据 (B, T, D)
            noise: 添加的高斯噪声 (B, T, D)
            x_t:  当前插值状态 (B, T, D)
            timesteps: 整数时间步索引 (B,)，用于计算 v* 的时变系数
                若为 None 则退回线性路径的常数目标（x₀ - ε）
        返回:
            标量 loss
        """
        # ── 计算速度场目标 v* ──────────────────────────────────────────
        if timesteps is not None and self.path_type != "linear":
            # 需要 t 的连续值来计算时变系数
            t = timesteps.float() / self.num_train_timesteps  # (B,)
            while t.dim() < x0.dim():
                t = t.unsqueeze(-1)  # (B, 1, 1)
            target = self._velocity_target(x0, noise, t)
        else:
            # linear 路径：v* = noise - x0（常数，无需 t）
            # 其他路径若未传入 timesteps 也退回此简化形式
            target = noise - x0

        loss_main = F.mse_loss(pred_velocity, target.detach())

        if self.lyapunov_reg_weight <= 0.0:
            return loss_main

        # ── §7.1 Lyapunov 正则项 ──────────────────────────────────────
        # 模拟走一小步后到达的位置
        dt_approx = 1.0 / self.num_train_timesteps
        x_next = x_t + pred_velocity * dt_approx          # 走一步（t 增大方向）
        # 注：训练时 dt > 0（t 增大），推理时 dt < 0（t 减小）
        # Lyapunov 正则惩罚"走一步后距 x₀ 的距离"
        # 应惩罚 x_t - v_θ·dt 距 x₀ 的距离（推理方向）
        x_next_infer = x_t - pred_velocity * dt_approx
        loss_lyap = F.mse_loss(x_next_infer, x0.detach())

        return loss_main + self.lyapunov_reg_weight * loss_lyap

    # ------------------------------------------------------------------
    # 兼容 SDEBase 的占位方法（供 modeling_ctrlflow.py 统一查询）
    # ------------------------------------------------------------------

    def _beta_t(self, t: Tensor) -> Tensor:
        """占位：Flow Matching 无随机项，返回零"""
        return torch.zeros_like(t)


# ---------------------------------------------------------------------------
# 内部 config 容器（供 modeling_ctrlflow.py 的 sde.config 查询）
# ---------------------------------------------------------------------------

class _ODEConfig:
    def __init__(self, num_train_timesteps: int):
        self.num_train_timesteps = num_train_timesteps
