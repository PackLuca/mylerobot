"""扩散模型的SDE基类"""

import math
import torch
from torch import Tensor
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SDEResult:
    """SDE步进函数的结果"""

    prev_sample: Tensor


class SDEBase(ABC):
    """SDE调度器基类

    子类必须实现:
    - _alpha: 均值系数α(t)
    - _sigma: 标准差σ(t)
    - _beta_t: 瞬时β(t)
    - _reverse_drift: 逆向漂移系数
    - _diffusion_coeff: 扩散系数
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: str = "epsilon",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.prediction_type = prediction_type

        self.timesteps: Tensor | None = None
        self.num_inference_steps: int | None = None

    @abstractmethod
    def _alpha(self, t: Tensor) -> Tensor:
        """均值系数α(t)"""
        pass

    @abstractmethod
    def _sigma(self, t: Tensor) -> Tensor:
        """标准差σ(t)"""
        pass

    @abstractmethod
    def _beta_t(self, t: Tensor) -> Tensor:
        """瞬时β(t)"""
        pass

    def set_timesteps(self, num_inference_steps: int) -> None:
        """设置推理时的timesteps"""
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(
            1.0, 1.0 / num_inference_steps, num_inference_steps
        )

    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """前向扩散: x_t = α(t) x_0 + σ(t) ε

        参数:
            x0: 干净数据 (B, ...)
            noise: 要添加的噪声 (B, ...)
            timesteps: 时间步索引 (B,), 范围 [0, num_train_timesteps)
        """
        t = timesteps.float() / self.num_train_timesteps
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        alpha = self._alpha(t)
        sigma = self._sigma(t)
        return alpha * x0 + sigma * noise

    def step(
        self,
        model_output: Tensor,
        t: Tensor | float,
        sample: Tensor,
        generator: torch.Generator | None = None,
    ) -> SDEResult:
        """一步逆向SDE

        参数:
            model_output: 模型预测 (取决于prediction_type, 可以是epsilon或x0)
            t: 当前时间步
            sample: 当前样本
            generator: 随机数生成器

        返回:
            SDEResult, 包含 prev_sample
        """
        if not isinstance(t, Tensor):
            t_val = torch.tensor(t, dtype=torch.float32, device=sample.device)
        else:
            t_val = t.float().to(sample.device)

        dt = torch.tensor(-1.0 / self.num_inference_steps, device=sample.device)

        beta = self._beta_t(t_val)
        alpha = self._alpha(t_val)
        sigma = self._sigma(t_val)

        while beta.dim() < sample.dim():
            beta = beta.unsqueeze(-1)
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)

        if self.prediction_type == "epsilon":
            score = -model_output / sigma
        elif self.prediction_type == "sample":
            score = -(sample - alpha * model_output) / (sigma**2)
        elif self.prediction_type == "score":
            score = model_output
        else:
            raise ValueError(f"不支持的prediction_type: {self.prediction_type}")

        drift = self._reverse_drift(sample, beta, score, alpha, sigma, t_val) * dt
        diffusion_coeff = self._diffusion_coeff(beta, dt)

        z = torch.randn(
            sample.shape, device=sample.device, generator=generator, dtype=sample.dtype
        )
        prev_sample = sample + drift + diffusion_coeff * z

        if self.clip_sample:
            prev_sample = prev_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        return SDEResult(prev_sample=prev_sample)

    @abstractmethod
    def _reverse_drift(
        self, x: Tensor, beta: Tensor, score: Tensor, alpha: Tensor, sigma: Tensor, t: Tensor
    ) -> Tensor:
        """逆向漂移系数"""
        pass

    @abstractmethod
    def _diffusion_coeff(self, beta: Tensor, dt: Tensor) -> Tensor:
        """扩散系数"""
        pass


class VPSDE(SDEBase):
    """方差保持SDE (Variance-Preserving SDE)

    前向过程:  dx = -½ β(t) x dt + √β(t) dW
    边际分布:  q(x_t | x_0) = N(x_t; α(t) x_0, σ²(t) I)

    使用线性β调度: β(t) = β_min + t(β_max - β_min), t ∈ [0,1]
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: str = "epsilon",
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            prediction_type=prediction_type,
        )
        self.beta_min = beta_start * num_train_timesteps
        self.beta_max = beta_end * num_train_timesteps

    def _integral_beta(self, t: Tensor) -> Tensor:
        """∫₀ᵗ β(s) ds, 线性调度"""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2

    def _alpha(self, t: Tensor) -> Tensor:
        """均值系数α(t) = exp(-½ ∫₀ᵗ β(s) ds)"""
        return torch.exp(-0.5 * self._integral_beta(t))

    def _sigma(self, t: Tensor) -> Tensor:
        """标准差σ(t) = √(1 - α²(t))"""
        return torch.sqrt(torch.clamp(1.0 - self._alpha(t) ** 2, min=1e-5))

    def _beta_t(self, t: Tensor) -> Tensor:
        """瞬时β(t) = β_min + t(β_max - β_min)"""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _reverse_drift(
        self, x: Tensor, beta: Tensor, score: Tensor, alpha: Tensor, sigma: Tensor, t: Tensor
    ) -> Tensor:
        """逆向漂移: f̃(x,t) = -½ β x - β · score

        逆向SDE公式: f̃ = f(x,t) - g²(t)·s_θ，VP-SDE中 g²(t)=β(t)。
        score 已是 ∇_x log p，量纲正确，不应再乘 σ²。
        """
        return -0.5 * beta * x - beta * score

    def _diffusion_coeff(self, beta: Tensor, dt: Tensor) -> Tensor:
        """扩散系数: g(t) = √(β |dt|)"""
        return torch.sqrt(beta * torch.abs(dt))


class DSSDE(SDEBase):
    """Decoupled-Schedule SDE (DS-SDE)

    前向过程: dx = -γ(t) x dt + g(t) dW
    漂移收缩 γ(t) 与扩散 g(t) 完全解耦，各自独立设计。

    γ(t) = γ_min + (γ_max - γ_min) · sin²(πt/2)   # 正弦曲线，早弱后强
    g²(t) = g_min² + (g_max² - g_min²) · t          # 线性增长，数值稳定

    边际分布仍为高斯，α(t) 和 σ(t) 由 ODE 数值积分预计算为 lookup table。

    【扩展】
    _precompute 额外用中心差分存储：
      _alpha_prime_table  = dα/dt
      _sigma_prime_table  = dσ/dt
    供 Mixed-W₂→KL 调度器在 velocity 坐标系下计算混合 loss target 使用。
    这两张表对纯 DS-SDE 训练/推理没有任何影响（从不调用）。
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        gamma_min: float = 0.5,
        gamma_max: float = 3.0,
        g_min: float = 0.01,
        g_max: float = 1.0,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: str = "epsilon",
        num_precompute_steps: int = 1000,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            prediction_type=prediction_type,
        )
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.g_min_sq = g_min ** 2
        self.g_max_sq = g_max ** 2

        self._precompute(num_precompute_steps)

    # ------------------------------------------------------------------
    # 调度函数（纯 Python/Tensor，不依赖 lookup table）
    # ------------------------------------------------------------------
    def _gamma(self, t: Tensor) -> Tensor:
        """收缩强度 γ(t) = γ_min + (γ_max - γ_min)·sin²(πt/2)"""
        return self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sin(
            torch.tensor(math.pi / 2, dtype=t.dtype, device=t.device) * t
        ) ** 2

    def _g_sq(self, t: Tensor) -> Tensor:
        """扩散强度平方 g²(t) = g_min² + (g_max² - g_min²)·t"""
        return self.g_min_sq + (self.g_max_sq - self.g_min_sq) * t

    # ------------------------------------------------------------------
    # 预计算 lookup table
    # ------------------------------------------------------------------
    def _precompute(self, N: int) -> None:
        """数值积分预计算 α(t)、σ(t) 及其关于 t 的一阶导数。

        【原有逻辑，未做任何修改】
        ─────────────────────────────────────────────────────────────
        α(t) = exp(-∫₀ᵗ γ(s) ds)
        P(t) = E[x_t²]，满足 dP/dt = -2γ(t)P + g²(t)，P(0) = 1
        σ(t) = √(P(t) - α²(t))

        【新增：有限差分求导数表】
        ─────────────────────────────────────────────────────────────
        对已经算好的 alpha[0..N] 和 sigma[0..N]（步长 h = 1/N）
        用中心差分（内点）+ 单侧差分（端点）：

          alpha'[i] ≈ (alpha[i+1] - alpha[i-1]) / (2h),  1 ≤ i ≤ N-1
          alpha'[0] ≈ (alpha[1]   - alpha[0])   / h       前向差分
          alpha'[N] ≈ (alpha[N]   - alpha[N-1]) / h       后向差分

        sigma' 同理。

        精度分析：
          中心差分 O(h²)，单侧差分 O(h)，N=1000 时 h=0.001。
          α'(t) 的绝对值量级约为 O(γ_max)≈3，相对误差 < 0.1%，
          足够用于 loss 权重计算，无需更高阶方法。
        """
        ts = torch.linspace(0.0, 1.0, N + 1)  # (N+1,)
        dt = 1.0 / N

        gamma_vals = self._gamma(ts)  # (N+1,)

        # ── α(t)：梯形积分 ──────────────────────────────────────────
        int_gamma = torch.cumsum(
            (gamma_vals[:-1] + gamma_vals[1:]) / 2.0 * dt, dim=0
        )
        int_gamma = torch.cat([torch.zeros(1), int_gamma])  # (N+1,)
        alpha = torch.exp(-int_gamma)                        # (N+1,)

        # ── P(t) 和 σ(t)：欧拉法 ────────────────────────────────────
        P = torch.ones(N + 1)
        for i in range(N):
            P[i + 1] = P[i] + dt * (
                -2.0 * gamma_vals[i].item() * P[i].item()
                + self._g_sq(ts[i]).item()
            )

        sigma = torch.sqrt(torch.clamp(P - alpha ** 2, min=1e-5))  # (N+1,)

        # ── 有限差分求 α'(t) ─────────────────────────────────────────
        # 内点：中心差分（二阶精度）
        alpha_prime = torch.empty(N + 1)
        alpha_prime[1:-1] = (alpha[2:] - alpha[:-2]) / (2.0 * dt)
        # 端点：单侧差分（一阶精度）
        alpha_prime[0]  = (alpha[1] - alpha[0])  / dt   # 前向
        alpha_prime[-1] = (alpha[-1] - alpha[-2]) / dt   # 后向

        # ── 有限差分求 σ'(t) ─────────────────────────────────────────
        sigma_prime = torch.empty(N + 1)
        sigma_prime[1:-1] = (sigma[2:] - sigma[:-2]) / (2.0 * dt)
        sigma_prime[0]  = (sigma[1] - sigma[0])  / dt
        sigma_prime[-1] = (sigma[-1] - sigma[-2]) / dt

        # ── 存储所有表（不参与梯度）────────────────────────────────────
        self._ts = ts
        self._alpha_table       = alpha
        self._sigma_table       = sigma
        self._alpha_prime_table = alpha_prime   # 新增
        self._sigma_prime_table = sigma_prime   # 新增

    def _lookup(self, t: Tensor, table: Tensor) -> Tensor:
        """对 lookup table 做线性插值，支持任意形状的 t。

        对导数表和原始表使用完全相同的插值逻辑，保持一致性。
        """
        t_clamped = t.clamp(0.0, 1.0)
        N = len(table) - 1
        idx_float = t_clamped * N
        idx_lo = idx_float.long().clamp(0, N - 1)
        idx_hi = (idx_lo + 1).clamp(0, N)
        frac = idx_float - idx_lo.float()
        table = table.to(t.device)
        return table[idx_lo] * (1.0 - frac) + table[idx_hi] * frac

    # ------------------------------------------------------------------
    # SDEBase 抽象方法实现（原有，未改动）
    # ------------------------------------------------------------------
    def _alpha(self, t: Tensor) -> Tensor:
        return self._lookup(t, self._alpha_table)

    def _sigma(self, t: Tensor) -> Tensor:
        return self._lookup(t, self._sigma_table)

    def _beta_t(self, t: Tensor) -> Tensor:
        """返回 g²(t)，供基类 step() 用作扩散系数平方。"""
        return self._g_sq(t)

    def _reverse_drift(
        self, x: Tensor, beta: Tensor, score: Tensor, alpha: Tensor, sigma: Tensor, t: Tensor
    ) -> Tensor:
        """逆向漂移: f̃(x,t) = -γ(t)·x - g²(t)·score

        由 Anderson(1982) 逆向 SDE 公式:
            f̃ = f(x,t) - g²(t)·∇_x log p_t
              = -γ(t)·x  - g²(t)·score
        beta 此处即 g²(t)(由 _beta_t 返回，已完成 broadcast)。
        """
        gamma = self._gamma(t)
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(-1)
        return -gamma * x - beta * score

    def _diffusion_coeff(self, beta: Tensor, dt: Tensor) -> Tensor:
        """随机项系数: g(t)·√|dt| = √(g²(t)·|dt|)"""
        return torch.sqrt(beta * torch.abs(dt))

    # ------------------------------------------------------------------
    # 新增公开接口：供 MixedW2KLScheduler 使用
    # ------------------------------------------------------------------
    def _alpha_prime(self, t: Tensor) -> Tensor:
        """dα/dt，通过有限差分表插值。

        调用方式与 _alpha(t) 完全相同，支持任意形状的 t。
        仅供 Mixed-W₂→KL 调度器在构造 velocity target 时使用；
        纯 DS-SDE 训练/推理路径不调用此方法。
        """
        return self._lookup(t, self._alpha_prime_table)

    def _sigma_prime(self, t: Tensor) -> Tensor:
        """dσ/dt，通过有限差分表插值。

        同 _alpha_prime，仅供混合调度器使用。
        """
        return self._lookup(t, self._sigma_prime_table)

    def velocity_target(self, x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """在 DS-SDE 路径下计算 velocity 坐标系的训练目标。

        定义：v*(x_t, t) = dx_t/dt = α'(t)·x₀ + σ'(t)·ε

        与 W2FlowODE._velocity_target 接口和语义完全一致，
        区别仅在于 α'、σ' 来自 DS-SDE 的有限差分表而非解析公式。

        参数:
            x0:    干净数据     (B, T, D) 或任意与 noise 对齐的形状
            noise: 标准高斯噪声 (B, T, D)
            t:     连续时间     (B, 1, 1)，范围 [0, 1]，已完成广播扩展

        返回:
            v*: 与 x0 同形状的 velocity target
        """
        alpha_p = self._alpha_prime(t)  # (B, 1, 1)
        sigma_p = self._sigma_prime(t)  # (B, 1, 1)
        return alpha_p * x0 + sigma_p * noise
