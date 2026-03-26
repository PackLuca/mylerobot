"""扩散模型的SDE基类"""

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

        drift = self._reverse_drift(sample, beta, score, alpha, sigma) * dt
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
        self, x: Tensor, beta: Tensor, score: Tensor, alpha: Tensor, sigma: Tensor
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
        self, x: Tensor, beta: Tensor, score: Tensor, alpha: Tensor, sigma: Tensor
    ) -> Tensor:
        """逆向漂移: f(x,t) = -½ β x - β score · σ²"""
        return -0.5 * beta * x - beta * score * (sigma**2)

    def _diffusion_coeff(self, beta: Tensor, dt: Tensor) -> Tensor:
        """扩散系数: g(t) = √(β |dt|)"""
        return torch.sqrt(beta * torch.abs(dt))
