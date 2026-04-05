"""混合 SDE/ODE 调度器（CTRL-Flow 框架 §5.3）

将 W2-ODE（最优传输粗对齐）与 VP-SDE（扩散精细调优）组合为统一的
混合生成动力学，通过两阶段权重调度实现从全局结构到局部密度的渐进优化。

===========================================================================
【理论基础 —— CTRL-Flow §5.3】
===========================================================================

混合 Lyapunov 泛函：
    L_hybrid(p_t) = w_W2(t) · L_W2(p_t) + w_SDE(t) · L_SDE(p_t)

组合动力学：
    f_combined = w_W2 · f_W2 + w_SDE · f_SDE
    (gg^T)_combined = w_W2 · 0 + w_SDE · g_SDE^2

稳定性保证（Theorem 1）：
    dL_hybrid/dt ≤ -(w_W2·α_W2·Φ_W2 + w_SDE·α_SDE·Φ_SDE) ≤ 0

===========================================================================
【两阶段权重调度】
===========================================================================

推理时 t 从 1→0（噪声→数据）：

Phase 1（粗对齐，t > T + δ/2）：
    w_W2 ≈ 1, w_SDE ≈ 0
    W2-ODE 主导，最优传输直线路径快速建立全局结构

Phase 2（精细调优，t < T - δ/2）：
    w_W2 ≈ 0, w_SDE ≈ 1
    VP-SDE 主导，扩散过程优化局部概率密度

过渡区（T - δ/2 ≤ t ≤ T + δ/2）：
    smoothstep 插值，C1 连续，避免动力学突变

===========================================================================
【UNet 输出语义 —— 方案 A：单输出 ε + 数学转换】
===========================================================================

UNet 统一输出 ε_pred（噪声估计）。

VP-SDE 直接使用：
    score = -ε_pred / σ
    drift_SDE = (-½βx + β·ε_pred/σ) · dt

W2-ODE 从 ε_pred 推导（仅 linear 路径完全等价）：
    x₀_est = (x_t - σ·ε_pred) / α
    v_pred = -x₀_est + ε_pred = -x_t/α + (σ/α + 1)·ε_pred
    drift_W2 = v_pred · dt

对 linear 路径 x_t = (1-t)·x₀ + t·ε：
    v* = -x₀ + ε
    v_pred = -x₀ + ε_pred
    当 ε_pred = ε 时 v_pred = v*，完全等价，无矛盾。

===========================================================================
【推理步进公式】
===========================================================================

    prev_sample = sample + f_combined + diff_combined · z

其中：
    f_combined   = w_W2 · drift_W2 + w_SDE · drift_SDE
    diff_combined = √(w_W2 · 0 + w_SDE · g_SDE² · |dt|)
    z ~ N(0, I)

===========================================================================
【训练 Loss 组合】
===========================================================================

    L = w_W2(t) · L_W2 + w_SDE(t) · L_SDE

其中：
    L_SDE = MSE(ε_pred, ε)
    L_W2  = MSE(v_pred, v*) = (σ/α + 1)² · MSE(ε_pred, ε)  （linear 路径）

对 linear 路径，两者本质上是同一 loss 的缩放版本，不会冲突。
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.ctrlflow.mysde import VPSDE
from lerobot.policies.ctrlflow.myode import W2FlowODE


# ---------------------------------------------------------------------------
# 返回结果容器（与 SDEResult / ODEResult 接口兼容）
# ---------------------------------------------------------------------------


@dataclass
class HybridSDEStepResult:
    """混合 SDE 步进结果"""

    prev_sample: Tensor


# ---------------------------------------------------------------------------
# 平滑过渡函数
# ---------------------------------------------------------------------------


def _smoothstep(x: Tensor) -> Tensor:
    """C1 连续 smoothstep：3x² - 2x³，将 x ∈ [0,1] 映射到 [0,1]"""
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ---------------------------------------------------------------------------
# HybridSDE
# ---------------------------------------------------------------------------


class HybridSDE:
    """W2-ODE + VP-SDE 混合调度器（CTRL-Flow §5.3）

    训练：UNet 学习 ε_pred，混合 loss = w_W2(t)·L_W2 + w_SDE(t)·L_SDE
    推理：从 ε_pred 同时推导 W2-ODE drift 和 VP-SDE drift，加权组合步进

    与 SDEBase / W2FlowODE 接口完全兼容：
    - add_noise(x0, noise, timesteps) → x_t
    - set_timesteps(num_inference_steps)
    - step(model_output, t, sample) → HybridSDEStepResult
    - compute_loss(pred_epsilon, x0, noise, x_t, timesteps) → scalar
    """

    # 固定为 epsilon 预测，VP-SDE 和 W2-ODE 都从此出发
    prediction_type: str = "epsilon"

    def __init__(
        self,
        num_train_timesteps: int = 100,
        # VP-SDE 参数
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        # W2-ODE 参数
        w2_path_type: str = "linear",
        w2_path_poly_order: float = 2.0,
        # 混合调度参数
        hybrid_transition_step: float = 0.5,
        hybrid_transition_width: float = 0.15,
        # 通用参数
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        # 训练 loss 权重缩放（0 → 只用 VP-SDE loss）
        w2_loss_weight: float = 1.0,
    ):
        """
        参数:
            num_train_timesteps: 训练离散步数
            beta_start / beta_end: VP-SDE 噪声调度范围
            w2_path_type: W2-ODE 路径类型（推荐 "linear"，数学等价）
            w2_path_poly_order: poly 路径阶数（仅 path_type="poly" 时生效）
            hybrid_transition_step: 过渡中心点 T（归一化 [0,1]）
            hybrid_transition_width: 过渡区宽度 δ
            clip_sample: 推理时是否裁剪输出
            clip_sample_range: 裁剪范围
            w2_loss_weight: W2-ODE loss 的额外缩放因子
                （linear 路径下 L_W2 = (σ/α+1)²·L_SDE，
                  此参数用于控制两 loss 的相对贡献）
        """
        assert w2_path_type == "linear", (
            f"HybridSDE 仅支持 linear 路径（方案 A 数学等价要求），"
            f"得到 w2_path_type={w2_path_type}"
        )

        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.w2_loss_weight = w2_loss_weight
        self.transition_step = hybrid_transition_step
        self.transition_width = hybrid_transition_width

        # 内部子模块
        self.vp_sde = VPSDE(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            prediction_type="epsilon",
        )
        self.w2_ode = W2FlowODE(
            num_train_timesteps=num_train_timesteps,
            path_type=w2_path_type,
            path_poly_order=w2_path_poly_order,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            lyapunov_reg_weight=0.0,
            inference_noise_scale=0.0,
        )

        self.timesteps: Tensor | None = None
        self.num_inference_steps: int | None = None

    # ------------------------------------------------------------------
    # 权重调度
    # ------------------------------------------------------------------

    def get_weights(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """计算当前时间步的混合权重 (w_W2, w_SDE)。

        t 是归一化时间 [0, 1]，t=1 是纯噪声，t=0 是纯数据。
        推理时 t 从 1→0，所以：
          t > T + δ/2 → Phase 1（W2-ODE 主导）
          t < T - δ/2 → Phase 2（VP-SDE 主导）

        参数:
            t: 归一化时间，任意形状 tensor
        返回:
            (w_W2, w_SDE)，每个与 t 同形状
        """
        # 过渡区下界和上界
        lo = self.transition_step - self.transition_width / 2.0
        hi = self.transition_step + self.transition_width / 2.0

        # w_SDE 在 t < lo 时为 0，t > hi 时为 1，中间 smoothstep
        # 注意：推理 t 从 1→0，所以 w_SDE 在 t 小时更大
        x = (hi - t) / (hi - lo)  # t=hi → x=0, t=lo → x=1
        w_sde = _smoothstep(x)
        w_w2 = 1.0 - w_sde
        return w_w2, w_sde

    # ------------------------------------------------------------------
    # 前向过程（训练）
    # ------------------------------------------------------------------

    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """前向加噪：使用 VP-SDE 的加噪方式（更稳定）。

        与 SDEBase.add_noise 接口完全一致。
        """
        return self.vp_sde.add_noise(x0, noise, timesteps)

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int) -> None:
        """设置推理时间节点。"""
        self.num_inference_steps = num_inference_steps
        self.vp_sde.set_timesteps(num_inference_steps)
        self.w2_ode.set_timesteps(num_inference_steps)
        self.timesteps = self.vp_sde.timesteps

    def step(
        self,
        model_output: Tensor,
        t: Tensor | float,
        sample: Tensor,
        generator: torch.Generator | None = None,
    ) -> HybridSDEStepResult:
        """混合步进：从 ε_pred 同时计算 W2-ODE 和 VP-SDE 的 drift，加权组合。

        参数:
            model_output: ε_pred（UNet 输出的噪声估计）(B, ...)
            t: 当前归一化时间步（标量，范围 [0, 1]）
            sample: 当前状态 x_t (B, ...)
            generator: 随机数生成器
        返回:
            HybridSDEStepResult，包含 prev_sample
        """
        if not isinstance(t, Tensor):
            t_val = torch.tensor(t, dtype=torch.float32, device=sample.device)
        else:
            t_val = t.float().to(sample.device)

        dt = torch.tensor(-1.0 / self.num_inference_steps, device=sample.device)

        # ── 计算混合权重 ──────────────────────────────────────────────
        t_scalar = t_val.item() if t_val.dim() == 0 else t_val
        w_w2, w_sde = self.get_weights(t_val)
        while w_w2.dim() < sample.dim():
            w_w2 = w_w2.unsqueeze(-1)
            w_sde = w_sde.unsqueeze(-1)

        # ── VP-SDE 分量 ───────────────────────────────────────────────
        beta = self.vp_sde._beta_t(t_val)
        alpha = self.vp_sde._alpha(t_val)
        sigma = self.vp_sde._sigma(t_val)
        while beta.dim() < sample.dim():
            beta = beta.unsqueeze(-1)
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)

        # drift_SDE = (-½βx + β·ε_pred/σ) · dt
        drift_sde = (-0.5 * beta * sample + beta * model_output / sigma) * dt
        diff_sde = torch.sqrt(beta * torch.abs(dt))

        # ── W2-ODE 分量（从 ε_pred 推导 v_pred）───────────────────────
        # x₀_est = (x_t - σ·ε_pred) / α
        # v_pred = -x₀_est + ε_pred  （linear 路径 v* = -x₀ + ε）
        x0_est = (sample - sigma * model_output) / alpha.clamp(min=1e-5)
        v_pred = -x0_est + model_output
        drift_w2 = v_pred * dt
        diff_w2 = torch.zeros_like(diff_sde)  # ODE 无扩散

        # ── 组合 ──────────────────────────────────────────────────────
        f_combined = w_w2 * drift_w2 + w_sde * drift_sde
        diff_combined = torch.sqrt(w_w2 * diff_w2**2 + w_sde * diff_sde**2).clamp(
            min=1e-8
        )

        z = torch.randn(
            sample.shape,
            device=sample.device,
            dtype=sample.dtype,
            generator=generator,
        )
        prev_sample = sample + f_combined + diff_combined * z

        if self.clip_sample:
            prev_sample = prev_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        return HybridSDEStepResult(prev_sample=prev_sample)

    # ------------------------------------------------------------------
    # 训练 loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        pred_epsilon: Tensor,
        x0: Tensor,
        noise: Tensor,
        x_t: Tensor,
        timesteps: Tensor | None = None,
    ) -> Tensor:
        """混合训练 loss

        L = w_W2(t) · L_W2 + w_SDE(t) · L_SDE

        其中：
            L_SDE = MSE(ε_pred, ε)
            L_W2  = MSE(v_pred, v*)  （从 ε_pred 推导）

        对 linear 路径，L_W2 = (σ/α + 1)² · L_SDE，
        两者本质上是同一 loss 的缩放版本。

        参数:
            pred_epsilon: ε_pred，UNet 输出 (B, ...)
            x0:   干净数据 (B, ...)
            noise: 添加的高斯噪声 ε (B, ...)
            x_t:  当前加噪状态 (B, ...)
            timesteps: 整数时间步索引 (B,)
        返回:
            标量 loss
        """
        assert timesteps is not None, "HybridSDE.compute_loss 需要 timesteps"

        t = timesteps.float() / self.num_train_timesteps  # (B,)
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)

        # ── VP-SDE loss ───────────────────────────────────────────────
        loss_sde = F.mse_loss(pred_epsilon, noise.detach(), reduction="none")

        # ── W2-ODE loss（从 ε_pred 推导 v_pred）───────────────────────
        alpha = self.vp_sde._alpha(t)
        sigma = self.vp_sde._sigma(t)
        while alpha.dim() < x0.dim():
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)

        x0_est = (x_t - sigma * pred_epsilon) / alpha.clamp(min=1e-5)
        v_pred = -x0_est + pred_epsilon
        v_target = noise - x0  # linear 路径 v* = ε - x₀
        loss_w2 = F.mse_loss(v_pred, v_target.detach(), reduction="none")

        # ── 权重组合 ──────────────────────────────────────────────────
        w_w2, w_sde = self.get_weights(t)
        while w_w2.dim() < loss_w2.dim():
            w_w2 = w_w2.unsqueeze(-1)
            w_sde = w_sde.unsqueeze(-1)

        loss = w_sde * loss_sde + w_w2 * self.w2_loss_weight * loss_w2
        return loss.mean()

    # ------------------------------------------------------------------
    # 兼容 SDEBase 的占位方法
    # ------------------------------------------------------------------

    def _beta_t(self, t: Tensor) -> Tensor:
        """返回 VP-SDE 的 β(t)，供外部查询。"""
        return self.vp_sde._beta_t(t)

    def _alpha(self, t: Tensor) -> Tensor:
        return self.vp_sde._alpha(t)

    def _sigma(self, t: Tensor) -> Tensor:
        return self.vp_sde._sigma(t)
