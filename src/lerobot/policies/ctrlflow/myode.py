"""纯 MMD 自然梯度流 ODE（CTRL-Flow 框架 §5.2.2）

===========================================================================
【时间方向约定——与 modeling_ctrlflow.py 对齐】
===========================================================================

modeling_ctrlflow.py 的 conditional_sample() 逻辑：
  - 起点 sample = randn(...)              ← 纯噪声
  - timesteps = linspace(1.0, 1/N, N)    ← t 从大到小（1 → 0）
  - 每步 step(model_output, t, sample)   ← t 大时是噪声态，t 小时是数据态

因此 modeling_ctrlflow.py 隐含的时间约定是：
  t=1 → 噪声（x_noise），t≈0 → 数据（x_data）

本文件的所有实现必须与此对齐：

  前向过程 add_noise：
    x_t = (1-t)·x_data + t·noise       ← t=1 时为噪声，t=0 时为数据 ✓

  速度场方向（v 应把 x_t 从噪声推向数据，即 t 减小方向）：
    v*(x, t) = -∇_x ĥ_t(x)
    其中 ĥ_t(x) = E_{X~p_t}[k(x,X)] - E_{Y~p_data}[k(x,Y)]

    当 x_t 接近噪声（t≈1）时，ĥ_t(x) > 0（p_t 偏向噪声，远离 p_data），
    ∇_x ĥ_t 指向"更像 p_t"的方向，取负则指向"更像 p_data"的方向 ✓

  推理步进 step（dt < 0，t 从 1 减小到 0）：
    x_{t+dt} = x_t + v_θ(x_t, t) · dt
    dt = -1/N < 0，sample 从噪声逐步变为数据 ✓

===========================================================================
【target 量级归一化——解决训练信号爆炸问题】
===========================================================================

∇_x ĥ_t(x_i) ≈ (1/N) Σ_j k(x_i,x_j) · (x_j - x_i) / h²

当带宽 h 较小时（如 h=0.1），梯度量级 ∝ 1/h²=100，远超
Flow Matching target（x_data - x_noise，量级 O(1)），导致 loss 爆炸。

修复方案：对 target 做 L2 归一化后乘以参考量级：
    target_norm = target / (‖target‖ + ε)   ← 单位方向
    target_scaled = target_norm · ref_scale  ← 恢复合理量级

ref_scale 默认取 1.0（与 Flow Matching 的 target 量级对齐），
可通过 target_scale 参数调整。

===========================================================================
【与 modeling_ctrlflow.py 的接口】
===========================================================================
  完全兼容，无需修改上层代码：
  - add_noise(x0, noise, timesteps) → x_t = (1-t)·x0 + t·noise（t=1 为噪声）
  - set_timesteps(N)               → timesteps = linspace(1, 1/N, N)（从大到小）
  - step(model_output, t, sample)  → Euler 步，dt < 0，t 从 1→0
  - compute_loss(pred, x0, noise, x_t) → MMD 梯度监督 + 可选 Lyapunov 正则
  - prediction_type = "velocity"   → isinstance(self.sde, MMDODE) 分支识别
  - self.config.num_train_timesteps → 属性兼容
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 返回结果容器（与 SDEBase.SDEResult 接口相同）
# ---------------------------------------------------------------------------

@dataclass
class ODEResult:
    """ODE 步进结果，与 SDEResult 接口兼容。"""
    prev_sample: Tensor


# ---------------------------------------------------------------------------
# 核函数（可替换）
# ---------------------------------------------------------------------------

def _kernel(x: Tensor, y: Tensor, bandwidth: float) -> Tensor:
    """RBF 核 k(x,y) = exp(-‖x-y‖² / (2h²))

    只需修改此函数即可切换核类型（Matérn、多项式等），
    梯度计算 _witness_grad 依赖 autograd，无需改动。

    参数:
        x: (N, D)
        y: (M, D)
        bandwidth: 核带宽 h
    返回:
        K: (N, M)
    """
    x_sq  = (x ** 2).sum(dim=-1, keepdim=True)   # (N, 1)
    y_sq  = (y ** 2).sum(dim=-1, keepdim=True)   # (M, 1)
    cross = x @ y.t()                             # (N, M)
    dist_sq = (x_sq + y_sq.t() - 2.0 * cross).clamp(min=0.0)
    return torch.exp(-dist_sq / (2.0 * bandwidth ** 2))


# ---------------------------------------------------------------------------
# Witness function 及其梯度（autograd 方案）
# ---------------------------------------------------------------------------

def _witness_function(
    x: Tensor,
    x_t: Tensor,
    x_data: Tensor,
    bandwidth: float,
) -> Tensor:
    """估计 MMD witness function ĥ_t(x)

    ĥ_t(x) = E_{X~p_t}[k(x,X)] - E_{Y~p_data}[k(x,Y)]
            ≈ (1/N) Σ_j k(x, x_j^{(t)}) - (1/M) Σ_j k(x, y_j)

    参数:
        x:      查询点 (N, D)，需 requires_grad=True
        x_t:    p_t 的样本 (N, D)，在函数内部 detach
        x_data: p_data 的样本 (M, D)，在函数内部 detach
        bandwidth: 核带宽
    返回:
        h_hat: (N,)
    """
    term_pt    = _kernel(x, x_t.detach(),    bandwidth).mean(dim=1)  # (N,)
    term_pdata = _kernel(x, x_data.detach(), bandwidth).mean(dim=1)  # (N,)
    return term_pt - term_pdata


def _witness_grad(
    x_t_flat: Tensor,
    x0_flat: Tensor,
    bandwidth: float,
) -> Tensor:
    """用 autograd 计算 ∇_{x_i} ĥ_t(x_i)

    x_query 是 x_t_flat 的独立副本，与 pred_velocity 的梯度图隔离，
    确保 target 不会对 U-Net 产生非预期的反向传播。

    参数:
        x_t_flat: (N, D)，当前 p_t 样本
        x0_flat:  (M, D)，目标 p_data 样本
        bandwidth: 核带宽
    返回:
        grad: (N, D)，∇_{x_i} ĥ_t(x_i)
    """
    x_query = x_t_flat.detach().requires_grad_(True)
    h_hat = _witness_function(x_query, x_t_flat, x0_flat, bandwidth)
    grad, = torch.autograd.grad(
        outputs=h_hat.sum(),
        inputs=x_query,
        create_graph=False,
    )
    return grad  # (N, D)


# ---------------------------------------------------------------------------
# 带宽估计
# ---------------------------------------------------------------------------

def _estimate_bandwidth(x: Tensor) -> float:
    """中值启发式带宽：h = sqrt(median_sq / (2 log N))"""
    with torch.no_grad():
        N = min(x.shape[0], 512)
        idx = torch.randperm(x.shape[0], device=x.device)[:N]
        x_sub = x[idx]
        dist_sq = torch.cdist(x_sub, x_sub, p=2) ** 2
        mask = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1
        )
        median_sq = dist_sq[mask].median()
        h = (median_sq / (2.0 * math.log(N + 1)) + 1e-8) ** 0.5
    return max(h.item(), 1e-3)   # 防止 h 过小导致梯度爆炸


# ---------------------------------------------------------------------------
# 纯 MMD 自然梯度流 ODE
# ---------------------------------------------------------------------------

class MMDODE:
    """MMD 自然梯度流 ODE（CTRL-Flow §5.2.2）

    时间约定（与 modeling_ctrlflow.py 对齐）：
      t=1 → 纯噪声，t=0 → 干净数据
      推理时 t 从 1 减小到 0（dt < 0），sample 从噪声变为数据
    """

    prediction_type: str = "velocity"

    def __init__(
        self,
        num_train_timesteps: int = 100,
        bandwidth: float = 1.0,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        mmd_reg_weight: float = 0.0,
        bandwidth_auto: bool = True,
        target_scale: float = 1.0,
    ):
        """
        参数:
            num_train_timesteps: 训练离散步数
            bandwidth: 固定核带宽（bandwidth_auto=True 时忽略）
            clip_sample: 推理时是否裁剪输出
            clip_sample_range: 裁剪范围 [-r, r]
            mmd_reg_weight: Lyapunov 正则项权重（0 → 纯梯度监督）
            bandwidth_auto: True → 中值启发式自动估计带宽
            target_scale: target 归一化后的参考量级（默认 1.0，与 FM 对齐）
        """
        self.num_train_timesteps = num_train_timesteps
        self.bandwidth = bandwidth
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.mmd_reg_weight = mmd_reg_weight
        self.bandwidth_auto = bandwidth_auto
        self.target_scale = target_scale

        self.timesteps: Tensor | None = None
        self.num_inference_steps: int | None = None

        self.config = _ODEConfig(num_train_timesteps=num_train_timesteps)

    # ------------------------------------------------------------------
    # 前向过程：t=1 为噪声，t=0 为数据
    # ------------------------------------------------------------------

    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """前向插值：x_t = (1-t)·x0 + t·noise

        t=1 时 x_t = noise（纯噪声），t=0 时 x_t = x0（干净数据）。
        与 modeling_ctrlflow.py 的 conditional_sample 起点（randn）和
        timesteps 方向（1→0）保持一致。

        参数:
            x0: 干净数据 (B, T, D)
            noise: 标准高斯噪声 (B, T, D)
            timesteps: 整数索引 (B,)，范围 [0, num_train_timesteps)
        返回:
            x_t: (B, T, D)
        """
        t = timesteps.float() / self.num_train_timesteps   # → [0, 1)
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)                            # → (B, 1, 1)
        return (1.0 - t) * x0 + t * noise

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int) -> None:
        """设置推理时间节点。

        与 modeling_ctrlflow.py 的 conditional_sample 一致：
        timesteps 从 1.0 降到 1/N（t 大→小，对应噪声→数据）。
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
        """一步 Euler ODE 推进：x_{t+dt} = x_t + v_θ · dt

        dt < 0（t 从 1→0），v_θ 方向指向 p_data，
        因此 sample 每步向数据方向靠近。

        参数:
            model_output: v_θ(x_t, t) (B, T, D)
            t: 当前时间步（标量）
            sample: 当前状态 x_t (B, T, D)
            generator: 忽略（纯 ODE）
        返回:
            ODEResult，包含 prev_sample
        """
        dt = -1.0 / self.num_inference_steps   # dt < 0

        prev_sample = sample + model_output * dt

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
    ) -> Tensor:
        """纯 MMD 自然梯度流训练 loss

        主 loss：
            target = normalize(-∇_x ĥ_t(x_t)) * target_scale
            L_main = MSE(v_θ, target)

            target 做 L2 归一化防止量级爆炸（当带宽 h 小时 ∇ĥ 量级 ∝ 1/h²）。

        速度场方向验证：
            ĥ_t(x) = E_pt[k] - E_pdata[k]
            当 x_t 远离 x0（t≈1）时，ĥ_t > 0，∇ĥ 指向"更像 p_t"的方向，
            -∇ĥ 指向"更像 p_data"的方向，即推进方向正确 ✓

        辅助 loss（可选）：
            L_lyap = MMD²(x_t + v_θ·dt, x0)   （§7.1 Lyapunov 正则）

        参数:
            pred_velocity: v_θ(x_t, t)，U-Net 输出 (B, T, D)
            x0:   干净数据 (B, T, D)
            noise: 初始噪声（保留接口兼容性，本版本不直接用于构造 target）
            x_t:  当前插值状态 (B, T, D)
        返回:
            标量 loss
        """
        B, T, D = x0.shape

        # ── 带宽估计 ──────────────────────────────────────────────────
        if self.bandwidth_auto:
            combined = torch.cat(
                [x_t.reshape(B * T, D).detach(),
                 x0.reshape(B * T, D).detach()],
                dim=0,
            )
            h = _estimate_bandwidth(combined)
        else:
            h = max(self.bandwidth, 1e-3)

        # ── witness function 梯度 ─────────────────────────────────────
        xt_flat = x_t.reshape(B * T, D)
        x0_flat = x0.reshape(B * T, D)

        grad_h = _witness_grad(xt_flat, x0_flat, h)   # (N, D)，∇_x ĥ_t

        # ── 量级归一化：防止 h 小时梯度爆炸 ──────────────────────────
        # target 方向 = -∇ĥ（推向 p_data），量级归一化后乘 target_scale
        raw_target = -grad_h   # (N, D)
        norms = raw_target.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (N, 1)
        target = (raw_target / norms * self.target_scale).reshape(B, T, D)

        loss_main = F.mse_loss(pred_velocity, target.detach())

        if self.mmd_reg_weight <= 0.0:
            return loss_main

        # ── 辅助 Lyapunov 正则（§7.1）────────────────────────────────
        dt_approx = 1.0 / self.num_train_timesteps
        x_next_flat = (x_t + pred_velocity * dt_approx).reshape(B * T, D)
        loss_lyap = self._mmd_squared(x_next_flat, x0_flat.detach(), h)

        return loss_main + self.mmd_reg_weight * loss_lyap

    def _mmd_squared(self, x: Tensor, y: Tensor, bandwidth: float) -> Tensor:
        """MMD²(p, q) 无偏 U-统计量估计（用于 Lyapunov 正则项）"""
        K_xx = _kernel(x, x, bandwidth)
        K_yy = _kernel(y, y, bandwidth)
        K_xy = _kernel(x, y, bandwidth)
        N, M = x.shape[0], y.shape[0]
        term_xx = (K_xx.sum() - K_xx.trace()) / (N * (N - 1) + 1e-8)
        term_yy = (K_yy.sum() - K_yy.trace()) / (M * (M - 1) + 1e-8)
        term_xy = K_xy.mean()
        return term_xx - 2.0 * term_xy + term_yy

    # ------------------------------------------------------------------
    # 兼容 SDEBase 的占位方法
    # ------------------------------------------------------------------

    def _alpha(self, t: Tensor) -> Tensor:
        """均值系数 α(t) = 1-t（t=1 时为噪声）"""
        return 1.0 - t

    def _sigma(self, t: Tensor) -> Tensor:
        """噪声系数 σ(t) = t"""
        return t


# ---------------------------------------------------------------------------
# 内部 config 容器
# ---------------------------------------------------------------------------

class _ODEConfig:
    def __init__(self, num_train_timesteps: int):
        self.num_train_timesteps = num_train_timesteps
