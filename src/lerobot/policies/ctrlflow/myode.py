"""纯 MMD 自然梯度流 ODE（CTRL-Flow 框架 §5.2.2）

===========================================================================
理论来源：CTRL-Flow 论文 §5.2.2 + Proposition 2 + Table 1
===========================================================================

【核心设计：witness function + autograd，而非手推解析梯度】

  §5.2.2 的实现路径（论文隐含）：

    Step 1. 用 batch 内样本估计 witness function ĥ_t(x)（U-统计量）：
                ĥ_t(x) = (1/N) Σ_j k(x, x_j^{(t)}) - (1/M) Σ_j k(x, y_j)
            其中 x_j^{(t)} ~ p_t（batch 内 x_t），y_j ~ p_data（batch 内 x0）

    Step 2. 对 x_t 做 autograd，得到 ∇_x ĥ_t(x_t)：
                torch.autograd.grad(ĥ_t(x_t).sum(), x_t)

    Step 3. 自然梯度流速度场（欧氏近似 G=I）：
                v*(x_i, t) = -∇_{x_i} ĥ_t(x_i)

    Step 4. U-Net 学习 v_θ ≈ v*，loss = MSE(v_θ, v*)

  为什么用 autograd 而非手推解析梯度？

    ① 灵活性：换核函数（如 Matérn、多项式、混合核）只改 _kernel()，
              梯度计算无需改动，解析版则需重推每个核的 ∇_x k
    ② 一致性：论文§5.2.2 提到 "natural-gradient preconditioning"，
              未来扩展 G(p_t) ≠ I 时，autograd 可无缝支持
    ③ 可读性：_witness_function() 直接对应论文公式，代码即文档
    ④ 数值稳定：由 PyTorch 统一管理，避免手动 clamp 引入的不一致

  数学等价性（对 RBF 核）：
    autograd 结果与手推解析梯度完全等价：
        ∇_{x_i} k(x_i, y_j) = k(x_i,y_j) · (y_j - x_i) / h²
    但 autograd 方案无需硬编码此公式，更易维护和扩展。

【Lyapunov 稳定性保证（Proposition 2）】

  选 L = MMD²(p_t, p_data) 为 Lyapunov 泛函，
  令 v = -∇_x ĥ_t（即 -∇_{W2} L），则：
      d/dt L(p_t) = ∫ <v, ∇_x δL/δρ> p_t dx
                  = -∫ ‖∇_x ĥ_t‖² p_t dx
                  = -‖∇_{W2} L‖²_{L²(p_t)}
                  ≤ -2κ L(p_t)     （κ-测地凸性）
  → L(p_t) ≤ e^{-2κt} L(p_0)，指数收敛。

【与 modeling_ctrlflow.py 的接口】

  完全兼容，无需修改上层代码：
  - add_noise(x0, noise, timesteps) → 线性插值构造 x_t（前向过程）
  - step(model_output, t, sample)  → Euler ODE 推进（逆向过程）
  - set_timesteps(N)               → 推理步数设置
  - compute_loss(pred, x0, noise, x_t) → 主 loss + 可选 Lyapunov 正则
  - prediction_type = "velocity"   → modeling_ctrlflow.py 的 isinstance 分支识别
  - self.config.num_train_timesteps → 属性兼容

【与旧版 myode.py 的对比】

  ┌────────────────────┬──────────────────────┬────────────────────────────┐
  │                    │ 旧 myode.py          │ 本版本                     │
  ├────────────────────┼──────────────────────┼────────────────────────────┤
  │ 训练 target        │ x0 - noise (FM速度)  │ -∇_x ĥ_t(x_t) via autograd│
  │ ∇_x ĥ_t 计算      │ 不计算               │ witness fn + autograd      │
  │ 核函数扩展性       │ N/A                  │ 只改 _kernel() 即可        │
  │ MMD 作用           │ 可选正则项           │ Lyapunov 泛函，驱动 target │
  │ 论文对应           │ FM + MMD 混合        │ 纯 §5.2.2 自然梯度流       │
  └────────────────────┴──────────────────────┴────────────────────────────┘
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
    """RBF（高斯）核 k(x, y) = exp(-‖x-y‖² / (2h²))

    替换说明：若要换核函数（Matérn、多项式等），只修改此函数即可，
    梯度计算（_witness_grad）无需改动，因为它依赖 autograd。

    参数:
        x: (N, D)
        y: (M, D)
        bandwidth: 核带宽 h
    返回:
        K: (N, M)
    """
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)    # (N, 1)
    y_sq = (y ** 2).sum(dim=-1, keepdim=True)    # (M, 1)
    cross = x @ y.t()                             # (N, M)
    dist_sq = (x_sq + y_sq.t() - 2.0 * cross).clamp(min=0.0)
    return torch.exp(-dist_sq / (2.0 * bandwidth ** 2))


# ---------------------------------------------------------------------------
# Witness function 及其梯度（autograd 方案）
# ---------------------------------------------------------------------------

def _witness_function(x: Tensor, x_t: Tensor, x0: Tensor, bandwidth: float) -> Tensor:
    """估计 MMD witness function ĥ_t(x)（U-统计量）

    ĥ_t(x) = E_{X~p_t}[k(x, X)] - E_{Y~p_data}[k(x, Y)]
            ≈ (1/N) Σ_j k(x, x_j^{(t)}) - (1/M) Σ_j k(x, y_j)

    其中 x_j^{(t)} 用 batch 内 x_t 近似 p_t，
         y_j       用 batch 内 x0 近似 p_data。

    参数:
        x:    查询点 (N, D)，需 requires_grad=True 供 autograd 使用
        x_t:  p_t 的样本 (N, D)，detach（不参与梯度）
        x0:   p_data 的样本 (N, D)，detach（不参与梯度）
        bandwidth: 核带宽
    返回:
        h_hat: (N,) — 每个查询点 x_i 的 ĥ_t(x_i) 值
    """
    # p_t 项：(1/N) Σ_j k(x_i, x_j^{(t)})
    K_pt     = _kernel(x, x_t.detach(), bandwidth)    # (N, N)
    term_pt  = K_pt.mean(dim=1)                        # (N,)

    # p_data 项：(1/M) Σ_j k(x_i, y_j)
    K_pd       = _kernel(x, x0.detach(), bandwidth)   # (N, M)
    term_pdata = K_pd.mean(dim=1)                      # (N,)

    return term_pt - term_pdata                        # (N,)


def _witness_grad(
    x_t_flat: Tensor,
    x0_flat: Tensor,
    bandwidth: float,
) -> Tensor:
    """用 autograd 计算 ∇_{x_i} ĥ_t(x_i)（witness function 对 x_t 的梯度）

    实现步骤：
      1. 令 x_query（x_t_flat 的独立副本）参与梯度图
      2. 计算 ĥ_t(x_query) 并求和，得到标量
      3. autograd.grad 一次性得到所有 ∂ĥ/∂x_i

    关键细节：
      - x_query 是 x_t_flat.detach() 的副本，与 pred_velocity 的梯度图隔离，
        确保 target 不会对 U-Net 产生非预期的反向传播
      - create_graph=False：target 不需要二阶梯度，节省内存
      - x_t 和 x0 在 _witness_function 内部 detach，只有 x_query 保留梯度

    参数:
        x_t_flat: (N, D)，当前 p_t 样本（来自 x_t.reshape(B*T, D)）
        x0_flat:  (M, D)，目标 p_data 样本（来自 x0.reshape(B*T, D)）
        bandwidth: 核带宽
    返回:
        grad: (N, D)，∇_{x_i} ĥ_t(x_i)
    """
    # 独立的查询点，建立新计算图（与 pred_velocity 的图隔离）
    x_query = x_t_flat.detach().requires_grad_(True)

    # ĥ_t(x_query)：形状 (N,)
    h_hat = _witness_function(x_query, x_t_flat, x0_flat, bandwidth)

    # ∂(Σ_i ĥ_t(x_i)) / ∂x_i = ∇_{x_i} ĥ_t(x_i)
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
    """中值启发式带宽：h = sqrt(median(‖x_i - x_j‖²) / (2 log N))

    论文§5.2.2 提到 "U-statistics, yielding low variance (O(1/n))"，
    带宽采用中值启发式（核估计标准做法）。

    参数:
        x: (N, D)，建议先将 (B,T,D) 展平后传入
    """
    with torch.no_grad():
        N = min(x.shape[0], 512)
        idx = torch.randperm(x.shape[0], device=x.device)[:N]
        x_sub = x[idx]
        dist_sq = torch.cdist(x_sub, x_sub, p=2) ** 2   # (N, N)
        mask = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1
        )
        median_sq = dist_sq[mask].median()
        h = (median_sq / (2.0 * math.log(N + 1)) + 1e-8) ** 0.5
    return h.item()


# ---------------------------------------------------------------------------
# 纯 MMD 自然梯度流 ODE
# ---------------------------------------------------------------------------

class MMDODE:
    """MMD 自然梯度流 ODE（CTRL-Flow §5.2.2）

    Lyapunov 泛函：L = MMD²(p_t, p_data)
    速度场目标：v*(x,t) = -∇_x ĥ_t(x)，由 witness function + autograd 计算
    训练：U-Net 学习 v_θ ≈ v*，loss = MSE(v_θ, v*)

    与 modeling_ctrlflow.py 接口完全兼容，无需修改上层代码。
    """

    # 供 modeling_ctrlflow.py 的 isinstance(self.sde, MMDODE) 分支识别
    prediction_type: str = "velocity"

    def __init__(
        self,
        num_train_timesteps: int = 100,
        bandwidth: float = 1.0,
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        mmd_reg_weight: float = 0.1,
        bandwidth_auto: bool = True,
    ):
        """
        参数:
            num_train_timesteps: 训练离散步数（用于归一化 t）
            bandwidth: 固定核带宽 h（bandwidth_auto=True 时忽略，自动估计）
            clip_sample: 是否裁剪推理输出
            clip_sample_range: 裁剪范围 [-r, r]
            mmd_reg_weight: Lyapunov 正则项权重（§7.1 stability regularizer）
                            0 → 只用 MMD 梯度监督 loss
                            >0 → 额外加 MMD²(p_t_pred, p_data) 惩罚项
            bandwidth_auto: True → 中值启发式自动估计带宽（推荐）
        """
        self.num_train_timesteps = num_train_timesteps
        self.bandwidth = bandwidth
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.mmd_reg_weight = mmd_reg_weight
        self.bandwidth_auto = bandwidth_auto

        self.timesteps: Tensor | None = None
        self.num_inference_steps: int | None = None

        # 供 modeling_ctrlflow.py 读取（兼容 SDE 接口）
        self.config = _ODEConfig(num_train_timesteps=num_train_timesteps)

    # ------------------------------------------------------------------
    # 前向过程：线性插值路径
    # ------------------------------------------------------------------

    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """前向插值：x_t = (1-t)·noise + t·x0

        t=0 为纯噪声，t=1 为干净数据。线性路径确保 x_t 关于 t 连续可微。

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
        return (1.0 - t) * noise + t * x0

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int) -> None:
        """设置推理时间节点（t 从 1 降到 0，逆向生成）。"""
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
        """一步 Euler ODE 推进：x_{t+dt} = x_t + v_θ(x_t, t) · dt

        推理方向：t 从 1→0，dt < 0。

        参数:
            model_output: v_θ(x_t, t) (B, T, D)
            t: 当前时间步（标量）
            sample: 当前状态 x_t (B, T, D)
            generator: 忽略（纯 ODE，无随机项）
        返回:
            ODEResult，包含 prev_sample = x_{t+dt}
        """
        dt = -1.0 / self.num_inference_steps

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

        ┌─────────────────────────────────────────────────────────────────┐
        │ 主 loss：witness function + autograd 梯度监督                   │
        │                                                                 │
        │   target = -∇_{x_i} ĥ_t(x_i)   （witness fn 的负梯度）        │
        │   L_main = MSE(v_θ, target)                                    │
        │                                                                 │
        │   直接满足 Proposition 2 的耗散条件：                           │
        │   ∫ <v*, ∇_x δL/δρ> p_t dx = -‖∇_{W2}L‖²_{L²(p_t)} ≤ 0      │
        └─────────────────────────────────────────────────────────────────┘
        ┌─────────────────────────────────────────────────────────────────┐
        │ 辅助 loss：Lyapunov 正则（§7.1，可选）                         │
        │                                                                 │
        │   L_lyap = MMD²(x_t + v_θ·dt, x0)                             │
        │   mmd_reg_weight=0 时退化为纯梯度监督                          │
        └─────────────────────────────────────────────────────────────────┘

        参数:
            pred_velocity: v_θ(x_t, t)，U-Net 输出 (B, T, D)
            x0:   干净数据（p_data 样本）(B, T, D)
            noise: 初始噪声（保留以兼容接口，本版本不用于构造 target）
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
            h = self.bandwidth

        # ── 展平为 (N, D) 供核计算 ───────────────────────────────────
        xt_flat = x_t.reshape(B * T, D)   # ~ p_t
        x0_flat = x0.reshape(B * T, D)    # ~ p_data

        # ── 主 loss：witness function + autograd ──────────────────────
        # grad_h = ∇_{x_i} ĥ_t(x_i)，(N, D)
        # _witness_grad 内部对 x_query（xt_flat 的独立副本）求梯度，
        # 与 pred_velocity 的梯度图完全隔离。
        grad_h = _witness_grad(xt_flat, x0_flat, h)
        target = (-grad_h).reshape(B, T, D)   # v* = -∇ĥ_t，(B, T, D)

        loss_main = F.mse_loss(pred_velocity, target.detach())

        if self.mmd_reg_weight <= 0.0:
            return loss_main

        # ── 辅助 loss：Lyapunov 正则项（§7.1）────────────────────────
        dt_approx = 1.0 / self.num_train_timesteps
        x_next_flat = (x_t + pred_velocity * dt_approx).reshape(B * T, D)

        loss_lyap = self._mmd_squared(x_next_flat, x0_flat.detach(), h)

        return loss_main + self.mmd_reg_weight * loss_lyap

    def _mmd_squared(self, x: Tensor, y: Tensor, bandwidth: float) -> Tensor:
        """MMD²(p, q) 无偏 U-统计量估计（用于 Lyapunov 正则项）

        参数:
            x: p 的样本 (N, D)
            y: q 的样本 (M, D)
            bandwidth: RBF 核带宽
        返回:
            标量 MMD²（无偏估计，可为负）
        """
        K_xx = _kernel(x, x, bandwidth)
        K_yy = _kernel(y, y, bandwidth)
        K_xy = _kernel(x, y, bandwidth)

        N, M = x.shape[0], y.shape[0]

        K_xx_sum = K_xx.sum() - K_xx.trace()
        K_yy_sum = K_yy.sum() - K_yy.trace()

        term_xx = K_xx_sum / (N * (N - 1) + 1e-8)
        term_yy = K_yy_sum / (M * (M - 1) + 1e-8)
        term_xy = K_xy.mean()

        return term_xx - 2.0 * term_xy + term_yy

    # ------------------------------------------------------------------
    # 兼容 SDEBase 的占位方法（避免 AttributeError）
    # ------------------------------------------------------------------

    def _alpha(self, t: Tensor) -> Tensor:
        """线性插值路径的均值系数 α(t) = t"""
        return t

    def _sigma(self, t: Tensor) -> Tensor:
        """线性插值路径的噪声系数 σ(t) = 1 - t"""
        return 1.0 - t


# ---------------------------------------------------------------------------
# 内部 config 容器
# ---------------------------------------------------------------------------

class _ODEConfig:
    def __init__(self, num_train_timesteps: int):
        self.num_train_timesteps = num_train_timesteps
