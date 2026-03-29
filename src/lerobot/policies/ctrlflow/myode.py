"""基于 MMD 距离的确定性 ODE 流（CTRL-Flow 框架）

理论来源：CTRL-Flow 论文 §5.2.2
    选 MMD² 为 Lyapunov 泛函 L(p_t, p_data)，其 Lions 导数为
        δL/δρ (x) = 2 ĥ_t(x)
        ĥ_t(x) = E_{X~p_t}[k(x,X)] - E_{Y~p_data}[k(x,Y)]
    Natural Gradient Flow 给出速度场：
        v(x,t) = -G(p_t)^{-1} ∇_x ĥ_t(x)
    令 g=0（纯 ODE），稳定条件退化为 Prop.2 的漂移耗散条件：
        ∫ <v, ∇_x δL/δρ> p dx ≤ -2κ L(p_t)
    由 κ-测地凸性自动满足。

网络预测目标：
    U-Net 在时刻 t 学习速度场 v_θ(x_t, t)，
    使 x_t 沿 v_θ 方向移动后更接近 p_data 的样本。
    训练 target 通过"flow matching"构造：
        target = (x_data - x_noise) / 1   （线性插值路径的速度）
    其中 x_t = (1-t)*x_noise + t*x_data（常数速度路径，t∈[0,1]）。

与 modeling_ctrlflow.py 的接口对齐：
    - add_noise(x0, noise, timesteps)：构造 x_t（前向插值）
    - step(model_output, t, sample)：一步 ODE 推进（Euler 法，无随机项）
    - set_timesteps(N)：设置推理步数
    - prediction_type = "velocity"：U-Net 输出速度场 v_θ

注意：MMDODE 不继承 SDEBase（SDEBase 假设存在随机项），
      而是实现完全相同的公开接口，使 modeling_ctrlflow.py 无需修改。
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
# RBF 核工具函数
# ---------------------------------------------------------------------------

def _rbf_kernel(x: Tensor, y: Tensor, bandwidth: float) -> Tensor:
    """径向基函数核 k(x,y) = exp(-‖x-y‖²/(2h²))

    参数:
        x: (N, D)
        y: (M, D)
        bandwidth: 核带宽 h

    返回:
        K: (N, M)
    """
    # ‖x_i - y_j‖² = ‖x_i‖² + ‖y_j‖² - 2 x_i·y_j
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)   # (N, 1)
    y_sq = (y ** 2).sum(dim=-1, keepdim=True)   # (M, 1)
    cross = x @ y.t()                            # (N, M)
    dist_sq = x_sq + y_sq.t() - 2.0 * cross
    dist_sq = dist_sq.clamp(min=0.0)             # 数值保护
    return torch.exp(-dist_sq / (2.0 * bandwidth ** 2))


def _rbf_kernel_grad_x(x: Tensor, y: Tensor, bandwidth: float) -> Tensor:
    """∇_x k(x, y) = k(x,y) · (-(x-y)/h²)

    参数:
        x: (N, D)
        y: (M, D)
        bandwidth: 核带宽 h

    返回:
        grad: (N, M, D)，grad[i,j] = ∇_{x_i} k(x_i, y_j)
    """
    K = _rbf_kernel(x, y, bandwidth)            # (N, M)
    # x_i - y_j：(N, M, D)
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    # ∇_{x_i} k = -diff / h² * K[i,j]
    grad = -diff / (bandwidth ** 2) * K.unsqueeze(-1)
    return grad


# ---------------------------------------------------------------------------
# MMDODE
# ---------------------------------------------------------------------------

class MMDODE:
    """MMD Natural Gradient Flow ODE（CTRL-Flow §5.2.2）

    前向过程（线性插值路径）：
        x_t = (1 - t) · x_noise + t · x_data,   t ∈ [0, 1]
        速度场（常数）：v = x_data - x_noise

    逆向过程（ODE，Euler 积分，t: 1→0）：
        dx = v_θ(x_t, t) dt

    训练目标：
        网络预测速度场 v_θ(x_t, t)，target = x_data - x_noise
        主 loss：MSE(v_θ, target)
        MMD 正则项：惩罚当前分布与数据分布的 MMD² 偏差（可选）

    与 modeling_ctrlflow.py 的接口：
        - prediction_type 固定为 "velocity"，compute_loss 需特殊处理
        - step() 返回 ODEResult（.prev_sample 属性）
        - set_timesteps / add_noise / step 与 SDEBase 接口完全兼容
    """

    # 固定预测类型，供 modeling_ctrlflow.py 的 compute_loss 识别
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
            num_train_timesteps: 训练时的离散步数（用于归一化 t）
            bandwidth: RBF 核带宽 h；bandwidth_auto=True 时忽略此值，
                       改用中值启发式 h = median(‖x_i - x_j‖)
            clip_sample: 是否裁剪输出范围
            clip_sample_range: 裁剪范围 [-r, r]
            mmd_reg_weight: MMD 正则项权重（训练时使用）
            bandwidth_auto: 是否自动估计带宽（推荐 True）
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
    # 带宽估计
    # ------------------------------------------------------------------

    def _estimate_bandwidth(self, x: Tensor) -> float:
        """中值启发式带宽估计：h = median(‖x_i - x_j‖) / sqrt(2 log N)

        参数:
            x: (N, D)，将 (B, T, D) 展平后传入
        """
        with torch.no_grad():
            N = min(x.shape[0], 512)       # 避免 O(N²) 内存爆炸，最多采 512 个点
            idx = torch.randperm(x.shape[0], device=x.device)[:N]
            x_sub = x[idx]                 # (N, D)
            dist_sq = torch.cdist(x_sub, x_sub, p=2) ** 2   # (N, N)
            # 取上三角（不含对角线）的中值
            mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1)
            median_sq = dist_sq[mask].median()
            h = (median_sq / (2.0 * math.log(N + 1)) + 1e-8) ** 0.5
        return h.item()

    # ------------------------------------------------------------------
    # 前向过程：线性插值路径
    # ------------------------------------------------------------------

    def add_noise(self, x0: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """前向插值：x_t = (1-t)·noise + t·x0

        对应线性路径（常数速度场），t=0 时为纯噪声，t=1 时为数据。

        参数:
            x0: 干净数据 (B, T, D)
            noise: 标准高斯噪声 (B, T, D)
            timesteps: 整数索引 (B,)，范围 [0, num_train_timesteps)

        返回:
            x_t: (B, T, D)
        """
        t = timesteps.float() / self.num_train_timesteps   # (B,) → [0, 1)
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)                            # → (B, 1, 1)
        return (1.0 - t) * noise + t * x0

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int) -> None:
        """设置推理时的时间节点（t 从 1 降到 0）。"""
        self.num_inference_steps = num_inference_steps
        # 从 t=1（数据端）到 t=1/N（接近噪声端），逆序推进
        # 注意：推理方向与训练相反，t 从 1→0
        self.timesteps = torch.linspace(
            1.0, 1.0 / num_inference_steps, num_inference_steps
        )

    def step(
        self,
        model_output: Tensor,
        t: Tensor | float,
        sample: Tensor,
        generator: torch.Generator | None = None,  # 保留参数，ODE 不使用
    ) -> ODEResult:
        """一步 ODE 推进（Euler 法）：x_{t-dt} = x_t + v_θ(x_t, t) · dt

        dt 为负值（t 从 1→0），model_output 为速度场 v_θ。

        参数:
            model_output: U-Net 预测的速度场 v_θ (B, T, D)
            t: 当前时间步（标量，float 或 0-d Tensor）
            sample: 当前状态 x_t (B, T, D)
            generator: 忽略（ODE 无随机项）

        返回:
            ODEResult，包含 prev_sample = x_{t+dt}（dt < 0）
        """
        # dt < 0：从 t 往 0 方向走
        dt = -1.0 / self.num_inference_steps

        # Euler step：x_{t+dt} = x_t + v_θ · dt
        prev_sample = sample + model_output * dt

        if self.clip_sample:
            prev_sample = prev_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        return ODEResult(prev_sample=prev_sample)

    # ------------------------------------------------------------------
    # 训练 loss（供 compute_loss 直接调用）
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        pred_velocity: Tensor,
        x0: Tensor,
        noise: Tensor,
        x_t: Tensor,
    ) -> Tensor:
        """MMD-ODE 训练 loss

        主 loss：速度场 MSE
            L_flow = E[‖v_θ(x_t, t) - (x0 - noise)‖²]

        MMD 正则项（CTRL-Flow §5.2.2）：
            L_mmd = MMD²(p_t_pred, p_data)
            其中 p_t_pred 由 x_t + v_θ·dt 的分布近似，
            p_data 由 x0 近似。
            这一项惩罚预测速度场推进后的分布与真实数据分布的偏差，
            直接对应论文中 MMD 作为 Lyapunov 泛函的耗散条件。

        参数:
            pred_velocity: v_θ(x_t, t)，U-Net 输出 (B, T, D)
            x0: 干净数据（p_data 的样本）(B, T, D)
            noise: 初始噪声（p_0 的样本）(B, T, D)
            x_t: 当前插值状态 (B, T, D)

        返回:
            标量 loss
        """
        # 速度场目标：常数速度路径下 target = x0 - noise
        target_velocity = x0 - noise                          # (B, T, D)
        loss_flow = F.mse_loss(pred_velocity, target_velocity)

        if self.mmd_reg_weight <= 0.0:
            return loss_flow

        # MMD 正则：将 (B, T, D) 展平为 (B*T, D) 后计算
        B, T, D = x0.shape
        x0_flat = x0.reshape(B * T, D).detach()              # p_data 样本

        # 用 v_θ 推进一小步后的分布作为 p_t 的近似
        dt_approx = 1.0 / self.num_train_timesteps
        x_next_flat = (x_t + pred_velocity * dt_approx).reshape(B * T, D)

        # 自动带宽估计
        if self.bandwidth_auto:
            h = self._estimate_bandwidth(
                torch.cat([x_next_flat.detach(), x0_flat], dim=0)
            )
        else:
            h = self.bandwidth

        loss_mmd = self._mmd_squared(x_next_flat, x0_flat, h)

        return loss_flow + self.mmd_reg_weight * loss_mmd

    def _mmd_squared(self, x: Tensor, y: Tensor, bandwidth: float) -> Tensor:
        """MMD²(p, q) 的无偏 U-统计量估计

        MMD²(p,q) = E_{x,x'~p}[k(x,x')] - 2E_{x~p,y~q}[k(x,y)]
                  + E_{y,y'~q}[k(y,y')]

        参数:
            x: p 的样本 (N, D)
            y: q 的样本 (M, D)
            bandwidth: RBF 核带宽

        返回:
            标量 MMD²（可为负，是无偏估计的正常现象）
        """
        K_xx = _rbf_kernel(x, x, bandwidth)   # (N, N)
        K_yy = _rbf_kernel(y, y, bandwidth)   # (M, M)
        K_xy = _rbf_kernel(x, y, bandwidth)   # (N, M)

        N, M = x.shape[0], y.shape[0]

        # 无偏估计：去掉对角线（自身与自身的核值）
        # E[k(x,x')] ≈ (1/(N(N-1))) Σ_{i≠j} k(x_i, x_j)
        K_xx_sum = K_xx.sum() - K_xx.trace()
        K_yy_sum = K_yy.sum() - K_yy.trace()

        term_xx = K_xx_sum / (N * (N - 1) + 1e-8)
        term_yy = K_yy_sum / (M * (M - 1) + 1e-8)
        term_xy = K_xy.mean()

        return term_xx - 2.0 * term_xy + term_yy

    # ------------------------------------------------------------------
    # 兼容 SDEBase 的占位方法（modeling_ctrlflow.py 不会调用，但避免 AttributeError）
    # ------------------------------------------------------------------

    def _alpha(self, t: Tensor) -> Tensor:
        """线性插值路径的均值系数 α(t) = t"""
        return t

    def _sigma(self, t: Tensor) -> Tensor:
        """线性插值路径的噪声系数 σ(t) = 1 - t"""
        return 1.0 - t


# ---------------------------------------------------------------------------
# 内部 config 容器（让 modeling_ctrlflow.py 的 self.sde.config.num_train_timesteps 可用）
# ---------------------------------------------------------------------------

class _ODEConfig:
    def __init__(self, num_train_timesteps: int):
        self.num_train_timesteps = num_train_timesteps
