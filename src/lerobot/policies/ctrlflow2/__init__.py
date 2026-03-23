"""
CTRL-Flow2: Lyapunov-Stable Regularized Diffusion Policy for Robot Learning.

Based on "CTRL-Flow: Generative Modeling via Lyapunov-Stable Regularized
Probability Flows on Wasserstein Space".

Quick start
-----------
Replace `diffusion` with `ctrlflow2` in your lerobot config:

    policy:
      name: ctrlflow2
      lyapunov_functional: kl          # "kl" | "chi2" | "hybrid"
      lyapunov_lambda: 0.01            # stability regularizer weight
      use_adaptive_beta: true          # adaptive noise schedule
      use_state_dependent_diffusion: false

To reproduce vanilla DiffusionPolicy behaviour set lyapunov_lambda=0,
use_adaptive_beta=false, use_state_dependent_diffusion=false.
"""

from lerobot.policies.ctrlflow2.configuration_ctrlflow2 import CtrlFlow2Config
from lerobot.policies.ctrlflow2.modeling_ctrlflow2 import CtrlFlow2Policy

__all__ = ["CtrlFlow2Config", "CtrlFlow2Policy"]
