import torch
import torch.nn as nn
import torch.nn.functional as F


class VPHead(nn.Module):
    """Spatial probability over the spherical viewing manifold (paper eq. 7).
        gamma_vp = softmax( phi_vp^T H_out )
    """

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.phi = nn.Linear(d_model, num_classes)

    def forward(self, H_out: torch.Tensor) -> torch.Tensor:
        h = H_out.mean(dim=1)
        return self.phi(h)


class ABRHead(nn.Module):
    """Bitrate selection logits for RL policy. Q_t (paper eq. 8) is computed in the env loop."""

    def __init__(self, d_model: int, num_actions: int):
        super().__init__()
        self.policy = nn.Linear(d_model, num_actions)
        self.value = nn.Linear(d_model, 1)

    def forward(self, H_out: torch.Tensor):
        h = H_out[:, -1]
        return self.policy(h), self.value(h).squeeze(-1)

    @staticmethod
    def qoe(R_t: torch.Tensor, R_prev: torch.Tensor, T_buf: torch.Tensor,
            alpha: float = 1.0, beta: float = 1.0) -> torch.Tensor:
        # Q_t = q(R_t) - alpha * |q(R_t) - q(R_{t-1})| - beta * T_buf  (paper eq. 8)
        q = torch.log1p(R_t)
        q_prev = torch.log1p(R_prev)
        return q - alpha * (q - q_prev).abs() - beta * T_buf


class CJSHead(nn.Module):
    """Per-node priority score for cluster job scheduling (paper eq. 9):
        p_i = sigmoid( W_cjs H_out * u_i )

    Returns both pre-sigmoid logits (for masked Categorical sampling in RL) and
    sigmoid scores (for paper-faithful priority interpretation).
    """

    def __init__(self, d_model: int, urgency_dim: int = 1):
        super().__init__()
        self.W_cjs = nn.Linear(d_model, urgency_dim)
        self.value = nn.Linear(d_model, 1)

    def forward(self, H_out: torch.Tensor, u: torch.Tensor | None = None):
        score = self.W_cjs(H_out)
        if u is not None:
            score = score * u
        logits = score.squeeze(-1)
        v = self.value(H_out.mean(dim=1)).squeeze(-1)
        return {"logits": logits, "scores": torch.sigmoid(logits), "value": v}
