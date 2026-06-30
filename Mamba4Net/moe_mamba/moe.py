import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MambaBlock


class HistoryAwareRouter(nn.Module):
    """Routing features with temporal awareness (paper eq. 10):
        Z_t = sigma( W_h [E_t || POOL(H_{t-k:t-1}) ] )
    then noisy Top-K gating (paper eq. 4, 11):
        G(Z) = Softmax( TopK( W_g Z + Noise, k ) )
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2,
                 history_len: int = 4, noise_std: float = 1.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.history_len = history_len
        self.noise_std = noise_std

        self.fuse = nn.Linear(2 * d_model, d_model)
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.noise = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, E: torch.Tensor):
        B, L, D = E.shape
        k = min(self.history_len, L)
        cum = E.cumsum(dim=1)
        pad = torch.zeros(B, k, D, device=E.device, dtype=E.dtype)
        cum_pad = torch.cat([pad, cum], dim=1)
        window_sum = cum_pad[:, k:k + L] - cum_pad[:, :L]
        denom = torch.arange(1, L + 1, device=E.device).clamp(max=k).view(1, L, 1).to(E.dtype)
        H_pool = window_sum / denom

        Z = torch.tanh(self.fuse(torch.cat([E, H_pool], dim=-1)))

        clean_logits = self.gate(Z)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(clean_logits) * F.softplus(self.noise(Z)) * self.noise_std
            logits = clean_logits + noise
        else:
            logits = clean_logits

        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)
        gates = F.softmax(mask, dim=-1)

        return gates, topk_idx, clean_logits


class MoEMambaLayer(nn.Module):
    """Sparse Mamba MoE block with residual + LayerNorm fusion (paper eq. 5, 13):
        Y_out = LayerNorm( E + sum_{i in TopK} G(Z)_i * MM_i(E) )
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int,
                 history_len: int, d_state: int, d_conv: int, expand: int,
                 noise_std: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = HistoryAwareRouter(d_model, n_experts, top_k, history_len, noise_std)
        self.experts = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_experts)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, E: torch.Tensor):
        gates, topk_idx, clean_logits = self.router(E)
        B, L, D = E.shape

        out = torch.zeros_like(E)
        for e_id, expert in enumerate(self.experts):
            mask = (topk_idx == e_id).any(dim=-1)
            if not mask.any():
                continue
            y_e = expert(E)
            w = gates[..., e_id].unsqueeze(-1)
            out = out + w * y_e

        out = self.drop(out)
        Y = self.norm(E + out)
        return Y, self._load_balance_loss(clean_logits, topk_idx)

    @staticmethod
    def _load_balance_loss(clean_logits: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        # Switch-Transformer style auxiliary loss to discourage expert collapse.
        n_experts = clean_logits.size(-1)
        probs = F.softmax(clean_logits, dim=-1).mean(dim=(0, 1))
        one_hot = F.one_hot(topk_idx, num_classes=n_experts).float()
        usage = one_hot.sum(dim=2).clamp(max=1.0).mean(dim=(0, 1))
        return n_experts * (probs * usage).sum()
