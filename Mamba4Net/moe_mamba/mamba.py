"""Mamba block for MoE-Mamba.

Backend selection (set env MOE_MAMBA_BACKEND=pytorch to force fallback):
    1. mamba-ssm (official CUDA kernel)            -- fast, requires CUDA + Linux build
    2. pure-PyTorch sequential scan (this module) -- correct everywhere; slow on long L

Both expose the same `MambaBlock(d_model, d_state, d_conv, expand)` constructor and
`forward(x: (B, L, D)) -> (B, L, D)`, so the rest of the codebase is backend-agnostic.
"""
from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


_BACKEND_PREF = os.environ.get("MOE_MAMBA_BACKEND", "auto").lower()


def _try_import_mamba_ssm():
    if _BACKEND_PREF == "pytorch":
        return None
    try:
        from mamba_ssm import Mamba as _MambaCUDA
        return _MambaCUDA
    except Exception:
        return None


_MambaCUDA = _try_import_mamba_ssm()
HAS_MAMBA_SSM = _MambaCUDA is not None


class _SelectiveSSMPyTorch(nn.Module):
    """Selective S6 core in pure PyTorch: h_t = A_bar h_{t-1} + B_bar x_t ; y_t = C h_t.
    (paper eq. 3, 12). Reference fallback when mamba-ssm is unavailable.
    """

    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int | str = "auto"):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        if dt_rank == "auto":
            dt_rank = max(1, math.ceil(d_inner / 16))
        self.dt_rank = dt_rank

        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        nn.init.uniform_(self.dt_proj.weight, -1.0 / math.sqrt(dt_rank), 1.0 / math.sqrt(dt_rank))
        with torch.no_grad():
            dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = self.d_state

        x_dbl = self.x_proj(x)
        delta, Bm, Cm = torch.split(x_dbl, [self.dt_rank, N, N], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        A = -torch.exp(self.A_log.float())
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        B_bar = delta.unsqueeze(-1) * Bm.unsqueeze(2)

        h = x.new_zeros(B, D, N)
        ys = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            y = (h * Cm[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + self.D * x
        return y


class _MambaBlockPyTorch(nn.Module):
    """Mamba block: in_proj -> [conv1d + SiLU -> S6] x SiLU(z) -> out_proj. Pure PyTorch."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        self.ssm = _SelectiveSSMPyTorch(self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_part, z_part = xz.chunk(2, dim=-1)
        x_part = self.conv1d(x_part.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_part = self.act(x_part)
        y = self.ssm(x_part)
        y = y * self.act(z_part)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Backend dispatcher. Constructor and forward signature stable across backends.

    `in_proj.weight` is exposed as `self.in_proj.weight` for CWR seeding regardless of backend.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 backend: str | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        backend = (backend or _BACKEND_PREF).lower()
        if backend == "auto":
            backend = "cuda" if HAS_MAMBA_SSM else "pytorch"
        if backend == "cuda" and not HAS_MAMBA_SSM:
            backend = "pytorch"

        self.backend = backend
        if backend == "cuda":
            self.impl = _MambaCUDA(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.in_proj = self.impl.in_proj
        else:
            self.impl = _MambaBlockPyTorch(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.in_proj = self.impl.in_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)
