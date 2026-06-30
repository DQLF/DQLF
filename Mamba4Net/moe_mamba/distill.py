from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import MoEMambaLayer
from .mamba import MambaBlock


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      temperature: float = 2.0) -> torch.Tensor:
    """Cross-architecture KL distillation (paper eq. 6):
        L_kd = tau^2 * KL( softmax(z_s / tau) || softmax(z_t / tau) )
    """
    tau = temperature
    s = F.log_softmax(student_logits / tau, dim=-1)
    t = F.softmax(teacher_logits / tau, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (tau * tau)


def _svd_lowrank(W: torch.Tensor, rank: int) -> torch.Tensor:
    """Return a rank-truncated reconstruction U S V^T of W."""
    if W.dim() != 2:
        W = W.reshape(W.size(0), -1)
    rank = min(rank, min(W.shape))
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    return (U[:, :rank] * S[:rank]) @ Vh[:rank]


@torch.no_grad()
def cwr_seed_experts(student: nn.Module,
                     teacher_attn_weights: Iterable[torch.Tensor],
                     teacher_ffn_weights: Iterable[torch.Tensor],
                     rank: int = 16) -> int:
    """Cross-Architectural Weight Reuse (paper Sec. VI).

    Take the Transformer teacher's attention/FFN matrices, low-rank-SVD them, and project
    the resulting bases into each Mamba expert's input projection. Each expert gets a
    distinct slice -> "preliminary semantic differentiation" instead of cold-start.

    Returns the number of experts that were seeded.
    """
    attn_list = [w.detach() for w in teacher_attn_weights]
    ffn_list = [w.detach() for w in teacher_ffn_weights]
    if not attn_list and not ffn_list:
        return 0

    seeded = 0
    expert_id = 0
    for module in student.modules():
        if not isinstance(module, MoEMambaLayer):
            continue
        for expert in module.experts:
            assert isinstance(expert, MambaBlock)
            # alternate attention / FFN sources across experts
            pool = attn_list if expert_id % 2 == 0 and attn_list else (ffn_list or attn_list)
            src = pool[expert_id % len(pool)]
            low = _svd_lowrank(src, rank)

            target = expert.in_proj.weight  # (2 * d_inner, d_model)
            tgt_rows, tgt_cols = target.shape
            src_rows, src_cols = low.shape
            r = min(tgt_rows, src_rows)
            c = min(tgt_cols, src_cols)
            low = low.to(device=target.device, dtype=target.dtype)
            scale = target.std() / (low[:r, :c].std() + 1e-6)
            target[:r, :c] = low[:r, :c] * scale

            expert_id += 1
            seeded += 1

    return seeded
