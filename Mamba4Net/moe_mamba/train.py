"""Multi-task training skeleton for MoE-Mamba.

Each task uses synthetic data so the loop runs end-to-end without real environments.
Plug your VP dataset / Mahimahi-ABR / Spark-CJS simulators into the marked sections.

Loss design:
    L = L_task + lambda_kd * L_kd + lambda_lb * L_load_balance       (paper eq. 6)
"""
from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from .config import MoEMambaConfig
from .model import MoEMamba
from .heads import ABRHead
from .distill import distillation_loss


def _synth_vp_batch(cfg: MoEMambaConfig, B: int = 4, device="cpu"):
    x = torch.randn(B, cfg.vp_in_channels, cfg.vp_image_size, cfg.vp_image_size, device=device)
    y = torch.randint(0, cfg.vp_num_classes, (B,), device=device)
    return x, y


def _synth_abr_batch(cfg: MoEMambaConfig, B: int = 4, device="cpu"):
    x = torch.randn(B, cfg.abr_seq_len, cfg.abr_in_features, device=device)
    advantages = torch.randn(B, device=device)
    actions = torch.randint(0, cfg.abr_num_actions, (B,), device=device)
    returns = torch.randn(B, device=device)
    return x, actions, advantages, returns


def _synth_cjs_batch(cfg: MoEMambaConfig, B: int = 4, device="cpu"):
    N = cfg.cjs_max_nodes
    nf = torch.randn(B, N, cfg.cjs_node_features, device=device)
    adj = (torch.rand(B, N, N, device=device) < 0.05).float()
    adj = torch.tril(adj, diagonal=-1)
    target = torch.rand(B, N, device=device)
    urgency = torch.rand(B, N, 1, device=device)
    return nf, adj, target, urgency


def step_vp(model, teacher, x, y, cfg, lambda_kd=0.5, lambda_lb=0.01):
    out = model("vp", x)
    loss_task = F.cross_entropy(out["logits"], y)
    loss = loss_task + lambda_lb * out["aux"]
    if teacher is not None:
        with torch.no_grad():
            t_logits = teacher(x)
        loss = loss + lambda_kd * distillation_loss(out["logits"], t_logits, cfg.distill_temperature)
    return loss, {"task": loss_task.item()}


def step_abr(model, x, actions, adv, returns, cfg, lambda_lb=0.01):
    out = model("abr", x)
    log_p = F.log_softmax(out["policy"], dim=-1)
    chosen = log_p.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    pg = -(chosen * adv.detach()).mean()
    vf = F.mse_loss(out["value"], returns)
    ent = -(log_p.exp() * log_p).sum(dim=-1).mean()
    loss = pg + 0.5 * vf - 0.01 * ent + lambda_lb * out["aux"]
    return loss, {"pg": pg.item(), "vf": vf.item()}


def step_cjs(model, nf, adj, target, urgency, cfg, lambda_lb=0.01):
    out = model("cjs", nf, adj, urgency=urgency)
    loss_task = F.binary_cross_entropy(out["scores"], target)
    loss = loss_task + lambda_lb * out["aux"]
    return loss, {"task": loss_task.item()}


def train(steps: int = 100, batch: int = 2, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MoEMambaConfig()
    model = MoEMamba(cfg).to(device)
    opt = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    for it in range(steps):
        task = ["vp", "abr", "cjs"][it % 3]
        if task == "vp":
            x, y = _synth_vp_batch(cfg, batch, device)
            loss, info = step_vp(model, None, x, y, cfg)
        elif task == "abr":
            x, a, adv, ret = _synth_abr_batch(cfg, batch, device)
            loss, info = step_abr(model, x, a, adv, ret, cfg)
        else:
            nf, adj, tgt, u = _synth_cjs_batch(cfg, batch, device)
            loss, info = step_cjs(model, nf, adj, tgt, u, cfg)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 10 == 0:
            print(f"step {it:4d}  task={task}  loss={loss.item():.4f}  {info}")

    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    train(args.steps, args.batch, args.device)
