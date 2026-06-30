"""Supervised training of MoE-Mamba on the Viewport Prediction task.

Run with real data (NetLLM cooked viewports):
    python train_vp.py --vp_root path/to/NetLLM/viewport_prediction/data/viewports \
                       --vp_dataset wu2017 --epochs 20

Run with synthetic fallback:
    python train_vp.py --synth --epochs 5

Loss: KL divergence against a Gaussian-on-the-sphere soft target (better than one-hot
because adjacent tiles share visual content). Set `--hard` to use cross-entropy on
the one-hot tile id instead. Optional KD vs a teacher checkpoint via `--teacher`.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from moe_mamba import MoEMamba, MoEMambaConfig, distillation_loss
from envs import (
    VPDataset, ViewportTrajectory,
    load_netllm_viewports, synth_dataset, great_circle_mae_rad,
)


def build_loaders(args):
    if args.synth or not args.vp_root:
        trajs = synth_dataset(n_users=args.synth_users, n_seconds=args.synth_seconds,
                              sample_hz=args.sample_hz, seed=args.seed)
    else:
        trajs = load_netllm_viewports(args.vp_root, dataset=args.vp_dataset, max_files=args.max_files)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(trajs)
    n_train = max(1, int(len(trajs) * 0.9))
    train_trajs = trajs[:n_train]
    val_trajs = trajs[n_train:] or trajs[-1:]

    train_ds = VPDataset(train_trajs, seq_len=args.seq_len, horizon=args.horizon,
                         tile_h=args.tile_h, tile_w=args.tile_w, stride=args.stride)
    val_ds = VPDataset(val_trajs, seq_len=args.seq_len, horizon=args.horizon,
                       tile_h=args.tile_h, tile_w=args.tile_w, stride=args.stride * 4)
    return train_ds, val_ds


def iter_batches(ds: VPDataset, batch_size: int, rng: np.random.Generator, shuffle=True):
    idx = np.arange(len(ds))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = idx[i:i + batch_size]
        batch = ds.collate([ds[int(j)] for j in chunk])
        yield batch


def step(model, batch, device, args, teacher_logits=None):
    x = torch.from_numpy(batch["x"]).to(device)               # (B, T, 6)
    out = model("vp", x)
    logits = out["logits"]                                    # (B, H*W)

    if args.hard:
        target = torch.from_numpy(batch["tile"]).to(device)
        task_loss = F.cross_entropy(logits, target)
    else:
        soft = torch.from_numpy(batch["soft"]).to(device)     # (B, H*W)
        task_loss = F.kl_div(F.log_softmax(logits, dim=-1), soft, reduction="batchmean")

    loss = task_loss + args.lb_coef * out["aux"]
    if teacher_logits is not None:
        loss = loss + args.kd_coef * distillation_loss(logits, teacher_logits, args.kd_temperature)
    return loss, logits, task_loss


@torch.no_grad()
def validate(model, val_ds, device, args, max_batches=200):
    model.eval()
    rng = np.random.default_rng(0)
    all_pred = []
    all_yaw = []
    all_pitch = []
    losses = []
    for bi, batch in enumerate(iter_batches(val_ds, args.batch, rng, shuffle=False)):
        if bi >= max_batches:
            break
        x = torch.from_numpy(batch["x"]).to(device)
        out = model("vp", x)
        soft = torch.from_numpy(batch["soft"]).to(device)
        loss = F.kl_div(F.log_softmax(out["logits"], dim=-1), soft, reduction="batchmean")
        losses.append(float(loss.item()))
        pred = out["logits"].argmax(-1).cpu().numpy()
        yp = batch["yaw_pitch"]
        all_pred.append(pred)
        all_yaw.append(yp[:, 0])
        all_pitch.append(yp[:, 1])
    model.train()
    if not all_pred:
        return {"val_loss": float("nan"), "mae_rad": float("nan"), "mae_deg": float("nan")}
    pred = np.concatenate(all_pred)
    true_yaw = np.concatenate(all_yaw)
    true_pitch = np.concatenate(all_pitch)
    mae = great_circle_mae_rad(pred, true_yaw, true_pitch, args.tile_h, args.tile_w)
    return {"val_loss": float(np.mean(losses)), "mae_rad": mae, "mae_deg": math.degrees(mae)}


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MoEMambaConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_experts=args.n_experts,
        top_k=args.top_k,
        vp_tile_h=args.tile_h,
        vp_tile_w=args.tile_w,
        vp_traj_seq_len=args.seq_len,
        vp_traj_features=6,
        vp_horizon_steps=args.horizon,
    )
    model = MoEMamba(cfg).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_ds, val_ds = build_loaders(args)
    print(f"train samples = {len(train_ds)} | val samples = {len(val_ds)} | tiles = {cfg.vp_num_classes}")

    rng = np.random.default_rng(args.seed)
    best_mae = float("inf")
    for epoch in range(args.epochs):
        loss_acc, n = 0.0, 0
        for bi, batch in enumerate(iter_batches(train_ds, args.batch, rng)):
            loss, logits, task_loss = step(model, batch, device, args)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_acc += float(task_loss.item()) * len(batch["tile"])
            n += len(batch["tile"])
            if args.max_batches and bi + 1 >= args.max_batches:
                break

        train_loss = loss_acc / max(1, n)
        val = validate(model, val_ds, device, args)
        print(f"epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val['val_loss']:.4f} "
              f"| MAE={val['mae_rad']:.3f} rad ({val['mae_deg']:.2f} deg)")

        if val["mae_rad"] < best_mae and args.save:
            best_mae = val["mae_rad"]
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, args.save)
            print(f"  saved -> {args.save}")

    return model, val_ds


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--vp_root", default=None,
                   help="path to NetLLM viewport_prediction/data/viewports (or compatible)")
    p.add_argument("--vp_dataset", choices=["wu2017", "jin2022"], default="wu2017")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--synth", action="store_true", help="use synthetic fallback dataset")
    p.add_argument("--synth_users", type=int, default=30)
    p.add_argument("--synth_seconds", type=int, default=60)
    p.add_argument("--sample_hz", type=float, default=5.0)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--tile_h", type=int, default=8)
    p.add_argument("--tile_w", type=int, default=32)
    p.add_argument("--hard", action="store_true", help="cross-entropy on hard tile id (default: soft KL)")
    # model
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_experts", type=int, default=4)
    p.add_argument("--top_k", type=int, default=2)
    # opt
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max_batches", type=int, default=0, help="0 = full epoch")
    p.add_argument("--lb_coef", type=float, default=0.01)
    p.add_argument("--kd_coef", type=float, default=0.0)
    p.add_argument("--kd_temperature", type=float, default=2.0)
    # misc
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None)
    args = p.parse_args()
    train(args)
