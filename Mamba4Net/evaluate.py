"""Evaluate trained MoE-Mamba checkpoints on ABR / CJS / VP."""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from moe_mamba import MoEMamba, MoEMambaConfig
from envs import (ABREnv, ABRTrace, CJSEnv,
                  VPDataset, load_netllm_viewports, synth_dataset, great_circle_mae_rad)
from train_abr import rollout as abr_rollout, load_traces
from train_cjs import rollout as cjs_rollout


def load_model(ckpt_path: str, device: str):
    blob = torch.load(ckpt_path, map_location=device)
    cfg = MoEMambaConfig(**blob["cfg"])
    m = MoEMamba(cfg).to(device).eval()
    m.load_state_dict(blob["model"])
    return m, cfg


def eval_abr(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)
    traces = load_traces(args.traces)
    env = ABREnv(traces=traces, history_len=cfg.abr_seq_len, seed=args.seed)
    qoes = []
    for _ in range(args.episodes):
        with torch.no_grad():
            roll = abr_rollout(env, model, device, deterministic=True)
        qoes.append(roll["qoe_sum"])
    print(f"[abr] eps={args.episodes}  qoe_mean={np.mean(qoes):.3f}  qoe_std={np.std(qoes):.3f}")


@torch.no_grad()
def eval_vp(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)
    model.eval()
    if args.synth or not args.vp_root:
        trajs = synth_dataset(n_users=args.synth_users, n_seconds=args.synth_seconds,
                              sample_hz=args.sample_hz, seed=args.seed)
    else:
        trajs = load_netllm_viewports(args.vp_root, dataset=args.vp_dataset, max_files=args.max_files)
    ds = VPDataset(trajs, seq_len=cfg.vp_traj_seq_len, horizon=cfg.vp_horizon_steps,
                   tile_h=cfg.vp_tile_h, tile_w=cfg.vp_tile_w, stride=args.stride)
    preds = []; yaws = []; pitches = []; losses = []
    for i in range(0, len(ds), args.batch):
        chunk = [ds[j] for j in range(i, min(i + args.batch, len(ds)))]
        batch = ds.collate(chunk)
        x = torch.from_numpy(batch["x"]).to(device)
        out = model("vp", x)
        soft = torch.from_numpy(batch["soft"]).to(device)
        loss = F.kl_div(F.log_softmax(out["logits"], dim=-1), soft, reduction="batchmean")
        losses.append(float(loss.item()))
        preds.append(out["logits"].argmax(-1).cpu().numpy())
        yaws.append(batch["yaw_pitch"][:, 0])
        pitches.append(batch["yaw_pitch"][:, 1])
    pred = np.concatenate(preds)
    yaw = np.concatenate(yaws); pitch = np.concatenate(pitches)
    mae = great_circle_mae_rad(pred, yaw, pitch, cfg.vp_tile_h, cfg.vp_tile_w)
    print(f"[vp] samples={len(pred)}  loss={np.mean(losses):.4f}  "
          f"MAE={mae:.3f} rad ({math.degrees(mae):.2f} deg)")


def eval_cjs(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)
    env = CJSEnv(n_executors=args.n_executors, n_jobs=args.n_jobs,
                 max_stages_per_job=args.max_stages, max_nodes=cfg.cjs_max_nodes, seed=args.seed)
    jcts, makespans = [], []
    for _ in range(args.episodes):
        with torch.no_grad():
            roll = cjs_rollout(env, model, device, max_steps=args.max_steps)
        jcts.append(roll["avg_jct"])
        makespans.append(roll["makespan"])
    print(f"[cjs] eps={args.episodes}  avg_jct={np.mean(jcts):.2f}s  makespan={np.mean(makespans):.2f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="task", required=True)

    a = sub.add_parser("abr")
    a.add_argument("--ckpt", required=True)
    a.add_argument("--traces", default=None)
    a.add_argument("--episodes", type=int, default=20)
    a.add_argument("--device", default=None)
    a.add_argument("--seed", type=int, default=42)

    c = sub.add_parser("cjs")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--episodes", type=int, default=20)
    c.add_argument("--n_executors", type=int, default=8)
    c.add_argument("--n_jobs", type=int, default=4)
    c.add_argument("--max_stages", type=int, default=6)
    c.add_argument("--max_steps", type=int, default=2000)
    c.add_argument("--device", default=None)
    c.add_argument("--seed", type=int, default=42)

    v = sub.add_parser("vp")
    v.add_argument("--ckpt", required=True)
    v.add_argument("--vp_root", default=None)
    v.add_argument("--vp_dataset", choices=["wu2017", "jin2022"], default="wu2017")
    v.add_argument("--max_files", type=int, default=None)
    v.add_argument("--synth", action="store_true")
    v.add_argument("--synth_users", type=int, default=10)
    v.add_argument("--synth_seconds", type=int, default=60)
    v.add_argument("--sample_hz", type=float, default=5.0)
    v.add_argument("--stride", type=int, default=4)
    v.add_argument("--batch", type=int, default=64)
    v.add_argument("--device", default=None)
    v.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.task == "abr":
        eval_abr(args)
    elif args.task == "vp":
        eval_vp(args)
    else:
        eval_cjs(args)
