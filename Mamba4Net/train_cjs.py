"""REINFORCE-with-baseline training of MoE-Mamba on the Decima-style CJS environment.

Run:
    python train_cjs.py --episodes 200 --device cuda
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.distributions import Categorical

from moe_mamba import MoEMamba, MoEMambaConfig
from envs import CJSEnv


def obs_to_tensors(obs, device):
    nf = torch.from_numpy(obs["node_feats"]).unsqueeze(0).to(device)
    adj = torch.from_numpy(obs["adj"]).unsqueeze(0).to(device)
    mask = torch.from_numpy(obs["node_mask"]).unsqueeze(0).to(device)
    urg = torch.from_numpy(obs["urgency"]).unsqueeze(0).to(device)
    return nf, adj, mask, urg


def rollout(env: CJSEnv, model: MoEMamba, device: str, max_steps: int = 1000):
    log_probs, values, rewards = [], [], []
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        nf, adj, mask, urg = obs_to_tensors(obs, device)
        if not bool(mask.any()):
            # nothing schedulable yet -- step env with a no-op (action=0); env will time-skip.
            obs, r, done, info = env.step(0)
            rewards.append(float(r))
            log_probs.append(torch.zeros((), device=device))
            values.append(0.0)
            steps += 1
            continue
        out = model("cjs", nf, adj, urgency=urg)
        logits = out["logits"][0]  # (N,)
        logits = logits.masked_fill(~mask[0], float("-inf"))
        dist = Categorical(logits=logits)
        a = dist.sample()
        log_probs.append(dist.log_prob(a))
        values.append(float(out["value"].item()))
        obs, r, done, info = env.step(int(a.item()))
        rewards.append(float(r))
        steps += 1
    return {"log_probs": log_probs, "values": values, "rewards": rewards,
            "makespan": env.now_s, "avg_jct": env.avg_jct()}


def returns_to_go(rewards: List[float], gamma: float = 0.99) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float32)
    G = 0.0
    for i in reversed(range(len(rewards))):
        G = rewards[i] + gamma * G
        out[i] = G
    return out


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MoEMambaConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_experts=args.n_experts,
        top_k=args.top_k,
        cjs_max_nodes=args.max_nodes,
        cjs_node_features=8,
    )
    model = MoEMamba(cfg).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    env = CJSEnv(n_executors=args.n_executors, n_jobs=args.n_jobs,
                 max_stages_per_job=args.max_stages, max_nodes=args.max_nodes, seed=args.seed)

    rolling_jct = []
    for ep in range(args.episodes):
        roll = rollout(env, model, device, max_steps=args.max_steps)
        if not roll["log_probs"]:
            continue
        ret = returns_to_go(roll["rewards"], gamma=args.gamma)
        ret = (ret - ret.mean()) / (ret.std() + 1e-6)
        ret_t = torch.from_numpy(ret).to(device)
        log_probs_t = torch.stack(roll["log_probs"])
        pg = -(log_probs_t * ret_t).mean()

        # auxiliary load-balance comes from each forward; we approximate via one extra forward
        loss = pg

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        rolling_jct.append(roll["avg_jct"])
        rolling_jct = rolling_jct[-20:]
        if ep % args.log_every == 0:
            print(f"ep {ep:4d} | makespan={roll['makespan']:7.2f}s "
                  f"| avg_jct={roll['avg_jct']:7.2f}s | jct_avg20={np.mean(rolling_jct):7.2f} "
                  f"| pg={pg.item():.4f}")

    if args.save:
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, args.save)
        print(f"saved -> {args.save}")
    return model, env


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_experts", type=int, default=4)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--n_executors", type=int, default=8)
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--max_stages", type=int, default=6)
    p.add_argument("--max_nodes", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save", default=None)
    args = p.parse_args()
    train(args)
