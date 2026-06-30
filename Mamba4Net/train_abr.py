"""A2C training of MoE-Mamba on the Pensieve-style ABR environment.

Run:
    python train_abr.py --episodes 200 --device cuda
    python train_abr.py --traces path/to/trace_dir       # use real traces
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.distributions import Categorical

from moe_mamba import MoEMamba, MoEMambaConfig
from envs import ABREnv, ABRTrace, BITRATES_KBPS


def load_traces(trace_dir: str | None) -> List[ABRTrace]:
    if not trace_dir:
        return [ABRTrace.synth(seed=i) for i in range(16)]
    paths = sorted(glob.glob(os.path.join(trace_dir, "*")))
    if not paths:
        raise SystemExit(f"no trace files found under {trace_dir}")
    return [ABRTrace.from_file(p) for p in paths]


def compute_gae(rewards: List[float], values: List[float], gamma: float = 0.99,
                lam: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_v = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        next_v = values[t]
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


def rollout(env: ABREnv, model: MoEMamba, device: str, deterministic: bool = False):
    """
    执行单次环境交互轨迹（Rollout），收集状态、动作、奖励等数据。

    参数:
        env (ABREnv): 自适应码率强化学习环境实例。
        model (MoEMamba): 用于决策的策略模型。
        device (str): 运行模型的设备类型（如 'cpu' 或 'cuda'）。
        deterministic (bool): 是否采用确定性策略。若为 True，则选择概率最大的动作；
                              若为 False，则根据概率分布采样动作。默认为 False。

    返回:
        dict: 包含以下键值的字典：
            - "states" (list): 历史状态列表。
            - "actions" (list): 历史动作列表。
            - "log_probs" (list): 每个动作的对数概率列表。
            - "values" (list): 每个状态的状态价值估计列表。
            - "rewards" (list): 每一步获得的奖励列表。
            - "qoe_sum" (float): 整个轨迹的累积服务质量体验（QoE）总和。
    """
    states, actions, log_probs, values, rewards = [], [], [], [], []
    s = env.reset()
    done = False
    total_qoe = 0.0
    while not done:
        # 将当前状态转换为张量并移至指定设备，增加批次维度以适配模型输入
        x = torch.from_numpy(s).unsqueeze(0).to(device)  # (1, T, F)
        out = model("abr", x)
        logits = out["policy"]
        dist = Categorical(logits=logits)
        # 根据确定性标志选择动作：确定性模式下取最大概率动作，否则从分布中采样
        a = logits.argmax(-1) if deterministic else dist.sample()
        states.append(s)
        actions.append(int(a.item()))
        log_probs.append(float(dist.log_prob(a).item()))
        values.append(float(out["value"].item()))
        s, r, done, info = env.step(int(a.item()))
        rewards.append(float(r))
        total_qoe += r
    return {
        "states": states, "actions": actions, "log_probs": log_probs,
        "values": values, "rewards": rewards, "qoe_sum": total_qoe,
    }


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MoEMambaConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_experts=args.n_experts,
        top_k=args.top_k,
        abr_seq_len=args.history_len,
        abr_in_features=6,
        abr_num_actions=len(BITRATES_KBPS),
    )
    model = MoEMamba(cfg).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    traces = load_traces(args.traces)
    env = ABREnv(traces=traces, history_len=args.history_len, alpha=args.alpha, beta=args.beta,
                 utility=args.utility, seed=args.seed)

    rolling = []
    for ep in range(args.episodes):
        roll = rollout(env, model, device)
        adv, ret = compute_gae(roll["rewards"], roll["values"], gamma=args.gamma, lam=args.lam)

        T = len(roll["actions"])
        x = torch.from_numpy(np.stack(roll["states"], axis=0)).to(device)  # (T, T_hist, F)
        a = torch.tensor(roll["actions"], device=device, dtype=torch.long)
        adv_t = torch.from_numpy(adv).to(device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)
        ret_t = torch.from_numpy(ret).to(device)

        out = model("abr", x)
        logp = F.log_softmax(out["policy"], dim=-1)
        chosen = logp.gather(1, a.unsqueeze(-1)).squeeze(-1)
        pg_loss = -(chosen * adv_t).mean()
        v_loss = F.mse_loss(out["value"], ret_t)
        ent = -(logp.exp() * logp).sum(dim=-1).mean()
        loss = pg_loss + args.value_coef * v_loss - args.entropy_coef * ent + args.lb_coef * out["aux"]

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        rolling.append(roll["qoe_sum"])
        rolling = rolling[-20:]
        if ep % args.log_every == 0:
            print(f"ep {ep:4d} | qoe_ep={roll['qoe_sum']:8.3f} | qoe_avg20={np.mean(rolling):8.3f} "
                  f"| pg={pg_loss.item():.3f} v={v_loss.item():.3f} ent={ent.item():.3f} aux={float(out['aux']):.3f}")

    if args.save:
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, args.save)
        print(f"saved -> {args.save}")
    return model, env


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--traces", default=None, help="dir of pensieve-format trace files; else synth")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_experts", type=int, default=4)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--history_len", type=int, default=8)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=4.3)
    p.add_argument("--utility", choices=["linear", "log"], default="linear")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--lb_coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save", default=None)
    args = p.parse_args()
    train(args)
