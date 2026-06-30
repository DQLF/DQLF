"""End-to-end smoke check: model forward/backward + ABR & CJS env rollouts + VP dataset."""
import numpy as np
import torch

from moe_mamba import MoEMamba, MoEMambaConfig, distillation_loss, cwr_seed_experts
from moe_mamba.mamba import HAS_MAMBA_SSM
from envs import ABREnv, ABRTrace, BITRATES_KBPS, CJSEnv, VPDataset, synth_dataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | mamba-ssm CUDA backend available: {HAS_MAMBA_SSM}")

    cfg = MoEMambaConfig(
        d_model=128, n_layers=2, n_experts=4, top_k=2,
        vp_image_size=64, vp_patch_size=16,
        vp_tile_h=4, vp_tile_w=8,         # 32-tile spherical grid for the smoke
        abr_seq_len=8, cjs_max_nodes=24,
    )
    model = MoEMamba(cfg).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"MoE-Mamba params: {n/1e6:.2f}M")

    # ---------- VP (frame path) forward + KD loss ----------
    x = torch.randn(2, cfg.vp_in_channels, cfg.vp_image_size, cfg.vp_image_size, device=device)
    out = model("vp", x)
    teacher_logits = torch.randn_like(out["logits"])
    kd = distillation_loss(out["logits"], teacher_logits, cfg.distill_temperature)
    print(f"vp[frame] logits {tuple(out['logits'].shape)} | distill={kd.item():.4f} | aux={float(out['aux']):.3f}")

    # ---------- VP (trajectory path) on synthetic head-motion data ----------
    trajs = synth_dataset(n_users=4, n_seconds=20, sample_hz=5.0, seed=0)
    ds = VPDataset(trajs, seq_len=cfg.vp_traj_seq_len, horizon=cfg.vp_horizon_steps,
                   tile_h=cfg.vp_tile_h, tile_w=cfg.vp_tile_w, stride=4)
    batch = ds.collate([ds[i] for i in range(min(8, len(ds)))])
    xt = torch.from_numpy(batch["x"]).to(device)
    out_t = model("vp", xt)
    print(f"vp[traj]  logits {tuple(out_t['logits'].shape)}  ds_size={len(ds)}  "
          f"target_tile={int(batch['tile'][0])}/{cfg.vp_num_classes}")

    # ---------- CWR seeding ----------
    fake_attn = [torch.randn(cfg.d_model, cfg.d_model) for _ in range(2)]
    fake_ffn = [torch.randn(cfg.d_model * 2, cfg.d_model) for _ in range(2)]
    seeded = cwr_seed_experts(model, fake_attn, fake_ffn, rank=cfg.cwr_rank)
    print(f"CWR seeded experts: {seeded}")

    # ---------- ABR env rollout ----------
    env = ABREnv(traces=[ABRTrace.synth(seed=i, n_steps=200) for i in range(4)],
                 history_len=cfg.abr_seq_len, seed=0)
    s = env.reset()
    qoe_sum = 0.0
    for _ in range(8):
        with torch.no_grad():
            xt = torch.from_numpy(s).unsqueeze(0).to(device)
            out = model("abr", xt)
        a = int(out["policy"].argmax(-1).item())
        s, r, done, info = env.step(a)
        qoe_sum += r
        if done:
            break
    print(f"abr 8-step: qoe={qoe_sum:.3f} bitrate={info['bitrate_kbps']}kbps buffer={info['buffer_s']:.2f}s")

    # ---------- CJS env rollout ----------
    cjs = CJSEnv(n_executors=4, n_jobs=3, max_stages_per_job=4, max_nodes=cfg.cjs_max_nodes, seed=0)
    obs = cjs.reset()
    rew = 0.0
    for _ in range(50):
        nf = torch.from_numpy(obs["node_feats"]).unsqueeze(0).to(device)
        adj = torch.from_numpy(obs["adj"]).unsqueeze(0).to(device)
        urg = torch.from_numpy(obs["urgency"]).unsqueeze(0).to(device)
        mask = torch.from_numpy(obs["node_mask"]).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model("cjs", nf, adj, urgency=urg)
        if not bool(mask.any()):
            obs, r, done, _ = cjs.step(0)
        else:
            logits = out["logits"][0].masked_fill(~mask[0], float("-inf"))
            a = int(logits.argmax().item())
            obs, r, done, _ = cjs.step(a)
        rew += r
        if done:
            break
    print(f"cjs 50-step: reward={rew:.2f} makespan={cjs.now_s:.2f}s avg_jct={cjs.avg_jct():.2f}s")

    # ---------- backward smoke ----------
    obs = cjs.reset()
    nf = torch.from_numpy(obs["node_feats"]).unsqueeze(0).to(device)
    adj = torch.from_numpy(obs["adj"]).unsqueeze(0).to(device)
    urg = torch.from_numpy(obs["urgency"]).unsqueeze(0).to(device)
    out = model("cjs", nf, adj, urgency=urg)
    target = torch.rand_like(out["scores"])
    loss = torch.nn.functional.binary_cross_entropy(out["scores"], target) + 0.01 * out["aux"]
    loss.backward()
    print(f"backward ok. loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
