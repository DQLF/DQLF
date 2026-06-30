# MoE-Mamba

Reference implementation of *Mamba4Net-MoE / MoE-Mamba* — a unified networking
foundation model that distills multi-task expertise from Transformer/LLM teachers
into a single Mamba-based student equipped with Mixture-of-Experts. Three
representative tasks: **Viewport Prediction (VP)**, **Adaptive Bitrate streaming
(ABR)**, **Cluster Job Scheduling (CJS)**.

This repo deliberately mirrors the data conventions of two upstream open-source
projects so you can swap their datasets in directly:

- **NetLLM** (SIGCOMM '24) — <https://github.com/duowuyms/NetLLM>
- **Mamba4Net** (arXiv 2510.17147)

## Layout

```
moe_mamba/        model code (mamba block, MoE, encoders, heads, distillation)
envs/             environments and datasets for ABR / CJS / VP
train_abr.py      A2C training on the Pensieve-style ABR env
train_cjs.py      REINFORCE training on the Decima-style CJS env
train_vp.py       Supervised training on Wu2017 / Jin2022 viewport datasets
evaluate.py       Evaluator with vp / abr / cjs subcommands
demo.py           End-to-end smoke check (no real data required)
```

## Quick start

```bash
python -m pip install torch numpy
python demo.py                    # forward + backward + every env sanity check
python train_vp.py --synth --epochs 5
python train_abr.py --episodes 60
python train_cjs.py --episodes 60
```

To use the official CUDA Mamba kernel (Linux + CUDA only):

```bash
pip install mamba-ssm causal-conv1d
```

`MambaBlock` autodetects this at import time and falls back to a pure-PyTorch
selective scan when the kernel is unavailable. You can force the fallback with
`MOE_MAMBA_BACKEND=pytorch`.

---

## Data sources

We do **not** redistribute datasets. Each task points at the canonical upstream
release used by NetLLM / Mamba4Net. The loaders in this repo are written to
read those files unchanged so you can `git clone` the upstream repo (or its
data drop) and just point at the directory.

### 1. Viewport Prediction — Wu2017 / Jin2022

| Resource | Origin | Citation |
| -------- | ------ | -------- |
| `wu2017` viewport traces | Wu et al., MMSys '17 | *A dataset for exploring user behaviors in VR spherical video streaming.* |
| `jin2022` viewport traces | Jin et al., ACM MM '22 | *Where Are You Looking? A Large-Scale Dataset of Head and Gaze Behavior for 360° Videos.* |
| Video frames + ViT features | Cooked by NetLLM authors | SharePoint link below |

**How to get the cooked CSVs (ready-to-train).** They are bundled in NetLLM:

```bash
git clone https://github.com/duowuyms/NetLLM.git
ls NetLLM/viewport_prediction/data/viewports
# wu2017/  jin2022/
#   <video_id>/<user_id>.csv     # rows of: t_s, yaw_rad, pitch_rad
```

**How to get the raw video frames / ViT features** (only needed if you want to
fuse visual saliency; trajectory-only training does not need them):

```
SharePoint:
https://cuhko365-my.sharepoint.com/:f:/g/personal/223015061_link_cuhk_edu_cn/Es2wxUodbWNDvpTb0Vk3zQwBrbq4aaxTLuVVp2jUtroXsA?e=zPgYvu
```

Place them at `NetLLM/viewport_prediction/data/images` and
`NetLLM/viewport_prediction/data/image_features` per the upstream README.

**Train against real data:**

```bash
python train_vp.py \
    --vp_root /path/to/NetLLM/viewport_prediction/data/viewports \
    --vp_dataset wu2017 \
    --epochs 30 --batch 64 --tile_h 8 --tile_w 32
```

**Or use the synthetic fallback** (sphere random-walk with salient attractors;
preserves the 1/f temporal structure of head motion for smoke tests / CI):

```bash
python train_vp.py --synth --epochs 5
```

The loader is `envs.load_netllm_viewports(root, dataset="wu2017"|"jin2022")`.
A second loader `ViewportTrajectory.from_xyz_csv` parses the original Wu2017
unit-vector format `t_s, x, y, z` if you prefer the raw release.

### 2. Adaptive Bitrate streaming — FCC + Norway HSDPA

| Resource | Origin |
| -------- | ------ |
| Bandwidth traces | Pensieve (NSDI '17) → Genet (SIGCOMM '22) → NetLLM |
| Real-world sources | FCC Measuring Broadband America (USA) + Telenor 3G/HSDPA traces (Norway) |
| Video chunk specs | Pensieve `video_size_*` files (DASH-encoded `Envivio-Dash3`) |

**File format** for every trace, used unmodified by Pensieve / NetLLM / this repo:

```
<time_s>\t<throughput_mbps>
<time_s>\t<throughput_mbps>
...
```

**How to get them.** The simplest path is NetLLM's drop:

```bash
ls NetLLM/adaptive_bitrate_streaming/data/traces
# fcc-train/  fcc-test/  norway-train/  norway-test/  ...
```

If you want the upstream releases:

- **Pensieve** (FCC + Norway already split): <https://github.com/hongzimao/pensieve>
  → `pensieve/sim/cooked_traces/` and `pensieve/sim/cooked_test_traces/`
- **FCC** raw broadband CSVs (need to cook to time/throughput pairs):
  <https://www.fcc.gov/oet/mba/raw-data-releases>
- **Norway HSDPA** raw 4G/HSDPA traces:
  <http://skuld.cs.umass.edu/traces/mmsys/2013/pathbandwidth/>

**Train against real data:**

```bash
python train_abr.py --traces /path/to/NetLLM/adaptive_bitrate_streaming/data/traces/fcc-train \
                    --episodes 200
```

**Synthetic fallback** (a 2-state Markov bandwidth process with bursts /
dropouts) runs automatically when `--traces` is omitted.

The video chunk-size table in this repo is generated procedurally around the 6
canonical Pensieve bitrates `[300, 750, 1200, 1850, 2850, 4300] kbps` × 48
4-second chunks. To use Pensieve's exact `video_size_<bitrate>` files, drop them
into a directory and adapt `_build_video_chunk_sizes` in `envs/abr.py` to read
them — straightforward; one for-loop.

### 3. Cluster Job Scheduling — TPC-H via spark-sched-sim

| Resource | Origin |
| -------- | ------ |
| Workload | TPC-H benchmark queries (decision-support SQL workloads) |
| DAG generator + simulator | `spark-sched-sim` ← Decima (SIGCOMM '19) |
| NetLLM wrapper | <https://github.com/duowuyms/NetLLM/tree/master/cluster_job_scheduling> |

**How to get it.** Two equivalent options:

- **NetLLM drop** — `NetLLM/cluster_job_scheduling/data/tpch/` already contains
  the precomputed task-duration tables (one file per query / scale factor).
- **Upstream simulator** — clone <https://github.com/ArchieGertsman/spark-sched-sim>
  and use its `data/tpch/` directory.

The original TPC-H schema and queries (if you want to regenerate raw runtimes):
<http://www.tpc.org/tpch/>

**Note on this repo's CJS env.** `envs/cjs.py` is a self-contained discrete-event
DAG scheduler that follows Decima's MDP (random-DAG generator with chain / fan-out
/ tree topologies, executor pool, event-driven advancement, reward = `−Δt × #active_jobs`).
This works out of the box for training. To swap in real TPC-H DAGs, replace
`_gen_job` in `envs/cjs.py` with a routine that walks the TPC-H stage tables —
the rest of the simulator stays the same.

**Train:**

```bash
python train_cjs.py --episodes 200 --n_executors 8 --n_jobs 4 --max_stages 6
```

---

## Distillation (optional)

`distill.py` provides:

- **KL distillation** (paper eq. 6) — pass `teacher_logits` to `step_*` and the
  loss adds `τ² · KL(softmax(z_s/τ) ‖ softmax(z_t/τ))`.
- **Cross-Architectural Weight Reuse (CWR)** — `cwr_seed_experts(model, attn_W,
  ffn_W)` does a low-rank SVD of teacher Transformer matrices and seeds each
  Mamba expert's input projection. Use this when you have a Llama-family
  Transformer teacher. NetLLM's released checkpoint at
  <https://drive.google.com/file/d/17UyXJ9rGc0wKUkAhQ4wMrYDEbRPRjil0/view> is a
  reasonable seed.

## Citation pointers

If you use this code, please credit the upstream papers:

- NetLLM — Wu et al., SIGCOMM 2024 (DOI: `10.1145/3651890.3672268`).
- Mamba4Net — Mamba4Net authors, arXiv:2510.17147.
- Mamba — Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces", 2023.
- Wu2017 — Wu et al., MMSys 2017.
- Jin2022 — Jin et al., ACM MM 2022.
- Pensieve — Mao et al., SIGCOMM 2017.
- Decima — Mao et al., SIGCOMM 2019.
