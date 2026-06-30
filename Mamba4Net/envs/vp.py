"""Viewport Prediction (VP) dataset, tile utilities, and trajectory featurizer.

Data sources (see README for download instructions):
    Wu2017   -- Wu et al., "A dataset for exploring user behaviors in VR
                spherical video streaming," MMSys 2017. Per-user yaw/pitch
                traces over 48 videos at ~5 Hz.
    Jin2022  -- Jin et al., "Where Are You Looking? ...", ACM MM 2022.
                Larger-scale 360-deg head/gaze dataset.

NetLLM ships *cooked* CSVs of both at
    <NetLLM>/viewport_prediction/data/viewports/{wu2017,jin2022}/<video_id>/<user_id>.csv
each row of which is `t_s, yaw_rad, pitch_rad` (one sample per video frame).
This file's `ViewportTrajectory.from_netllm_csv` parses that format directly.

If real data is unavailable (CI or quick smoke), `synth_dataset` produces sphere
random-walks with moving salient attractors that retain the 1/f temporal
structure that head-motion datasets exhibit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------- #
# Tile utilities  (paper eq. 7: gamma_vp = softmax(phi^T H_out) over H*W tiles)  #
# ----------------------------------------------------------------------------- #

def yaw_pitch_to_tile(yaw_rad: np.ndarray, pitch_rad: np.ndarray,
                      H: int, W: int) -> np.ndarray:
    """Map (yaw, pitch) in radians to a flat tile index in [0, H*W).

    yaw  in [-pi, pi]  -> column in [0, W)
    pitch in [-pi/2, pi/2] -> row in [0, H)
    """
    yaw = np.mod(yaw_rad + np.pi, 2 * np.pi) - np.pi
    pitch = np.clip(pitch_rad, -np.pi / 2 + 1e-6, np.pi / 2 - 1e-6)
    col = ((yaw + np.pi) / (2 * np.pi) * W).astype(np.int64)
    row = ((pitch + np.pi / 2) / np.pi * H).astype(np.int64)
    col = np.clip(col, 0, W - 1)
    row = np.clip(row, 0, H - 1)
    return row * W + col


def tile_to_yaw_pitch(tile_idx: int, H: int, W: int) -> Tuple[float, float]:
    row = tile_idx // W
    col = tile_idx % W
    yaw = (col + 0.5) / W * 2 * np.pi - np.pi
    pitch = (row + 0.5) / H * np.pi - np.pi / 2
    return float(yaw), float(pitch)


def gaussian_tile_target(yaw: float, pitch: float, H: int, W: int,
                         sigma_rad: float = 0.25) -> np.ndarray:
    """Soft target: Gaussian on the sphere centered at (yaw, pitch), tiled to H*W.

    Better learning signal than one-hot because adjacent tiles should also score high.
    """
    cols = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
    rows = (np.arange(H) + 0.5) / H * np.pi - np.pi / 2
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    # great-circle distance on unit sphere
    cos_d = np.sin(yy) * np.sin(pitch) + np.cos(yy) * np.cos(pitch) * np.cos(xx - yaw)
    d = np.arccos(np.clip(cos_d, -1.0, 1.0))
    p = np.exp(-(d ** 2) / (2 * sigma_rad ** 2))
    p = p / p.sum()
    return p.astype(np.float32).reshape(-1)


# ----------------------------------------------------------------------------- #
# Trajectory record + featurizer                                                #
# ----------------------------------------------------------------------------- #

@dataclass
class ViewportTrajectory:
    times_s: np.ndarray
    yaw_rad: np.ndarray
    pitch_rad: np.ndarray
    video_id: str = ""
    user_id: str = ""

    def __len__(self) -> int:
        return len(self.times_s)

    @classmethod
    def from_netllm_csv(cls, path: str | Path, video_id: str = "", user_id: str = "") -> "ViewportTrajectory":
        """Parse NetLLM's cooked viewport CSV: each row `t_s, yaw_rad, pitch_rad`.

        Header line is ignored. Non-numeric / short rows are silently skipped.
        """
        ts, yaws, pitches = [], [], []
        with open(path) as f:
            for line in f:
                parts = [p.strip() for p in line.replace(",", " ").split()]
                if len(parts) < 3:
                    continue
                try:
                    ts.append(float(parts[0]))
                    yaws.append(float(parts[1]))
                    pitches.append(float(parts[2]))
                except ValueError:
                    continue  # header / blank line
        return cls(np.asarray(ts), np.asarray(yaws), np.asarray(pitches), video_id, user_id)

    @classmethod
    def from_xyz_csv(cls, path: str | Path, video_id: str = "", user_id: str = "") -> "ViewportTrajectory":
        """Parse the raw Wu2017 unit-vector format: `t_s, x, y, z`."""
        ts, xs, ys, zs = [], [], [], []
        with open(path) as f:
            for line in f:
                parts = [p.strip() for p in line.replace(",", " ").split()]
                if len(parts) < 4:
                    continue
                try:
                    ts.append(float(parts[0]))
                    xs.append(float(parts[1]))
                    ys.append(float(parts[2]))
                    zs.append(float(parts[3]))
                except ValueError:
                    continue
        x, y, z = np.asarray(xs), np.asarray(ys), np.asarray(zs)
        yaw = np.arctan2(y, x)
        pitch = np.arcsin(np.clip(z, -1.0, 1.0))
        return cls(np.asarray(ts), yaw, pitch, video_id, user_id)


def featurize(traj: ViewportTrajectory, idx_lo: int, idx_hi: int) -> np.ndarray:
    """6-feature trajectory frame: [sin(yaw), cos(yaw), sin(pitch), cos(pitch), dyaw, dpitch].

    Slicing convention: timesteps idx_lo .. idx_hi-1 inclusive.
    """
    yaw = traj.yaw_rad[idx_lo:idx_hi]
    pitch = traj.pitch_rad[idx_lo:idx_hi]
    dyaw = np.diff(yaw, prepend=yaw[:1])
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi  # unwrap to [-pi, pi]
    dpitch = np.diff(pitch, prepend=pitch[:1])
    return np.stack([np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch), dyaw, dpitch], axis=-1).astype(np.float32)


# ----------------------------------------------------------------------------- #
# Dataset                                                                        #
# ----------------------------------------------------------------------------- #

class VPDataset:
    """Sliding-window VP dataset over a list of trajectories.

    Each sample = (window of `seq_len` frames, target tile index `horizon` frames ahead).
    """

    def __init__(self, trajectories: Sequence[ViewportTrajectory],
                 seq_len: int = 16, horizon: int = 5,
                 tile_h: int = 8, tile_w: int = 32,
                 stride: int = 1):
        self.trajs = [t for t in trajectories if len(t) >= seq_len + horizon]
        if not self.trajs:
            raise ValueError("no trajectories long enough for given seq_len + horizon")
        self.seq_len = seq_len
        self.horizon = horizon
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.stride = stride
        self._index: List[Tuple[int, int]] = []
        for ti, t in enumerate(self.trajs):
            n_starts = len(t) - seq_len - horizon + 1
            for s in range(0, n_starts, stride):
                self._index.append((ti, s))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        ti, s = self._index[i]
        t = self.trajs[ti]
        x = featurize(t, s, s + self.seq_len)
        target_yaw = float(t.yaw_rad[s + self.seq_len + self.horizon - 1])
        target_pitch = float(t.pitch_rad[s + self.seq_len + self.horizon - 1])
        tile = int(yaw_pitch_to_tile(np.array([target_yaw]), np.array([target_pitch]),
                                     self.tile_h, self.tile_w)[0])
        soft = gaussian_tile_target(target_yaw, target_pitch, self.tile_h, self.tile_w)
        return {
            "x": x,                  # (seq_len, 6)
            "tile": tile,            # int
            "soft": soft,            # (H*W,)
            "yaw_pitch": np.array([target_yaw, target_pitch], dtype=np.float32),
        }

    def collate(self, batch):
        x = np.stack([b["x"] for b in batch], axis=0)
        tile = np.array([b["tile"] for b in batch], dtype=np.int64)
        soft = np.stack([b["soft"] for b in batch], axis=0)
        yp = np.stack([b["yaw_pitch"] for b in batch], axis=0)
        return {"x": x, "tile": tile, "soft": soft, "yaw_pitch": yp}


# ----------------------------------------------------------------------------- #
# Loaders                                                                        #
# ----------------------------------------------------------------------------- #

def load_netllm_viewports(root: str | Path, dataset: str = "wu2017",
                          max_files: int | None = None) -> List[ViewportTrajectory]:
    """Walk NetLLM-style directory: <root>/<dataset>/<video_id>/<user_id>.csv.

    Returns a list of ViewportTrajectory; skips empty/short files silently.
    """
    base = Path(root) / dataset
    if not base.exists():
        raise FileNotFoundError(f"viewport dataset root not found: {base}")
    out: List[ViewportTrajectory] = []
    for csv_path in sorted(base.glob("**/*.csv")):
        try:
            video_id = csv_path.parent.name
            user_id = csv_path.stem
            t = ViewportTrajectory.from_netllm_csv(csv_path, video_id=video_id, user_id=user_id)
            if len(t) >= 4:
                out.append(t)
        except Exception:
            continue
        if max_files and len(out) >= max_files:
            break
    if not out:
        raise FileNotFoundError(f"no usable .csv viewports under {base}")
    return out


def synth_dataset(n_users: int = 30, n_seconds: int = 60, sample_hz: float = 5.0,
                  seed: int = 0) -> List[ViewportTrajectory]:
    """Synthetic VP traces: 1/f-ish random walk on the sphere with a moving salient attractor.

    Used for CI / quick smoke when real Wu2017/Jin2022 isn't available.
    """
    rng = np.random.default_rng(seed)
    out: List[ViewportTrajectory] = []
    n_steps = int(n_seconds * sample_hz)
    times = np.arange(n_steps) / sample_hz
    for u in range(n_users):
        # salient attractor drifts slowly
        att_yaw = np.cumsum(rng.normal(0, 0.05, size=n_steps))
        att_pitch = 0.4 * np.sin(2 * np.pi * times * rng.uniform(0.05, 0.2) + rng.uniform(0, 6.28))
        # gaze tracks attractor with momentum + noise
        yaw = np.zeros(n_steps); pitch = np.zeros(n_steps)
        v_yaw = 0.0; v_pitch = 0.0
        for i in range(1, n_steps):
            v_yaw = 0.85 * v_yaw + 0.10 * (att_yaw[i] - yaw[i - 1]) + rng.normal(0, 0.03)
            v_pitch = 0.85 * v_pitch + 0.10 * (att_pitch[i] - pitch[i - 1]) + rng.normal(0, 0.02)
            yaw[i] = yaw[i - 1] + v_yaw
            pitch[i] = np.clip(pitch[i - 1] + v_pitch, -1.4, 1.4)
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        out.append(ViewportTrajectory(times, yaw, pitch, video_id=f"synth", user_id=f"user{u:03d}"))
    return out


# ----------------------------------------------------------------------------- #
# Metrics                                                                        #
# ----------------------------------------------------------------------------- #

def great_circle_mae_rad(pred_tile: np.ndarray, true_yaw: np.ndarray, true_pitch: np.ndarray,
                         H: int, W: int) -> float:
    """Mean great-circle distance (radians) between predicted tile center and ground truth."""
    yp = np.array([tile_to_yaw_pitch(int(t), H, W) for t in pred_tile])
    pred_yaw, pred_pitch = yp[:, 0], yp[:, 1]
    cos_d = (np.sin(pred_pitch) * np.sin(true_pitch) +
             np.cos(pred_pitch) * np.cos(true_pitch) * np.cos(pred_yaw - true_yaw))
    d = np.arccos(np.clip(cos_d, -1.0, 1.0))
    return float(d.mean())
