"""Pensieve-style Adaptive Bitrate (ABR) streaming environment.

State (per timestep, exposed as (T, F) for the ThroughputEncoder):
    For each of the last `history_len` chunks:
        [throughput_mbps, download_time_s, buffer_s, last_bitrate_idx_norm,
         chunks_left_norm, next_chunk_size_avg_kb_norm]

Reward (paper eq. 8): Q_t = q(R_t) - alpha * |q(R_t) - q(R_{t-1})| - beta * T_buf
where q(R) is the bitrate utility (linear-in-Mbps by default; switch to log via cfg).

Bandwidth traces: load Pensieve-format text traces (`time_s\tthroughput_mbps` per line)
or synthesize via a 2-state Markov-modulated process with random dropouts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


BITRATES_KBPS = [300, 750, 1200, 1850, 2850, 4300]
VIDEO_CHUNK_LEN_S = 4.0
TOTAL_CHUNKS = 48
BUFFER_THRESH_S = 60.0
DRAIN_BUFFER_SLEEP_S = 0.5
LINK_RTT_S = 0.080
PACKET_PAYLOAD_PORTION = 0.95
NOISE_LOW = 0.9
NOISE_HIGH = 1.1


@dataclass
class ABRTrace:
    times_s: np.ndarray
    bw_mbps: np.ndarray

    @classmethod
    def from_file(cls, path: str | Path) -> "ABRTrace":
        """
        从文件中读取数据并创建 ABRTrace 实例。

        参数:
            path (str | Path): 输入文件的路径，文件每行应包含至少两个由空白字符分隔的数值，
                               分别代表时间戳和带宽。

        返回:
            ABRTrace: 包含解析后的时间戳和带宽数据的 ABRTrace 实例。
        """
        ts, bw = [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                ts.append(float(parts[0]))
                bw.append(float(parts[1]))
        return cls(np.asarray(ts, dtype=np.float64), np.asarray(bw, dtype=np.float64))

    @classmethod
    def synth(cls, n_steps: int = 600, dt: float = 1.0, seed: int = 0) -> "ABRTrace":
        rng = np.random.default_rng(seed)
        times = np.arange(n_steps) * dt
        # 2-state Markov: high (~5 Mbps) / low (~1.5 Mbps) with sticky transitions, plus dropouts
        bw = np.empty(n_steps)
        state = rng.integers(0, 2)
        for i in range(n_steps):
            if rng.random() < 0.05:
                state = 1 - state
            mu = 5.0 if state == 1 else 1.5
            bw[i] = max(0.05, mu + rng.normal(0, 0.4))
            if rng.random() < 0.02:
                bw[i] = 0.05
        return cls(times, bw)


def _build_video_chunk_sizes(rng: np.random.Generator, n_bitrates: int = 6,
                             n_chunks: int = TOTAL_CHUNKS) -> np.ndarray:
    """chunk_kbits[bitrate, chunk_idx]; per-chunk size jitters around bitrate*chunk_len."""
    out = np.zeros((n_bitrates, n_chunks))
    for bi, br in enumerate(BITRATES_KBPS):
        base = br * VIDEO_CHUNK_LEN_S  # kbits per chunk on average
        out[bi] = base * rng.uniform(0.85, 1.20, size=n_chunks)
    return out


class ABREnv:
    """Single-stream ABR simulator with paper-faithful QoE reward."""

    def __init__(self, traces: Sequence[ABRTrace] | None = None,
                 history_len: int = 8,
                 alpha: float = 1.0, beta: float = 4.3,
                 utility: str = "linear",
                 seed: int = 0):
        self.history_len = history_len
        self.alpha = alpha
        self.beta = beta
        self.utility = utility
        self.rng = np.random.default_rng(seed)
        self.traces = list(traces) if traces else [ABRTrace.synth(seed=i) for i in range(8)]
        self._reset_internal_state()

    # ---- public API ----------------------------------------------------------

    @property
    def state_shape(self) -> Tuple[int, int]:
        return (self.history_len, 6)

    @property
    def num_actions(self) -> int:
        return len(BITRATES_KBPS)

    def reset(self) -> np.ndarray:
        self._reset_internal_state()
        return self._make_state()

    def step(self, action: int):
        assert 0 <= action < self.num_actions
        chunk_size_kbits = self.video_chunk_sizes[action, self.chunk_idx]
        chunk_size_bytes = chunk_size_kbits * 125.0  # kb -> bytes (1 kbit = 125 bytes)

        delay_s = self._download(chunk_size_bytes)

        # buffer dynamics
        rebuf_s = max(0.0, delay_s - self.buffer_s)
        self.buffer_s = max(0.0, self.buffer_s - delay_s) + VIDEO_CHUNK_LEN_S

        # cap buffer (sleep)
        sleep_s = 0.0
        while self.buffer_s > BUFFER_THRESH_S:
            sleep_s += DRAIN_BUFFER_SLEEP_S
            self.buffer_s -= DRAIN_BUFFER_SLEEP_S
            self._advance_trace_time(DRAIN_BUFFER_SLEEP_S)

        reward = self._qoe(action, rebuf_s)
        self.last_action = action
        self.chunk_idx += 1
        self.history_throughput_mbps.append((chunk_size_bytes * 8 / 1e6) / max(delay_s, 1e-6))
        self.history_download_s.append(delay_s)

        done = self.chunk_idx >= TOTAL_CHUNKS
        info = {"rebuf_s": rebuf_s, "sleep_s": sleep_s, "bitrate_kbps": BITRATES_KBPS[action],
                "buffer_s": self.buffer_s}
        return self._make_state(), reward, done, info

    # ---- internals -----------------------------------------------------------

    def _reset_internal_state(self):
        self.trace = self.rng.choice(self.traces)
        self.trace_idx = self.rng.integers(0, len(self.trace.times_s) - 1)
        self.trace_time_offset_s = 0.0
        self.video_chunk_sizes = _build_video_chunk_sizes(self.rng)
        self.chunk_idx = 0
        self.buffer_s = 0.0
        self.last_action = self.rng.integers(0, self.num_actions)
        self.history_throughput_mbps: List[float] = [0.0] * self.history_len
        self.history_download_s: List[float] = [0.0] * self.history_len

    def _advance_trace_time(self, dt: float):
        # advance the running pointer; wrap when we run out of trace
        self.trace_time_offset_s += dt
        seg_len = self.trace.times_s[(self.trace_idx + 1) % len(self.trace.times_s)] - \
                  self.trace.times_s[self.trace_idx]
        seg_len = max(seg_len, 1e-3)
        while self.trace_time_offset_s >= seg_len:
            self.trace_time_offset_s -= seg_len
            self.trace_idx = (self.trace_idx + 1) % len(self.trace.times_s)
            seg_len = self.trace.times_s[(self.trace_idx + 1) % len(self.trace.times_s)] - \
                      self.trace.times_s[self.trace_idx]
            seg_len = max(seg_len, 1e-3)

    def _download(self, chunk_size_bytes: float) -> float:
        """Walk through trace segments, deliver bytes at current bandwidth, return delay in s."""
        bytes_left = chunk_size_bytes
        delay_s = 0.0
        while bytes_left > 0:
            bw_mbps = self.trace.bw_mbps[self.trace_idx]
            bw_mbps *= self.rng.uniform(NOISE_LOW, NOISE_HIGH)
            throughput_Bps = bw_mbps * 1e6 / 8.0 * PACKET_PAYLOAD_PORTION
            seg_end_s = self.trace.times_s[(self.trace_idx + 1) % len(self.trace.times_s)]
            seg_start_s = self.trace.times_s[self.trace_idx]
            seg_remaining_s = max(seg_end_s - seg_start_s - self.trace_time_offset_s, 1e-3)

            packet_payload = throughput_Bps * seg_remaining_s
            if packet_payload >= bytes_left:
                used_s = bytes_left / max(throughput_Bps, 1e-6)
                delay_s += used_s
                self._advance_trace_time(used_s)
                bytes_left = 0.0
            else:
                delay_s += seg_remaining_s
                self._advance_trace_time(seg_remaining_s)
                bytes_left -= packet_payload
        return delay_s + LINK_RTT_S

    def _qoe(self, action: int, rebuf_s: float) -> float:
        R = BITRATES_KBPS[action] / 1000.0  # Mbps
        R_prev = BITRATES_KBPS[self.last_action] / 1000.0
        if self.utility == "log":
            q = float(np.log(max(R, 1e-3)))
            q_prev = float(np.log(max(R_prev, 1e-3)))
        else:
            q = R
            q_prev = R_prev
        # paper eq. 8: q(R_t) - alpha * |q(R_t) - q(R_{t-1})| - beta * T_buf
        return q - self.alpha * abs(q - q_prev) - self.beta * rebuf_s

    def _make_state(self) -> np.ndarray:
        T = self.history_len
        s = np.zeros((T, 6), dtype=np.float32)
        thr = self.history_throughput_mbps[-T:]
        dl = self.history_download_s[-T:]
        if self.chunk_idx >= TOTAL_CHUNKS:
            avg_next_size_kb = 0.0
        else:
            avg_next_size_kb = float(self.video_chunk_sizes[:, self.chunk_idx].mean())
        chunks_left_norm = max(0.0, (TOTAL_CHUNKS - self.chunk_idx) / TOTAL_CHUNKS)
        last_bitrate_norm = self.last_action / max(1, self.num_actions - 1)
        for i in range(T):
            s[i, 0] = thr[i] / 10.0
            s[i, 1] = dl[i]
            s[i, 2] = self.buffer_s / BUFFER_THRESH_S
            s[i, 3] = last_bitrate_norm
            s[i, 4] = chunks_left_norm
            s[i, 5] = avg_next_size_kb / 1000.0
        return s
