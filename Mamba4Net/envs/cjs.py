"""Decima-style Cluster Job Scheduling (CJS) environment.

Each Job is a DAG of Stages; each Stage has `n_tasks` tasks each lasting
`task_duration_s`. A pool of executors runs one task at a time. The scheduler is
asked, at every event (task finish or job arrival), to pick the next *ready*
stage to assign one freed executor to. Reward = -delta_t * (#active jobs),
which is the per-step contribution to total job completion time.

State (NN-facing):
    node_feats : (max_nodes, F=8)  -- per-stage features:
        [tasks_remaining_norm, task_dur_norm, parents_done_frac, ready_flag,
         num_parents_norm, num_children_norm, age_in_system_norm, exec_running_norm]
    adj        : (max_nodes, max_nodes)  -- parent->child binary
    node_mask  : (max_nodes,) bool       -- True for "scheduleable now" stages
    urgency    : (max_nodes, 1)          -- u_i in paper eq. 9 (we use 1/(parents_done_frac+eps))
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class Stage:
    n_tasks: int
    task_duration_s: float
    parents: List[int] = field(default_factory=list)  # local indices within job
    completed_tasks: int = 0
    running_tasks: int = 0

    @property
    def remaining(self) -> int:
        return self.n_tasks - self.completed_tasks - self.running_tasks


@dataclass
class Job:
    stages: List[Stage]
    arrival_time_s: float = 0.0
    completion_time_s: float = 0.0


def _gen_dag_topology(n_stages: int, rng: np.random.Generator) -> List[List[int]]:
    """Random DAG: stage i can have parents only among 0..i-1. Mix chain/fan-out/tree."""
    parents = [[] for _ in range(n_stages)]
    for i in range(1, n_stages):
        n_p = max(1, int(rng.integers(1, min(3, i) + 1)))
        choices = rng.choice(i, size=n_p, replace=False)
        parents[i] = sorted(int(c) for c in choices)
    return parents


def _gen_job(rng: np.random.Generator, max_stages: int) -> Job:
    n_stages = int(rng.integers(2, max_stages + 1))
    parents = _gen_dag_topology(n_stages, rng)
    stages = []
    for s in range(n_stages):
        stages.append(Stage(
            n_tasks=int(rng.integers(1, 6)),
            task_duration_s=float(rng.uniform(0.5, 4.0)),
            parents=parents[s],
        ))
    return Job(stages=stages)


class CJSEnv:
    """Discrete-event DAG scheduler. One executor commitment per step."""

    def __init__(self, n_executors: int = 8, n_jobs: int = 4, max_stages_per_job: int = 6,
                 max_nodes: int = 64, seed: int = 0):
        self.n_executors = n_executors
        self.n_jobs = n_jobs
        self.max_stages = max_stages_per_job
        self.max_nodes = max_nodes
        self.rng = np.random.default_rng(seed)
        self._reset_state()

    # ---- public API ----------------------------------------------------------

    @property
    def node_feature_dim(self) -> int:
        return 8

    def reset(self):
        self._reset_state()
        return self._make_obs()

    def step(self, action: int):
        """`action` is a global node index in 0..max_nodes-1; must be in node_mask=True.

        If nothing is schedulable right now, auto-advance to the next event
        (executor finish or job arrival) instead of returning a stuck no-op.
        """
        nodes = self._global_nodes()
        any_ready = any(self._is_ready(n) for n in nodes)

        if not any_ready:
            prev_t = self.now_s
            self._advance_to_next_event()
            if self.now_s == prev_t:
                # truly nothing left to do (defensive: shouldn't happen if not done)
                return self._make_obs(), 0.0, self._all_done(), {"idle": True, "stuck": True}
            delta_t = self.now_s - prev_t
            active = sum(1 for j in self.jobs if j.completion_time_s == 0.0 and j.arrival_time_s <= self.now_s)
            return self._make_obs(), -delta_t * active, self._all_done(), {"idle": True}

        if action >= len(nodes) or not self._is_ready(nodes[action]):
            # invalid action when valid options exist -> shaping penalty, no-op
            return self._make_obs(), -1.0, self._all_done(), {"invalid": True}

        job_id, local_id = nodes[action]
        stage = self.jobs[job_id].stages[local_id]

        # commit one free executor: schedule a single task; advance time to next event
        free_exec_id = self._take_free_executor()
        finish_t = self.now_s + stage.task_duration_s
        self._executor_busy_until[free_exec_id] = finish_t
        self._executor_assigned_to[free_exec_id] = (job_id, local_id)
        stage.running_tasks += 1

        prev_t = self.now_s
        self._advance_to_next_event()
        delta_t = self.now_s - prev_t
        active_jobs = sum(1 for j in self.jobs if j.completion_time_s == 0.0 and j.arrival_time_s <= self.now_s)
        reward = -delta_t * active_jobs

        done = self._all_done()
        info = {"makespan_s": self.now_s, "active_jobs": active_jobs}
        return self._make_obs(), reward, done, info

    # ---- internals -----------------------------------------------------------

    def _reset_state(self):
        self.jobs: List[Job] = []
        for k in range(self.n_jobs):
            j = _gen_job(self.rng, self.max_stages)
            j.arrival_time_s = float(k) * float(self.rng.uniform(0.0, 2.0))
            self.jobs.append(j)
        self.now_s: float = 0.0
        self._executor_busy_until = np.zeros(self.n_executors)
        self._executor_assigned_to: List[Tuple[int, int] | None] = [None] * self.n_executors
        # advance clock to first job arrival
        first_arrival = min(j.arrival_time_s for j in self.jobs)
        self.now_s = first_arrival

    def _global_nodes(self) -> List[Tuple[int, int]]:
        nodes: List[Tuple[int, int]] = []
        for ji, job in enumerate(self.jobs):
            if job.arrival_time_s > self.now_s:
                continue
            if job.completion_time_s > 0.0:
                continue
            for si, stage in enumerate(job.stages):
                if stage.completed_tasks >= stage.n_tasks:
                    continue
                nodes.append((ji, si))
                if len(nodes) >= self.max_nodes:
                    return nodes
        return nodes

    def _is_ready(self, node: Tuple[int, int]) -> bool:
        ji, si = node
        stage = self.jobs[ji].stages[si]
        if stage.remaining <= 0:
            return False
        if not all(self.jobs[ji].stages[p].completed_tasks >= self.jobs[ji].stages[p].n_tasks
                   for p in stage.parents):
            return False
        return self._has_free_executor()

    def _has_free_executor(self) -> bool:
        return any(t <= self.now_s for t in self._executor_busy_until)

    def _take_free_executor(self) -> int:
        for i, t in enumerate(self._executor_busy_until):
            if t <= self.now_s:
                return i
        raise RuntimeError("no free executor")

    def _advance_to_next_event(self):
        # next event = earliest of (busy executor finishes) or (new job arrives)
        future_arrivals = [j.arrival_time_s for j in self.jobs if j.arrival_time_s > self.now_s]
        future_finishes = [t for t in self._executor_busy_until if t > self.now_s]
        # if we still have free executors and at least one schedulable stage, don't advance time
        if self._has_free_executor() and len(self._global_nodes()) > 0 and \
                any(self._is_ready(n) for n in self._global_nodes()):
            return
        candidates = future_arrivals + future_finishes
        if not candidates:
            return
        next_t = min(candidates)
        self.now_s = next_t
        # complete tasks whose finish time has arrived
        for i, t in enumerate(self._executor_busy_until):
            if t <= self.now_s and self._executor_assigned_to[i] is not None:
                ji, si = self._executor_assigned_to[i]
                stage = self.jobs[ji].stages[si]
                stage.completed_tasks += 1
                stage.running_tasks = max(0, stage.running_tasks - 1)
                self._executor_assigned_to[i] = None
                # job done?
                job = self.jobs[ji]
                if all(s.completed_tasks >= s.n_tasks for s in job.stages) and job.completion_time_s == 0.0:
                    job.completion_time_s = self.now_s

    def _all_done(self) -> bool:
        return all(j.completion_time_s > 0.0 for j in self.jobs)

    def _make_obs(self):
        N = self.max_nodes
        F = self.node_feature_dim
        feats = np.zeros((N, F), dtype=np.float32)
        adj = np.zeros((N, N), dtype=np.float32)
        mask = np.zeros((N,), dtype=bool)
        urgency = np.zeros((N, 1), dtype=np.float32)

        nodes = self._global_nodes()
        idx_of: dict[Tuple[int, int], int] = {n: i for i, n in enumerate(nodes)}
        for i, (ji, si) in enumerate(nodes):
            job = self.jobs[ji]
            stage = job.stages[si]
            n_p = len(stage.parents)
            n_c = sum(1 for s in job.stages if si in s.parents)
            parents_done = sum(1 for p in stage.parents
                               if job.stages[p].completed_tasks >= job.stages[p].n_tasks)
            parents_done_frac = parents_done / max(1, n_p) if n_p > 0 else 1.0
            ready = 1.0 if (parents_done_frac >= 1.0 and stage.remaining > 0) else 0.0
            feats[i] = [
                stage.remaining / max(1, stage.n_tasks),
                stage.task_duration_s / 5.0,
                parents_done_frac,
                ready,
                n_p / 5.0,
                n_c / 5.0,
                (self.now_s - job.arrival_time_s) / 30.0,
                stage.running_tasks / 5.0,
            ]
            mask[i] = bool(ready) and self._has_free_executor()
            urgency[i, 0] = 1.0 / (1.0 + stage.task_duration_s)
            for p in stage.parents:
                if (ji, p) in idx_of:
                    adj[idx_of[(ji, p)], i] = 1.0

        return {"node_feats": feats, "adj": adj, "node_mask": mask, "urgency": urgency}

    # ---- diagnostics ---------------------------------------------------------

    def avg_jct(self) -> float:
        completed = [j.completion_time_s - j.arrival_time_s for j in self.jobs if j.completion_time_s > 0]
        return float(np.mean(completed)) if completed else 0.0
