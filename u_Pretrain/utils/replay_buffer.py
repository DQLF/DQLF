"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import pickle
import os
import json

class EpisodeReplayBuffer(object):
    def __init__(self, capacity, trajectories=[]):
        self.capacity = capacity
        if len(trajectories) <= self.capacity:
            self.trajectories = trajectories
        else:
            returns = [traj["rewards"].sum() for traj in trajectories]
            sorted_inds = np.argsort(returns)  # lowest to highest
            self.trajectories = [
                trajectories[ii] for ii in sorted_inds[-self.capacity :]
            ]

        self.start_idx = 0

    def __len__(self):
        return len(self.trajectories)

    def add_new_trajs(self, new_trajs):
        if len(self.trajectories) < self.capacity:
            self.trajectories.extend(new_trajs)
            self.trajectories = self.trajectories[-self.capacity :]
        else:
            self.trajectories[
                self.start_idx : self.start_idx + len(new_trajs)
            ] = new_trajs
            self.start_idx = (self.start_idx + len(new_trajs)) % self.capacity

        assert len(self.trajectories) <= self.capacity

pool_paths = {
        "bba": '/home/ubuntu/kevin/q_pretrain/check_q/exp_pool/train/bba/seed_42_trace_num_-1_fixed_False/bba.pkl',
        "mpc": '/home/ubuntu/kevin/q_pretrain/check_q/exp_pool/train/mpc/seed_42_trace_num_-1_fixed_False/mpc.pkl',
        "genet": '/home/ubuntu/kevin/q_pretrain/check_q/exp_pool/train/genet/seed_42_trace_num_-1_fixed_False/genet.pkl',
        "pensieve": '/home/ubuntu/kevin/q_pretrain/check_q/exp_pool/train/pensieve/seed_42_trace_num_-1_fixed_False/pensieve.pkl',
        "merina": '/home/ubuntu/kevin/q_pretrain/check_q/exp_pool/train/merina/seed_42_trace_num_-1_fixed_False/merina.pkl',
    }
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.rtg = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.dp = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    def sample_use_rtg(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.rtg[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    def sample_use_dp(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.rtg[ind]).to(self.device),
            torch.FloatTensor(self.dp[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
        
    def sample_sarsa(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.next_action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    
    def convert_abr(self, exp_pool):
        dp_path='/home/ubuntu/kevin/q_pretrain/check_q/seg_dp_all_trace'

        algo = ['bba', 'mpc', 'genet', 'pensieve', 'merina']
        k=0
        for algo_name in algo:
            with open(f'{dp_path}/{algo_name}_all.json', 'r') as f:
                dp = json.load(f)
            pool = pickle.load(open(pool_paths[algo_name], 'rb'))
            num_trajs = len(pool.states) // 47
            for i in range(num_trajs):
                if i % 1 == 0:
                    step=k+i//1
                    print(f"[Progress] {i}/{num_trajs} for {algo_name}",end='\r')
                    state = pool.states[i*47:(i+1)*47]
                    state = np.array(state).reshape(len(state), -1)
                    self.state[step*47:(step+1)*47] = state
                    action = np.array(pool.actions[i*47:(i+1)*47]) # (19928,)
                    actions_onehot = np.eye(8)[action]  # (19928, 8)
                    self.action[step*47:(step+1)*47] = actions_onehot
                    self.reward[step*47:(step+1)*47] = np.array(pool.rewards[i*47:(i+1)*47]).reshape(-1, 1)
                    self.not_done[step*47:(step+1)*47] = 1. - np.array(pool.dones[i*47:(i+1)*47]).reshape(-1, 1)
                    self.size = (step+1)*47

                    # 构造 next_state：将当前状态往后一位平移
                    self.next_state = np.zeros_like(self.state)
                    self.next_state[:-1] = self.state[1:]
                    # 如果当前 step 是终止状态，则 next_state 仍然保留为0或state本身
                    dones = np.array(pool.dones[i*47:(i+1)*47]).reshape(-1)
                    for j in range(len(dones) - 1):
                        if dones[j]:
                            self.next_state[step*47+j] = 0#self.state[i]  # 或者置为0
                    traj_name = pool.traj_names[i*47]
                    traj_name = traj_name+f"_{i}"
                    traj_dp=dp[traj_name]
                    # self.dp.append(np.array(traj_dp).reshape(-1, 1))
                    self.dp[step*47:(step+1)*47] = np.array(traj_dp).reshape(-1, 1)
            k=step+1

        #计算rtg
        rtg = np.zeros_like(self.reward)
        running_return = 0
        for j in reversed(range(self.size)):
            if not self.not_done[j]:
                running_return = 0
            running_return += self.reward[j]
            rtg[j] = running_return
        self.rtg = rtg.reshape(-1, 1)

    def convert_abr_use_reward(self, exp_pool):
        self.state = np.array(exp_pool.states).reshape(len(exp_pool.states), -1)  # (19928, 48)
        action = np.array(exp_pool.actions)  # (19928,)
        actions_onehot = np.eye(8)[action]  # (19928, 8)
        self.action = actions_onehot  # (19928, 8)
        
        # 奖励处理
        rewards = np.array(exp_pool.rewards).reshape(-1) # 先转为1维方便计算
        self.reward = rewards.reshape(-1, 1)
        
        dones = np.array(exp_pool.dones).reshape(-1)
        self.not_done = 1. - dones.reshape(-1, 1)
        self.size = self.state.shape[0]

        # --- 计算 RTG (Reward-to-Go) ---
        rtg = np.zeros_like(rewards)
        running_return = 0
        # 从后往前遍历
        for i in reversed(range(self.size)):
            if dones[i]:
                running_return = 0 # 如果当前是终止状态，重置累积奖励
            running_return += rewards[i]
            rtg[i] = running_return
        
        self.rtg = rtg.reshape(-1, 1) # (19928, 1)
        # ------------------------------

        # 构造 next_state
        self.next_state = np.zeros_like(self.state)
        self.next_state[:-1] = self.state[1:]
        for i in range(len(dones) - 1):
            if dones[i]:
                self.next_state[i] = self.state[i]
    
