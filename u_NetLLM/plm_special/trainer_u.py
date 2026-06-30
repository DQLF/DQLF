import numpy as np
import torch
import time

from munch import Munch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from plm_special.utils.utils import process_batch
import random
from baseline_special.utils.constants import BITRATE_LEVELS
class Trainer_q:
    def __init__(self, args, model,qf,qf_dp,vf, q_info,optimizer, exp_dataset, loss_fn, device, batch_size=1, grad_accum_steps=1, lr_scheduler=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.qf = qf
        self.qf_dp = qf_dp
        self.vf = vf
        self.q_info = q_info
        self.qloss_mean = q_info["q_loss_mean"]

        self.exp_dataset = exp_dataset
        self.loss_fn = loss_fn
        self.device = device
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.lr_scheduler = lr_scheduler
        
        self.exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
        self.dataloader = DataLoader(exp_dataset, batch_size, shuffle=True, pin_memory=True)

    def train_epoch(self, report_loss_per_steps=200):
        train_losses = []
        action_losses = []
        w1s=[]
        w2s=[]
        q_losses = []
        weighted_q_losses = []
        logs = dict()

        train_start = time.time()
        dataset_size = len(self.dataloader)

        self.model.train()
        time_step=time.time()
        for step, batch in enumerate(self.dataloader):

            train_loss, action_loss,w1, q_loss = self.train_step(batch)
            # print(train_loss, action_loss, q_loss,weighted_q_loss,weighted_action_loss)
            # exit()
            #train_loss = self.train_step(batch)
            train_losses.append(train_loss.item())
            action_losses.append(action_loss.item())
            # w2s.append(w2.item())
            w1s.append(w1.item())
            q_losses.append(q_loss.item())
            # weighted_q_losses.append(q_loss.item())

            #debug
            # devices = set(p.device for p in self.model.parameters())
            # dtypes = set(p.dtype for p in self.model.parameters())
            # print("Devices:", devices)
            # print("Dtypes:", dtypes)
            # exit()

            # perform gradient accumulation update
            train_loss = train_loss / self.grad_accum_steps
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            if ((step + 1) % self.grad_accum_steps == 0) or (step + 1 == dataset_size):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if step % report_loss_per_steps == 0:                
                mean_train_loss = np.mean(train_losses)
                mean_w1 = np.mean(w1s)
                # mean_w2 = np.mean(w2s)
                mean_q_loss = np.mean(q_losses)
                # mean_w_q_loss = np.mean(weighted_q_losses)
                mean_action_loss = np.mean(action_losses)
                print(f'Step {step:>4} loss:{mean_train_loss:.4f} \taction_loss:{mean_action_loss:.4f} \t w1:{mean_w1:.4f}\t q_loss:{mean_q_loss:.4f} \t time {time.time()-time_step:.1f}s')
                time_step=time.time()

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/action_loss_mean'] = np.mean(action_losses)
        logs['training/action_loss_std'] = np.std(action_losses)
        logs['training/q_loss_mean'] = np.mean(q_losses)
        logs['training/q_loss_std'] = np.std(q_losses)
        logs['training/weighted_q_loss_mean'] = np.mean(weighted_q_losses)
        logs['training/weighted_q_loss_std'] = np.std(weighted_q_losses)

        return logs, train_losses

    def train_step(self, batch):
        states, actions, returns, timesteps, labels, traj_upbounds, traj_rewards = process_batch(batch, device=self.device)
        B, T, s_info,s_len = states.shape
        A = BITRATE_LEVELS
        # print(states.shape, actions.shape, returns.shape, timesteps.shape, labels.shape, reward_upbound.shape, traj_rewards.shape)
        # #torch.Size([1, 10, 6, 8]) torch.Size([1, 10, 1]) torch.Size([1, 10, 1]) torch.Size([1, 10]) torch.Size([1, 10]) torch.Size([1, 10, 1]) torch.Size([1, 10, 1])
        # exit()

        #计算经验池中s，a对应的q值
        q1, q2 = self.qf(states.reshape(B,T, -1), F.one_hot(labels.squeeze(-1), num_classes=A).float().reshape(B,T,A))
        q_exp_pool = torch.min(q1, q2)  # (B, T, 1)
        # print("q_values:",q_values.squeeze(-1))
         # ------------------------------------------------------------
        # Step 1: LLM 前向传播 (Policy Forward)
        # ------------------------------------------------------------
        actions_pred = self.model(states, actions, returns, timesteps) # (B, T, A)
        # 得到概率分布 pi (用于计算期望 Q 值，保留梯度)
        pi = F.softmax(actions_pred, dim=-1) # (B, T, A)

        # ------------------------------------------------------------
        # Step 2: 计算 Action Loss (RCSL / 模仿学习)
        # ------------------------------------------------------------  
        # 展平以适配 CrossEntropyLoss
        actions_pred_flat = actions_pred.reshape(-1, A) # (B*T, A)
        labels_flat = labels.reshape(-1)                # (B*T,)

        action_loss = self.loss_fn(actions_pred_flat, labels_flat) # (B*T,)
        action_loss = action_loss.reshape(B, T) # 变回 (B, T)

        # ------------------------------------------------------------
        # Step 3: 计算 Q-Aid Loss (RL / 缝合修正)
        # ------------------------------------------------------------
        # 逻辑：利用 Q 网络计算所有动作的 Q 值，引导 LLM 偏向高 Q 值的动作
        
        # 3.1 构造所有可能的动作输入 (Broadcasting)
        # 将 state 重复 A 次
        states_flat = states.reshape(B,T, -1) # (B, T, s_info, s_len)--> (B, T, s_info*s_len)
        states_flat = states_flat.reshape(-1, states_flat.shape[-1]) # (B*T, s_info*s_len)
        states_repeated = states_flat.repeat_interleave(A, dim=0) # (B*T*A, s_info*s_len)
        
        # 构造对应的动作 one-hot: (0, 1, ... 7, 0, 1, ... 7)
        all_actions_idx = torch.arange(A, device=self.device).repeat(states_flat.shape[0])
        all_actions_onehot = F.one_hot(all_actions_idx, num_classes=A).float() # (B*T*A, A)
        
        with torch.no_grad():
            # 并行计算所有动作的 Q 值 (只需调用这一次 Q 网络)

            q1_all, q2_all = self.qf(states_repeated, all_actions_onehot)
            q_all = torch.min(q1_all, q2_all) # (B*T*A, 1)
            #==========q_dp==========
            q1dp_all, q2dp_all = self.qf_dp(states_repeated, all_actions_onehot)
            qdp_all = torch.min(q1dp_all, q2dp_all) # (B*T*A, 1)
            qdp_all = torch.sigmoid(torch.tensor(qdp_all))
            #============================
            
            # 变回矩阵形状 (B*T, A)
            # 每一行代表一个 state 在 8 个动作上的 Q 值
            q_all_matrix = q_all.reshape(states_flat.shape[0], A)
            #==========q_dp==========
            qdp_all_matrix = qdp_all.reshape(states_flat.shape[0], A)
            #============================

        # 3.2 计算期望 Q 值 (Expected Q)
        # sum( prob(a) * Q(s,a) )
        # 这一步保证了梯度可以传回 LLM，告诉它提高高 Q 值动作的概率
        pi_flat = pi.reshape(-1, A) #(B, T, A) -> (B*T, A)
        expected_q = torch.sum(pi_flat * q_all_matrix, dim=-1) # (B*T,)
        #针对每个state计算minmax归一化的expected q值
        # 1获取每行（每个state）在所有动作上的 最小值 和 最大值
        # torch.min/max 返回 (values, indices)，我们只需要 values ([0])
        min_q = torch.min(q_all_matrix, dim=-1)[0] # 形状 (B*T,)
        max_q = torch.max(q_all_matrix, dim=-1)[0] # 形状 (B*T,)
        normalized_expected_q=(expected_q - min_q) / (max_q - min_q )  # 形状 (B*T,)
        #==========q_dp==========
        normalized_expected_q_dp=torch.sum(pi_flat * qdp_all_matrix, dim=-1) # (B*T,)
        #============================
        
        # 3.3 计算 Q Loss (最大化 Q -> 最小化 -Q)
        # 归一化: 除以数据集平均 Q 值绝对值 (防止 Q Loss 梯度过大主导 Action Loss)
        q_loss = -normalized_expected_q
        # q_loss = q_loss.reshape(B, T) # 变回 (B, T)以便和 w2 结合

        #==========q_dp==========
        qdp_loss = -normalized_expected_q_dp
        qloss = qdp_loss.reshape(B,T) 
        #============================
        

        # ------------------------------------------------------------
        # Step 4: 计算权重 w1 (Trajectory Quality)
        # ------------------------------------------------------------
        # 逻辑：轨迹越差 (距离 DP 上界越远)，w2 越大，越依赖 Q Loss 进行修正
        # 输入形状检查: traj_rewards 和 traj_upbounds 均为 (B, T, 1)

        # 1. 维度调整: (B, T, 1) -> (B, T)
        # 这一步是为了和 q_loss_elementwise (B, T) 维度对齐
        current_returns = traj_rewards.squeeze(-1) 
        target_returns = traj_upbounds.squeeze(-1)
        # print("reward:",current_returns)
        # print("dp_upbound:",target_returns)
        # 2. 计算归一化的轨迹回报差距
        min_reward=0.  # 假设最小回报为 0
        normalized_returns = (current_returns-min_reward) / (target_returns-min_reward + 1e-8)  # 假设最小回报为 0
        normalized_returns=torch.clamp(normalized_returns, min=0.0001)
        #用q值计算权重
        #计算经验池中s，a对应的q值

        exp_pool_max_q=self.q_info["max_q"]
        exp_pool_min_q=self.q_info["min_q"]
        normalized_q=(q_exp_pool.reshape(-1)-min_q)/(max_q - min_q)
        normalized_q=normalized_q.reshape(B,T)
        normalized_q=torch.clamp(normalized_q, min=0.0001)

        # print(normalized_returns)
        # 3. 计算 w2
        q_scale = self.args.q_scale
        
        # w2 = q_scale * (normalized_max_return - normalized_returns) * 100  # (B, T)
        # w2 = torch.clamp(w2, min=min_q)
        # print("w2:",w2)
        # print(q_loss)
        
        # weighted_q_loss = (w2 * q_loss).mean()

        # 3. 计算 w1
        # 用reward计算权重
        normalized_max_return= self.args.normalized_max_return
        w1=(normalized_returns+1.0-normalized_max_return)
        # weighted_action_loss = (w1 * action_loss).mean()

        #用q值计算权重w1
        w11=(normalized_q)
        weighted_action_loss = (w11 * action_loss).mean()
        
        # 总loss
        # loss=q_scale*q_loss.mean()
        loss= weighted_action_loss + q_scale*q_loss.mean()
        # loss = action_loss.mean() + weighted_q_loss

        return loss,weighted_action_loss,w11.mean(), q_loss.mean()
