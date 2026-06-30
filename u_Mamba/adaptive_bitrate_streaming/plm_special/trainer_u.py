import numpy as np
import torch
import time

from munch import Munch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from plm_special.utils.utils import process_batch


class Trainer:
    def __init__(self, args, model, qf, q_info,optimizer, exp_dataset, loss_fn, device, batch_size=1, grad_accum_steps=1, lr_scheduler=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.qf = qf
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
        self.alpha_kl = args.alpha_kl  # KL损失权重，默认1.0
        self.kl_div = torch.nn.KLDivLoss(reduction='batchmean')  # KL散度损失函数
    def train_epoch(self, report_loss_per_steps=100):
        train_losses = []
        weighted_action_losses = []
        layer_kl_losses = []
        q_losses = []
        logs = dict()

        train_start = time.time()
        dataset_size = len(self.dataloader)

        self.model.train()
        for step, batch in enumerate(self.dataloader):

            train_loss, weighted_action_loss, layer_kl_loss, q_loss = self.train_step(batch)
            train_losses.append(train_loss.item())
            weighted_action_losses.append(weighted_action_loss.item())
            q_losses.append(q_loss.item())
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
                mean_weighted_action_loss = np.mean(weighted_action_losses)
                mean_layer_kl_loss = np.mean(layer_kl_losses)
                mean_q_loss = np.mean(q_losses)
                print(f'Step {step:<4} -train loss:  {mean_train_loss:.4f} -action loss: {mean_weighted_action_loss:.4f} -layer KL loss: {mean_layer_kl_loss:.4f} -Q loss: {mean_q_loss:.4f}')

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        return logs, train_losses

    def train_step(self, batch):
        states, actions, returns, timesteps, labels, traj_upbounds, traj_rewards = process_batch(batch, device=self.device)
        B, T, s_info,s_len = states.shape
        A = 8
        #计算经验池中s，a对应的q值
        q1, q2 = self.qf(states.reshape(B,T, -1), F.one_hot(labels.squeeze(-1), num_classes=A).float().reshape(B,T,A))
        q_exp_pool = torch.min(q1, q2)  # (B, T, 1)

        teacher_action_pred, student_action_pred, out=\
            self.model(states, actions, returns, timesteps)

        # 1.分别计算教师模型和学生模型的损失
        teacher_action_pred = teacher_action_pred.permute(0, 2, 1)
        student_action_pred = student_action_pred.permute(0, 2, 1)

        loss_teacher = self.loss_fn(teacher_action_pred, labels)
        loss_student = self.loss_fn(student_action_pred, labels)
        pi= F.softmax(student_action_pred, dim=-1)  # (B, T, A)
        # 2.计算加权的 Q 值损失
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
            # 变回矩阵形状 (B*T, A)
            # 每一行代表一个 state 在 8 个动作上的 Q 值
            q_all_matrix = q_all.reshape(states_flat.shape[0], A)
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

        # 3.3 计算 Q Loss (最大化 Q -> 最小化 -Q)
        # 归一化: 除以数据集平均 Q 值绝对值 (防止 Q Loss 梯度过大主导 Action Loss)
        # qloss = - expected_q / (self.qloss_mean)  # (B*T,)        
        q_loss = -normalized_expected_q
        q_loss = q_loss.reshape(B, T) # 变回 (B, T)以便和 w2 结合

        # 计算权重 w1 (Trajectory Quality)
        current_returns = traj_rewards.squeeze(-1) 
        target_returns = traj_upbounds.squeeze(-1)
        # print("reward:",current_returns)
        # print("dp_upbound:",target_returns)
        # 2. 计算归一化的轨迹回报差距
        min_reward=0.  # 假设最小回报为 0
        normalized_returns = (current_returns-min_reward) / (target_returns-min_reward + 1e-8)  # 假设最小回报为 0
        normalized_returns=torch.clamp(normalized_returns, min=0.0001)
        # 用reward计算权重
        normalized_max_return= 0.8  # 可以调整这个值
        # 计算加权的 Q Loss
        w1=(normalized_returns+1.0-normalized_max_return)
        weighted_action_loss = (w1 * loss_student).mean()
        # 3.计算逐层隐藏状态的 KL 散度损失
        # 假设 teacher_out.hidden_states 和 student_out.hidden_states 的长度相同
        # 如果层数不一致，需要自己做好层的对齐和映射
        layer_kl_loss = 0.0

        # 通常 hidden_states[0] 是 embedding 输出, hidden_states[-1] 是最后一层输出
        # 大多数情况下会对中间的几层做蒸馏(可以自己控制哪些层做对齐)
        t_states = out['teacher_outputs'].hidden_states
        s_states = out['student_outputs'].hidden_states

        num_t = len(t_states)    # 33
        num_s = len(s_states)    # 10

        # 均匀采样教师的对应层
        selected_t_indices = [
            round((i + 1) * num_t / (num_s + 1)) - 1  # -1 因为从0开始
            for i in range(num_s)
        ]
        for i, s_hidden in enumerate(s_states):
            # 对每一层的隐藏状态计算 KL 散度
            #先对每一层执行
            t_hidden = t_states[selected_t_indices[i]]
            # print(f't_hidden shape: {t_hidden.shape}, s_hidden shape: {s_hidden.shape}')

            # 如果使用 residual 连接，则先加上输入的 layer norm 后再计算 KL 散度
            if out['residual']:
                t_hidden = t_hidden + out['stacked_inputs_ln']
                s_hidden = s_hidden + out['mamba_stacked_inputs_ln']
            # print(f'After residual add - t_hidden shape: {t_hidden.shape}, s_hidden shape: {s_hidden.shape}')

            # 形状：[batch_size, seq_len, hidden_size]
            t_hidden_used = t_hidden[:,out['mamba_action_embed_positions']-2]  # 只取动作位置的隐藏状态
            s_hidden_used = s_hidden[:,out['mamba_action_embed_positions']-2]  # 只取动作位置的隐藏状态
            # print(f'After indexing action positions - t_hidden_used shape: {t_hidden_used.shape}, s_hidden_used shape: {s_hidden_used.shape}')
            
            t_action_pred = out['t_action_head'](t_hidden_used)
            s_action_pred = out['s_action_head'](s_hidden_used)
            # print(f'Action predictions - t_action_pred shape: {t_action_pred.shape}, s_action_pred shape: {s_action_pred.shape}')
            

            # 计算动作预测的 KL 散度
            # 这里我们先对动作预测做一个简单的线性变换，得到概率分布
            # KLDivLoss 需要 log_prob vs prob
            # 所以对学生做 log_softmax，对教师做 softmax
            s_log_prob = F.log_softmax(s_action_pred, dim=-1)
            t_prob = F.softmax(t_action_pred, dim=-1)
            layer_kl_loss += self.kl_div(s_log_prob, t_prob)
      
        q_scale = 4.0
        # 总loss
        # student_loss = weighted_action_loss + 1.0 * layer_kl_loss+ q_scale*q_loss.mean()
        # student_loss = weighted_action_loss + q_scale*q_loss.mean()
        weighted_action_loss=loss_student.mean()
        student_loss = weighted_action_loss + q_scale*q_loss.mean()

        return student_loss,weighted_action_loss, 0, q_loss.mean()
