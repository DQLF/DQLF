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

            train_losses.append(train_loss.item())

            w1s.append(w1.item())
            q_losses.append(q_loss.item())

            train_loss = train_loss / self.grad_accum_steps
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            if ((step + 1) % self.grad_accum_steps == 0) or (step + 1 == dataset_size):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

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

        q1, q2 = self.qf(states.reshape(B,T, -1), F.one_hot(labels.squeeze(-1), num_classes=A).float().reshape(B,T,A))
        q_exp_pool = torch.min(q1, q2)  #

        actions_pred_flat = actions_pred.reshape(-1, A) #
        labels_flat = labels.reshape(-1)                # 

        action_loss = self.loss_fn(actions_pred_flat, labels_flat) # 
        action_loss = action_loss.reshape(B, T) #

        states_flat = states.reshape(B,T, -1) # 
        states_flat = states_flat.reshape(-1, states_flat.shape[-1]) # 

        pi_flat = pi.reshape(-1, A) #
        expected_q = torch.sum(pi_flat * qdp_all_matrix, dim=-1) #

        min_q = torch.min(qdp_all_matrix, dim=-1)[0] # 
        max_q = torch.max(qdp_all_matrix, dim=-1)[0] 
        normalized_expected_q=(expected_q - min_q) / (max_q - min_q )  

        normalized_expected_q_dp=torch.sum(pi_flat * qdp_all_matrix, dim=-1) 
        q_loss = -normalized_expected_q
        qdp_loss = -normalized_expected_q_dp
        qloss = qdp_loss.reshape(B,T) 

        current_returns = traj_rewards.squeeze(-1) 
        target_returns = traj_upbounds.squeeze(-1)

        min_reward=0.  
        normalized_returns = (current_returns-min_reward) / (target_returns-min_reward + 1e-8)  
        normalized_returns=torch.clamp(normalized_returns, min=0.0001)

        normalized_q=(q_exp_pool.reshape(-1)-min_q_exp_pool)/(max_q_exp_pool - min_q_exp_pool)
        normalized_q=normalized_q.reshape(B,T)
        q_scale = self.args.q_scale
        w1=(normalized_q)
        weighted_action_loss = (w1 * action_loss).mean()
        
        loss= weighted_action_loss + q_scale*q_loss.mean()


        return loss,
