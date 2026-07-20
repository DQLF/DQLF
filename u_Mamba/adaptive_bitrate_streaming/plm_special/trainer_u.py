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
        self.alpha_kl = args.alpha_kl  
        self.kl_div = torch.nn.KLDivLoss(reduction='batchmean')  
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

        q1, q2 = self.qf(states.reshape(B,T, -1), F.one_hot(labels.squeeze(-1), num_classes=A).float().reshape(B,T,A))
        q_exp_pool = torch.min(q1, q2) 

        

        layer_kl_loss = 0.0


        t_states = out['teacher_outputs'].hidden_states
        s_states = out['student_outputs'].hidden_states

        num_t = len(t_states)    
        num_s = len(s_states)    

        selected_t_indices = [
            round((i + 1) * num_t / (num_s + 1)) - 1  
            for i in range(num_s)
        ]
        for i, s_hidden in enumerate(s_states):

            t_hidden = t_states[selected_t_indices[i]]

            if out['residual']:
                t_hidden = t_hidden + out['stacked_inputs_ln']
                s_hidden = s_hidden + out['mamba_stacked_inputs_ln']

            t_hidden_used = t_hidden[:,out['mamba_action_embed_positions']-2]  
            s_hidden_used = s_hidden[:,out['mamba_action_embed_positions']-2]  
            
            t_action_pred = out['t_action_head'](t_hidden_used)
            s_action_pred = out['s_action_head'](s_hidden_used)
            
            s_log_prob = F.log_softmax(s_action_pred, dim=-1)
            t_prob = F.softmax(t_action_pred, dim=-1)
            layer_kl_loss += self.kl_div(s_log_prob, t_prob)
      
        q_scale = 4.0
        student_loss = weighted_action_loss + 1.0 * layer_kl_loss+ q_scale*q_loss.mean()


        return student_loss
