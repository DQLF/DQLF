import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
S_LEN=8

INF = 1e5


class OfflineRLPolicy(nn.Module):
    def __init__(
            self,
            state_feature_dim,
            bitrate_levels,
            state_encoder,
            plm,
            mamba_plm, #这里用你的 MambaModel 或者对应类 (学生模型)
            plm_embed_size, #llama 4096
            mamba_embed_size, #mamba 512
            max_length=None,
            max_ep_len=100,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            device_out = None,
            residual = False, 
            conv_size = 4,  
            which_layer = -1,  # for early stopping: specify which layer to stop
            **kwargs
    ):
        super().__init__()
        
        if device_out is None:
            device_out = device

        self.bitrate_levels = bitrate_levels
        self.max_length = max_length

        self.plm = plm
        self.plm_embed_size = plm_embed_size
        self.mamba_plm = mamba_plm  #学生模型
        self.mamba_embed_size = mamba_embed_size
        

        # =========== multimodal encoder (start) ===========
        self.state_encoder = state_encoder
        self.state_feature_dim = state_feature_dim

        self.embed_timestep = nn.Embedding(max_ep_len + 1, plm_embed_size).to(device)
        self.embed_return = nn.Linear(1, plm_embed_size).to(device)
        self.embed_action = nn.Linear(1, plm_embed_size).to(device)
        self.embed_state1 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        self.embed_state2 = nn.Linear(state_feature_dim, plm_embed_size).to(device)    
        self.embed_state3 = nn.Linear(state_feature_dim * (S_LEN - conv_size + 1), plm_embed_size).to(device)    
        self.embed_state4 = nn.Linear(state_feature_dim * (S_LEN - conv_size + 1), plm_embed_size).to(device)    
        self.embed_state5 = nn.Linear(state_feature_dim, plm_embed_size).to(device)
        self.embed_state6 = nn.Linear(state_feature_dim, plm_embed_size).to(device)    
        self.embed_ln = nn.LayerNorm(plm_embed_size).to(device)

        # for mamba
        self.mamba_embed_timestep = nn.Embedding(max_ep_len + 1, mamba_embed_size).to(device)
        self.mamba_embed_return = nn.Linear(1, mamba_embed_size).to(device)
        self.mamba_embed_action = nn.Linear(1, mamba_embed_size).to(device)
        self.mamba_embed_state1 = nn.Linear(state_feature_dim, mamba_embed_size).to(device)
        self.mamba_embed_state2 = nn.Linear(state_feature_dim, mamba_embed_size).to(device)    
        self.mamba_embed_state3 = nn.Linear(state_feature_dim * (S_LEN - conv_size + 1), mamba_embed_size).to(device)    
        self.mamba_embed_state4 = nn.Linear(state_feature_dim * (S_LEN - conv_size + 1), mamba_embed_size).to(device)    
        self.mamba_embed_state5 = nn.Linear(state_feature_dim, mamba_embed_size).to(device)
        self.mamba_embed_state6 = nn.Linear(state_feature_dim, mamba_embed_size).to(device)    
        self.mamba_embed_ln = nn.LayerNorm(mamba_embed_size).to(device)
        
        self.action_head = nn.Linear(plm_embed_size, bitrate_levels).to(device)  # the so-called networking head in our paper
        self.mamba_action_head = nn.Linear(mamba_embed_size, bitrate_levels).to(device)  # the so-called networking head in our paper

        self.device = device
        self.device_out = device_out

        self.mamba_states_dq = deque([torch.zeros((1, 0, mamba_embed_size), device=device)], maxlen=max_length)
        self.mamba_returns_dq = deque([torch.zeros((1, 0, mamba_embed_size), device=device)], maxlen=max_length)
        self.mamba_actions_dq = deque([torch.zeros((1, 0, mamba_embed_size), device=device)], maxlen=max_length)

        self.residual = residual
        self.which_layer = which_layer
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.state_encoder, self.embed_timestep, self.embed_return, self.embed_action, self.embed_ln, 
            self.embed_state1, self.embed_state2, self.embed_state3, self.embed_state4, self.embed_state5,
            self.embed_state6, self.action_head
        ])

        # for mamba
        self.mamba_modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.state_encoder,self.mamba_embed_timestep, self.mamba_embed_return, self.mamba_embed_action, self.mamba_embed_ln, 
            self.mamba_embed_state1, self.mamba_embed_state2, self.mamba_embed_state3, self.mamba_embed_state4, 
            self.mamba_embed_state5, self.mamba_embed_state6, self.mamba_action_head
        ])

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        """
        Forward function, used for training.
        """
        assert actions.shape[0] == 1, 'batch size should be 1 to avoid CUDA memory exceed'

        # Step 1: process actions, returns and timesteps first as they are simple
        actions = actions.to(self.device)  # shape: (1, seq_len, 1)
        returns = returns.to(self.device)  # shape: (1, seq_len, 1)
        timesteps = timesteps.to(self.device)  # shape: (1, seq_len)

        # 1.1 embed action, return, timestep
        action_embeddings = self.embed_action(actions)  # shape: (1, seq_len, embed_size)
        returns_embeddings = self.embed_return(returns)  # shape: (1, seq_len, embed_size)
        time_embeddings = self.embed_timestep(timesteps)  # shape: (1, seq_len, embed_size)
        # for mamba
        mamba_action_embeddings = self.mamba_embed_action(actions)  # shape: (1, seq_len, embed_size)
        mamba_returns_embeddings = self.mamba_embed_return(returns)  # shape: (1, seq_len, embed_size)
        mamba_time_embeddings = self.mamba_embed_timestep(timesteps)  # shape: (1, seq_len, embed_size)

        # 1.2 time embeddings are treated similar to positional embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        # for mamba
        mamba_action_embeddings = mamba_action_embeddings + mamba_time_embeddings
        mamba_returns_embeddings = mamba_returns_embeddings + mamba_time_embeddings

        # Step 2: process states, turn them into embeddings.
        states = states.to(self.device)  # shape: (1, seq_len, 6, 6)
        states_features = self.state_encoder(states)
        states_embeddings1 = self.embed_state1(states_features[0]) + time_embeddings
        states_embeddings2 = self.embed_state2(states_features[1]) + time_embeddings
        states_embeddings3 = self.embed_state3(states_features[2]) + time_embeddings
        states_embeddings4 = self.embed_state4(states_features[3]) + time_embeddings
        states_embeddings5 = self.embed_state5(states_features[4]) + time_embeddings
        states_embeddings6 = self.embed_state6(states_features[5]) + time_embeddings
        # for mamba
        mamba_states_embeddings1 = self.mamba_embed_state1(states_features[0]) + mamba_time_embeddings
        mamba_states_embeddings2 = self.mamba_embed_state2(states_features[1]) + mamba_time_embeddings
        mamba_states_embeddings3 = self.mamba_embed_state3(states_features[2]) + mamba_time_embeddings
        mamba_states_embeddings4 = self.mamba_embed_state4(states_features[3]) + mamba_time_embeddings
        mamba_states_embeddings5 = self.mamba_embed_state5(states_features[4]) + mamba_time_embeddings
        mamba_states_embeddings6 = self.mamba_embed_state6(states_features[5]) + mamba_time_embeddings
        
        # Step 3: stack returns, states, actions embeddings.
        stacked_inputs = []
        action_embed_positions = np.zeros(returns_embeddings.shape[1])  # record the positions of action embeddings
        for i in range(returns_embeddings.shape[1]):
            stacked_input = torch.cat((returns_embeddings[0, i:i + 1], states_embeddings1[0, i:i + 1], states_embeddings2[0, i:i + 1], 
                                       states_embeddings3[0, i:i + 1], states_embeddings4[0, i:i + 1], states_embeddings5[0, i:i + 1], 
                                       states_embeddings6[0, i:i + 1], action_embeddings[0, i:i + 1]), dim=0)
            stacked_inputs.append(stacked_input)
            action_embed_positions[i] = (i + 1) * (2 + 6)
        stacked_inputs = torch.cat(stacked_inputs, dim=0).unsqueeze(0)
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]  # truncate sequence length (should not exceed plm embed size)
        stacked_inputs_ln = self.embed_ln(stacked_inputs)  # layer normalization
        mamba_stacked_inputs = []
        mamba_action_embed_positions = np.zeros(mamba_returns_embeddings.shape[1])  # record the positions of action embeddings
        for i in range(mamba_returns_embeddings.shape[1]):
            mamba_stacked_input = torch.cat((mamba_returns_embeddings[0, i:i + 1], mamba_states_embeddings1[0, i:i + 1], 
                                            mamba_states_embeddings2[0, i:i + 1], mamba_states_embeddings3[0, i:i + 1], 
                                            mamba_states_embeddings4[0, i:i + 1], mamba_states_embeddings5[0, i:i + 1], 
                                            mamba_states_embeddings6[0, i:i + 1], mamba_action_embeddings[0, i:i + 1]), dim=0)
            mamba_stacked_inputs.append(mamba_stacked_input)
            mamba_action_embed_positions[i] = (i + 1) * (2 + 6)
        mamba_stacked_inputs = torch.cat(mamba_stacked_inputs, dim=0).unsqueeze(0)
        mamba_stacked_inputs = mamba_stacked_inputs[:, -self.mamba_embed_size:, :]  # truncate sequence length (should not exceed plm embed size)
        mamba_stacked_inputs_ln = self.mamba_embed_ln(mamba_stacked_inputs)  # layer normalization

        # Step 4: feed stacked embeddings into the plm
        # 4.1 create attention mask
        if attention_mask is None:
            # 1 if can be attended to, 0 if not
            attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)
            mamba_attention_mask = torch.ones((mamba_stacked_inputs.shape[0], mamba_stacked_inputs.shape[1]), dtype=torch.long, device=self.device)
        
        # we feed in the input embeddings (not word indices as in NLP) to the model
        teacher_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )

        student_outputs = self.mamba_plm(  #学生模型的输出
            inputs_embeds=mamba_stacked_inputs_ln,
            attention_mask=mamba_attention_mask,
            output_hidden_states=True,
        )
        teacher_logits = teacher_outputs['last_hidden_state']
        student_logits = student_outputs['logits']  # 就是学生模型的last hidden state

        if self.residual:
            teacher_logits = teacher_logits + stacked_inputs_ln  # residual add
            student_logits = student_logits + mamba_stacked_inputs_ln  # residual add

        # Step 5: predict actions
        teacher_logits_used = teacher_logits[:, action_embed_positions - 2]
        student_logits_used = student_logits[:, mamba_action_embed_positions - 2]
        
        teacher_action_pred = self.action_head(teacher_logits_used)
        student_action_pred = self.mamba_action_head(student_logits_used)

        return teacher_action_pred, student_action_pred, out

    def sample(self, state, target_return, timestep,training=False, **kwargs):
        """
        Sample action function, used for evaluation/testing.
        """
        mamba_prev_stacked_inputs = []
        for i in range(len(self.mamba_states_dq)):
            mamba_prev_return_embeddings = self.mamba_returns_dq[i]
            mamba_prev_state_embeddings = self.mamba_states_dq[i]
            mamba_prev_action_embeddings = self.mamba_actions_dq[i]
            mamba_prev_stacked_inputs.append(torch.cat((mamba_prev_return_embeddings, mamba_prev_state_embeddings, mamba_prev_action_embeddings), dim=1))
        mamba_prev_stacked_inputs = torch.cat(mamba_prev_stacked_inputs, dim=1)
        # Step 2: process target return and timesteps
        target_return = torch.as_tensor(target_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        timestep = torch.as_tensor(timestep, dtype=torch.int32, device=self.device).reshape(1, 1)

        mamba_return_embeddings = self.mamba_embed_return(target_return)
        mamba_time_embeddings = self.mamba_embed_timestep(timestep)
        mamba_return_embeddings = mamba_return_embeddings + mamba_time_embeddings

        # Step 4: process state
        state = state.to(self.device)
        state_features = self.state_encoder(state)                            state_embeddings5, state_embeddings6], dim=1)
        mamba_state_embeddings1 = self.mamba_embed_state1(state_features[0]) + mamba_time_embeddings
        mamba_state_embeddings2 = self.mamba_embed_state2(state_features[1]) + mamba_time_embeddings
        mamba_state_embeddings3 = self.mamba_embed_state3(state_features[2]) + mamba_time_embeddings
        mamba_state_embeddings4 = self.mamba_embed_state4(state_features[3]) + mamba_time_embeddings
        mamba_state_embeddings5 = self.mamba_embed_state5(state_features[4]) + mamba_time_embeddings
        mamba_state_embeddings6 = self.mamba_embed_state6(state_features[5]) + mamba_time_embeddings
        mamba_state_embeddings = torch.cat([mamba_state_embeddings1, mamba_state_embeddings2, mamba_state_embeddings3, mamba_state_embeddings4,
                                           mamba_state_embeddings5, mamba_state_embeddings6], dim=1)


        mamba_stacked_inputs = torch.cat((mamba_return_embeddings, mamba_state_embeddings), dim=1)  # mind the order
        mamba_stacked_inputs = torch.cat((mamba_prev_stacked_inputs, mamba_stacked_inputs), dim=1)  # mind the order
        mamba_stacked_inputs = mamba_stacked_inputs[:, -self.mamba_embed_size:, :]  # truncate sequence length (should not exceed plm embed size)
        mamba_stacked_inputs_ln = self.mamba_embed_ln(mamba_stacked_inputs)  # layer normalization

        mamba_attention_mask = torch.ones((mamba_stacked_inputs_ln.shape[0], mamba_stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)


        mamba_plm_outputs = self.mamba_plm(  #学生模型的输出
            inputs_embeds=mamba_stacked_inputs_ln,
            attention_mask=mamba_attention_mask,
            output_hidden_states=True,
        )
        mamba_plm_logits = mamba_plm_outputs['logits']  # 就是学生模型的last hidden state
        if self.residual:
            mamba_plm_logits = mamba_plm_logits + mamba_stacked_inputs_ln  # residual add

        # Step 6: predict the bitrate for next chunk
        logits_used = mamba_plm_logits[:, -1:]
        action_pred = self.mamba_action_head(logits_used)
        action_pred = action_pred.reshape(-1)
        if training==True:
            bitrate, _ = self._sample(action_pred)  # for training, use sampling
        else:
            bitrate, _ = self._argmax(action_pred)  # for testing, use argmax

        mamba_action_tensor = torch.zeros(1, 1, 1, dtype=torch.float32, device=self.device)
        mamba_action_tensor[..., 0] = (bitrate + 1) / self.bitrate_levels
        mamba_action_embeddings = self.mamba_embed_action(mamba_action_tensor) + mamba_time_embeddings
        
        self.mamba_returns_dq.append(mamba_return_embeddings)
        self.mamba_states_dq.append(mamba_state_embeddings) 
        self.mamba_actions_dq.append(mamba_action_embeddings)

        return bitrate
    
    def clear_dq(self):

        self.mamba_states_dq.clear()
        self.mamba_actions_dq.clear()
        self.mamba_returns_dq.clear()   
        self.mamba_states_dq.append(torch.zeros((1, 0, self.mamba_embed_size), device=self.device))
        self.mamba_actions_dq.append(torch.zeros((1, 0, self.mamba_embed_size), device=self.device))
        self.mamba_returns_dq.append(torch.zeros((1, 0, self.mamba_embed_size), device=self.device))

    def _sample(self, logits):
        pi = F.softmax(logits, 0).cpu().numpy()
        idx = random.choices(np.arange(pi.size), pi)[0]
        lgprob = np.log(pi[idx])
        return idx, lgprob
    def _argmax(self, logits):
        pi = F.softmax(logits, 0).cpu().numpy()
        idx = np.argmax(pi)
        lgprob = np.log(pi[idx])
        return idx, lgprob