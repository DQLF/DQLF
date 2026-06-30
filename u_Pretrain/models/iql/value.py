import torch
import torch.nn as nn
import torch.nn.functional as F

def create_layers(in_dim, hidden_dim, n_hiddens, use_layer_norm, dropout_p=0.1):
    layers = []

    # 第一层
    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout_p))   # ⭐ 加这里
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))

    # 中间层
    for _ in range(n_hiddens - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_p))  # ⭐ 每层都加
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))

    # 输出层（不要加 dropout ❗）
    layers.append(nn.Linear(hidden_dim, 1))

    return layers

class ValueNetwork(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_hiddens,
        layernorm,
        dropout_p=0.1,
        **kwargs
    ) -> None:
        super().__init__()
        layers = create_layers(in_dim, hidden_dim, n_hiddens, layernorm, dropout_p)
        self.v = nn.Sequential(*layers)

    def forward(self, state):
        return self.v(state)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, n_hiddens, layernorm,dropout_p=0.1):
        super(QNetwork, self).__init__()
        
        if layernorm:
            layers = create_layers(state_dim + action_dim, hidden_dim, n_hiddens, layernorm,dropout_p)
            self.q1 = nn.Sequential(*layers)
            self.q2 = nn.Sequential(*layers)
        else:
            layers1 = create_layers(state_dim + action_dim, hidden_dim, n_hiddens, layernorm,dropout_p)
            layers2 = create_layers(state_dim + action_dim, hidden_dim, n_hiddens, layernorm,dropout_p)
            self.q1 = nn.Sequential(*layers1)
            self.q2 = nn.Sequential(*layers2)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        
        q1 = self.q1(state_action)
        q2 = self.q2(state_action)
        
        return q1, q2