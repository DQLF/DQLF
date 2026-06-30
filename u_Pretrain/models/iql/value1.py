import torch
import torch.nn as nn
import torch.nn.functional as F

def create_layers(in_dim, hidden_dim, n_hiddens, use_layer_norm,output_dim=None):
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    for _ in range(n_hiddens-1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
    if output_dim is not None:
        layers.append(nn.Linear(hidden_dim, output_dim))
    return layers

class ValueNetwork(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_hiddens,
        layernorm,
        **kwargs
    ) -> None:
        super().__init__()
        layers = create_layers(in_dim, hidden_dim, n_hiddens, layernorm,output_dim=1)
        self.v = nn.Sequential(*layers)

    def forward(self, state):
        return self.v(state)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, n_hiddens, layernorm):
        super(QNetwork, self).__init__()

        if layernorm:#
            layers = create_layers(state_dim, hidden_dim, n_hiddens, layernorm, output_dim=None)
            self.q1 = nn.Sequential(*layers, nn.Linear(hidden_dim, action_dim))
            self.q2 = nn.Sequential(*layers, nn.Linear(hidden_dim, action_dim))
        else:
            # 不使用layernorm时，创建两套独立的网络结构
            # 这样可以避免在forward中重复计算同一层
            # 只输入state，输出action_dim个Q值
            layers1 = create_layers(state_dim, hidden_dim, n_hiddens, layernorm, output_dim=None)
            layers2 = create_layers(state_dim, hidden_dim, n_hiddens, layernorm, output_dim=None)
            self.q1 = nn.Sequential(*layers1, nn.Linear(hidden_dim, action_dim))
            self.q2 = nn.Sequential(*layers2, nn.Linear(hidden_dim, action_dim))

    def forward(self, state):

        q1 = self.q1(state)  # [batch, action_dim]
        q2 = self.q2(state)  # [batch, action_dim]
        return q1, q2
    
    # def __init__(self, state_dim, action_dim, hidden_dim, n_hiddens, layernorm):
    #     super(QNetwork, self).__init__()
        
    #     if layernorm:
    #         layers = create_layers(state_dim + action_dim, hidden_dim, n_hiddens, layernorm)
    #         self.q1 = nn.Sequential(*layers)
    #         self.q2 = nn.Sequential(*layers)
    #     else:
    #         layers1 = create_layers(state_dim + action_dim, hidden_dim, n_hiddens, layernorm)
    #         layers2 = create_layers(state_dim + action_dim, hidden_dim, n_hiddens, layernorm)
    #         self.q1 = nn.Sequential(*layers1)
    #         self.q2 = nn.Sequential(*layers2)

    # def forward(self, state, action):
    #     state_action = torch.cat([state, action], dim=-1)
        
    #     q1 = self.q1(state_action)
    #     q2 = self.q2(state_action)
        
    #     return q1, q2