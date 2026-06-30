from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MoEMambaConfig:
    d_model: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    n_experts: int = 4
    top_k: int = 2
    history_len: int = 4
    router_noise_std: float = 1.0

    n_layers: int = 4

    vp_tile_h: int = 8
    vp_tile_w: int = 32
    abr_num_actions: int = 6
    cjs_max_nodes: int = 64

    vp_image_size: int = 224
    vp_patch_size: int = 16
    vp_in_channels: int = 3
    vp_traj_seq_len: int = 16
    vp_traj_features: int = 6
    vp_horizon_steps: int = 5

    @property
    def vp_num_classes(self) -> int:
        return self.vp_tile_h * self.vp_tile_w

    abr_seq_len: int = 32
    abr_in_features: int = 6

    cjs_node_features: int = 8

    distill_temperature: float = 2.0
    distill_alpha: float = 0.5

    cwr_rank: int = 16

    dropout: float = 0.1
