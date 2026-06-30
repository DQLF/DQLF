import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewportEncoder(nn.Module):
    """ViT-style patch embedder for VP frames -> token sequence (paper eq. 1: u_a)."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, d_out: int):
        super().__init__()
        assert image_size % patch_size == 0
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_out, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_out))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() == 5:
            B, T, C, H, W = frames.shape
            x = frames.reshape(B * T, C, H, W)
            x = self.proj(x).flatten(2).transpose(1, 2) + self.pos
            return x.reshape(B, T * self.n_patches, -1)
        x = self.proj(frames).flatten(2).transpose(1, 2)
        return x + self.pos


class ViewportTrajectoryEncoder(nn.Module):
    """Trajectory encoder for VP head-motion sequences (yaw/pitch + derivatives) -> tokens.

    Input: (B, T, F=traj_features). Output: (B, T, d_out).
    """

    def __init__(self, in_features: int, d_out: int, kernel_size: int = 3):
        super().__init__()
        self.lift = nn.Linear(in_features, d_out)
        self.conv = nn.Sequential(
            nn.Conv1d(d_out, d_out, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(d_out, d_out, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        h = self.lift(traj)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        return h


class ThroughputEncoder(nn.Module):
    """1D CNN encoder for ABR throughput / chunk-size traces (paper eq. 1: u_b)."""

    def __init__(self, in_features: int, d_out: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, d_out, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(d_out, d_out, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, T, d_out)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class DAGEncoder(nn.Module):
    """Lightweight GNN encoder for CJS DAG nodes (paper eq. 1: u_c)."""

    def __init__(self, node_features: int, d_out: int, n_layers: int = 2):
        super().__init__()
        self.lift = nn.Linear(node_features, d_out)
        self.layers = nn.ModuleList([nn.Linear(2 * d_out, d_out) for _ in range(n_layers)])

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # node_feats: (B, N, F); adj: (B, N, N) (DAG adjacency, no self-loops needed -- we add)
        h = self.lift(node_feats)
        eye = torch.eye(adj.size(1), device=adj.device).unsqueeze(0).expand_as(adj)
        a = adj + eye
        deg = a.sum(dim=-1, keepdim=True).clamp(min=1.0)
        a_norm = a / deg
        for layer in self.layers:
            agg = torch.bmm(a_norm, h)
            h = F.gelu(layer(torch.cat([h, agg], dim=-1)))
        return h


class UnifiedProjection(nn.Module):
    """E_md = W_p R_md + b_p  (paper eq. 2). Maps each modality to the unified D-dim space."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MultimodalEncoder(nn.Module):
    """Dispatches to the right modality encoder, then projects to the unified manifold.

    For VP, the input tensor rank picks the encoder:
        (B, T, F)              -> trajectory encoder (head-motion sequence)
        (B, C, H, W)           -> single-frame ViT encoder
        (B, T, C, H, W)        -> frame-sequence ViT encoder
    """

    def __init__(self, cfg):
        super().__init__()
        D = cfg.d_model
        self.vp_frame_enc = ViewportEncoder(cfg.vp_image_size, cfg.vp_patch_size, cfg.vp_in_channels, D)
        self.vp_traj_enc = ViewportTrajectoryEncoder(cfg.vp_traj_features, D)
        self.abr_enc = ThroughputEncoder(cfg.abr_in_features, D)
        self.cjs_enc = DAGEncoder(cfg.cjs_node_features, D)
        self.vp_proj = UnifiedProjection(D, D)
        self.abr_proj = UnifiedProjection(D, D)
        self.cjs_proj = UnifiedProjection(D, D)

    def forward(self, task: str, *args, **kwargs) -> torch.Tensor:
        if task == "vp":
            x = args[0] if args else kwargs.get("x")
            if x.dim() == 3:
                tokens = self.vp_traj_enc(x)
            else:
                tokens = self.vp_frame_enc(x)
            return self.vp_proj(tokens)
        if task == "abr":
            return self.abr_proj(self.abr_enc(*args, **kwargs))
        if task == "cjs":
            return self.cjs_proj(self.cjs_enc(*args, **kwargs))
        raise ValueError(f"unknown task: {task}")
