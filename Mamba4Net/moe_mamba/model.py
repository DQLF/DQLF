import torch
import torch.nn as nn

from .config import MoEMambaConfig
from .encoders import MultimodalEncoder
from .moe import MoEMambaLayer
from .heads import VPHead, ABRHead, CJSHead


class MoEMamba(nn.Module):
    """Unified MoE-Mamba student model: multimodal encoder -> stacked MoE-Mamba layers -> task head."""

    def __init__(self, cfg: MoEMambaConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = MultimodalEncoder(cfg)

        self.layers = nn.ModuleList([
            MoEMambaLayer(
                d_model=cfg.d_model,
                n_experts=cfg.n_experts,
                top_k=cfg.top_k,
                history_len=cfg.history_len,
                d_state=cfg.d_state,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
                noise_std=cfg.router_noise_std,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])

        self.vp_head = VPHead(cfg.d_model, cfg.vp_num_classes)
        self.abr_head = ABRHead(cfg.d_model, cfg.abr_num_actions)
        self.cjs_head = CJSHead(cfg.d_model)

    def backbone(self, E: torch.Tensor):
        aux = E.new_zeros(())
        H = E
        for layer in self.layers:
            H, lb = layer(H)
            aux = aux + lb
        return H, aux

    def forward(self, task: str, *args, **kwargs):
        urgency = kwargs.pop("urgency", None)
        E = self.encoder(task, *args, **kwargs)
        H_out, aux = self.backbone(E)

        if task == "vp":
            logits = self.vp_head(H_out)
            return {"logits": logits, "aux": aux, "hidden": H_out}
        if task == "abr":
            policy, value = self.abr_head(H_out)
            return {"policy": policy, "value": value, "aux": aux, "hidden": H_out}
        if task == "cjs":
            head = self.cjs_head(H_out, urgency)
            return {**head, "aux": aux, "hidden": H_out}
        raise ValueError(f"unknown task: {task}")
