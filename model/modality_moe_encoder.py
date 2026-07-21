"""
Modality-only MoE encoder for age prediction.

Replicates local_encoder + modality_experts + mod_gate from LifespanMoEMAE
but removes the age MoE entirely, so no age_group_ids are needed.
This avoids label leakage when age is the prediction target.

Forward: (x, modality_ids) → encoded features (B, embed_dim, H/4, W/4, D/4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks import ExtResNetBlock, Encoder


class ModalityExpert(nn.Module):
    """Expert specialised for one imaging modality."""

    def __init__(self, embed_dim: int = 512, expert_id: str = "T1w"):
        super().__init__()
        self.expert_id = expert_id

        self.expert_layers = nn.Sequential(
            ExtResNetBlock(embed_dim, embed_dim, stride=1, num_groups=32),
            ExtResNetBlock(embed_dim, embed_dim, stride=1, num_groups=32),
            nn.Conv3d(embed_dim, embed_dim, 1),
            nn.GroupNorm(32, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.attention = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim // 8, embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.expert_layers(x)
        feat = feat * self.attention(feat)
        return feat + x


class ModalityMoEGate(nn.Module):
    """Soft gating that combines spatial and global context."""

    def __init__(self, embed_dim: int = 512, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts

        self.spatial_gate = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // 4, 3, padding=1),
            nn.GroupNorm(16, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim // 4, num_experts, 1),
        )
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, num_experts),
        )
        # One-hot initialisation: modality i prefers expert i
        self.modality_routing = nn.Parameter(torch.eye(num_experts))

    def forward(self, x: torch.Tensor, modality_id: torch.Tensor):
        """
        Args:
            x           : (B, embed_dim, H, W, D)
            modality_id : (B,) LongTensor, values in [0, num_experts)
        Returns:
            spatial_gates : (B, num_experts, H, W, D)
            global_gates  : (B, num_experts)
        """
        B = x.shape[0]
        modality_gates = self.modality_routing[modality_id]          # (B, E)
        spatial_gates  = modality_gates.view(B, self.num_experts, 1, 1, 1)
        spatial_gates  = spatial_gates.expand(-1, -1, x.shape[2],
                                               x.shape[3], x.shape[4])
        return spatial_gates, modality_gates


class ModalityMoEEncoder(nn.Module):
    """
    Shared ResNet encoder followed by soft modality-MoE blending.

    Input  : (B, 1, H, W, D)
    Output : (B, embed_dim, H/4, W/4, D/4)   (embed_dim = 512 by default)

    modality_ids : list/tuple of strings e.g. ['T1w', 'T2w', ...]
                   OR single string for the whole batch
                   OR LongTensor of modality indices
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        embed_dim         = cfg.model.embed_dim          # typically 512
        self.embed_dim    = embed_dim
        self.modalities   = cfg.model.modalities         # e.g. ['T1w','T2w','FA','MD']
        self.num_experts  = len(self.modalities)
        self.modality_to_id = {m: i for i, m in enumerate(self.modalities)}

        # Two-stage shared encoder (stride 2 × stride 2 = 4× downsampling)
        intermediate_dim = embed_dim // 2
        self.local_encoder = nn.ModuleList([
            Encoder(1, intermediate_dim,
                    basic_module=ExtResNetBlock,
                    conv_kernel_size=3, conv_stride_size=2,
                    conv_layer_order='gcr', num_groups=32, padding=1),
            Encoder(intermediate_dim, embed_dim,
                    basic_module=ExtResNetBlock,
                    conv_kernel_size=3, conv_stride_size=2,
                    conv_layer_order='gcr', num_groups=32, padding=1),
        ])

        # Modality experts
        self.modality_experts = nn.ModuleDict({
            mod: ModalityExpert(embed_dim=embed_dim, expert_id=mod)
            for mod in self.modalities
        })

        # Gating network
        self.mod_gate = ModalityMoEGate(
            embed_dim=embed_dim,
            num_experts=self.num_experts,
        )

    def _parse_modality_ids(self, modality_ids, batch_size, device):
        if isinstance(modality_ids, str):
            return torch.full((batch_size,), self.modality_to_id[modality_ids],
                              device=device, dtype=torch.long)
        if isinstance(modality_ids, (list, tuple)):
            return torch.tensor([self.modality_to_id[m] for m in modality_ids],
                                device=device, dtype=torch.long)
        return modality_ids.to(device)

    def forward(self, x: torch.Tensor,
                modality_ids=None) -> torch.Tensor:
        """
        Args:
            x            : (B, 1, H, W, D)
            modality_ids : string / list of strings / LongTensor
        Returns:
            features : (B, embed_dim, H/4, W/4, D/4)
        """
        # Shared encoding
        for blk in self.local_encoder:
            x = blk(x)                      # (B, embed_dim, H/4, W/4, D/4)

        if modality_ids is None:
            return x

        B, device = x.shape[0], x.device
        mod_id = self._parse_modality_ids(modality_ids, B, device)

        spatial_gates, global_gates = self.mod_gate(x, mod_id)

        moe_out = torch.zeros_like(x)
        for i, mod in enumerate(self.modalities):
            expert_feat = self.modality_experts[mod](x)           # (B, C, H, W, D)
            s_w = spatial_gates[:, i:i+1]                         # (B, 1, H, W, D)
            g_w = global_gates[:, i:i+1, None, None, None]        # (B, 1, 1, 1, 1)
            moe_out = moe_out + expert_feat * (s_w * g_w)

        return moe_out
