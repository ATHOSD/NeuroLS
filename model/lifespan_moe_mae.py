"""
Lifespan Foundation Model with Duo MoE Architecture

A unified foundation model that handles multiple brain imaging modalities (T1w, T2w, FA, MD)
across the full lifespan (fetal, infant, child, adult) for various downstream tasks:
1. MAE pre-training (reconstruction)
2. Subcortical segmentation
3. Age prediction
4. Sex prediction
5. Synthesis (T1w<->T2w, (T1w+T2w)->FA)

Key Features:
- Single modality input [B, 1, H, W, D] + modality_id + age_group_id
- Modality-agnostic MAE encoder
- Duo MoE: one expert per (modality, age_group) combination
  e.g. T1w_fetal, T1w_infant, T1w_child, T1w_adult,
       T2w_fetal, ..., MD_adult  → 4 x 4 = 16 experts
- Direct routing: each sample routes to exactly one expert
- Task-specific heads for different downstream applications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from model.blocks import ExtResNetBlock, res_decoders, Encoder




class ModalityExpert(nn.Module):
    """
    Individual expert that specializes in processing specific modality features
    Each expert learns modality-specific patterns:
    - T1w expert: Structural features (gray/white matter)
    - T2w expert: Pathological features (lesions, tissue contrast)
    - FA expert: White matter integrity, diffusion patterns
    """

    def __init__(self, embed_dim: int = 512, expert_id: str = "T1w"):
        super().__init__()

        self.expert_id = expert_id
        self.embed_dim = embed_dim

        # Expert-specific processing layers
        self.expert_layers = nn.Sequential(
            ExtResNetBlock(embed_dim, embed_dim, stride=1, num_groups=32),
            ExtResNetBlock(embed_dim, embed_dim, stride=1, num_groups=32),
            nn.Conv3d(embed_dim, embed_dim, 1),  # 1x1 conv for feature refinement
            nn.GroupNorm(32, embed_dim),
            nn.ReLU(inplace=True)
        )

        # Self-attention for expert specialization
        self.attention = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim // 8, embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Encoded features [B, embed_dim, H, W, D]
        Returns:
            expert_features: [B, embed_dim, H, W, D]
        """
        # Expert processing
        expert_feat = self.expert_layers(x)

        # Self-attention for feature refinement
        attention_weights = self.attention(expert_feat)
        expert_feat = expert_feat * attention_weights

        # Residual connection
        return expert_feat + x


class ModalitySoftGate(nn.Module):
    """
    Level-1 soft gate: learns how much each modality expert contributes.
    Input: encoded features (global avg pooled).
    Output: softmax weights over num_modalities experts [B, num_modalities].
    """
    def __init__(self, embed_dim: int, num_modalities: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, num_modalities)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.gate(x), dim=1)  # [B, num_modalities]


class AgeSoftGate(nn.Module):
    """
    Level-2 soft gate: learns how much each age expert contributes.
    Conditioned on both features AND age group embedding — mirrors the
    cat([feat, age]) pattern from AgeMoE but uses a learned embedding
    for the discrete age group (fetal/infant/child/adult).

    Output: softmax weights over num_age_groups experts [B, num_age_groups].
    """
    def __init__(self, embed_dim: int, num_age_groups: int, age_embed_dim: int = 16):
        super().__init__()
        self.age_embedding = nn.Embedding(num_age_groups, age_embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim + age_embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, num_age_groups)
        )
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x: torch.Tensor, age_id: torch.Tensor) -> torch.Tensor:
        """
        x:      [B, embed_dim, H, W, D]
        age_id: [B]  integer age group index
        """
        feat = self.pool(x).flatten(1)              # [B, embed_dim]
        age_emb = self.age_embedding(age_id)         # [B, age_embed_dim]
        gate_input = torch.cat([feat, age_emb], dim=1)  # [B, embed_dim + age_embed_dim]
        return F.softmax(self.gate(gate_input), dim=1)   # [B, num_age_groups]


class DuoMoEGate(nn.Module):
    """
    Gating network for Duo MoE: routes to expert indexed by
    (modality, age_group) combination.

    combo_id = modality_idx * num_age_groups + age_group_idx

    Supports:
    - Direct routing (one-hot, default during training)
    - Soft learned routing (for analysis / fine-tuning)
    """

    def __init__(self, embed_dim: int = 512, num_modalities: int = 4,
                 num_age_groups: int = 4):
        super().__init__()

        self.num_modalities = num_modalities
        self.num_age_groups = num_age_groups
        self.num_experts = num_modalities * num_age_groups

        # Direct one-hot routing table (identity matrix, not trained)
        # combo_id → one-hot [num_experts]
        self.register_buffer('routing_table', torch.eye(self.num_experts))

        # Optional learned soft gate (for downstream fine-tuning / analysis)
        self.soft_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, self.num_experts),
        )

    def get_combo_id(self, modality_id: torch.Tensor,
                     age_group_id: torch.Tensor) -> torch.Tensor:
        """Compute combined expert index from modality and age group indices."""
        return modality_id * self.num_age_groups + age_group_id

    def forward(self, x: torch.Tensor, modality_id: torch.Tensor,
                age_group_id: torch.Tensor, use_direct_routing: bool = True):
        """
        Args:
            x: Encoded features [B, embed_dim, H, W, D]
            modality_id: Modality index [B] (0..num_modalities-1)
            age_group_id: Age group index [B] (0..num_age_groups-1)
            use_direct_routing: If True, one-hot route to exact expert
        Returns:
            gates: [B, num_experts]  (one-hot for direct, softmax for soft)
        """
        combo_id = self.get_combo_id(modality_id, age_group_id)  # [B]

        if use_direct_routing:
            gates = self.routing_table[combo_id]  # [B, num_experts]
        else:
            gates = torch.softmax(self.soft_gate(x), dim=1)  # [B, num_experts]

        return gates




class DeepLabHead(nn.Module):
    """Segmentation head using DeepLab-style ASPP"""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        # ASPP with different dilation rates
        self.aspp1 = nn.Conv3d(in_channels, 256, 1)
        self.aspp2 = nn.Conv3d(in_channels, 256, 3, padding=6, dilation=6)
        self.aspp3 = nn.Conv3d(in_channels, 256, 3, padding=12, dilation=12)
        self.aspp4 = nn.Conv3d(in_channels, 256, 3, padding=18, dilation=18)

        # Global average pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, 256, 1)
        )

        # Feature fusion
        self.project = nn.Conv3d(256 * 5, 256, 1)
        self.classifier = nn.Conv3d(256, num_classes, 1)

    def forward(self, x: torch.Tensor):
        size = x.shape[-3:]

        feat1 = self.aspp1(x)
        feat2 = self.aspp2(x)
        feat3 = self.aspp3(x)
        feat4 = self.aspp4(x)
        feat5 = F.interpolate(self.global_pool(x), size=size, mode='trilinear', align_corners=False)

        # Concatenate all features
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        out = self.project(out)
        out = self.classifier(out)

        return out


class SimpleRegressionHead(nn.Module):
    """Simple head for age prediction"""

    def __init__(self, embed_dim: int = 512, output_dim: int = 1):
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embed_dim // 4, output_dim)
        )

    def forward(self, x: torch.Tensor):
        x = self.global_pool(x)
        x = self.flatten(x)
        return self.fc(x)


class SimpleClassificationHead(nn.Module):
    """Simple head for sex prediction"""

    def __init__(self, embed_dim: int = 512, num_classes: int = 2):
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embed_dim // 4, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.global_pool(x)
        x = self.flatten(x)
        return self.fc(x)


class SimpleSynthesisHead(nn.Module):
    """Simple head for modality synthesis"""

    def __init__(self, embed_dim: int = 512):
        super().__init__()

        # Simple decoder for synthesis
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, 256, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)


class LifespanMoEMAE(nn.Module):
    """
    Unified Lifespan Foundation Model with MoE Architecture

    Supports:
    1. MAE pre-training on individual modalities
    2. Downstream fine-tuning for multiple tasks
    3. Flexible modality combinations
    """

    def __init__(self, cfg):
        super().__init__()

        # Use original MAE configuration
        self.cfg = cfg
        embed_dim = cfg.model.embed_dim
        self.embed_dim = embed_dim
        self.modalities = cfg.model.modalities           # ['T1w', 'T2w', 'FA', 'MD']
        self.age_groups = cfg.model.age_groups           # ['fetal', 'infant', 'child', 'adult']
        # moe_type: 'hierarchical' (two-layer, 8 experts) or 'flat' (16 combined experts)
        self.moe_type = getattr(cfg.model, 'moe_type', 'hierarchical')
        self.num_modalities = len(self.modalities)
        self.num_age_groups = len(self.age_groups)

        # Create modality and age_group to ID mappings
        self.modality_to_id = {mod: i for i, mod in enumerate(self.modalities)}
        self.age_group_to_id = {ag: i for i, ag in enumerate(self.age_groups)}

        # --------------------------------------------------------------------------
        # Original MAE_CNN encoder architecture
        intermediate_dim = embed_dim // 2
        self.local_encoder = nn.ModuleList([
            Encoder(1, intermediate_dim, basic_module=ExtResNetBlock,
                    conv_kernel_size=3, conv_stride_size=2, conv_layer_order='gcr',
                    num_groups=32, padding=1),
            Encoder(intermediate_dim, embed_dim, basic_module=ExtResNetBlock,
                    conv_kernel_size=3, conv_stride_size=2, conv_layer_order='gcr',
                    num_groups=32, padding=1)
        ])

        # --------------------------------------------------------------------------
        # MoE experts
        if self.moe_type == 'hierarchical':
            # Two-layer soft MoE: 4 modality experts + 4 age experts = 8 total
            # Level 1: all modality experts run, blended by ModalitySoftGate
            # Level 2: all age experts run on blended output, weighted by AgeSoftGate
            #          (age gate conditioned on features + age group embedding)
            self.modality_experts = nn.ModuleDict({
                mod: ModalityExpert(embed_dim=embed_dim, expert_id=mod)
                for mod in self.modalities
            })
            self.age_experts = nn.ModuleDict({
                ag: ModalityExpert(embed_dim=embed_dim, expert_id=ag)
                for ag in self.age_groups
            })
            self.mod_gate = ModalitySoftGate(embed_dim, self.num_modalities)
            self.age_gate = AgeSoftGate(embed_dim, self.num_age_groups)
        else:
            # Flat: one expert per (modality × age_group) = 16 total
            self.experts = nn.ModuleDict({
                f'{mod}_{ag}': ModalityExpert(embed_dim=embed_dim, expert_id=f'{mod}_{ag}')
                for mod in self.modalities
                for ag in self.age_groups
            })
            self.gate = DuoMoEGate(
                embed_dim=embed_dim,
                num_modalities=self.num_modalities,
                num_age_groups=self.num_age_groups
            )

        # --------------------------------------------------------------------------
        # Original MAE_CNN decoder architecture
        # First upsampling: (512,24) -> (64,48)
        self.local_upsample1 = nn.ConvTranspose3d(in_channels=embed_dim, out_channels=64,
                                                 kernel_size=3, stride=2, padding=1, output_padding=1)

        # First decoder block: (64,48) -> (32,48) with multiple convs
        self.local_decoder1 = res_decoders(in_channels=64, f_maps=[32],
                                          basic_module=ExtResNetBlock, conv_kernel_size=3, conv_stride_size=1,
                                          conv_padding=1, layer_order='gcr', num_groups=8)

        # Second upsampling: (32,48) -> (32,96)
        self.local_upsample2 = nn.ConvTranspose3d(in_channels=32, out_channels=32,
                                                 kernel_size=3, stride=2, padding=1, output_padding=1)

        # Second decoder block: (32,96) -> (16,96) with multiple convs
        self.local_decoder2 = res_decoders(in_channels=32, f_maps=[16],
                                          basic_module=ExtResNetBlock, conv_kernel_size=3, conv_stride_size=1,
                                          conv_padding=1, layer_order='gcr', num_groups=8)

        # Final layers
        self.final_projection_local_recon = nn.Conv3d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.final_norm_local_recon = nn.GroupNorm(
            num_groups=8, num_channels=16)

        self.avgpool = nn.AdaptiveAvgPool3d((3, 1, 1))

    def patchify(self, imgs, p):
        """
        Original patchify method from MAE_CNN
        imgs: (N, 1, H, W, D)
        x: (N, H*W*D/P^3, patch_size^3)
        """
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0
        h, w, d = [i//p for i in self.cfg.data.patch_size]

        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
        x = torch.einsum('nchpwqdr->nhwdpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3))
        return x

    def unpatchify(self, x, p):
        """
        Original unpatchify method from MAE_CNN
        x: (N, H*W*D/P^3, patch_size^3)
        imgs: (N, 1, H, W, D)
        """
        h, w, d = [i//p for i in self.cfg.data.patch_size]
        assert h * w * d == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p))
        x = torch.einsum('nhwdpqr->nhpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p, d * p))
        return imgs

    def random_masking(self, x, mask_ratio, p):
        """
        Original random_masking method from MAE_CNN
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        x = self.patchify(x, p)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask_ = torch.zeros_like(x_masked)
        # generate the binary mask: 0 is keep, 1 is remove

        x_empty = torch.zeros((N, L - len_keep, D)).cuda()
        mask = torch.ones_like(x_empty)
        x_ = torch.cat([x_masked, x_empty], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        mask_ = torch.cat([mask_, mask], dim=1)
        mask_ = torch.gather(
            mask_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x_masked = self.unpatchify(x_, p)

        mask = self.unpatchify(mask_, p)

        return x_masked, mask

    def forward_encoder(self, x, mask_ratio, p):
        """
        Original MAE encoder with MoE components added
        Based on original MAE_CNN forward_encoder
        """
        # masking: length -> length * mask_ratio
        x, mask = self.random_masking(x, mask_ratio, p)

        # apply original MAE encoder blocks
        for blk in self.local_encoder:
            x = blk(x)

        return x, mask

    def _parse_ids(self, ids, id_map: dict, batch_size: int,
                   device: torch.device) -> torch.Tensor:
        """Convert string / list / tensor IDs to a LongTensor [B]."""
        if isinstance(ids, str):
            return torch.full((batch_size,), id_map[ids], device=device, dtype=torch.long)
        elif isinstance(ids, (list, tuple)):
            return torch.tensor([id_map[v] if isinstance(v, str) else int(v)
                                 for v in ids], device=device, dtype=torch.long)
        else:
            return ids.to(device)

    def _route_experts(self, features: torch.Tensor, experts: nn.ModuleDict,
                       id_tensor: torch.Tensor, id_to_name: list) -> torch.Tensor:
        """
        Direct per-sample routing: apply expert[id_tensor[b]] to features[b].
        Handles mixed batches efficiently by grouping samples per expert.
        """
        out = torch.zeros_like(features)
        for idx, name in enumerate(id_to_name):
            mask = (id_tensor == idx)
            if not mask.any():
                continue
            out[mask] = experts[name](features[mask])
        return out

    def forward_encoder_with_moe(self, x, modality_ids, age_group_ids, mask_ratio, p):
        """
        Encoder + MoE routing (hierarchical or flat).

        Hierarchical (default):
          encoded → modality_expert[mod] → age_expert[age] → output
          8 experts total, 2 active per sample.

        Flat:
          encoded → combined_expert[mod_age] → output
          16 experts total, 1 active per sample.
        """
        encoded_features, mask = self.forward_encoder(x, mask_ratio, p)

        batch_size = x.shape[0]
        device = x.device

        mod_id = self._parse_ids(modality_ids, self.modality_to_id, batch_size, device)
        ag_id = self._parse_ids(age_group_ids, self.age_group_to_id, batch_size, device)

        if self.moe_type == 'hierarchical':
            # Level 1: soft modality gating
            # All modality experts run; outputs blended by learned weights
            mod_weights = self.mod_gate(encoded_features)  # [B, num_modalities]
            mod_features = torch.zeros_like(encoded_features)
            for i, mod in enumerate(self.modalities):
                w = mod_weights[:, i].view(-1, 1, 1, 1, 1)
                mod_features = mod_features + self.modality_experts[mod](encoded_features) * w

            # Level 2: soft age gating conditioned on features + age embedding
            # All age experts run on modality-blended features
            age_weights = self.age_gate(mod_features, ag_id)  # [B, num_age_groups]
            moe_features = torch.zeros_like(mod_features)
            for i, ag in enumerate(self.age_groups):
                w = age_weights[:, i].view(-1, 1, 1, 1, 1)
                moe_features = moe_features + self.age_experts[ag](mod_features) * w
        else:
            # Flat: one-hot gate → single combined expert
            gates = self.gate(encoded_features, mod_id, ag_id, use_direct_routing=True)
            expert_keys = [f'{mod}_{ag}'
                           for mod in self.modalities
                           for ag in self.age_groups]
            moe_features = torch.zeros_like(encoded_features)
            for i, key in enumerate(expert_keys):
                weight = gates[:, i].view(-1, 1, 1, 1, 1)
                if weight.sum() == 0:
                    continue
                moe_features = moe_features + self.experts[key](encoded_features) * weight

        return moe_features, mask

    def forward_local_decoder(self, x):
        """
        Original MAE decoder from MAE_CNN
        """
        # First upsampling: (512,24) -> (64,48)
        x = self.local_upsample1(x)
        # First decoder blocks: (64,48) -> (32,48) with conv1,2,3
        for blk in self.local_decoder1:
            x = blk(x)

        # Second upsampling: (32,48) -> (32,96)
        x = self.local_upsample2(x)
        # Second decoder blocks: (32,96) -> (16,96) with conv1,2,3
        for blk in self.local_decoder2:
            x = blk(x)

        x = self.final_norm_local_recon(x)
        x = self.final_projection_local_recon(x)  # (16,96) -> (1,96)
        x = torch.sigmoid(x)

        return x

    def recon_loss(self, imgs, pred, mask):
        """
        Reconstruction loss for MAE following original implementation

        Args:
            imgs: Target images [N, C, H, W, D]
            pred: Predicted images [N, C, H, W, D]
            mask: Mask [N, C, H, W, D], 0 is keep, 1 is remove
        """
        loss = (pred - imgs) ** 2
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_train(self, local_patch, global_img, modality_ids, age_group_ids,
                      mask_ratio=0.75):
        """
        Forward pass for MAE training with Duo MoE.

        Args:
            local_patch: Local patches [B, 1, H, W, D]
            global_img: Global images [B, 1, H, W, D]
            modality_ids: Modality name(s) or index tensor [B]
            age_group_ids: Age-group name(s) or index tensor [B]
                           ('fetal', 'infant', 'child', 'adult')
            mask_ratio: Masking ratio for MAE
        """
        # Forward local patch with Duo MoE
        local_latent, local_mask = self.forward_encoder_with_moe(
            local_patch, modality_ids, age_group_ids,
            mask_ratio, self.cfg.train.local_mae_patch)
        local_pred = self.forward_local_decoder(local_latent)
        local_loss = self.recon_loss(local_patch, local_pred, local_mask)

        # Forward global image with Duo MoE
        global_latent, global_mask = self.forward_encoder_with_moe(
            global_img, modality_ids, age_group_ids,
            mask_ratio, self.cfg.train.global_mae_patch)
        global_pred = self.forward_local_decoder(global_latent)
        global_loss = self.recon_loss(global_img, global_pred, global_mask)

        return local_loss, global_loss, local_pred, global_pred, local_mask, global_mask

    def forward(self, x: torch.Tensor, modality: str, age_group: str, task: str,
                mask_ratio: float = 0.0, target: torch.Tensor = None):
        """
        Forward pass for downstream tasks.

        Args:
            x: Input volume [B, 1, H, W, D]
            modality: Input modality ('T1w', 'T2w', 'FA', 'MD')
            age_group: Age group ('fetal', 'infant', 'child', 'adult')
            task: Task type ('mae', 'segmentation', 'age', 'sex', 'synthesis')
            mask_ratio: For MAE training
            target: Ground truth for loss computation
        """
        if mask_ratio > 0:
            moe_features, mask = self.forward_encoder_with_moe(
                x, modality, age_group, mask_ratio, self.cfg.train.local_mae_patch)
        else:
            moe_features, _ = self.forward_encoder_with_moe(
                x, modality, age_group, 0.0, self.cfg.train.local_mae_patch)
            mask = None

        task_head = self.task_heads[task]
        output = task_head(moe_features)

        loss = None
        if target is not None:
            if task == 'mae':
                if mask is not None:
                    loss = self.mae_loss(output * mask, target * mask) / (mask.sum() + 1e-8)
                else:
                    loss = self.mae_loss(output, target)
            elif task == 'segmentation':
                loss = self.seg_loss(output, target)
            elif task == 'age':
                loss = self.age_loss(output.squeeze(), target.float())
            elif task == 'sex':
                loss = self.sex_loss(output, target)
            elif task == 'synthesis':
                loss = self.synthesis_loss(output, target)

        return output, loss

    def get_expert_usage(self, x: torch.Tensor, modality: str, age_group: str):
        """Analyze expert utilization for interpretability."""
        batch_size = x.shape[0]
        device = x.device

        mod_id = torch.full((batch_size,), self.modality_to_id[modality],
                            device=device, dtype=torch.long)
        ag_id = torch.full((batch_size,), self.age_group_to_id[age_group],
                           device=device, dtype=torch.long)

        for blk in self.local_encoder:
            x = blk(x)
        gates = self.gate(x, mod_id, ag_id, use_direct_routing=False)

        return {
            'gates': gates.detach().cpu(),
            'expert_usage': gates.mean(dim=0).detach().cpu(),
            'active_expert': f'{modality}_{age_group}'
        }