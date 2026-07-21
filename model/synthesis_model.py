"""
Synthesis Model: MoE Encoder + Upfold3D Decoder
Connects LifespanMoEMAE encoder to Upfold3D decoder (like PTNet) for T1<->T2, (T1+T2)->FA

Architecture:
Input: [B, 1, 96, 96, 96]
Encoder: 96 -> 48 -> 24 (with MoE at 24)
Connectors: 512 -> 32 (channel adaptation)
Decoder: 24 -> 48 -> 96 (with Upfold3D + skip connections)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.lifespan_moe_mae import LifespanMoEMAE
from model.transformer_block import Upfold3D


class SynthesisDecoder(nn.Module):
    """
    Decoder for synthesis task with Upfold3D blocks and skip connections
    Based on PTNet_local_trans decoder architecture

    Flow:
    [32, 24] + skip[32, 48] -> Upfold3D -> [32, 48]
    [32, 48] + skip[16, 96] -> Upfold3D -> [16, 96]
    [16, 96] + input[1, 96]  -> Upfold3D -> [16, 96]
    [16, 96] -> Linear -> [1, 96]
    """

    def __init__(self, img_size=[96, 96, 96], channels=[1, 16, 32, 32],
                 down_ratio=[1, 1, 2, 4], patch=[3, 3, 3]):
        super().__init__()

        self.size = img_size
        self.ratio = down_ratio

        # Upfold3D blocks with skip connections (following PTNet structure)
        # Stage 1: 24->48, [32, 24] upsampled + skip[32, 48] -> concat[64, 48] -> [32, 48]
        self.up_blocks = nn.ModuleList([
            Upfold3D(up_scale=2, in_channel=channels[-1]+channels[-1], out_channel=channels[-1],
                    patch=patch[-1], stride=1, padding=1),
            # Stage 2: 48->96, [32, 48] upsampled + skip[16, 96] -> concat[48, 96] -> [16, 96]
            Upfold3D(up_scale=2, in_channel=channels[-1]+channels[1], out_channel=channels[1],
                    patch=patch[0], stride=1, padding=1),
            # Stage 3: 96->96, [16, 96] + input[1, 96] -> concat[17, 96] -> [16, 96]
            Upfold3D(up_scale=1, in_channel=channels[1]+channels[0], out_channel=channels[1],
                    patch=patch[0], stride=1, padding=1)
            #Upfold3D(up_scale=1, in_channel=channels[1], out_channel=channels[1],
            #        patch=patch[0], stride=1, padding=1)
        ])

        # Final projection
        self.final_proj = nn.Linear(channels[1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skip_48=None, skip_96=None, x0=None):
        """
        Args:
            x: [B, 32, 24, 24, 24] - connector output
            skip_48: [B, 32, 48, 48, 48] - connector_1 output
            skip_96: [B, 16, 96, 96, 96] - connector_3 output
            x0: [B, 1, 96, 96, 96] - original input for final skip connection
        """
        # Stage 1: 24->48 with skip[32, 48]
        x = self.up_blocks[0](x, size=[x.shape[2], x.shape[3], x.shape[4]],
                             SC=skip_48, reshape=True)

        # Stage 2: 48->96 with skip[16, 96]
        x = self.up_blocks[1](x, size=[x.shape[2], x.shape[3], x.shape[4]],
                             SC=skip_96, reshape=True)

        # Stage 3: 96->96 with original input [1, 96]
        x = self.up_blocks[2](x, SC=x0, reshape=False)
        #x = self.up_blocks[2](x, SC=None, reshape=False)

        # Final projection
        x = self.final_proj(x).transpose(1, 2)
        B, C, HW = x.shape
        x = x.reshape(B, C, self.size[0], self.size[1], self.size[2])

        return self.sigmoid(x)


class MoESynthesisModel(nn.Module):
    """
    Complete synthesis model combining MoE encoder and Upfold3D decoder
    Following PTNet_local_trans architecture with connectors

    Architecture Flow:
    Input [B, 1, 96, 96, 96]
        ↓ encoder[0] (stride=2)
    [B, 256, 48, 48, 48] ← skip_48
        ↓ encoder[1] (stride=2)
    [B, 512, 24, 24, 24]
        ↓ MoE experts
    [B, 512, 24, 24, 24] ← moe_features
        ↓ connectors (adapt channels)
    SC[1]: [B, 32, 48] (connector_1)
    SC[0]: [B, 16, 96] (connector_3)
    x: [B, 32, 24] (connector_2)
        ↓ Upfold3D decoder with skip connections
    [B, 1, 96, 96, 96] ← output
    """

    def __init__(self, cfg, pretrained_encoder_path=None):
        super().__init__()

        self.cfg = cfg
        self.img_size = cfg.data.patch_size

        # MoE Encoder (can load pretrained MAE weights)
        self.encoder = LifespanMoEMAE(cfg)

        if pretrained_encoder_path is not None:
            self._load_pretrained_encoder(pretrained_encoder_path)

        # Connectors - adapt encoder features for decoder
        # Connector for x_24 -> skip_48: [512, 24] -> [32, 48]
        self.connector_skip_48 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.ConvTranspose3d(512, 32, kernel_size=3, stride=2,
                             padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )

        # Connector for x_48 -> skip_96: [256, 48] -> [16, 96]
        self.connector_skip_96 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.ConvTranspose3d(256, 16, kernel_size=3, stride=2,
                             padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )

        # Connector for moe_features -> decoder input: [512, 24] -> [32, 24]
        self.connector_bottleneck = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.Conv3d(512, 32, kernel_size=3, stride=1,
                     padding=1, bias=True),
            nn.ReLU(True)
        )

        # Synthesis decoder with Upfold3D
        self.decoder = SynthesisDecoder(
            img_size=self.img_size,
            channels=[1, 16, 32, 32],
            down_ratio=[1, 1, 2, 4],
            patch=[3, 3, 3]
        )

        # Freeze encoder if needed
        self.freeze_encoder = False

    def _load_pretrained_encoder(self, pretrained_path):
        """Load pretrained MAE encoder weights with detailed logging"""
        print(f"Loading encoder weights from {pretrained_path}")

        # Load checkpoint
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Get target encoder state dict
        target_dict = self.encoder.state_dict()

        # Filter only matching keys
        matched_dict = {
            k: v for k, v in state_dict.items()
            if k in target_dict and v.shape == target_dict[k].shape
        }

        print(f"Matched {len(matched_dict)} / {len(target_dict)} MAE encoder weights")

        # Update and load
        target_dict.update(matched_dict)
        self.encoder.load_state_dict(target_dict)

        print("✅ MAE encoder weights loaded into MoESynthesisModel.")

    def set_freeze_encoder(self, freeze=True):
        """Freeze/unfreeze encoder for fine-tuning"""
        self.freeze_encoder = freeze
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def forward_encoder_with_skips(self, x, modality_ids):
        """
        Forward through encoder, MoE, and collect skip connections via connectors

        Flow:
        x_24 [B, 512, 24] → connector_skip_48 → [B, 32, 48] → SC[1]
        x_48 [B, 256, 48] → connector_skip_96 → [B, 16, 96] → SC[0]
        moe_features [B, 512, 24] → connector_bottleneck → [B, 32, 24] → decoder

        Returns:
            x_24_out: [B, 32, 24, 24, 24] - adapted bottleneck from MoE
            skip_48: [B, 32, 48, 48, 48] - from x_24 (before MoE)
            skip_96: [B, 16, 96, 96, 96] - from x_48
        """
        # First encoder block: 96 -> 48
        x_48 = self.encoder.local_encoder[0](x)  # [B, 256, 48, 48, 48]

        # Second encoder block: 48 -> 24
        x_24 = self.encoder.local_encoder[1](x_48)  # [B, 512, 24, 24, 24]

        # MoE processing at 24x24x24
        batch_size = x.shape[0]
        device = x.device

        # Handle modality IDs
        if isinstance(modality_ids, str):
            modality_id = torch.full((batch_size,), self.encoder.modality_to_id[modality_ids],
                                   device=device, dtype=torch.long)
        elif isinstance(modality_ids, (list, tuple)):
            modality_id = torch.tensor([self.encoder.modality_to_id[mod] for mod in modality_ids],
                                     device=device, dtype=torch.long)
        else:
            modality_id = modality_ids.to(device)

        # Route through MoE experts
        spatial_gates, global_gates = self.encoder.gate(x_24, modality_id)

        # Apply experts
        moe_features = torch.zeros_like(x_24)
        for i, expert_modality in enumerate(self.encoder.modalities):
            expert = self.encoder.experts[expert_modality]
            expert_output = expert(x_24)

            spatial_weight = spatial_gates[:, i:i+1]
            global_weight = global_gates[:, i:i+1, None, None, None]
            combined_weight = spatial_weight * global_weight

            moe_features += expert_output * combined_weight

        # Apply connectors
        # x_24 (before MoE) -> skip_48
        skip_48 = self.connector_skip_48(x_24)      # [B, 512, 24] → [B, 32, 48]
        # x_48 -> skip_96
        skip_96 = self.connector_skip_96(x_48)      # [B, 256, 48] → [B, 16, 96]

        # Decoder input from MoE features
        x_24_out = self.connector_bottleneck(moe_features)  # [B, 512, 24] → [B, 32, 24]

        return x_24_out, skip_48, skip_96

    def forward(self, x, modality_ids):
        """
        Forward pass for synthesis

        Args:
            x: [B, 1, 96, 96, 96] - input modality
            modality_ids: modality identifier(s)
        Returns:
            output: [B, 1, 96, 96, 96] - synthesized modality
        """
        x0 = x  # Save input for final skip connection

        # Encode with skip connections via connectors
        if self.freeze_encoder:
            with torch.no_grad():
                x_24, skip_48, skip_96 = self.forward_encoder_with_skips(x, modality_ids)
        else:
            x_24, skip_48, skip_96 = self.forward_encoder_with_skips(x, modality_ids)

        # Decode with skip connections
        # Following PTNet order: SC[1] = skip_48, SC[0] = skip_96, x0 = original input
        output = self.decoder(x_24, skip_48=skip_48, skip_96=skip_96, x0=x0)

        return output


class MoEMultiModalSynthesisModel(nn.Module):
    """
    Multi-modal synthesis: (T1 + T2) -> FA

    IMPORTANT: Each modality goes through its own MoE expert first,
    then features are fused after MoE processing

    Architecture:
    T1 [B,1,96³] → encoder → MoE(T1w expert) → [B,512,24³]
                                                            ↓ fusion
    T2 [B,1,96³] → encoder → MoE(T2w expert) → [B,512,24³]
                                                            ↓
                                                [B,512,24³] → decoder → FA [B,1,96³]
    """

    def __init__(self, cfg, pretrained_encoder_path=None):
        super().__init__()

        self.cfg = cfg
        self.img_size = cfg.data.patch_size

        # Shared MoE encoder (with experts for T1w and T2w)
        self.encoder = LifespanMoEMAE(cfg)

        if pretrained_encoder_path is not None:
            self._load_pretrained_encoder_multi(pretrained_encoder_path)

        # Fusion after MoE: concat [512+512] -> 512
        self.fusion_bottleneck = nn.Sequential(
            nn.Conv3d(1024, 512, kernel_size=3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True)
        )

        # Skip fusion for 48x48x48 level (before MoE)
        self.fusion_skip_48 = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True)
        )

        # Connectors - adapt fused features for decoder
        self.connector_skip_48 = nn.Sequential(
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.ConvTranspose3d(256, 32, kernel_size=3, stride=2,
                             padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )

        self.connector_skip_96 = nn.Sequential(
            nn.GroupNorm(num_groups=16, num_channels=256),
            nn.ConvTranspose3d(256, 16, kernel_size=3, stride=2,
                             padding=1, output_padding=1, bias=True),
            nn.ReLU(True)
        )

        self.connector_bottleneck = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.Conv3d(512, 32, kernel_size=3, stride=1,
                     padding=1, bias=True),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = SynthesisDecoder(
            img_size=self.img_size,
            channels=[1, 16, 32, 32],
            down_ratio=[1, 1, 2, 4],
            patch=[3, 3, 3]
        )

        self.freeze_encoder = False

    def _load_pretrained_encoder_multi(self, pretrained_path):
        """Load pretrained MAE encoder weights with detailed logging"""
        print(f"Loading encoder weights from {pretrained_path}")

        # Load checkpoint
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Get target encoder state dict
        target_dict = self.encoder.state_dict()

        # Filter only matching keys
        matched_dict = {
            k: v for k, v in state_dict.items()
            if k in target_dict and v.shape == target_dict[k].shape
        }

        print(f"Matched {len(matched_dict)} / {len(target_dict)} MAE encoder weights")

        # Update and load
        target_dict.update(matched_dict)
        self.encoder.load_state_dict(target_dict)

        print("✅ MAE encoder weights loaded into MoEMultiModalSynthesisModel.")

    def set_freeze_encoder(self, freeze=True):
        """Freeze/unfreeze encoder for fine-tuning"""
        self.freeze_encoder = freeze
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def forward_encoder_moe(self, x, modality_id):
        """
        Process single modality through encoder and its MoE expert

        Args:
            x: [B, 1, 96, 96, 96]
            modality_id: 'T1w' or 'T2w'
        Returns:
            moe_features: [B, 512, 24, 24, 24] after MoE
            x_48: [B, 256, 48, 48, 48] skip connection
            x_24: [B, 512, 24, 24, 24] before MoE
        """
        batch_size = x.shape[0]
        device = x.device

        # Encoder: 96 -> 48 -> 24
        x_48 = self.encoder.local_encoder[0](x)  # [B, 256, 48, 48, 48]
        x_24 = self.encoder.local_encoder[1](x_48)  # [B, 512, 24, 24, 24]

        # MoE routing with modality-specific expert
        if isinstance(modality_id, str):
            modality_id_tensor = torch.full((batch_size,),
                                           self.encoder.modality_to_id[modality_id],
                                           device=device, dtype=torch.long)
        else:
            modality_id_tensor = modality_id

        spatial_gates, global_gates = self.encoder.gate(x_24, modality_id_tensor)

        # Apply MoE experts
        moe_features = torch.zeros_like(x_24)
        for i, expert_modality in enumerate(self.encoder.modalities):
            expert = self.encoder.experts[expert_modality]
            expert_output = expert(x_24)

            spatial_weight = spatial_gates[:, i:i+1]
            global_weight = global_gates[:, i:i+1, None, None, None]
            combined_weight = spatial_weight * global_weight

            moe_features += expert_output * combined_weight

        return moe_features, x_48, x_24

    def forward(self, t1_img, t2_img):
        """
        Args:
            t1_img: [B, 1, 96, 96, 96]
            t2_img: [B, 1, 96, 96, 96]
        Returns:
            fa_img: [B, 1, 96, 96, 96]
        """
        # Process T1 through T1w expert
        if self.freeze_encoder:
            with torch.no_grad():
                t1_moe, t1_48, t1_24 = self.forward_encoder_moe(t1_img, 'T1w')
                t2_moe, t2_48, t2_24 = self.forward_encoder_moe(t2_img, 'T2w')
        else:
            t1_moe, t1_48, t1_24 = self.forward_encoder_moe(t1_img, 'T1w')
            t2_moe, t2_48, t2_24 = self.forward_encoder_moe(t2_img, 'T2w')

        # Fuse MoE outputs at 24x24x24
        fused_moe = torch.cat([t1_moe, t2_moe], dim=1)  # [B, 1024, 24, 24, 24]
        fused_moe = self.fusion_bottleneck(fused_moe)  # [B, 512, 24, 24, 24]

        # Fuse skip connections at 48x48x48 (before MoE)
        fused_48 = torch.cat([t1_48, t2_48], dim=1)  # [B, 512, 48, 48, 48]
        fused_48 = self.fusion_skip_48(fused_48)  # [B, 256, 48, 48, 48]

        # Apply connectors
        skip_48 = self.connector_skip_48(fused_48)  # [B, 256, 48] -> [B, 32, 96]
        skip_96 = self.connector_skip_96(fused_48)  # [B, 256, 48] -> [B, 16, 96]
        x_24 = self.connector_bottleneck(fused_moe)  # [B, 512, 24] -> [B, 32, 24]

        # Decode to FA
        output = self.decoder(x_24, skip_48=skip_48, skip_96=skip_96, x0=t1_img)

        return output


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)



class SynthesisLoss(nn.Module):
    """
    Combined loss for synthesis: MSE or L1 + perceptual loss
    """
    def __init__(self, loss_type='mse', weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight
        self.perceptual_weight = perceptual_weight

        if loss_type == 'mse':
            self.recon_loss = nn.MSELoss()
        elif loss_type == 'l1':
            self.recon_loss = nn.L1Loss()
        else:
            raise ValueError(f"loss_type must be 'mse' or 'l1', got {loss_type}")

    def forward(self, pred, target, pred_features=None, target_features=None):
        """
        Args:
            pred: Predicted image [B, 1, H, W, D]
            target: Target image [B, 1, H, W, D]
            pred_features: Encoded features of prediction (optional)
            target_features: Encoded features of target (optional)
        """
        # Reconstruction loss (MSE or L1)
        recon_loss = self.recon_loss(pred, target)

        # Perceptual loss (if features provided)
        perceptual_loss = 0
        if pred_features is not None and target_features is not None:
            perceptual_loss = self.recon_loss(pred_features, target_features)

        total_loss = self.weight * recon_loss + self.perceptual_weight * perceptual_loss

        return total_loss, recon_loss, perceptual_loss

