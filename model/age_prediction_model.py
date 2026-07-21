"""
Age Prediction Model for Lifespan Foundation

Architecture:
    Input (192³) → Downsample (96³) → Pretrained MAE Encoder →
    → (512, 24, 24, 24) → Connector → (64, 24, 24, 24) →
    → Modified ResNet-18 (from 24³) → 6 classes

The model uses:
1. Pretrained MAE encoder (can be frozen or fine-tuned) to extract features
2. Connector to reduce channels 512→64
3. Modified ResNet-18 starting from 24³ input

Note: Images are first padded/cropped to 192³ (suitable for fetal brains),
      then downsampled to 96³ before passing through the encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.lifespan_moe_mae import LifespanMoEMAE
from model.ResNet3D import BasicBlock


class Connector(nn.Module):
    """
    Connector module to reduce channels from MAE output to ResNet input
    512 channels → 64 channels, keeping spatial size (24³)
    """
    def __init__(self, in_channels=512, out_channels=64):
        super().__init__()

        self.connector = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.connector(x)


class ModifiedResNet18(nn.Module):
    """
    Modified ResNet-18 that starts from 24³ input with 64 channels

    Skips the initial conv1, bn1, relu, maxpool layers
    Directly uses layer1, layer2, layer3, layer4 from ResNet-18

    Input: (B, 64, 24, 24, 24)
    Output: (B, num_classes)
    """
    def __init__(self, num_classes=6, dropout=True):
        super().__init__()

        # ResNet-18 configuration: [2, 2, 2, 2] BasicBlocks
        # Channel progression: 64 -> 128 -> 256 -> 512

        self.inplanes = 64  # Start with 64 channels from connector

        # Build ResNet layers (skip initial conv/pool)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)   # 24³ → 24³
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  # 24³ → 12³
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  # 12³ → 6³
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)  # 6³ → 3³

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 3³ → 1³

        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = None

        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """Build a residual layer with given number of blocks"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input features (B, 64, 24, 24, 24)
        Returns:
            logits: Class logits (B, num_classes)
        """
        x = self.layer1(x)  # (B, 64, 24, 24, 24)
        x = self.layer2(x)  # (B, 128, 12, 12, 12)
        x = self.layer3(x)  # (B, 256, 6, 6, 6)
        x = self.layer4(x)  # (B, 512, 3, 3, 3)

        x = self.avgpool(x)  # (B, 512, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.fc(x)  # (B, num_classes)

        return x


class AgePredictionModel(nn.Module):
    """
    Complete age prediction model with encoder (MAE + MoE)

    Pipeline:
        Input (96³) → MAE Encoder (frozen/trainable) → (512, 24³)
        → Connector → (64, 24³) → ResNet-18 → 6 classes
    """
    def __init__(self, cfg, pretrained_encoder_path=None):
        super().__init__()

        self.cfg = cfg

        # Load pretrained MAE encoder
        self.encoder = LifespanMoEMAE(cfg)

        if pretrained_encoder_path is not None:
            self._load_pretrained_encoder(pretrained_encoder_path)

        # Connector: 512 → 64 channels
        self.connector = Connector(in_channels=512, out_channels=64)

        # Modified ResNet-18 classifier
        self.resnet = ModifiedResNet18(
            num_classes=cfg.model.num_age_classes if hasattr(cfg.model, 'num_age_classes') else 7,
            dropout=True
        )

    def forward(self, x, modality_ids=None):
        """
        Args:
            x: Input images (B, 1, 96, 96, 96)
            modality_ids: Modality identifiers for MAE encoder
        Returns:
            logits: Class predictions (B, num_classes)
        """
        # Extract features using encoder with MoE (no masking for downstream tasks)
        # mask_ratio=0 means no masking, p=4 is the patch size
        # x shape output: (B, 512, 24, 24, 24)
        x, _ = self.encoder.forward_encoder_with_moe(x, modality_ids=modality_ids, mask_ratio=0.0, p=int(self.cfg.train.local_mae_patch))

        # Reduce channels: 512 → 64
        x = self.connector(x)  # (B, 64, 24, 24, 24)

        # ResNet classification
        logits = self.resnet(x)  # (B, num_classes)

        return logits
    
    def set_freeze_encoder(self, freeze=True):
        """Freeze/unfreeze encoder for fine-tuning"""
        self.freeze_encoder = freeze
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

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


    def train(self, mode=True):
        """Override train to keep encoder frozen if needed"""
        super().train(mode)
        # Keep encoder in eval mode if frozen
        if hasattr(self, 'freeze_encoder') and self.freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        return self
