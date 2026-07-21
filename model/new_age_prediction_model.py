"""
Brain age stage classification model using modality-only MoE encoder.

Stage 1 only: x (B,1,96³) → ModalityMoEEncoder → Connector → ResNet-18 → logits (B,6)

No age_group_ids passed to encoder (avoids label leakage).
No pretrained weights loaded; trains from scratch.
"""

import torch
import torch.nn as nn

from model.modality_moe_encoder import ModalityMoEEncoder
from model.ResNet3D import BasicBlock


class Connector(nn.Module):
    """1×1 conv to reduce encoder output channels to ResNet input channels."""

    def __init__(self, in_channels=512, out_channels=64):
        super().__init__()
        self.connector = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.connector(x)


class ModifiedResNet18(nn.Module):
    """
    ResNet-18 starting from (B, 64, 24³) — skips initial conv/pool.

    Input : (B, 64, 24, 24, 24)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes=6, dropout=True):
        super().__init__()
        self.inplanes = 64

        self.layer1 = self._make_layer(BasicBlock,  64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5) if dropout else None
        self.fc      = nn.Linear(512 * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc(x)


class NewAgePredictionModel(nn.Module):
    """
    Lifespan stage classifier using modality-only MoE encoder.

    x (B,1,96³) → ModalityMoEEncoder → Connector (512→64) →
    ModifiedResNet18 → logits (B, num_classes)

    Call signature matches the existing training loop:
        model(images, modality_ids=modalities)
    """

    def __init__(self, cfg, pretrained_encoder_path=None):
        super().__init__()
        self.cfg = cfg

        self.encoder   = ModalityMoEEncoder(cfg)
        if pretrained_encoder_path is not None:
            self._load_pretrained_encoder(pretrained_encoder_path)

        self.connector = Connector(in_channels=cfg.model.embed_dim,
                                   out_channels=64)
        num_classes = getattr(cfg.model, 'num_age_classes', 6)
        self.resnet  = ModifiedResNet18(num_classes=num_classes, dropout=True)

    def forward(self, x, modality_ids=None):
        """
        Args:
            x            : (B, 1, 96, 96, 96)
            modality_ids : list of strings / single string / LongTensor
        Returns:
            logits : (B, num_classes)
        """
        feat   = self.encoder(x, modality_ids)
        feat   = self.connector(feat)
        return self.resnet(feat)

    def set_freeze_encoder(self, freeze=True):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def _load_pretrained_encoder(self, pretrained_path):
        print(f"Loading pretrained encoder from {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)

        # Keep only keys that belong to the encoder components
        encoder_prefixes = ('local_encoder.', 'modality_experts.', 'mod_gate.')
        encoder_sd = {k: v for k, v in state_dict.items()
                      if any(k.startswith(p) for p in encoder_prefixes)}

        target_sd = self.encoder.state_dict()
        matched   = {k: v for k, v in encoder_sd.items()
                     if k in target_sd and v.shape == target_sd[k].shape}

        target_sd.update(matched)
        self.encoder.load_state_dict(target_sd)
        print(f"Matched {len(matched)} / {len(target_sd)} encoder parameters")
