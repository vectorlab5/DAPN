"""
Feature Encoder Module for DAPN
ResNet-12 backbone for extracting feature representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Basic residual block for ResNet-12"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet12(nn.Module):
    """ResNet-12 backbone for feature extraction"""
    def __init__(self, feature_dim=512):
        super(ResNet12, self).__init__()
        self.feature_dim = feature_dim
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 128, 2, stride=2)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        
        # Final pooling and projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(Block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(Block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FeatureEncoder(nn.Module):
    """Feature Encoder that extracts base features"""
    def __init__(self, feature_dim=512):
        super(FeatureEncoder, self).__init__()
        self.backbone = ResNet12(feature_dim=feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input images [batch_size, 3, H, W]
        Returns:
            features: Base features [batch_size, feature_dim]
        """
        return self.backbone(x)
