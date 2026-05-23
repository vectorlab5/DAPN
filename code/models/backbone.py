"""
ResNet-12 backbone for few-shot learning.

This is the standard backbone used in few-shot learning literature,
providing a good balance between representational capacity and 
computational efficiency for episodic training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding - the workhorse of ResNet."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-12.
    
    Uses 3 conv layers per block (slightly different from standard ResNet)
    which is common in few-shot learning implementations. This provides
    more capacity while keeping the network relatively shallow.
    """
    
    expansion = 1
    
    def __init__(
        self, 
        inplanes: int, 
        planes: int, 
        stride: int = 1, 
        downsample: Optional[nn.Module] = None,
        drop_rate: float = 0.0
    ):
        super().__init__()
        
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        
        # Max pooling for downsampling spatial dimensions
        self.maxpool = nn.MaxPool2d(stride) if stride > 1 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        if self.maxpool is not None:
            out = self.maxpool(out)
        
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        
        return out


class ResNet12(nn.Module):
    """
    ResNet-12 backbone commonly used in few-shot learning.
    
    This architecture has 4 residual blocks with increasing channel sizes
    (64 -> 128 -> 256 -> 512), producing 512-dimensional feature vectors.
    The design follows common conventions in the few-shot learning literature.
    """
    
    def __init__(
        self, 
        channels: list = [64, 128, 256, 512],
        drop_rate: float = 0.1,
        avg_pool: bool = True
    ):
        super().__init__()
        
        self.inplanes = 3  # RGB input
        self.drop_rate = drop_rate
        self.avg_pool = avg_pool
        
        # Build the 4 residual blocks
        self.layer1 = self._make_layer(channels[0], stride=2)
        self.layer2 = self._make_layer(channels[1], stride=2)
        self.layer3 = self._make_layer(channels[2], stride=2)
        self.layer4 = self._make_layer(channels[3], stride=2)
        
        # Global average pooling to get fixed-size output
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Output dimension
        self.out_dim = channels[-1]
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _make_layer(self, planes: int, stride: int = 1) -> nn.Module:
        """Create a residual block with downsampling if needed."""
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes),
            )
        
        layer = BasicBlock(
            self.inplanes, planes, stride=stride, 
            downsample=downsample, drop_rate=self.drop_rate
        )
        self.inplanes = planes
        
        return layer
    
    def _initialize_weights(self):
        """Kaiming initialization for conv layers, standard for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input images of shape (batch_size, 3, H, W)
            
        Returns:
            Feature vectors of shape (batch_size, 512)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.avg_pool:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
        
        return x


def resnet12(drop_rate: float = 0.1) -> ResNet12:
    """Convenience function to create a ResNet-12 backbone."""
    return ResNet12(drop_rate=drop_rate)
