"""
Custom lightweight ResNet for image modality feature extraction.
"""

from __future__ import annotations
from typing import Sequence
import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    A standard Residual Block with two 3x3 convolutions and a skip connection.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample is used when the input dimensions don't match the output (stride > 1 or channel change)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Skip connection: add input (identity) to the output of convolutions
        out += identity
        out = self.relu(out)
        return out


def make_layer(in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
    """Helper function to create a stage consisting of multiple residual blocks."""
    layers = [ResidualBlock(in_channels, out_channels, stride)]
    for _ in range(1, blocks):
        layers.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class ResidualBackbone(nn.Module):
    """
    Simplified ResNet architecture optimized for grayscale traffic images.
    Input shape is typically (Batch, 1, 64, 64).
    """

    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 32,
        layers: Sequence[int] = (2, 2, 2, 2),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = base_channels
        
        # Stem: Initial convolution and pooling to reduce spatial dimensions
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        channels = base_channels
        stride_plan = [1, 2, 2, 2]
        stages = []
        for idx, num_blocks in enumerate(layers):
            stride = stride_plan[idx]
            # Double the channels every time we increase stride (except the first stage)
            out_channels = channels * 2 if idx > 0 else channels
            stages.append(make_layer(channels, out_channels, num_blocks, stride=stride))
            channels = out_channels
            
        self.stages = nn.Sequential(*stages)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.output_dim = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Grayscale image tensor of shape (B, 1, H, W).
        Returns:
            torch.Tensor: Flattened spatial feature vector.
        """
        x = self.stem(x)
        x = self.stages(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return x