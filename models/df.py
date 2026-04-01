"""
Implementation of the Deep Fingerprinting (DF) model.
"""

from __future__ import annotations
from typing import Sequence
import torch
from torch import nn


def conv1d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dropout: float = 0.1,
) -> nn.Sequential:
    """
    Standard 1D Convolutional block including:
    Conv1d -> BatchNorm1d -> ReLU -> Dropout
    """
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )


class DFBackbone(nn.Module):
    """
    Deep Fingerprinting (DF) Feature Extraction Network.
    Optimized for sequence-based website fingerprinting and traffic classification.
    """

    def __init__(
        self,
        input_length: int,
        conv_channels: Sequence[int] = (32, 64, 128, 256),
        kernel_sizes: Sequence[int] = (8, 8, 4, 4),
        pooling_sizes: Sequence[int] = (4, 4, 2, 2),
        dropout: float = 0.1,
        fc_hidden: Sequence[int] = (512, 128),
    ) -> None:
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes) == len(pooling_sizes)
        layers: nn.ModuleList = nn.ModuleList()

        # Input channels = 3 (Direction, Delta Time, Normalized Payload Size)
        in_channels = 3
        for out_channels, k, pool in zip(conv_channels, kernel_sizes, pooling_sizes):
            padding = k // 2
            # Each block consists of two convolutional layers followed by a MaxPool
            layers.append(conv1d_block(in_channels, out_channels, k, padding=padding, dropout=dropout))
            layers.append(conv1d_block(out_channels, out_channels, k, padding=padding, dropout=dropout))
            layers.append(nn.MaxPool1d(kernel_size=pool, stride=pool))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        
        # Automatically calculate the dimension after flattening for the FC layers
        self.feature_dim = self._get_flatten_size(input_length)

        # Fully Connected (FC) layers for feature refinement
        fcs: nn.ModuleList = nn.ModuleList()
        in_dim = self.feature_dim
        for hidden_dim in fc_hidden:
            fcs.append(nn.Linear(in_dim, hidden_dim))
            fcs.append(nn.BatchNorm1d(hidden_dim))
            fcs.append(nn.ReLU(inplace=True))
            fcs.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.classifier = nn.Sequential(*fcs)
        self.output_dim = in_dim

    def _get_flatten_size(self, input_length: int) -> int:
        """
        Performs a forward pass with a dummy input to calculate the 
        flattened feature dimension automatically.
        """
        with torch.no_grad():
            # Dummy tensor shape: (Batch=1, Channels=3, Length=input_length)
            dummy_input = torch.zeros(1, 3, input_length)
            output = self.features(dummy_input)
            return output.numel()  # Total number of elements

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Sequence tensor of shape (B, 3, L).
        Returns:
            torch.Tensor: Refined feature vector.
        """
        features = self.features(x)
        features = torch.flatten(features, start_dim=1)
        return self.classifier(features)