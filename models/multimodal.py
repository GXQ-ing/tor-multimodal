"""
Multi-modal classifier with a Gated Fusion mechanism.
"""

from __future__ import annotations
from typing import Dict, Optional
import torch
from torch import nn
from .df import DFBackbone
from .resnet import ResidualBackbone

class SharedRepresentation(nn.Module):
    """
    A multi-layer perceptron (MLP) to process the fused multi-modal features.
    """
    def __init__(self, input_dim: int, hidden_dims: Optional[list[int]] = None, dropout: float = 0.3) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiModalClassifier(nn.Module):
    """
    Multi-modal classifier for Tor traffic.
    Includes a Gating Mechanism to automatically adjust the contribution 
    of sequence and image modalities.
    """
    def __init__(
        self,
        input_length: int,
        num_classes: int,
        df_kwargs: Optional[Dict] = None,
        resnet_kwargs: Optional[Dict] = None,
        shared_hidden: Optional[list[int]] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        
        # Backbone networks for feature extraction
        self.sequence_backbone = DFBackbone(input_length=input_length, **(df_kwargs or {}))
        self.image_backbone = ResidualBackbone(**(resnet_kwargs or {}))
        
        # Gating Mechanism: Learns a scalar weight for each branch based on the feature importance
        self.seq_gate = nn.Sequential(
            nn.Linear(self.sequence_backbone.output_dim, 1),
            nn.Sigmoid()
        )
        self.img_gate = nn.Sequential(
            nn.Linear(self.image_backbone.output_dim, 1),
            nn.Sigmoid()
        )

        # Fusion Layer: Concatenation followed by shared representation layers
        fusion_dim = self.sequence_backbone.output_dim + self.image_backbone.output_dim
        self.shared = SharedRepresentation(fusion_dim, hidden_dims=shared_hidden or [512, 256], dropout=dropout)
        
        # Final classification head
        self.classifier = nn.Linear(self.shared.output_dim, num_classes)

    def forward(self, sequence: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # 1. Extract modality-specific features
        seq_feat = self.sequence_backbone(sequence)
        img_feat = self.image_backbone(image)

        # 2. Compute gating weights
        g_seq = self.seq_gate(seq_feat)
        g_img = self.img_gate(img_feat)

        # 3. Weighted feature fusion
        # The gating mechanism ensures that a modality only contributes significantly 
        # when it provides useful information for the classification task.
        fused = torch.cat([seq_feat * g_seq, img_feat * g_img], dim=1)
        
        # 4. Classification
        shared_out = self.shared(fused)
        logits = self.classifier(shared_out)
        return logits

    def freeze_feature_extractors(self) -> None:
        """Freeze backbone weights; useful for training only the fusion and head."""
        for param in self.sequence_backbone.parameters():
            param.requires_grad = False
        for param in self.image_backbone.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractors(self) -> None:
        """Unfreeze backbones for full model fine-tuning."""
        for param in self.sequence_backbone.parameters():
            param.requires_grad = True
        for param in self.image_backbone.parameters():
            param.requires_grad = True