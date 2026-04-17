"""
model.py  —  EfficientNet-B0 classifier for 9-class CRC tissue typing.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model(num_classes: int = 9, dropout: float = 0.3) -> nn.Module:
    """
    ImageNet-pretrained EfficientNet-B0 with a custom classification head.

    Architecture change:
        Original: Dropout(0.2) → Linear(1280, 1000)
        Ours:     Dropout(dropout) → Linear(1280, num_classes)
    """
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace classifier head
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_gradcam_target_layer(model: nn.Module):
    """Returns the last convolutional block — best layer for Grad-CAM."""
    return model.features[-1]


def count_parameters(model: nn.Module) -> str:
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total params: {total/1e6:.2f}M  |  Trainable: {trainable/1e6:.2f}M"
