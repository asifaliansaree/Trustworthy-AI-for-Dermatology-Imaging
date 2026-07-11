"""
Configurable loss functions for imbalanced medical image classification.
Select via YAML: loss.name = cross_entropy | label_smoothing | focal | weighted_focal
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy examples so the model focuses on hard ones.
    Especially effective on imbalanced datasets like HAM10000.

    Lin et al. (2017) "Focal Loss for Dense Object Detection"

    Args:
        alpha:  per-class weight tensor (same as CrossEntropy weight)
        gamma:  focusing parameter. 0 = standard CrossEntropy. 2 is standard.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


def build_loss(cfg: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Build loss function from config dict.

    Config keys:
        loss.name:             cross_entropy | label_smoothing | focal | weighted_focal
        loss.label_smoothing:  float (default 0.1), used when name=label_smoothing
        loss.focal_gamma:      float (default 2.0), used when name=focal / weighted_focal
    """
    loss_cfg = cfg.get("loss", {})
    name     = loss_cfg.get("name", "cross_entropy")

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif name == "label_smoothing":
        smoothing = loss_cfg.get("label_smoothing", 0.1)
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=smoothing,
        )

    elif name == "focal":
        gamma = loss_cfg.get("focal_gamma", 2.0)
        return FocalLoss(alpha=None, gamma=gamma)

    elif name == "weighted_focal":
        gamma = loss_cfg.get("focal_gamma", 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)

    else:
        raise ValueError(
            f"Unknown loss '{name}'. "
            "Choose: cross_entropy | label_smoothing | focal | weighted_focal"
        )