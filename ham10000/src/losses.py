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
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.reduction        = reduction
        self.label_smoothing  = label_smoothing

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        # Weighted loss (used for the loss magnitude itself).
        # label_smoothing here directly stops per-example loss from being
        # driven toward zero on easy/confident examples -- combats the
        # train_loss-collapses-toward-zero pattern seen in v2-v10 runs.
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction="none"
        )

        # pt must come from the UNWEIGHTED cross-entropy. If alpha is folded
        # in here too (via weight=self.alpha), pt stops being the model's
        # true class-confidence and becomes pt**alpha_c instead. For
        # up-weighted rare classes (large alpha_c) that artificially crushes
        # pt, which drives (1 - pt)**gamma toward its max on TOP OF the
        # already alpha-scaled ce_loss -- i.e. alpha gets applied twice,
        # concentrated on exactly the rare classes it's meant to help.
        # That's a real source of the loss spikes seen on rare-class-heavy
        # batches, and it gets worse (not better) once a class-balanced
        # sampler puts rare classes in every batch instead of occasionally.
        with torch.no_grad():
            ce_loss_unweighted = F.cross_entropy(
                inputs, targets, weight=None, reduction="none"
            )
            pt = torch.exp(-ce_loss_unweighted)

        focal = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


def compute_effective_num_weights(
    class_counts,
    beta: float = 0.999,
    num_classes: Optional[int] = None,
    device=None,
) -> torch.Tensor:
    """
    Class-Balanced weights via "effective number of samples".
    Cui et al. (2019) "Class-Balanced Loss Based on Effective Number of Samples"

    weight_c = (1 - beta) / (1 - beta ** n_c)
    then normalized so weights sum to num_classes (same convention as a
    typical inverse-frequency class_weights.npy, so it's a drop-in swap).

    Gentler than raw 1/n_c on very small classes (e.g. HAM10000's `df`
    at 86 train images, `vasc` at 114) — those classes stop getting
    disproportionately large weights past a point of diminishing returns,
    which tends to generalize better than pure inverse-frequency weighting.

    Args:
        class_counts: array-like of per-class training sample counts,
                      ordered to match your label indices (e.g. from
                      np.bincount(labels, minlength=num_classes)).
        beta:         closer to 1.0 = weights approach true inverse
                      frequency; smaller = gentler / closer to uniform.
                      0.999 is the value used in the original paper.
        num_classes:  inferred from len(class_counts) if not given.
    """
    class_counts = np.asarray(class_counts, dtype=np.float64)
    if num_classes is None:
        num_classes = len(class_counts)

    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loss(cfg: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Build loss function from config dict.

    Config keys:
        loss.name:             cross_entropy | label_smoothing | focal | weighted_focal
        loss.label_smoothing:  float (default 0.1), used when name=label_smoothing
        loss.focal_gamma:      float (default 2.0), used when name=focal / weighted_focal
        loss.alpha_mode:       inverse | effective_num (default: inverse)
                                Only used when name=weighted_focal.
                                "inverse"       -> use `class_weights` as-is
                                                   (unchanged prior behavior).
                                "effective_num" -> recompute gentler per-class
                                                   weights via Cui et al. (2019).
                                                   Requires loss.class_counts
                                                   and loss.effective_num_beta
                                                   (default 0.999) in the config.
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
        gamma           = loss_cfg.get("focal_gamma", 2.0)
        label_smoothing = loss_cfg.get("label_smoothing", 0.0)
        return FocalLoss(alpha=None, gamma=gamma, label_smoothing=label_smoothing)

    elif name == "weighted_focal":
        gamma           = loss_cfg.get("focal_gamma", 2.0)
        alpha_mode      = loss_cfg.get("alpha_mode", "inverse")
        label_smoothing = loss_cfg.get("label_smoothing", 0.0)

        if alpha_mode == "effective_num":
            class_counts = loss_cfg.get("class_counts")
            if class_counts is None:
                raise ValueError(
                    "loss.alpha_mode='effective_num' requires "
                    "loss.class_counts (list of per-class train sample "
                    "counts) in the config."
                )
            beta = loss_cfg.get("effective_num_beta", 0.999)
            device = class_weights.device if class_weights is not None else None
            alpha = compute_effective_num_weights(
                class_counts, beta=beta, device=device
            )
        elif alpha_mode == "inverse":
            alpha = class_weights
        else:
            raise ValueError(
                f"Unknown loss.alpha_mode '{alpha_mode}'. "
                "Choose: inverse | effective_num"
            )

        return FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)

    else:
        raise ValueError(
            f"Unknown loss '{name}'. "
            "Choose: cross_entropy | label_smoothing | focal | weighted_focal"
        )