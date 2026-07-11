"""
DermaNet — configurable backbone for HAM10000 classification.
Supported architectures: resnet18, efficientnet_b0, efficientnet_b1, efficientnet_v2_s
Supports optional metadata late-fusion and staged fine-tuning.
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


ARCH_REGISTRY = {
    "resnet18": (
        models.resnet18,
        models.ResNet18_Weights.IMAGENET1K_V1,
        512,
    ),
    "efficientnet_b0": (
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        1280,
    ),
    "efficientnet_b1": (
        models.efficientnet_b1,
        models.EfficientNet_B1_Weights.IMAGENET1K_V1,
        1280,
    ),
    "efficientnet_v2_s": (
        models.efficientnet_v2_s,
        models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        1280,
    ),
}


class DermaNet(nn.Module):
    """
    Unified backbone wrapper with optional metadata late-fusion.

    Args:
        num_classes:   number of output classes (7 for HAM10000)
        metadata_dim:  dimension of metadata vector; 0 = image-only
        pretrained:    whether to load ImageNet weights
        dropout:       dropout rate before classifier head
        arch:          architecture name (see ARCH_REGISTRY)
        freeze_epochs: freeze backbone for this many epochs (staged fine-tuning)
    """

    def __init__(
        self,
        num_classes:   int   = 7,
        metadata_dim:  int   = 0,
        pretrained:    bool  = True,
        dropout:       float = 0.3,
        arch:          str   = "resnet18",
        freeze_epochs: int   = 0,
    ):
        super().__init__()

        if arch not in ARCH_REGISTRY:
            raise ValueError(
                f"Unknown architecture '{arch}'. "
                f"Choose from: {list(ARCH_REGISTRY.keys())}"
            )

        builder, weights_enum, feature_dim = ARCH_REGISTRY[arch]
        self.arch         = arch
        self.feature_dim  = feature_dim
        self.metadata_dim = metadata_dim
        self.freeze_epochs = freeze_epochs

        backbone = builder(weights=weights_enum if pretrained else None)

        # Strip classifier head — keep only feature extractor
        if arch == "resnet18":
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
            # EfficientNet variants: replace classifier with Identity
            backbone.classifier = nn.Identity()
            self.backbone = backbone

        fused_dim = feature_dim + metadata_dim

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, num_classes),
        )

    # ── Staged fine-tuning helpers ──────────────────────────────
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_parameter_groups(
        self,
        lr_backbone: float = 1e-5,
        lr_classifier: float = 1e-3,
    ) -> list:
        """
        Discriminative learning rates:
          - backbone: lr_backbone
          - classifier head: lr_classifier
        Pass the returned list directly to an optimizer.
        """
        return [
            {"params": self.backbone.parameters(),   "lr": lr_backbone},
            {"params": self.classifier.parameters(), "lr": lr_classifier},
        ]

    # ── Forward ───────────────────────────────────────────────
    def forward(
        self,
        image: torch.Tensor,
        meta:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = self.backbone(image)

        # EfficientNet with AdaptiveAvgPool already flattens; ResNet needs squeeze
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        elif x.dim() == 2:
            pass
        else:
            x = x.view(x.size(0), -1)

        if self.metadata_dim > 0 and meta is not None:
            x = torch.cat([x, meta], dim=1)

        return self.classifier(x)


def build_model(cfg: dict) -> DermaNet:
    """Build DermaNet from a YAML config dict."""
    return DermaNet(
        num_classes=cfg["model"]["num_classes"],
        metadata_dim=cfg["model"].get("metadata_dim", 0),
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.3),
        arch=cfg["model"].get("architecture", "resnet18"),
        freeze_epochs=cfg["model"].get("freeze_epochs", 0),
    )