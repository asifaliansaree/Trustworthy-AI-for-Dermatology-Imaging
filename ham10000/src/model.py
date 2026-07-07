import torch
import torch.nn as nn
from torchvision import models

class DermaNet(nn.Module):
    def __init__(self, num_classes=7, metadata_dim=0, pretrained=True, dropout=0.3):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        self.metadata_dim = metadata_dim
        fused_dim = in_features + metadata_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(self, image, meta=None):
        x = self.backbone(image)              # (batch, 512)
        if self.metadata_dim > 0:
            if meta is None:
                raise ValueError("metadata_dim > 0 but meta is None")
            x = torch.cat([x, meta], dim=1)    # (batch, 512 + metadata_dim)
        return self.classifier(x)              # (batch, 7) logits