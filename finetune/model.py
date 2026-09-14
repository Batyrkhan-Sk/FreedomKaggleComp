"""DINOv2 backbone with a projection head, partially unfrozen.

Two choices worth explaining:

* **Only the last few blocks train.** Fine-tuning all 300M+ parameters on 20,000
  images would overwrite the general visual structure DINOv2 learned from 142M
  images and overfit badly. The early blocks encode edges and texture that
  transfer regardless; the task-specific part is how the final layers compose
  them.
* **The retrieval embedding is not the projection output.** Contrastive training
  is done through the head, but the head is discarded at inference and the
  pooled backbone features are submitted. This is standard in self-supervised
  learning: the projection layer absorbs loss-specific distortions, and the
  representation just below it transfers better.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gem_pool(patch_tokens: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """Generalised mean over patch tokens -- emphasises distinctive regions.

    Survives cropping better than the CLS token alone, because a crop removes
    some patches but leaves the informative ones dominating the mean.
    """
    return patch_tokens.clamp(min=eps).pow(p).mean(dim=1).pow(1.0 / p)


class RetrievalNet(nn.Module):
    def __init__(self, backbone_name: str = "dinov2_vitl14", trainable_blocks: int = 4,
                 proj_dim: int = 512):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name, verbose=False)
        embed_dim = self.backbone.embed_dim

        for param in self.backbone.parameters():
            param.requires_grad = False
        for block in self.backbone.blocks[-trainable_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        # CLS and GeM are concatenated, hence 2 * embed_dim.
        self.head = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, proj_dim),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled backbone representation -- this is what gets submitted."""
        out = self.backbone.forward_features(x)
        cls = F.normalize(out["x_norm_clstoken"], dim=1)
        gem = F.normalize(gem_pool(out["x_norm_patchtokens"]), dim=1)
        return torch.cat([cls, gem], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projected embedding -- used only for the contrastive loss."""
        return F.normalize(self.head(self.features(x)), dim=1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
