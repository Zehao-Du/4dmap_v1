"""DINOv3 ViT-S/16 visual encoder for Diffusion Policy.

Frozen DINOv3 backbone + trainable linear projection on patch-token mean.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_backbone(third_party_dir: str, weights_path: str, model: str):
    if third_party_dir not in sys.path:
        sys.path.insert(0, third_party_dir)
    from dinov3.hub.backbones import (
        dinov3_vits16,
        dinov3_vits16plus,
        dinov3_vitb16,
        dinov3_vitl16,
        dinov3_vith16plus,
    )
    builders = {
        "dinov3_vits16": dinov3_vits16,
        "dinov3_vits16plus": dinov3_vits16plus,
        "dinov3_vitb16": dinov3_vitb16,
        "dinov3_vitl16": dinov3_vitl16,
        "dinov3_vith16plus": dinov3_vith16plus,
    }
    if model not in builders:
        raise ValueError(f"Unsupported DINOv3 model {model!r}; choose from {list(builders)}")
    return builders[model](pretrained=True, weights=weights_path)


class DinoV3VisualEncoder(nn.Module):
    """Frozen DINOv3 ViT + trainable linear projection.

    Input: RGB images in [0, 1] with shape [B, 3, H, W] (H, W multiples of 16).
    Output: [B, out_dim] feature vector.
    """

    def __init__(
        self,
        out_dim: int = 256,
        weights_path: str = "",
        third_party_dir: str = "",
        model: str = "dinov3_vits16",
    ):
        super().__init__()
        if not weights_path or not Path(weights_path).exists():
            raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
        if not third_party_dir or not Path(third_party_dir).exists():
            raise FileNotFoundError(f"DINOv3 third_party dir not found: {third_party_dir}")

        self.backbone = _load_backbone(third_party_dir, weights_path, model)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.embed_dim = self.backbone.embed_dim
        self.proj = nn.Linear(self.embed_dim, out_dim)
        self.out_dim = out_dim

        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep backbone in eval mode regardless of outer train/eval state.
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] in [0, 1].
        x = (x - self._mean) / self._std
        with torch.no_grad():
            feats = self.backbone.forward_features(x)
        patch = feats["x_norm_patchtokens"]  # [B, N, embed_dim]
        pooled = patch.mean(dim=1)
        return self.proj(pooled)
