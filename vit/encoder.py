"""ViT Encoder: PatchEmbedding + stack of TransformerBlocks."""

import torch
import torch.nn as nn

from .patch_embed import PatchEmbedding
from .transformer import TransformerBlock


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder for segmentation.

    Returns all patch tokens (not just CLS) for dense prediction.

    Image [B, 3, H, W] → Tokens [B, num_patches, embed_dim]
    """

    def __init__(self, image_size: int, patch_size: int, embed_dim: int, num_layers: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches_per_side = image_size // patch_size

        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)

        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 3, H, W]

        x = self.patch_embed(x)
        # x: [batch, num_patches, embed_dim]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x
