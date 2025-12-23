"""Patch Embedding: Convert image to sequence of patch tokens."""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and projects them to embeddings.
    
    Image [B, 3, 224, 224] → Patch Tokens [B, 196, 384]
    """
    
    def __init__(self, image_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 3, 224, 224]
        
        x = self.projection(x)
        # x: [batch, 384, 14, 14]
        
        x = x.flatten(2)
        # x: [batch, 384, 196]
        
        x = x.transpose(1, 2)
        # x: [batch, 196, 384]
        
        x = x + self.pos_embed
        # Add position information to each patch
        
        return x

