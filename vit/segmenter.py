"""Segmenter: Complete model combining ViT encoder and progressive decoder."""

import torch
import torch.nn as nn

from .encoder import ViTEncoder
from .decoder import ProgressiveDecoder


class Segmenter(nn.Module):
    """
    Vision Transformer for semantic segmentation.
    
    Image [B, 3, H, W] → Segmentation mask [B, num_classes, H, W]
    """
    
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float
    ):
        super().__init__()
        
        self.encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        num_patches_per_side = image_size // patch_size
        
        self.decoder = ProgressiveDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_patches_per_side=num_patches_per_side
        )
    
    def forward(self, x: torch.Tensor, return_attention: bool = False, attention_layer_idx: int = -1):
        if return_attention:
            tokens, attention_weights = self.encoder(x, return_attention=True, attention_layer_idx=attention_layer_idx)
            mask = self.decoder(tokens)
            return mask, attention_weights
        else:
            tokens = self.encoder(x)
            mask = self.decoder(tokens)
            return mask

