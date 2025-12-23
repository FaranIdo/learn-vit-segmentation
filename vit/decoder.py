"""Progressive Decoder: Upsample tokens to per-pixel predictions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleBlock(nn.Module):
    """Single upsampling step: bilinear 2× + conv + norm + relu."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ProgressiveDecoder(nn.Module):
    """
    Upsamples encoder tokens to full resolution segmentation map.
    
    Tokens [B, 196, 384] → Mask [B, num_classes, 224, 224]
    
    4 upsampling steps: 14→28→56→112→224
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_patches_per_side: int
    ):
        super().__init__()
        self.num_patches_per_side = num_patches_per_side
        
        self.blocks = nn.Sequential(
            UpsampleBlock(embed_dim, embed_dim // 2),      # 14→28, 384→192
            UpsampleBlock(embed_dim // 2, embed_dim // 4), # 28→56, 192→96
            UpsampleBlock(embed_dim // 4, embed_dim // 8), # 56→112, 96→48
            UpsampleBlock(embed_dim // 8, embed_dim // 8), # 112→224, 48→48
        )
        
        self.head = nn.Conv2d(embed_dim // 8, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        x = self._reshape_tokens_to_spatial(x, batch_size)
        x = self.blocks(x)
        x = self.head(x)
        
        return x
    
    def _reshape_tokens_to_spatial(
        self, 
        x: torch.Tensor, 
        batch_size: int
    ) -> torch.Tensor:
        """[B, 196, 384] → [B, 384, 14, 14]"""
        h = w = self.num_patches_per_side
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(batch_size, -1, h, w)
        return x

