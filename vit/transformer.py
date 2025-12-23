"""Transformer Block: Self-attention + MLP with residual connections."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Feed-forward network: expand → GELU → shrink."""
    
    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block: Attention + MLP with residuals.
    
    Input/Output shape: [batch, num_patches, embed_dim]
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self._attention_block(self.norm1(x))
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def _attention_block(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(query=x, key=x, value=x)
        return attn_output

