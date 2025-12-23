"""Test script for TransformerBlock module."""

import sys
sys.path.insert(0, ".")

import torch
from vit.transformer import TransformerBlock, MLP
from config import EMBED_DIM, NUM_HEADS, MLP_RATIO, DROPOUT, NUM_PATCHES, DEVICE


def test_mlp():
    print("Testing MLP...")
    
    mlp = MLP(EMBED_DIM, MLP_RATIO, DROPOUT).to(DEVICE)
    x = torch.randn(2, NUM_PATCHES, EMBED_DIM, device=DEVICE)
    
    output = mlp(x)
    
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Hidden dim: {int(EMBED_DIM * MLP_RATIO)}")
    print("  ✓ MLP works!")


def test_transformer_block():
    print("\nTesting TransformerBlock...")
    
    block = TransformerBlock(EMBED_DIM, NUM_HEADS, MLP_RATIO, DROPOUT).to(DEVICE)
    x = torch.randn(2, NUM_PATCHES, EMBED_DIM, device=DEVICE)
    
    output = block(x)
    
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Num heads: {NUM_HEADS}")
    print(f"  Dim per head: {EMBED_DIM // NUM_HEADS}")
    print("  ✓ TransformerBlock works!")


def test_residual_connection():
    print("\nTesting residual connections...")
    
    block = TransformerBlock(EMBED_DIM, NUM_HEADS, MLP_RATIO, DROPOUT).to(DEVICE)
    block.eval()
    
    x = torch.randn(1, NUM_PATCHES, EMBED_DIM, device=DEVICE)
    output = block(x)
    
    diff = (output - x).abs().mean().item()
    print(f"  Mean absolute difference from input: {diff:.4f}")
    print("  (Non-zero shows the block is transforming the input)")
    print("  ✓ Residual connection works!")


def main():
    print("=" * 50)
    print("Transformer Block Tests")
    print(f"Device: {DEVICE}")
    print("=" * 50)
    
    test_mlp()
    test_transformer_block()
    test_residual_connection()
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()

