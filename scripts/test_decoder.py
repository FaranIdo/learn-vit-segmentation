"""Test script for ProgressiveDecoder module."""

import sys
sys.path.insert(0, ".")

import torch
from vit.decoder import ProgressiveDecoder, UpsampleBlock
from config import EMBED_DIM, NUM_CLASSES, NUM_PATCHES, DEVICE, IMAGE_SIZE, PATCH_SIZE


def test_upsample_block():
    print("Testing UpsampleBlock...")
    
    block = UpsampleBlock(384, 192).to(DEVICE)
    x = torch.randn(2, 384, 14, 14, device=DEVICE)
    
    output = block(x)
    
    expected = (2, 192, 28, 28)
    assert output.shape == expected, f"Expected {expected}, got {output.shape}"
    
    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")
    print("  ✓ UpsampleBlock works!")


def test_decoder():
    print("\nTesting ProgressiveDecoder...")
    
    num_patches_per_side = IMAGE_SIZE // PATCH_SIZE
    
    decoder = ProgressiveDecoder(
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        num_patches_per_side=num_patches_per_side
    ).to(DEVICE)
    
    tokens = torch.randn(2, NUM_PATCHES, EMBED_DIM, device=DEVICE)
    print(f"  Input (tokens): {tokens.shape}")
    
    output = decoder(tokens)
    print(f"  Output (mask):  {output.shape}")
    
    expected = (2, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)
    assert output.shape == expected, f"Expected {expected}, got {output.shape}"
    
    print(f"\n  ✓ ProgressiveDecoder works!")
    print(f"    {NUM_PATCHES} tokens → {IMAGE_SIZE}×{IMAGE_SIZE} mask")
    print(f"    {NUM_CLASSES} classes per pixel")


def count_parameters():
    print("\nDecoder parameters:")
    
    num_patches_per_side = IMAGE_SIZE // PATCH_SIZE
    decoder = ProgressiveDecoder(EMBED_DIM, NUM_CLASSES, num_patches_per_side)
    
    total = sum(p.numel() for p in decoder.parameters())
    print(f"  Total: {total:,} ({total/1e6:.2f}M)")


if __name__ == "__main__":
    test_upsample_block()
    test_decoder()
    count_parameters()

