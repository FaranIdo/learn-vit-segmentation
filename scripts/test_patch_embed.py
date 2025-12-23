"""Test script for PatchEmbedding module."""

import sys

sys.path.insert(0, ".")

import torch
from vit.patch_embed import PatchEmbedding
from config import IMAGE_SIZE, PATCH_SIZE, EMBED_DIM, DEVICE


def test_patch_embedding():
    print("Testing PatchEmbedding...")
    print(f"Device: {DEVICE}")

    patch_embed = PatchEmbedding(IMAGE_SIZE, PATCH_SIZE, EMBED_DIM).to(DEVICE)

    dummy_image = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    print(f"Input shape:  {dummy_image.shape}")

    output = patch_embed(dummy_image)
    print(f"Output shape: {output.shape}")

    expected = (2, 196, 384)
    assert output.shape == expected, f"Expected {expected}, got {output.shape}"

    print(f"\n✓ PatchEmbedding works correctly!")
    print(f"  - {IMAGE_SIZE}x{IMAGE_SIZE} image → {output.shape[1]} patches of {output.shape[2]} dims")


if __name__ == "__main__":
    test_patch_embedding()
