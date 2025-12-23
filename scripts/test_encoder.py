"""Test script for ViTEncoder module."""

import sys

sys.path.insert(0, ".")

import torch
from vit.encoder import ViTEncoder
from config import IMAGE_SIZE, PATCH_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, MLP_RATIO, DROPOUT, DEVICE


def test_encoder():
    print("Testing ViTEncoder...")
    print(f"Device: {DEVICE}")

    encoder = ViTEncoder(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, dropout=DROPOUT).to(DEVICE)

    dummy_image = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    print(f"\nInput shape:  {dummy_image.shape}")

    output = encoder(dummy_image)
    print(f"Output shape: {output.shape}")

    num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
    expected = (2, num_patches, EMBED_DIM)
    assert output.shape == expected, f"Expected {expected}, got {output.shape}"

    print(f"\n✓ ViTEncoder works!")
    print(f"  - {NUM_LAYERS} transformer blocks")
    print(f"  - {num_patches} patches × {EMBED_DIM} dims")


def count_parameters():
    print("\nParameter count:")

    encoder = ViTEncoder(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, dropout=DROPOUT)

    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"  Total parameters: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Size: ~{total * 4 / 1024 / 1024:.1f} MB (float32)")


def main():
    print("=" * 50)
    print("ViT Encoder Tests")
    print("=" * 50)

    test_encoder()
    count_parameters()

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
