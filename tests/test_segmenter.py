"""Test script for complete Segmenter model."""

import sys
sys.path.insert(0, ".")

import torch
from vit.segmenter import Segmenter
from config import (
    IMAGE_SIZE, PATCH_SIZE, NUM_CLASSES, EMBED_DIM,
    NUM_LAYERS, NUM_HEADS, MLP_RATIO, DROPOUT, DEVICE
)


def test_segmenter():
    print("Testing Segmenter (complete model)...")
    print(f"Device: {DEVICE}")
    
    model = Segmenter(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT
    ).to(DEVICE)
    
    dummy_image = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    print(f"\nInput:  {dummy_image.shape}")
    
    output = model(dummy_image)
    print(f"Output: {output.shape}")
    
    expected = (2, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)
    assert output.shape == expected, f"Expected {expected}, got {output.shape}"
    
    print(f"\n✓ Segmenter works!")
    print(f"  Image [{IMAGE_SIZE}×{IMAGE_SIZE}] → Mask [{NUM_CLASSES} classes, {IMAGE_SIZE}×{IMAGE_SIZE}]")


def count_parameters():
    print("\n" + "="*50)
    print("Model Parameters:")
    print("="*50)
    
    model = Segmenter(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT
    )
    
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = encoder_params + decoder_params
    
    print(f"  Encoder:  {encoder_params:>10,} ({encoder_params/1e6:.2f}M)")
    print(f"  Decoder:  {decoder_params:>10,} ({decoder_params/1e6:.2f}M)")
    print(f"  ─────────────────────────")
    print(f"  Total:    {total_params:>10,} ({total_params/1e6:.2f}M)")


def test_inference_speed():
    print("\n" + "="*50)
    print("Inference Speed (MPS):")
    print("="*50)
    
    model = Segmenter(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT
    ).to(DEVICE).eval()
    
    dummy_image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_image)
    
    import time
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    
    start = time.time()
    num_runs = 10
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_image)
            if DEVICE.type == "mps":
                torch.mps.synchronize()
    
    elapsed = time.time() - start
    avg_ms = (elapsed / num_runs) * 1000
    
    print(f"  Average: {avg_ms:.1f} ms per image")
    print(f"  Throughput: {1000/avg_ms:.1f} images/sec")


if __name__ == "__main__":
    test_segmenter()
    count_parameters()
    test_inference_speed()

