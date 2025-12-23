"""Sanity check script to verify all dependencies and GPU setup."""

import sys
sys.path.insert(0, ".")


def check_pytorch():
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    return torch.backends.mps.is_available()


def test_mps_operations():
    import torch
    device = torch.device("mps")
    x = torch.randn(3, 3, device=device)
    y = x @ x.T
    print(f"MPS tensor multiplication: SUCCESS")
    print(f"Result shape: {y.shape}")


def check_config():
    from config import DEVICE, IMAGE_SIZE, PATCH_SIZE, NUM_PATCHES, EMBED_DIM
    print(f"\nConfig loaded successfully:")
    print(f"  DEVICE: {DEVICE}")
    print(f"  IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"  PATCH_SIZE: {PATCH_SIZE}")
    print(f"  NUM_PATCHES: {NUM_PATCHES}")
    print(f"  EMBED_DIM: {EMBED_DIM}")


def check_dependencies():
    print("\nChecking dependencies:")
    
    import torchvision
    print(f"  torchvision: {torchvision.__version__}")
    
    import transformers
    print(f"  transformers: {transformers.__version__}")
    
    import datasets
    print(f"  datasets: {datasets.__version__}")
    
    import matplotlib
    print(f"  matplotlib: {matplotlib.__version__}")
    
    import numpy
    print(f"  numpy: {numpy.__version__}")


def main():
    print("=" * 50)
    print("ViT Segmentation Tutorial - Sanity Check")
    print("=" * 50)
    
    print("\n1. PyTorch & MPS Check:")
    mps_available = check_pytorch()
    
    if mps_available:
        print("\n2. MPS Operations Test:")
        test_mps_operations()
    else:
        print("\n2. MPS not available, skipping GPU test")
    
    print("\n3. Config Check:")
    check_config()
    
    check_dependencies()
    
    print("\n" + "=" * 50)
    print("All checks passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()

