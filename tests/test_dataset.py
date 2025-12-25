import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from data import OxfordPetsDataset, create_dataloaders
from config import IMAGE_SIZE, BATCH_SIZE, NUM_CLASSES, DATA_DIR, DEVICE


def test_single_sample():
    print("=" * 50)
    print("Testing Single Sample")
    print("=" * 50)
    
    dataset = OxfordPetsDataset(DATA_DIR, "train", IMAGE_SIZE)
    print(f"Dataset size: {len(dataset)}")
    
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Mask dtype: {mask.dtype}")
    print(f"Mask unique values: {torch.unique(mask).tolist()}")
    
    assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert mask.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert mask.dtype == torch.long
    print("✓ Single sample test passed\n")


def test_dataloader():
    print("=" * 50)
    print("Testing DataLoader")
    print("=" * 50)
    
    train_loader, val_loader = create_dataloaders(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    images, masks = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch masks shape: {masks.shape}")
    
    assert images.shape == (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert masks.shape == (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)
    print("✓ DataLoader test passed\n")


def test_gpu_transfer():
    print("=" * 50)
    print("Testing GPU Transfer")
    print("=" * 50)
    
    train_loader, _ = create_dataloaders(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
    images, masks = next(iter(train_loader))
    
    images = images.to(DEVICE)
    masks = masks.to(DEVICE)
    
    print(f"Images device: {images.device}")
    print(f"Masks device: {masks.device}")
    print("✓ GPU transfer test passed\n")


def show_class_distribution():
    print("=" * 50)
    print("Class Distribution (First Batch)")
    print("=" * 50)
    
    train_loader, _ = create_dataloaders(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
    _, masks = next(iter(train_loader))
    
    total_pixels = masks.numel()
    pet_pixels = (masks == 1).sum().item()
    bg_pixels = (masks == 0).sum().item()
    
    print(f"Pet (foreground): {pet_pixels:,} ({100*pet_pixels/total_pixels:.1f}%)")
    print(f"Background: {bg_pixels:,} ({100*bg_pixels/total_pixels:.1f}%)")
    print("This imbalance is why we use Dice Loss!\n")


def main():
    print(f"Using device: {DEVICE}")
    print(f"Dataset: Oxford Pets ({NUM_CLASSES} classes)\n")
    test_single_sample()
    test_dataloader()
    test_gpu_transfer()
    show_class_distribution()
    print("=" * 50)
    print("All dataset tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
