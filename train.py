import sys
sys.path.insert(0, ".")

import torch
from vit.segmenter import Segmenter
from training.losses import SegmentationLoss
from training.trainer import Trainer
from data.oxford_pets import create_dataloaders
from config import (
    DEVICE, IMAGE_SIZE, PATCH_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS,
    MLP_RATIO, DROPOUT, NUM_CLASSES, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, NUM_EPOCHS, DATA_DIR, CHECKPOINT_DIR
)


def create_model():
    """Create the segmentation model."""
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
    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print("=" * 60)
    print("ViT Segmentation - Training from Scratch")
    print("=" * 60)
    
    print("\n1. Loading Data...")
    train_loader, val_loader = create_dataloaders(
        root=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    print(f"   Batch size: {BATCH_SIZE}")
    
    print("\n2. Creating Model...")
    model = create_model()
    model = model.to(DEVICE)
    
    total_params, trainable_params = count_parameters(model)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Device: {DEVICE}")
    
    print("\n3. Setting up Training...")
    criterion = SegmentationLoss(dice_weight=1.0, ce_weight=1.0)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )
    
    print(f"   Loss: Dice + Cross Entropy")
    print(f"   Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"   Scheduler: CosineAnnealingLR (T_max={NUM_EPOCHS})")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Early stopping patience: 10")
    
    print("\n4. Training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        checkpoint_dir=CHECKPOINT_DIR,
        patience=10
    )
    
    best_miou = trainer.train(num_epochs=NUM_EPOCHS)
    
    print(f"\n5. Training Complete!")
    print(f"   Best Validation mIoU: {best_miou:.4f}")
    print(f"   Checkpoints saved to: {CHECKPOINT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

