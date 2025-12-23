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
    WEIGHT_DECAY, DATA_DIR
)


def main():
    print("=" * 60)
    print("Training Quick Test (2 epochs)")
    print("=" * 60)
    
    print("\nLoading small subset of data...")
    train_loader, val_loader = create_dataloaders(
        root=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    print("Creating model...")
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
    model = model.to(DEVICE)
    
    print(f"Device: {DEVICE}")
    
    criterion = SegmentationLoss(dice_weight=1.0, ce_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    
    print("\nTesting training loop for 2 epochs...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        checkpoint_dir="./checkpoints_test",
        patience=10
    )
    
    trainer.train(num_epochs=2)
    
    print("\n" + "=" * 60)
    print("✓ Training loop test passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

