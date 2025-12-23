import sys
sys.path.insert(0, ".")

import torch
from torch.utils.data import Subset
from vit.segmenter import Segmenter
from training.losses import SegmentationLoss
from training.trainer import Trainer
from data.oxford_pets import create_dataloaders
from config import (
    DEVICE, IMAGE_SIZE, PATCH_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS,
    MLP_RATIO, DROPOUT, NUM_CLASSES, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, DATA_DIR
)

# Test configuration
TEST_EPOCHS = 10
TEST_TRAIN_SAMPLES = 100
TEST_VAL_SAMPLES = 50
TEST_CHECKPOINT_DIR = "./checkpoints/test"


def main():
    print("=" * 60)
    print(f"Training Quick Test ({TEST_EPOCHS} epochs)")
    print("=" * 60)

    print(f"\nLoading small subset of data ({TEST_TRAIN_SAMPLES} train, {TEST_VAL_SAMPLES} val)...")
    train_loader, val_loader = create_dataloaders(
        root=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # Create subset for quick testing
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    train_subset = Subset(train_dataset, range(min(TEST_TRAIN_SAMPLES, len(train_dataset))))
    val_subset = Subset(val_dataset, range(min(TEST_VAL_SAMPLES, len(val_dataset))))

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train samples: {len(train_subset)}")
    print(f"  Val samples: {len(val_subset)}")

    print("\nCreating model...")
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TEST_EPOCHS)

    print(f"\nTesting training loop for {TEST_EPOCHS} epochs...")
    print(f"  Checkpoint dir: {TEST_CHECKPOINT_DIR}")
    print(f"  TensorBoard: Disabled (test mode)")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        checkpoint_dir=TEST_CHECKPOINT_DIR,
        patience=10,
        use_tensorboard=False
    )

    trainer.train(num_epochs=TEST_EPOCHS)

    print("\n" + "=" * 60)
    print("✓ Training loop test passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
