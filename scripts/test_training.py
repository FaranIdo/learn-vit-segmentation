import sys
sys.path.insert(0, ".")

import torch
import logging
import os
from datetime import datetime
from torch.utils.data import Subset, DataLoader
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


def setup_test_logging(checkpoint_dir: str):
    """Setup logging for test training."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(checkpoint_dir, f"test_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    log_file = os.path.join(run_dir, "training.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return run_dir, log_file


def main():
    run_dir, log_file = setup_test_logging(TEST_CHECKPOINT_DIR)
    
    logging.info("=" * 60)
    logging.info(f"Training Quick Test ({TEST_EPOCHS} epochs)")
    logging.info("=" * 60)
    logging.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Test run dir: {os.path.abspath(run_dir)}")
    logging.info(f"Log file: {os.path.abspath(log_file)}")

    logging.info(f"\nLoading small subset of data ({TEST_TRAIN_SAMPLES} train, {TEST_VAL_SAMPLES} val)...")
    train_loader, val_loader = create_dataloaders(
        root=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    train_subset = Subset(train_dataset, range(min(TEST_TRAIN_SAMPLES, len(train_dataset))))
    val_subset = Subset(val_dataset, range(min(TEST_VAL_SAMPLES, len(val_dataset))))

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    logging.info(f"  Train samples: {len(train_subset)}")
    logging.info(f"  Val samples: {len(val_subset)}")

    logging.info("\nCreating model...")
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

    logging.info(f"Device: {DEVICE}")

    criterion = SegmentationLoss(dice_weight=1.0, ce_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TEST_EPOCHS)

    logging.info(f"\nTesting training loop for {TEST_EPOCHS} epochs...")
    logging.info(f"  Checkpoint dir: {run_dir}")
    logging.info(f"  TensorBoard: Disabled (test mode)")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        checkpoint_dir=run_dir,
        patience=10,
        use_tensorboard=False
    )

    trainer.train(num_epochs=TEST_EPOCHS)

    logging.info("\n" + "=" * 60)
    logging.info("✓ Training loop test passed!")
    logging.info(f"Log saved to: {os.path.abspath(log_file)}")
    logging.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
