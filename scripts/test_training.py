import sys
sys.path.insert(0, ".")

import torch
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
    """Setup logging for test training using stdout redirection."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(checkpoint_dir, f"test_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "training.log")

    class DualOutput:
        """Write to both file and terminal."""
        def __init__(self, file_path, terminal):
            self.file = open(file_path, 'w')
            self.terminal = terminal
        
        def write(self, message):
            self.terminal.write(message)
            self.file.write(message)
            self.file.flush()
        
        def flush(self):
            self.terminal.flush()
            self.file.flush()
        
        def close(self):
            self.file.close()
    
    original_stdout = sys.stdout
    sys.stdout = DualOutput(log_file, original_stdout)

    return run_dir, log_file, original_stdout


def main():
    run_dir, log_file, original_stdout = setup_test_logging(TEST_CHECKPOINT_DIR)

    try:
        print("=" * 60)
        print(f"Training Quick Test ({TEST_EPOCHS} epochs)")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test run dir: {os.path.abspath(run_dir)}")
        print(f"Log file: {os.path.abspath(log_file)}")

        print(f"\nLoading small subset of data ({TEST_TRAIN_SAMPLES} train, {TEST_VAL_SAMPLES} val)...")
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
        print(f"  Checkpoint dir: {run_dir}")
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
            checkpoint_dir=run_dir,
            patience=10,
            use_tensorboard=False,
        )

        trainer.train(num_epochs=TEST_EPOCHS)

        print("\n" + "=" * 60)
        print("✓ Training loop test passed!")
        print(f"Log saved to: {os.path.abspath(log_file)}")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()
