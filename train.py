import sys
sys.path.insert(0, ".")

import argparse
import torch
import os
from datetime import datetime
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


def create_run_directory(base_dir: str, experiment_name: str = None) -> str:
    """
    Create a unique run directory with timestamp.
    
    Args:
        base_dir: Base checkpoint directory
        experiment_name: Optional experiment name (e.g., 'baseline', 'high_lr')
    
    Returns:
        Path to the run directory
        
    Examples:
        create_run_directory('./checkpoints', None)
        # → './checkpoints/run_2024-12-23_22-45-30'
        
        create_run_directory('./checkpoints', 'baseline')
        # → './checkpoints/baseline_2024-12-23_22-45-30'
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if experiment_name:
        run_name = f"{experiment_name}_{timestamp}"
    else:
        run_name = f"run_{timestamp}"
    
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


def setup_logging(run_dir: str):
    """
    Setup logging to both console and file using stdout redirection.
    
    Args:
        run_dir: Directory to save training.log
        
    Returns:
        Tuple of (log_file_path, original_stdout)
    """
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
    
    return log_file, original_stdout


def main():
    parser = argparse.ArgumentParser(description="Train ViT Segmentation from scratch")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (e.g., 'baseline', 'high_lr'). Timestamp will be appended automatically."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {NUM_EPOCHS} from config.py)"
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )
    args = parser.parse_args()
    
    use_tensorboard = not args.no_tensorboard
    num_epochs = args.epochs if args.epochs is not None else NUM_EPOCHS
    
    run_dir = create_run_directory(CHECKPOINT_DIR, args.name)
    log_file, original_stdout = setup_logging(run_dir)
    
    try:
        print("=" * 60)
        print("ViT Segmentation - Training from Scratch")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Experiment: {os.path.basename(run_dir)}")
        print(f"Checkpoint dir: {os.path.abspath(run_dir)}")
        print(f"Log file: {os.path.abspath(log_file)}")
        
        if use_tensorboard:
            print("\nTensorBoard logging enabled!")
            print(f"  Run: tensorboard --logdir={CHECKPOINT_DIR}")
            print(f"  URL: http://localhost:6006")
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
            T_max=num_epochs
        )
        
        print(f"   Loss: Dice + Cross Entropy")
        print(f"   Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
        print(f"   Scheduler: CosineAnnealingLR (T_max={num_epochs})")
        print(f"   Epochs: {num_epochs}")
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
            checkpoint_dir=run_dir,
            patience=10,
            use_tensorboard=use_tensorboard
        )
        
        best_miou = trainer.train(num_epochs=num_epochs)
        
        print(f"\n5. Training Complete!")
        print(f"   Best Validation mIoU: {best_miou:.4f}")
        print(f"   Checkpoints saved to: {os.path.abspath(run_dir)}/")
        print(f"   Log saved to: {os.path.abspath(log_file)}")
        print(f"   Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
    
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()

