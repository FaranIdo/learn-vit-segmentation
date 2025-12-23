import sys
sys.path.insert(0, ".")

import argparse
import torch
import os
import logging
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
    Setup logging to both console and file.
    
    Args:
        run_dir: Directory to save training.log
    """
    log_file = os.path.join(run_dir, "training.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file


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
    log_file = setup_logging(run_dir)
    
    logging.info("=" * 60)
    logging.info("ViT Segmentation - Training from Scratch")
    logging.info("=" * 60)
    logging.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Experiment: {os.path.basename(run_dir)}")
    logging.info(f"Checkpoint dir: {os.path.abspath(run_dir)}")
    logging.info(f"Log file: {os.path.abspath(log_file)}")
    
    if use_tensorboard:
        logging.info("\nTensorBoard logging enabled!")
        logging.info(f"  Run: tensorboard --logdir={CHECKPOINT_DIR}")
        logging.info(f"  URL: http://localhost:6006")
        logging.info("=" * 60)
    
    logging.info("\n1. Loading Data...")
    train_loader, val_loader = create_dataloaders(
        root=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    logging.info(f"   Train samples: {len(train_loader.dataset)}")
    logging.info(f"   Val samples: {len(val_loader.dataset)}")
    logging.info(f"   Batch size: {BATCH_SIZE}")
    
    logging.info("\n2. Creating Model...")
    model = create_model()
    model = model.to(DEVICE)
    
    total_params, trainable_params = count_parameters(model)
    logging.info(f"   Total parameters: {total_params:,}")
    logging.info(f"   Trainable parameters: {trainable_params:,}")
    logging.info(f"   Device: {DEVICE}")
    
    logging.info("\n3. Setting up Training...")
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
    
    logging.info(f"   Loss: Dice + Cross Entropy")
    logging.info(f"   Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    logging.info(f"   Scheduler: CosineAnnealingLR (T_max={num_epochs})")
    logging.info(f"   Epochs: {num_epochs}")
    logging.info(f"   Early stopping patience: 10")
    
    logging.info("\n4. Training...")
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
    
    logging.info(f"\n5. Training Complete!")
    logging.info(f"   Best Validation mIoU: {best_miou:.4f}")
    logging.info(f"   Checkpoints saved to: {os.path.abspath(run_dir)}/")
    logging.info(f"   Log saved to: {os.path.abspath(log_file)}")
    logging.info(f"   Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()

