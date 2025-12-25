import sys
sys.path.insert(0, ".")

import os
import torch
import matplotlib.pyplot as plt
from vit.segmenter import Segmenter
from data.oxford_pets import create_dataloaders
from visualization.predictions import (
    plot_prediction_comparison,
    visualize_batch_predictions
)
from config import (
    DEVICE, IMAGE_SIZE, PATCH_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS,
    MLP_RATIO, DROPOUT, NUM_CLASSES, BATCH_SIZE, DATA_DIR, parse_device
)

OUTPUT_DIR = "outputs/predictions"


def load_model_checkpoint(checkpoint_path: str, device) -> Segmenter:
    """Load trained model from checkpoint."""
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model


def visualize_untrained_model(device):
    """Visualize predictions from an untrained model (random weights)."""
    print("=" * 60)
    print("Visualizing Predictions: UNTRAINED Model")
    print("=" * 60)
    
    print("\nCreating untrained model...")
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
    model = model.to(device)
    model.eval()
    
    print("Loading validation data...")
    _, val_loader = create_dataloaders(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
    
    print("Generating predictions...")
    with torch.no_grad():
        images, masks = next(iter(val_loader))
        images = images.to(device)
        masks = masks.to(device)
        
        logits = model(images)
        predictions = torch.argmax(logits, dim=1)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\nCreating visualizations...")
    
    # Single sample comparison
    fig = plot_prediction_comparison(
        images[0],
        masks[0],
        predictions[0],
        num_classes=NUM_CLASSES,
        title="Untrained Model - Single Sample"
    )
    single_path = os.path.join(OUTPUT_DIR, "untrained_single.png")
    plt.savefig(single_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {single_path}")
    
    # Batch comparison
    fig = visualize_batch_predictions(
        images,
        masks,
        predictions,
        num_classes=NUM_CLASSES,
        num_samples=4
    )
    batch_path = os.path.join(OUTPUT_DIR, "untrained_batch.png")
    plt.savefig(batch_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {batch_path}")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Check '{OUTPUT_DIR}/' for the generated PNG files.")
    print("\nNote: Untrained model predictions will look random.")
    print("After training, run with --checkpoint to see real results.")
    print("=" * 60)


def visualize_trained_model(checkpoint_path: str, device):
    """Visualize predictions from a trained model."""
    print("=" * 60)
    print("Visualizing Predictions: TRAINED Model")
    print("=" * 60)
    
    print(f"\nLoading model from: {checkpoint_path}")
    model = load_model_checkpoint(checkpoint_path, device)
    
    print("Loading validation data...")
    _, val_loader = create_dataloaders(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
    
    print("Generating predictions...")
    with torch.no_grad():
        images, masks = next(iter(val_loader))
        images = images.to(device)
        masks = masks.to(device)
        
        logits = model(images)
        predictions = torch.argmax(logits, dim=1)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\nCreating visualizations...")
    
    # Single sample comparison
    fig = plot_prediction_comparison(
        images[0],
        masks[0],
        predictions[0],
        num_classes=NUM_CLASSES,
        title="Trained Model - Single Sample"
    )
    single_path = os.path.join(OUTPUT_DIR, "trained_single.png")
    plt.savefig(single_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {single_path}")
    
    # Batch comparison
    fig = visualize_batch_predictions(
        images,
        masks,
        predictions,
        num_classes=NUM_CLASSES,
        num_samples=4
    )
    batch_path = os.path.join(OUTPUT_DIR, "trained_batch.png")
    plt.savefig(batch_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {batch_path}")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Check '{OUTPUT_DIR}/' for the generated PNG files.")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize segmentation predictions")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (if not provided, uses untrained model)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use for visualization (default: auto-detect)"
    )
    args = parser.parse_args()
    
    device = parse_device(args.device)
    
    if args.checkpoint:
        visualize_trained_model(args.checkpoint, device)
    else:
        visualize_untrained_model(device)


if __name__ == "__main__":
    main()

