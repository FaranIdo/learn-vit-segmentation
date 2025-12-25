import sys
sys.path.insert(0, ".")

import os
import torch
import matplotlib.pyplot as plt
from vit.segmenter import Segmenter
from data.oxford_pets import create_dataloaders
from visualization.attention import (
    extract_attention_maps,
    average_attention_across_heads,
    get_mean_attention_map,
    upsample_attention_map,
    plot_attention_visualization,
    plot_multi_layer_attention
)
from config import (
    DEVICE, IMAGE_SIZE, PATCH_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS,
    MLP_RATIO, DROPOUT, NUM_CLASSES, BATCH_SIZE, DATA_DIR, parse_device
)

OUTPUT_DIR = "outputs/attention"


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


def visualize_untrained_attention(device):
    """Visualize attention from an untrained model."""
    print("=" * 60)
    print("Visualizing Attention: UNTRAINED Model")
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
    
    images, masks = next(iter(val_loader))
    images = images.to(device)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\nExtracting attention from last layer...")
    num_patches_per_side = IMAGE_SIZE // PATCH_SIZE
    
    attention_weights = extract_attention_maps(model, images[0], layer_idx=-1)
    
    print(f"  Attention shape: {attention_weights.shape}")
    print(f"  [batch, num_heads, num_patches, num_patches]")
    print(f"  [{attention_weights.shape[0]}, {attention_weights.shape[1]}, {attention_weights.shape[2]}, {attention_weights.shape[3]}]")
    
    print("\nCreating single image visualization...")
    fig = plot_attention_visualization(
        images[0],
        attention_weights.squeeze(0),
        num_patches_per_side,
        image_size=IMAGE_SIZE,
        title="Untrained Model - Attention Visualization"
    )
    single_path = os.path.join(OUTPUT_DIR, "untrained_single_attention.png")
    plt.savefig(single_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {single_path}")
    
    print("\nCreating multi-layer visualization...")
    layer_indices = [0, NUM_LAYERS // 2, NUM_LAYERS - 1]
    fig = plot_multi_layer_attention(
        model,
        images[0],
        layer_indices,
        num_patches_per_side,
        image_size=IMAGE_SIZE
    )
    multi_path = os.path.join(OUTPUT_DIR, "untrained_multi_layer.png")
    plt.savefig(multi_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {multi_path}")
    
    print("\n" + "=" * 60)
    print("Attention visualization complete!")
    print(f"Check '{OUTPUT_DIR}/' for the generated PNG files.")
    print("\nNote: Untrained model attention will look random/uniform.")
    print("After training, run with --checkpoint to see learned patterns.")
    print("=" * 60)


def visualize_trained_attention(checkpoint_path: str, device):
    """Visualize attention from a trained model."""
    print("=" * 60)
    print("Visualizing Attention: TRAINED Model")
    print("=" * 60)
    
    print(f"\nLoading model from: {checkpoint_path}")
    model = load_model_checkpoint(checkpoint_path, device)
    
    print("Loading validation data...")
    _, val_loader = create_dataloaders(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
    
    images, masks = next(iter(val_loader))
    images = images.to(device)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\nExtracting attention from last layer...")
    num_patches_per_side = IMAGE_SIZE // PATCH_SIZE
    
    attention_weights = extract_attention_maps(model, images[0], layer_idx=-1)
    
    print(f"  Attention shape: {attention_weights.shape}")
    
    print("\nCreating single image visualization...")
    fig = plot_attention_visualization(
        images[0],
        attention_weights.squeeze(0),
        num_patches_per_side,
        image_size=IMAGE_SIZE,
        title="Trained Model - Attention Visualization"
    )
    single_path = os.path.join(OUTPUT_DIR, "trained_single_attention.png")
    plt.savefig(single_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {single_path}")
    
    print("\nCreating multi-layer visualization...")
    layer_indices = [0, NUM_LAYERS // 2, NUM_LAYERS - 1]
    fig = plot_multi_layer_attention(
        model,
        images[0],
        layer_indices,
        num_patches_per_side,
        image_size=IMAGE_SIZE
    )
    multi_path = os.path.join(OUTPUT_DIR, "trained_multi_layer.png")
    plt.savefig(multi_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {multi_path}")
    
    print("\n" + "=" * 60)
    print("Attention visualization complete!")
    print(f"Check '{OUTPUT_DIR}/' for the generated PNG files.")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize attention maps")
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
        visualize_trained_attention(args.checkpoint, device)
    else:
        visualize_untrained_attention(device)


if __name__ == "__main__":
    main()

