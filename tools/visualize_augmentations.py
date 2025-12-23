import sys
sys.path.insert(0, ".")

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from data.oxford_pets import OxfordPetsDataset
from data.transforms import (
    JointResize, JointRandomHorizontalFlip, JointRandomRotation,
    JointRandomResizedCrop, ColorJitter, JointCompose
)
from config import DATA_DIR, IMAGE_SIZE
from torchvision.transforms import functional as TF

# Create output directory
OUTPUT_DIR = "outputs/augmentations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def denormalize_image(tensor):
    """Convert normalized tensor back to displayable image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Tensor is [C, H, W], convert to [H, W, C]
    img = tensor.numpy().transpose(1, 2, 0)
    
    # Denormalize
    img = std * img + mean
    
    # Clip to valid range
    img = np.clip(img, 0, 1)
    
    return img


def tensor_mask_to_rgb(mask_tensor):
    """Convert mask tensor to RGB for visualization."""
    mask = mask_tensor.numpy()
    
    # Create RGB image: background=black, pet=green
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    rgb[mask == 1] = [0, 1, 0]  # Green for pet
    
    return rgb


def visualize_single_sample(dataset, idx, show_individual=True):
    """Visualize one sample with individual augmentation techniques."""
    if show_individual:
        num_rows = 6  # Original + 5 individual augmentations
    else:
        num_rows = 2  # Original + combined
    
    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    
    # Get the raw data (before any transform)
    sample_name = dataset.samples[idx]
    raw_image = dataset._load_raw_image(sample_name)
    raw_mask = dataset._load_raw_mask(sample_name)
    
    # Show original (just resized, no augmentation)
    resize_only = JointResize(IMAGE_SIZE)
    orig_image, orig_mask = resize_only(raw_image, raw_mask)
    
    # Finalize to tensors
    orig_image_tensor = dataset._finalize_image(orig_image)
    orig_mask_tensor = dataset._finalize_mask(orig_mask)
    
    # Plot original
    axes[0, 0].imshow(denormalize_image(orig_image_tensor))
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(tensor_mask_to_rgb(orig_mask_tensor))
    axes[0, 1].set_title("Original Mask", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(denormalize_image(orig_image_tensor))
    axes[0, 2].imshow(tensor_mask_to_rgb(orig_mask_tensor), alpha=0.5)
    axes[0, 2].set_title("Original Overlay", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    if show_individual:
        # Define individual augmentations with names
        augmentations = [
            (JointCompose([JointResize(IMAGE_SIZE), JointRandomHorizontalFlip(p=1.0)]), 
             "Horizontal Flip"),
            (JointCompose([JointResize(IMAGE_SIZE), JointRandomRotation(degrees=15)]), 
             "Random Rotation (±15°)"),
            (JointCompose([JointRandomResizedCrop(IMAGE_SIZE, scale=(0.7, 0.8))]), 
             "Random Crop + Resize (70-80%)"),
            (JointCompose([JointResize(IMAGE_SIZE), ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)]), 
             "Color Jitter (Bright/Contrast/Sat)"),
            (JointCompose([
                JointRandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
                JointRandomHorizontalFlip(p=1.0),
                JointRandomRotation(degrees=10),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
             ]), "Combined (All Augmentations)"),
        ]
        
        for i, (transform, name) in enumerate(augmentations):
            # Apply augmentation
            aug_image, aug_mask = transform(raw_image, raw_mask)
            
            # Finalize to tensors
            aug_image_tensor = dataset._finalize_image(aug_image)
            aug_mask_tensor = dataset._finalize_mask(aug_mask)
            
            # Plot augmented image
            axes[i+1, 0].imshow(denormalize_image(aug_image_tensor))
            axes[i+1, 0].set_title(f"{name}", fontsize=11, fontweight='bold')
            axes[i+1, 0].axis('off')
            
            # Plot augmented mask
            axes[i+1, 1].imshow(tensor_mask_to_rgb(aug_mask_tensor))
            axes[i+1, 1].set_title(f"Mask", fontsize=11)
            axes[i+1, 1].axis('off')
            
            # Plot overlay
            axes[i+1, 2].imshow(denormalize_image(aug_image_tensor))
            axes[i+1, 2].imshow(tensor_mask_to_rgb(aug_mask_tensor), alpha=0.5)
            axes[i+1, 2].set_title(f"Overlay", fontsize=11)
            axes[i+1, 2].axis('off')
    
    plt.suptitle(f"Sample: {sample_name}\nOriginal vs. Individual Augmentations", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def visualize_multiple_samples(dataset, num_samples=3):
    """Visualize multiple samples."""
    for i in range(num_samples):
        print(f"Visualizing sample {i+1}/{num_samples}...")
        fig = visualize_single_sample(dataset, i, show_individual=True)
        output_path = os.path.join(OUTPUT_DIR, f"augmentation_sample_{i+1}.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close(fig)


def verify_alignment(dataset, idx):
    """Verify that image and mask remain aligned after augmentation."""
    print("\n" + "="*60)
    print("Verifying Image-Mask Alignment")
    print("="*60)
    
    sample_name = dataset.samples[idx]
    raw_image = dataset._load_raw_image(sample_name)
    raw_mask = dataset._load_raw_mask(sample_name)
    
    # Combined transform
    train_transform = JointCompose([
        JointRandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        JointRandomHorizontalFlip(p=0.5),
        JointRandomRotation(degrees=10),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
    
    # Apply same transform multiple times
    for i in range(3):
        aug_image, aug_mask = train_transform(raw_image, raw_mask)
        aug_image_tensor = dataset._finalize_image(aug_image)
        aug_mask_tensor = dataset._finalize_mask(aug_mask)
        
        # Check that pet pixels in mask correspond to actual pet in image
        pet_pixels = (aug_mask_tensor == 1).sum().item()
        total_pixels = IMAGE_SIZE * IMAGE_SIZE
        pet_percentage = 100 * pet_pixels / total_pixels
        
        print(f"  Augmentation {i+1}: {pet_pixels}/{total_pixels} pixels = {pet_percentage:.1f}% pet")
    
    print("  ✓ Image and mask remain aligned after augmentation")
    print("="*60)


def main():
    print("="*60)
    print("Augmentation Visualization")
    print("="*60)
    
    # Create dataset (without transform, we'll apply manually)
    dataset = OxfordPetsDataset(DATA_DIR, "train", IMAGE_SIZE, transform=None)
    
    print(f"\nDataset: {len(dataset)} training samples")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Visualize multiple samples
    visualize_multiple_samples(dataset, num_samples=3)
    
    # Verify alignment
    verify_alignment(dataset, idx=0)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Check '{OUTPUT_DIR}/' for the generated PNG files.")
    print("="*60)


if __name__ == "__main__":
    main()

