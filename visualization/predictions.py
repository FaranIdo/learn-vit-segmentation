import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from config import IMAGENET_MEAN, IMAGENET_STD


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalized tensor back to displayable image.
    Args:
        tensor: [C, H, W] normalized image tensor
    Returns:
        [H, W, C] numpy array in [0, 1] range
    """
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)

    return img


def mask_to_rgb(mask: torch.Tensor, num_classes: int = 2) -> np.ndarray:
    """
    Convert class mask to RGB for visualization.
    Args:
        mask: [H, W] class indices
        num_classes: Number of classes
    Returns:
        [H, W, 3] RGB array
    """
    mask = mask.cpu().numpy()

    if num_classes == 2:
        colors = np.array(
            [
                [0, 0, 0],  # Class 0: Black (background)
                [0, 255, 0],  # Class 1: Green (foreground/pet)
            ],
            dtype=np.uint8,
        )
    else:
        cmap = plt.cm.get_cmap("tab10", num_classes)
        colors = (cmap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)

    rgb = colors[mask]
    return rgb


def overlay_mask_on_image(image: torch.Tensor, mask: torch.Tensor, num_classes: int = 2, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay mask on image with transparency.
    Args:
        image: [C, H, W] normalized image tensor
        mask: [H, W] class indices
        num_classes: Number of classes
        alpha: Transparency of mask overlay (0=transparent, 1=opaque)
    Returns:
        [H, W, 3] blended image
    """
    img = denormalize_image(image)
    mask_rgb = mask_to_rgb(mask, num_classes) / 255.0

    overlay = img * (1 - alpha) + mask_rgb * alpha
    return np.clip(overlay, 0, 1)


def plot_prediction_comparison(image: torch.Tensor, ground_truth: torch.Tensor, prediction: torch.Tensor, num_classes: int = 2, title: str = None) -> plt.Figure:
    """
    Create a 4-panel comparison plot.
    Args:
        image: [C, H, W] normalized image
        ground_truth: [H, W] ground truth mask
        prediction: [H, W] predicted mask
        num_classes: Number of classes
        title: Optional title for the figure
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    img = denormalize_image(image)
    gt_rgb = mask_to_rgb(ground_truth, num_classes)
    pred_rgb = mask_to_rgb(prediction, num_classes)
    overlay = overlay_mask_on_image(image, prediction, num_classes, alpha=0.5)

    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    return fig


def visualize_batch_predictions(images: torch.Tensor, ground_truths: torch.Tensor, predictions: torch.Tensor, num_classes: int = 2, num_samples: int = 4) -> plt.Figure:
    """
    Visualize multiple predictions in a grid.
    Args:
        images: [B, C, H, W] batch of images
        ground_truths: [B, H, W] batch of ground truth masks
        predictions: [B, H, W] batch of predicted masks
        num_classes: Number of classes
        num_samples: Number of samples to visualize
    Returns:
        matplotlib Figure
    """
    num_samples = min(num_samples, images.shape[0])

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        img = denormalize_image(images[i])
        gt_rgb = mask_to_rgb(ground_truths[i], num_classes)
        pred_rgb = mask_to_rgb(predictions[i], num_classes)
        overlay = overlay_mask_on_image(images[i], predictions[i], num_classes, alpha=0.5)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {i+1}: Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_rgb)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Overlay")
        axes[i, 3].axis("off")

    plt.tight_layout()
    return fig
