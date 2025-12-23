import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from visualization.predictions import denormalize_image


def extract_attention_maps(
    model: torch.nn.Module,
    image: torch.Tensor,
    layer_idx: int = -1
) -> torch.Tensor:
    """
    Extract attention weights from a specific layer.
    Args:
        model: Segmenter model
        image: [C, H, W] or [B, C, H, W] input image
        layer_idx: Layer index to extract attention from (-1 = last layer)
    Returns:
        [B, num_heads, num_patches, num_patches] attention weights
    """
    model.eval()
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    with torch.no_grad():
        _, attention_weights = model(
            image,
            return_attention=True,
            attention_layer_idx=layer_idx
        )
    
    return attention_weights


def average_attention_across_heads(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Average attention weights across all heads.
    Args:
        attention_weights: [B, num_heads, num_patches, num_patches]
    Returns:
        [B, num_patches, num_patches] averaged attention
    """
    return attention_weights.mean(dim=1)


def get_attention_map_for_patch(
    attention_weights: torch.Tensor,
    patch_idx: int,
    num_patches_per_side: int
) -> torch.Tensor:
    """
    Get attention map showing what a specific patch attends to.
    Args:
        attention_weights: [B, num_patches, num_patches] averaged attention
        patch_idx: Index of source patch
        num_patches_per_side: Number of patches per side (e.g., 14)
    Returns:
        [B, H, W] spatial attention map
    """
    attn_from_patch = attention_weights[:, patch_idx, :]
    attn_spatial = attn_from_patch.reshape(-1, num_patches_per_side, num_patches_per_side)
    return attn_spatial


def get_mean_attention_map(
    attention_weights: torch.Tensor,
    num_patches_per_side: int
) -> torch.Tensor:
    """
    Get mean attention received by each patch (averaged across source patches).
    Args:
        attention_weights: [B, num_patches, num_patches]
        num_patches_per_side: Number of patches per side (e.g., 14)
    Returns:
        [B, H, W] spatial attention map
    """
    mean_attn = attention_weights.mean(dim=1)
    attn_spatial = mean_attn.reshape(-1, num_patches_per_side, num_patches_per_side)
    return attn_spatial


def upsample_attention_map(
    attention_map: torch.Tensor,
    target_size: int = 224
) -> torch.Tensor:
    """
    Upsample attention map to image resolution.
    Args:
        attention_map: [B, H, W] spatial attention map
        target_size: Target image size
    Returns:
        [B, target_size, target_size] upsampled attention
    """
    if attention_map.dim() == 2:
        attention_map = attention_map.unsqueeze(0)
    
    attention_map = attention_map.unsqueeze(1)
    upsampled = F.interpolate(
        attention_map,
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    )
    
    return upsampled.squeeze(1)


def create_attention_heatmap(
    attention_map: np.ndarray,
    cmap: str = 'jet'
) -> np.ndarray:
    """
    Convert attention map to RGB heatmap.
    Args:
        attention_map: [H, W] attention values
        cmap: Colormap name
    Returns:
        [H, W, 3] RGB heatmap in [0, 1] range
    """
    attention_normalized = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    cmap = plt.cm.get_cmap(cmap)
    heatmap = cmap(attention_normalized)[:, :, :3]
    
    return heatmap


def overlay_attention_on_image(
    image: torch.Tensor,
    attention_map: torch.Tensor,
    alpha: float = 0.5,
    cmap: str = 'jet'
) -> np.ndarray:
    """
    Overlay attention heatmap on image.
    Args:
        image: [C, H, W] normalized image tensor
        attention_map: [H, W] attention map
        alpha: Transparency of heatmap (0=transparent, 1=opaque)
        cmap: Colormap name
    Returns:
        [H, W, 3] blended image
    """
    img = denormalize_image(image)
    
    attn_np = attention_map.cpu().numpy()
    heatmap = create_attention_heatmap(attn_np, cmap=cmap)
    
    overlay = img * (1 - alpha) + heatmap * alpha
    return np.clip(overlay, 0, 1)


def plot_attention_visualization(
    image: torch.Tensor,
    attention_weights: torch.Tensor,
    num_patches_per_side: int,
    image_size: int = 224,
    title: str = None
) -> plt.Figure:
    """
    Create comprehensive attention visualization.
    Args:
        image: [C, H, W] normalized image
        attention_weights: [num_heads, num_patches, num_patches]
        num_patches_per_side: Number of patches per side
        image_size: Image size for upsampling
        title: Optional figure title
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    img = denormalize_image(image)
    
    avg_attn = attention_weights.mean(dim=0)
    
    mean_attn_map = get_mean_attention_map(
        avg_attn.unsqueeze(0),
        num_patches_per_side
    ).squeeze(0)
    
    mean_attn_upsampled = upsample_attention_map(
        mean_attn_map,
        target_size=image_size
    ).squeeze(0)
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(mean_attn_map.cpu().numpy(), cmap='jet')
    axes[0, 1].set_title(f"Mean Attention Map ({num_patches_per_side}×{num_patches_per_side})")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(mean_attn_upsampled.cpu().numpy(), cmap='jet')
    axes[0, 2].set_title(f"Upsampled Attention ({image_size}×{image_size})")
    axes[0, 2].axis("off")
    
    overlay = overlay_attention_on_image(image, mean_attn_upsampled, alpha=0.5)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Attention Overlay (α=0.5)")
    axes[1, 0].axis("off")
    
    center_patch = (num_patches_per_side * num_patches_per_side) // 2
    center_attn = get_attention_map_for_patch(
        avg_attn.unsqueeze(0),
        center_patch,
        num_patches_per_side
    ).squeeze(0)
    
    axes[1, 1].imshow(center_attn.cpu().numpy(), cmap='jet')
    axes[1, 1].set_title(f"Center Patch Attention")
    axes[1, 1].axis("off")
    
    per_head_attn = attention_weights.mean(dim=(1, 2))
    axes[1, 2].bar(range(len(per_head_attn)), per_head_attn.cpu().numpy())
    axes[1, 2].set_title("Mean Attention per Head")
    axes[1, 2].set_xlabel("Head Index")
    axes[1, 2].set_ylabel("Mean Attention")
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    return fig


def plot_multi_layer_attention(
    model: torch.nn.Module,
    image: torch.Tensor,
    layer_indices: list[int],
    num_patches_per_side: int,
    image_size: int = 224
) -> plt.Figure:
    """
    Visualize attention from multiple layers.
    Args:
        model: Segmenter model
        image: [C, H, W] or [B, C, H, W] input image
        layer_indices: List of layer indices to visualize
        num_patches_per_side: Number of patches per side
        image_size: Image size for upsampling
    Returns:
        matplotlib Figure
    """
    if image.dim() == 3:
        image_single = image
        image_batch = image.unsqueeze(0)
    else:
        image_single = image[0]
        image_batch = image
    
    num_layers = len(layer_indices)
    fig, axes = plt.subplots(2, num_layers + 1, figsize=(4 * (num_layers + 1), 8))
    
    img = denormalize_image(image_single)
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    
    for col_idx, layer_idx in enumerate(layer_indices):
        attn_weights = extract_attention_maps(model, image_batch, layer_idx=layer_idx)
        
        avg_attn = average_attention_across_heads(attn_weights)
        
        mean_attn_map = get_mean_attention_map(avg_attn, num_patches_per_side).squeeze(0)
        mean_attn_upsampled = upsample_attention_map(mean_attn_map, target_size=image_size).squeeze(0)
        
        axes[0, col_idx + 1].imshow(mean_attn_upsampled.cpu().numpy(), cmap='jet')
        axes[0, col_idx + 1].set_title(f"Layer {layer_idx} Attention")
        axes[0, col_idx + 1].axis("off")
        
        overlay = overlay_attention_on_image(image_single, mean_attn_upsampled, alpha=0.5)
        axes[1, col_idx + 1].imshow(overlay)
        axes[1, col_idx + 1].set_title(f"Layer {layer_idx} Overlay")
        axes[1, col_idx + 1].axis("off")
    
    plt.tight_layout()
    return fig

