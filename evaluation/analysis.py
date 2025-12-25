import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion
from visualization.predictions import denormalize_image, mask_to_rgb, overlay_mask_on_image


def calculate_sample_iou(pred: torch.Tensor, target: torch.Tensor, class_id: int = 1) -> float:
    pred_mask = pred == class_id
    target_mask = target == class_id

    intersection = (pred_mask & target_mask).sum().item()
    union = (pred_mask | target_mask).sum().item()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def select_samples_by_iou(all_ious: list[float], num_best: int = 5, num_median: int = 5, num_worst: int = 5) -> dict:
    sorted_indices = np.argsort(all_ious)

    worst_indices = sorted_indices[:num_worst].tolist()

    median_start = len(sorted_indices) // 2 - num_median // 2
    median_indices = sorted_indices[median_start : median_start + num_median].tolist()

    best_indices = sorted_indices[-num_best:].tolist()[::-1]

    return {"worst": worst_indices, "median": median_indices, "best": best_indices}


def plot_error_grid(images: list, masks: list, predictions: list, ious: list, category: str, num_classes: int = 2) -> plt.Figure:
    num_samples = len(images)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        img = denormalize_image(images[i])
        gt_rgb = mask_to_rgb(masks[i], num_classes)
        pred_rgb = mask_to_rgb(predictions[i], num_classes)
        overlay = overlay_mask_on_image(images[i], predictions[i], num_classes, alpha=0.5)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {i+1}: Original (IoU: {ious[i]:.3f})")
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

    fig.suptitle(f"{category.upper()} Performance Samples", fontsize=16, y=0.995)
    plt.tight_layout()

    return fig


def create_error_map(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    error_map = np.zeros((*pred.shape, 3), dtype=np.uint8)

    correct = pred == target
    false_positive = (pred == 1) & (target == 0)
    false_negative = (pred == 0) & (target == 1)

    error_map[correct] = [0, 255, 0]
    error_map[false_positive] = [255, 0, 0]
    error_map[false_negative] = [0, 0, 255]

    return error_map


def plot_error_map_comparison(image: torch.Tensor, mask: torch.Tensor, prediction: torch.Tensor, iou: float, sample_idx: int) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    img = denormalize_image(image)
    error_map = create_error_map(prediction, mask)

    axes[0].imshow(img)
    axes[0].set_title(f"Original Image (IoU: {iou:.3f})")
    axes[0].axis("off")

    axes[1].imshow(mask_to_rgb(mask, 2))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(mask_to_rgb(prediction, 2))
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(error_map)
    axes[3].set_title("Error Map\n(Green=Correct, Red=FP, Blue=FN)")
    axes[3].axis("off")

    fig.suptitle(f"Sample {sample_idx} - Error Analysis", fontsize=14, y=0.98)
    plt.tight_layout()

    return fig


def analyze_by_size(all_ious: list[float], mask_areas: list[float], total_pixels: int) -> dict:
    small_ious = []
    medium_ious = []
    large_ious = []

    for iou, area in zip(all_ious, mask_areas):
        area_ratio = area / total_pixels

        if area_ratio < 0.10:
            small_ious.append(iou)
        elif area_ratio < 0.30:
            medium_ious.append(iou)
        else:
            large_ious.append(iou)

    return {
        "small": {"count": len(small_ious), "mean_iou": np.mean(small_ious) if small_ious else 0.0},
        "medium": {"count": len(medium_ious), "mean_iou": np.mean(medium_ious) if medium_ious else 0.0},
        "large": {"count": len(large_ious), "mean_iou": np.mean(large_ious) if large_ious else 0.0},
    }


def extract_boundary_pixels(mask: np.ndarray, thickness: int = 3) -> tuple[np.ndarray, np.ndarray]:
    mask_binary = mask > 0

    if not mask_binary.any():
        return np.zeros_like(mask_binary), mask_binary

    dilated = binary_dilation(mask_binary, iterations=thickness)
    eroded = binary_erosion(mask_binary, iterations=thickness)

    boundary = dilated & ~eroded
    interior = eroded

    return boundary, interior


def analyze_boundary_vs_interior(all_masks: list, all_predictions: list) -> dict:
    boundary_correct = 0
    boundary_total = 0
    interior_correct = 0
    interior_total = 0

    for mask, pred in zip(all_masks, all_predictions):
        mask_np = mask.cpu().numpy()
        pred_np = pred.cpu().numpy()

        boundary, interior = extract_boundary_pixels(mask_np)

        boundary_pixels = boundary.sum()
        interior_pixels = interior.sum()

        if boundary_pixels > 0:
            boundary_correct += ((pred_np == mask_np) & boundary).sum()
            boundary_total += boundary_pixels

        if interior_pixels > 0:
            interior_correct += ((pred_np == mask_np) & interior).sum()
            interior_total += interior_pixels

    return {
        "boundary_accuracy": boundary_correct / boundary_total if boundary_total > 0 else 0.0,
        "interior_accuracy": interior_correct / interior_total if interior_total > 0 else 0.0,
        "boundary_pixels": boundary_total,
        "interior_pixels": interior_total,
    }


def plot_boundary_analysis(image: torch.Tensor, mask: torch.Tensor, prediction: torch.Tensor, sample_idx: int, iou: float) -> plt.Figure:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    img = denormalize_image(image)
    mask_np = mask.cpu().numpy()
    pred_np = prediction.cpu().numpy()

    boundary, interior = extract_boundary_pixels(mask_np)

    boundary_correct = (pred_np == mask_np) & boundary
    boundary_error = (pred_np != mask_np) & boundary
    interior_correct = (pred_np == mask_np) & interior
    interior_error = (pred_np != mask_np) & interior

    boundary_map = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    boundary_map[boundary_correct] = [0, 255, 0]
    boundary_map[boundary_error] = [255, 0, 0]

    interior_map = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    interior_map[interior_correct] = [0, 255, 0]
    interior_map[interior_error] = [255, 0, 0]

    boundary_acc = boundary_correct.sum() / boundary.sum() if boundary.sum() > 0 else 0
    interior_acc = interior_correct.sum() / interior.sum() if interior.sum() > 0 else 0

    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask_to_rgb(mask, 2))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(mask_to_rgb(prediction, 2))
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(boundary_map)
    axes[3].set_title(f"Boundary Errors\n(Acc: {boundary_acc:.2%})")
    axes[3].axis("off")

    axes[4].imshow(interior_map)
    axes[4].set_title(f"Interior Errors\n(Acc: {interior_acc:.2%})")
    axes[4].axis("off")

    fig.suptitle(f"Sample {sample_idx} - IoU: {iou:.3f} | Boundary: {boundary_acc:.2%} | Interior: {interior_acc:.2%}", fontsize=14, y=0.98)
    plt.tight_layout()

    return fig, boundary_acc, interior_acc


def calculate_confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    return max_probs


def calculate_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy


def analyze_confidence_by_correctness(all_logits: list, all_predictions: list, all_masks: list) -> dict:
    correct_confidences = []
    incorrect_confidences = []

    for logits, pred, mask in zip(all_logits, all_predictions, all_masks):
        confidence = calculate_confidence_from_logits(logits.unsqueeze(0)).squeeze(0)
        correct_mask = pred == mask
        
        correct_confidences.extend(confidence[correct_mask].cpu().numpy().tolist())
        incorrect_confidences.extend(confidence[~correct_mask].cpu().numpy().tolist())

    return {
        "correct_mean": np.mean(correct_confidences) if correct_confidences else 0.0,
        "correct_std": np.std(correct_confidences) if correct_confidences else 0.0,
        "incorrect_mean": np.mean(incorrect_confidences) if incorrect_confidences else 0.0,
        "incorrect_std": np.std(incorrect_confidences) if incorrect_confidences else 0.0,
        "correct_count": len(correct_confidences),
        "incorrect_count": len(incorrect_confidences),
    }


def analyze_uncertainty_zones(all_logits: list, all_predictions: list, all_masks: list, uncertainty_threshold: float = 0.5) -> dict:
    total_uncertain = 0
    uncertain_correct = 0
    total_certain = 0
    certain_correct = 0

    for logits, pred, mask in zip(all_logits, all_predictions, all_masks):
        confidence = calculate_confidence_from_logits(logits.unsqueeze(0)).squeeze(0)
        correct_mask = pred == mask
        
        uncertain_mask = confidence < uncertainty_threshold
        certain_mask = ~uncertain_mask
        
        total_uncertain += uncertain_mask.sum().item()
        uncertain_correct += (correct_mask & uncertain_mask).sum().item()
        
        total_certain += certain_mask.sum().item()
        certain_correct += (correct_mask & certain_mask).sum().item()

    return {
        "uncertain_accuracy": uncertain_correct / total_uncertain if total_uncertain > 0 else 0.0,
        "certain_accuracy": certain_correct / total_certain if total_certain > 0 else 0.0,
        "uncertain_pixels": total_uncertain,
        "certain_pixels": total_certain,
        "uncertainty_threshold": uncertainty_threshold,
    }


def plot_confidence_analysis(image: torch.Tensor, mask: torch.Tensor, prediction: torch.Tensor, logits: torch.Tensor, sample_idx: int, iou: float) -> tuple[plt.Figure, float, float]:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    img = denormalize_image(image)
    mask_np = mask.cpu().numpy()
    pred_np = prediction.cpu().numpy()
    
    confidence = calculate_confidence_from_logits(logits.unsqueeze(0)).squeeze(0).cpu().numpy()
    entropy = calculate_entropy(logits.unsqueeze(0)).squeeze(0).cpu().numpy()
    
    correct_mask = pred_np == mask_np
    incorrect_mask = ~correct_mask
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(mask_to_rgb(mask, 2))
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(mask_to_rgb(prediction, 2))
    axes[0, 2].set_title("Prediction")
    axes[0, 2].axis("off")
    
    conf_plot = axes[1, 0].imshow(confidence, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Confidence Map\n(Mean: {confidence.mean():.3f})")
    axes[1, 0].axis("off")
    plt.colorbar(conf_plot, ax=axes[1, 0], fraction=0.046)
    
    entropy_plot = axes[1, 1].imshow(entropy, cmap="hot")
    axes[1, 1].set_title(f"Uncertainty (Entropy)\n(Mean: {entropy.mean():.3f})")
    axes[1, 1].axis("off")
    plt.colorbar(entropy_plot, ax=axes[1, 1], fraction=0.046)
    
    confidence_error_map = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    confidence_error_map[correct_mask & (confidence > 0.5)] = [0, 255, 0]
    confidence_error_map[correct_mask & (confidence <= 0.5)] = [255, 255, 0]
    confidence_error_map[incorrect_mask & (confidence > 0.5)] = [255, 0, 0]
    confidence_error_map[incorrect_mask & (confidence <= 0.5)] = [255, 165, 0]
    
    axes[1, 2].imshow(confidence_error_map)
    axes[1, 2].set_title("Confidence vs Correctness\nGreen: Correct+High | Red: Wrong+High\nYellow: Correct+Low | Orange: Wrong+Low")
    axes[1, 2].axis("off")
    
    correct_conf_mean = confidence[correct_mask].mean() if correct_mask.any() else 0
    incorrect_conf_mean = confidence[incorrect_mask].mean() if incorrect_mask.any() else 0
    
    fig.suptitle(f"Sample {sample_idx} - IoU: {iou:.3f} | Conf(Correct): {correct_conf_mean:.3f} | Conf(Wrong): {incorrect_conf_mean:.3f}", fontsize=14)
    plt.tight_layout()
    
    return fig, correct_conf_mean, incorrect_conf_mean

