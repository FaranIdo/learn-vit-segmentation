import torch


def calculate_pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate pixel accuracy.
    
    Args:
        preds: Predicted class indices [B, H, W]
        targets: Ground truth class indices [B, H, W]
    
    Returns:
        Pixel accuracy as a float between 0 and 1
    """
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total


def calculate_iou_per_class(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> list[float]:
    """
    Calculate IoU for each class separately.
    
    Args:
        preds: Predicted class indices [B, H, W]
        targets: Ground truth class indices [B, H, W]
        num_classes: Number of classes
    
    Returns:
        List of IoU values, one per class
    """
    ious = []
    
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)
        
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return ious


def calculate_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    Calculate mean IoU (mIoU) across all classes.
    
    Args:
        preds: Predicted class indices [B, H, W]
        targets: Ground truth class indices [B, H, W]
        num_classes: Number of classes
    
    Returns:
        mIoU as a float between 0 and 1
    """
    ious = calculate_iou_per_class(preds, targets, num_classes)
    valid_ious = [iou for iou in ious if not torch.isnan(torch.tensor(iou))]
    
    if len(valid_ious) == 0:
        return 0.0
    
    return sum(valid_ious) / len(valid_ious)


class SegmentationMetrics:
    """
    Accumulator for segmentation metrics over multiple batches.
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.total_correct = 0
        self.total_pixels = 0
        self.class_intersection = [0] * self.num_classes
        self.class_union = [0] * self.num_classes
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with a new batch.
        
        Args:
            preds: Predicted class indices [B, H, W]
            targets: Ground truth class indices [B, H, W]
        """
        self.total_correct += (preds == targets).sum().item()
        self.total_pixels += targets.numel()
        
        for cls in range(self.num_classes):
            pred_mask = (preds == cls)
            target_mask = (targets == cls)
            
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            
            self.class_intersection[cls] += intersection
            self.class_union[cls] += union
    
    def get_pixel_accuracy(self) -> float:
        """Get accumulated pixel accuracy."""
        if self.total_pixels == 0:
            return 0.0
        return self.total_correct / self.total_pixels
    
    def get_iou_per_class(self) -> list[float]:
        """Get IoU for each class."""
        ious = []
        for cls in range(self.num_classes):
            if self.class_union[cls] == 0:
                ious.append(float('nan'))
            else:
                iou = self.class_intersection[cls] / self.class_union[cls]
                ious.append(iou)
        return ious
    
    def get_miou(self) -> float:
        """Get mean IoU across all classes."""
        ious = self.get_iou_per_class()
        valid_ious = [iou for iou in ious if not torch.isnan(torch.tensor(iou))]
        
        if len(valid_ious) == 0:
            return 0.0
        
        return sum(valid_ious) / len(valid_ious)
    
    def get_metrics_dict(self) -> dict:
        """Get all metrics as a dictionary."""
        ious = self.get_iou_per_class()
        
        metrics = {
            'pixel_accuracy': self.get_pixel_accuracy(),
            'miou': self.get_miou(),
        }
        
        for cls, iou in enumerate(ious):
            metrics[f'iou_class_{cls}'] = iou
        
        return metrics

