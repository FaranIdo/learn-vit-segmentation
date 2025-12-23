import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = self._get_class_probabilities(logits)
        targets_one_hot = self._one_hot_encode(targets, logits.shape[1])
        return self._compute_dice_loss(probs, targets_one_hot)
    
    def _get_class_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=1)
    
    def _one_hot_encode(self, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        B, H, W = targets.shape
        one_hot = torch.zeros(B, num_classes, H, W, device=targets.device)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)
        return one_hot
    
    def _compute_dice_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        return self.dice_weight * dice + self.ce_weight * ce

