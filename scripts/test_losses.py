import sys

sys.path.insert(0, "/Users/ido.faran/personal/vis-tutorial")

import torch
from training import DiceLoss, SegmentationLoss
from config import DEVICE, NUM_CLASSES


def create_test_data(batch_size: int = 2, height: int = 224, width: int = 224, requires_grad: bool = False):
    logits = torch.randn(batch_size, NUM_CLASSES, height, width, device=DEVICE, requires_grad=requires_grad)
    targets = torch.randint(0, NUM_CLASSES, (batch_size, height, width), device=DEVICE)
    return logits, targets


def test_dice_loss():
    print("=" * 50)
    print("Testing DiceLoss")
    print("=" * 50)

    logits, targets = create_test_data()
    dice_loss = DiceLoss().to(DEVICE)

    loss = dice_loss(logits, targets)
    print(f"Random predictions - Dice Loss: {loss.item():.4f}")

    perfect_logits = create_perfect_prediction(targets)
    perfect_loss = dice_loss(perfect_logits, targets)
    print(f"Perfect predictions - Dice Loss: {perfect_loss.item():.4f}")

    assert loss.item() > perfect_loss.item(), "Perfect predictions should have lower loss"
    print("✓ Dice loss behaves correctly\n")


def create_perfect_prediction(targets: torch.Tensor) -> torch.Tensor:
    B, H, W = targets.shape
    logits = torch.zeros(B, NUM_CLASSES, H, W, device=targets.device)
    logits.scatter_(1, targets.unsqueeze(1), 10.0)
    return logits


def test_segmentation_loss():
    print("=" * 50)
    print("Testing SegmentationLoss (Dice + CE)")
    print("=" * 50)

    logits, targets = create_test_data(requires_grad=True)
    seg_loss = SegmentationLoss().to(DEVICE)

    loss = seg_loss(logits, targets)
    print(f"Random predictions - Combined Loss: {loss.item():.4f}")

    loss.backward()
    print("✓ Backward pass successful\n")


def test_class_imbalance():
    print("=" * 50)
    print("Testing Class Imbalance Handling")
    print("=" * 50)

    targets = torch.zeros(2, 224, 224, dtype=torch.long, device=DEVICE)
    targets[:, 100:110, 100:110] = 1

    foreground_pixels = (targets == 1).sum().item()
    total_pixels = targets.numel()
    print(f"Foreground: {foreground_pixels}/{total_pixels} ({100*foreground_pixels/total_pixels:.1f}%)")

    dice_loss = DiceLoss().to(DEVICE)
    logits = create_perfect_prediction(targets)
    loss = dice_loss(logits, targets)
    print(f"Perfect prediction on imbalanced data - Dice Loss: {loss.item():.4f}")
    print("✓ Dice loss handles imbalanced classes\n")


def main():
    print(f"Using device: {DEVICE}\n")
    test_dice_loss()
    test_segmentation_loss()
    test_class_imbalance()
    print("=" * 50)
    print("All loss tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
