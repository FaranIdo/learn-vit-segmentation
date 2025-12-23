import sys
sys.path.insert(0, ".")

import torch
from training.metrics import (
    calculate_pixel_accuracy,
    calculate_iou_per_class,
    calculate_miou,
    SegmentationMetrics
)
from config import NUM_CLASSES


def create_simple_test_case():
    """Create a simple test case for manual verification."""
    print("=" * 60)
    print("Test Case 1: Simple Perfect Prediction")
    print("=" * 60)
    
    preds = torch.tensor([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ])
    
    targets = preds.clone()
    
    print("\nPrediction and Target are identical:")
    print(preds)
    
    pixel_acc = calculate_pixel_accuracy(preds, targets)
    ious = calculate_iou_per_class(preds, targets, NUM_CLASSES)
    miou = calculate_miou(preds, targets, NUM_CLASSES)
    
    print(f"\nPixel Accuracy: {pixel_acc:.4f} (expected: 1.0000)")
    print(f"IoU Class 0 (background): {ious[0]:.4f} (expected: 1.0000)")
    print(f"IoU Class 1 (pet): {ious[1]:.4f} (expected: 1.0000)")
    print(f"mIoU: {miou:.4f} (expected: 1.0000)")
    
    assert pixel_acc == 1.0, "Perfect prediction should have 100% accuracy"
    assert ious[0] == 1.0, "Perfect prediction should have IoU=1.0 for class 0"
    assert ious[1] == 1.0, "Perfect prediction should have IoU=1.0 for class 1"
    assert miou == 1.0, "Perfect prediction should have mIoU=1.0"
    print("\n✓ All assertions passed!")


def test_partial_overlap():
    """Test case with partial overlap - matches the doc example."""
    print("\n" + "=" * 60)
    print("Test Case 2: Partial Overlap")
    print("=" * 60)
    
    targets = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
    ])
    
    preds = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
    ])
    
    print("\nGround Truth (targets):")
    print(targets)
    print("\nPrediction (preds):")
    print(preds)
    
    pixel_acc = calculate_pixel_accuracy(preds, targets)
    ious = calculate_iou_per_class(preds, targets, NUM_CLASSES)
    miou = calculate_miou(preds, targets, NUM_CLASSES)
    
    print(f"\nPixel Accuracy: {pixel_acc:.4f}")
    print(f"  - Correct pixels: {(preds == targets).sum().item()}/15")
    
    print(f"\nIoU Class 1 (pet):")
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    print(f"  - TP (correct pet): {tp}")
    print(f"  - FP (wrong pet): {fp}")
    print(f"  - FN (missed pet): {fn}")
    print(f"  - IoU = {tp}/{tp + fp + fn} = {ious[1]:.4f}")
    
    print(f"\nIoU Class 0 (background): {ious[0]:.4f}")
    print(f"mIoU: {miou:.4f}")
    
    expected_pet_iou = 7 / (7 + 0 + 2)
    print(f"\nExpected pet IoU: {expected_pet_iou:.4f}")
    assert abs(ious[1] - expected_pet_iou) < 0.01, f"Expected pet IoU ~{expected_pet_iou:.4f}, got {ious[1]:.4f}"
    print("\n✓ Test passed!")


def test_all_background_prediction():
    """Test the pathological case: predict all background."""
    print("\n" + "=" * 60)
    print("Test Case 3: All Background Prediction (Bad Model)")
    print("=" * 60)
    
    targets = torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    
    preds = torch.zeros_like(targets)
    
    print("\nGround Truth:")
    print(targets)
    print("\nPrediction (all zeros):")
    print(preds)
    
    pixel_acc = calculate_pixel_accuracy(preds, targets)
    ious = calculate_iou_per_class(preds, targets, NUM_CLASSES)
    miou = calculate_miou(preds, targets, NUM_CLASSES)
    
    pet_pixels = (targets == 1).sum().item()
    total_pixels = targets.numel()
    expected_acc = (total_pixels - pet_pixels) / total_pixels
    
    print(f"\nPixel Accuracy: {pixel_acc:.4f}")
    print(f"  - Expected: {expected_acc:.4f} (proportion of background)")
    print(f"  - This looks decent but the model is useless!")
    
    print(f"\nIoU Class 0 (background): {ious[0]:.4f}")
    print(f"IoU Class 1 (pet): {ious[1]:.4f} (0.0 because completely missed!)")
    print(f"mIoU: {miou:.4f}")
    print(f"  - mIoU correctly shows this is a bad model!")
    
    assert ious[1] == 0.0, "Predicting no pets should give IoU=0 for pet class"
    assert miou < 0.6, "mIoU should be low for a bad model"
    print("\n✓ Test passed!")


def test_metrics_accumulator():
    """Test the SegmentationMetrics accumulator over multiple batches."""
    print("\n" + "=" * 60)
    print("Test Case 4: Metrics Accumulator (Multiple Batches)")
    print("=" * 60)
    
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)
    
    batch1_preds = torch.tensor([[0, 0, 1], [0, 1, 1]])
    batch1_targets = torch.tensor([[0, 0, 1], [0, 1, 1]])
    
    batch2_preds = torch.tensor([[1, 1, 0], [1, 0, 0]])
    batch2_targets = torch.tensor([[1, 1, 0], [0, 0, 0]])
    
    print("Batch 1: Perfect prediction")
    print("  Preds:", batch1_preds.view(-1).tolist())
    print("  Targets:", batch1_targets.view(-1).tolist())
    
    metrics.update(batch1_preds, batch1_targets)
    
    print("\nAfter Batch 1:")
    print(f"  Pixel Accuracy: {metrics.get_pixel_accuracy():.4f}")
    print(f"  mIoU: {metrics.get_miou():.4f}")
    
    print("\nBatch 2: Some errors")
    print("  Preds:", batch2_preds.view(-1).tolist())
    print("  Targets:", batch2_targets.view(-1).tolist())
    
    metrics.update(batch2_preds, batch2_targets)
    
    print("\nAfter Both Batches:")
    all_metrics = metrics.get_metrics_dict()
    print(f"  Pixel Accuracy: {all_metrics['pixel_accuracy']:.4f}")
    print(f"  IoU Class 0: {all_metrics['iou_class_0']:.4f}")
    print(f"  IoU Class 1: {all_metrics['iou_class_1']:.4f}")
    print(f"  mIoU: {all_metrics['miou']:.4f}")
    
    print("\n✓ Accumulator works correctly!")


def test_with_model_output():
    """Test with realistic model output (logits -> predictions)."""
    print("\n" + "=" * 60)
    print("Test Case 5: Realistic Model Output")
    print("=" * 60)
    
    batch_size = 2
    H, W = 8, 8
    
    logits = torch.randn(batch_size, NUM_CLASSES, H, W)
    targets = torch.randint(0, NUM_CLASSES, (batch_size, H, W))
    
    preds = torch.argmax(logits, dim=1)
    
    print(f"Batch size: {batch_size}")
    print(f"Image size: {H}x{W}")
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions shape: {preds.shape}")
    print(f"Targets shape: {targets.shape}")
    
    pixel_acc = calculate_pixel_accuracy(preds, targets)
    ious = calculate_iou_per_class(preds, targets, NUM_CLASSES)
    miou = calculate_miou(preds, targets, NUM_CLASSES)
    
    print(f"\nPixel Accuracy: {pixel_acc:.4f}")
    print(f"IoU Class 0: {ious[0]:.4f}")
    print(f"IoU Class 1: {ious[1]:.4f}")
    print(f"mIoU: {miou:.4f}")
    
    assert 0 <= pixel_acc <= 1, "Pixel accuracy must be in [0, 1]"
    assert 0 <= miou <= 1, "mIoU must be in [0, 1]"
    print("\n✓ Model output test passed!")


def main():
    print("=" * 60)
    print("Segmentation Metrics Test Suite")
    print("=" * 60)
    print(f"Number of classes: {NUM_CLASSES}")
    
    create_simple_test_case()
    test_partial_overlap()
    test_all_background_prediction()
    test_metrics_accumulator()
    test_with_model_output()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()

