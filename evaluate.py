import sys

sys.path.insert(0, ".")

import argparse
import torch
from tqdm import tqdm

from vit.segmenter import Segmenter
from data.oxford_pets import create_test_loader
from training.metrics import SegmentationMetrics
from config import DEVICE, IMAGE_SIZE, PATCH_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, MLP_RATIO, DROPOUT, NUM_CLASSES, BATCH_SIZE, DATA_DIR


def load_checkpoint(checkpoint_path: str, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint.get("epoch", "unknown"), checkpoint.get("metrics", {})


def evaluate_model(model, test_loader, device, num_classes):
    model.eval()
    metrics = SegmentationMetrics(num_classes)

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating on Test Set")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            metrics.update(preds, masks)

            pbar.set_postfix({"miou": f"{metrics.get_miou():.4f}", "acc": f"{metrics.get_pixel_accuracy():.4f}"})

    return metrics.get_metrics_dict()


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT Segmentation on official test set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (e.g., checkpoints/run_*/best_model.pth)")
    args = parser.parse_args()

    print("=" * 70)
    print("ViT Segmentation - Final Test Set Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {DEVICE}")
    print()

    print("1. Loading test data...")
    test_loader = create_test_loader(root=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
    print(f"   Test samples: {len(test_loader.dataset)}")
    print(f"   Batch size: {BATCH_SIZE}")

    print("\n2. Creating model...")
    model = Segmenter(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, dropout=DROPOUT)
    model = model.to(DEVICE)

    print("\n3. Loading checkpoint...")
    epoch, train_metrics = load_checkpoint(args.checkpoint, model, DEVICE)
    print(f"   Loaded model from epoch: {epoch}")
    if train_metrics:
        val_miou = train_metrics.get("miou", "N/A")
        print(f"   Validation mIoU (during training): {val_miou}")

    print("\n4. Evaluating on test set...")
    print("-" * 70)
    test_metrics = evaluate_model(model, test_loader, DEVICE, NUM_CLASSES)

    print("\n" + "=" * 70)
    print("FINAL TEST SET RESULTS")
    print("=" * 70)
    print(f"Pixel Accuracy: {test_metrics['pixel_accuracy']:.4f}")
    print(f"Mean IoU (mIoU): {test_metrics['miou']:.4f}")
    print()
    print("Per-Class IoU:")
    for c in range(NUM_CLASSES):
        class_name = "background" if c == 0 else "pet"
        iou = test_metrics[f"iou_class_{c}"]
        print(f"  Class {c} ({class_name}): {iou:.4f}")
    print("=" * 70)
    print()
    print("✓ Evaluation complete!")
    print("  These are your final test metrics.")
    print("=" * 70)


if __name__ == "__main__":
    main()
