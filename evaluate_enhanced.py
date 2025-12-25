import sys

sys.path.insert(0, ".")

import os
import argparse
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from vit.segmenter import Segmenter
from data.oxford_pets import create_test_loader
from training.metrics import SegmentationMetrics
from evaluation.analysis import calculate_sample_iou, select_samples_by_iou, plot_error_grid, plot_error_map_comparison, analyze_by_size, analyze_boundary_vs_interior, plot_boundary_analysis, analyze_confidence_by_correctness, analyze_uncertainty_zones, plot_confidence_analysis
from config import DEVICE, IMAGE_SIZE, PATCH_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, MLP_RATIO, DROPOUT, NUM_CLASSES, BATCH_SIZE, DATA_DIR, parse_device


class TeeOutput:
    """Write to both stdout and a file simultaneously."""

    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def get_output_dir(checkpoint_path: str) -> str:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    eval_dir = os.path.join(checkpoint_dir, "evaluation")
    return eval_dir


def load_checkpoint(checkpoint_path: str, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint.get("epoch", "unknown"), checkpoint.get("metrics", {})


def collect_all_predictions(model, test_loader, device):
    model.eval()

    all_images = []
    all_masks = []
    all_predictions = []
    all_ious = []
    all_mask_areas = []
    all_logits = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Collecting predictions")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            for i in range(images.shape[0]):
                img = images[i].cpu()
                mask = masks[i].cpu()
                pred = preds[i].cpu()
                logit = logits[i].cpu()

                iou = calculate_sample_iou(pred, mask, class_id=1)
                mask_area = (mask == 1).sum().item()

                all_images.append(img)
                all_masks.append(mask)
                all_predictions.append(pred)
                all_ious.append(iou)
                all_mask_areas.append(mask_area)
                all_logits.append(logit)

    return all_images, all_masks, all_predictions, all_ious, all_mask_areas, all_logits


def evaluate_with_metrics(model, test_loader, device, num_classes):
    model.eval()
    metrics = SegmentationMetrics(num_classes)

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Computing metrics")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            metrics.update(preds, masks)
            pbar.set_postfix({"miou": f"{metrics.get_miou():.4f}", "acc": f"{metrics.get_pixel_accuracy():.4f}"})

    return metrics.get_metrics_dict()


def run_phase1_analysis(all_images, all_masks, all_predictions, all_ious, all_mask_areas, output_dir):
    print("\n" + "=" * 70)
    print("PHASE 1: ERROR VISUALIZATION & SIZE ANALYSIS")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "error_maps"), exist_ok=True)

    print("\n1. Selecting samples by IoU...")
    selected = select_samples_by_iou(all_ious, num_best=5, num_median=5, num_worst=5)

    worst_ious_str = [f"{all_ious[i]:.3f}" for i in selected["worst"]]
    median_ious_str = [f"{all_ious[i]:.3f}" for i in selected["median"]]
    best_ious_str = [f"{all_ious[i]:.3f}" for i in selected["best"]]

    print(f"   Worst IoUs: {worst_ious_str}")
    print(f"   Median IoUs: {median_ious_str}")
    print(f"   Best IoUs: {best_ious_str}")

    print("\n2. Creating error visualization grids...")
    for category, indices in selected.items():
        images = [all_images[i] for i in indices]
        masks = [all_masks[i] for i in indices]
        preds = [all_predictions[i] for i in indices]
        ious = [all_ious[i] for i in indices]

        fig = plot_error_grid(images, masks, preds, ious, category, NUM_CLASSES)
        output_path = os.path.join(output_dir, f"error_grid_{category}.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved: {output_path}")

    print("\n3. Creating individual error maps...")
    all_selected = selected["worst"] + selected["median"] + selected["best"]
    for idx, sample_idx in enumerate(all_selected):
        fig = plot_error_map_comparison(all_images[sample_idx], all_masks[sample_idx], all_predictions[sample_idx], all_ious[sample_idx], sample_idx)
        output_path = os.path.join(output_dir, "error_maps", f"sample_{sample_idx:04d}_iou_{all_ious[sample_idx]:.3f}.png")
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"   Saved 15 error maps to: {os.path.join(output_dir, 'error_maps/')}")

    print("\n4. Size-based analysis...")
    total_pixels = IMAGE_SIZE * IMAGE_SIZE
    size_stats = analyze_by_size(all_ious, all_mask_areas, total_pixels)

    print("\n   Results by Pet Size:")
    print("   " + "-" * 60)
    print(f"   Small pets (<10% of image):")
    print(f"      Count: {size_stats['small']['count']}")
    print(f"      Mean IoU: {size_stats['small']['mean_iou']:.4f}")
    print(f"   Medium pets (10-30% of image):")
    print(f"      Count: {size_stats['medium']['count']}")
    print(f"      Mean IoU: {size_stats['medium']['mean_iou']:.4f}")
    print(f"   Large pets (>30% of image):")
    print(f"      Count: {size_stats['large']['count']}")
    print(f"      Mean IoU: {size_stats['large']['mean_iou']:.4f}")
    print("   " + "-" * 60)

    return size_stats


def run_phase2_analysis(all_images, all_masks, all_predictions, all_ious, selected, output_dir):
    print("\n" + "=" * 70)
    print("PHASE 2: BOUNDARY VS INTERIOR ERROR ANALYSIS")
    print("=" * 70)

    os.makedirs(os.path.join(output_dir, "boundary_analysis"), exist_ok=True)

    print("\n1. Computing boundary vs interior accuracy...")
    boundary_stats = analyze_boundary_vs_interior(all_masks, all_predictions)

    print("\n   Boundary vs Interior Results:")
    print("   " + "-" * 60)
    print(f"   Boundary Accuracy: {boundary_stats['boundary_accuracy']:.4f}")
    print(f"   Interior Accuracy: {boundary_stats['interior_accuracy']:.4f}")
    print(f"   Accuracy Gap: {boundary_stats['interior_accuracy'] - boundary_stats['boundary_accuracy']:.4f}")
    print(f"   Boundary Pixels: {boundary_stats['boundary_pixels']:,}")
    print(f"   Interior Pixels: {boundary_stats['interior_pixels']:,}")
    print("   " + "-" * 60)

    print("\n2. Creating boundary analysis visualizations...")
    all_selected = selected["worst"][:3] + selected["median"][:3] + selected["best"][:3]

    for idx, sample_idx in enumerate(all_selected):
        iou = all_ious[sample_idx]
        fig, boundary_acc, interior_acc = plot_boundary_analysis(all_images[sample_idx], all_masks[sample_idx], all_predictions[sample_idx], sample_idx, iou)
        output_path = os.path.join(output_dir, "boundary_analysis", f"sample_{sample_idx:04d}_iou_{iou:.3f}_b_{boundary_acc:.2f}_i_{interior_acc:.2f}.png")
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"   Saved {len(all_selected)} boundary analysis visualizations")

    return boundary_stats


def run_phase3_analysis(all_images, all_masks, all_predictions, all_logits, all_ious, selected, output_dir):
    print("\n" + "=" * 70)
    print("PHASE 3: CONFIDENCE & UNCERTAINTY ANALYSIS")
    print("=" * 70)

    os.makedirs(os.path.join(output_dir, "confidence_analysis"), exist_ok=True)

    print("\n1. Analyzing confidence by correctness...")
    conf_stats = analyze_confidence_by_correctness(all_logits, all_predictions, all_masks)

    print("\n   Confidence Analysis Results:")
    print("   " + "-" * 60)
    print(f"   Correct Predictions:")
    print(f"      Mean Confidence: {conf_stats['correct_mean']:.4f}")
    print(f"      Std Confidence: {conf_stats['correct_std']:.4f}")
    print(f"      Count: {conf_stats['correct_count']:,}")
    print(f"   Incorrect Predictions:")
    print(f"      Mean Confidence: {conf_stats['incorrect_mean']:.4f}")
    print(f"      Std Confidence: {conf_stats['incorrect_std']:.4f}")
    print(f"      Count: {conf_stats['incorrect_count']:,}")
    print(f"   Confidence Gap: {conf_stats['correct_mean'] - conf_stats['incorrect_mean']:.4f}")
    print("   " + "-" * 60)

    print("\n2. Analyzing uncertainty zones...")
    uncertainty_stats = analyze_uncertainty_zones(all_logits, all_predictions, all_masks, uncertainty_threshold=0.5)

    print("\n   Uncertainty Zone Results:")
    print("   " + "-" * 60)
    print(f"   High Uncertainty (confidence < 0.5):")
    print(f"      Accuracy: {uncertainty_stats['uncertain_accuracy']:.4f}")
    print(f"      Pixels: {uncertainty_stats['uncertain_pixels']:,}")
    print(f"   Low Uncertainty (confidence >= 0.5):")
    print(f"      Accuracy: {uncertainty_stats['certain_accuracy']:.4f}")
    print(f"      Pixels: {uncertainty_stats['certain_pixels']:,}")
    print(f"   Accuracy Gap: {uncertainty_stats['certain_accuracy'] - uncertainty_stats['uncertain_accuracy']:.4f}")
    print("   " + "-" * 60)

    print("\n3. Creating confidence visualizations...")
    all_selected = selected["worst"][:3] + selected["median"][:3] + selected["best"][:3]

    for idx, sample_idx in enumerate(all_selected):
        iou = all_ious[sample_idx]
        fig, correct_conf, incorrect_conf = plot_confidence_analysis(
            all_images[sample_idx], 
            all_masks[sample_idx], 
            all_predictions[sample_idx], 
            all_logits[sample_idx],
            sample_idx, 
            iou
        )
        output_path = os.path.join(
            output_dir, 
            "confidence_analysis", 
            f"sample_{sample_idx:04d}_iou_{iou:.3f}_conf_c_{correct_conf:.3f}_w_{incorrect_conf:.3f}.png"
        )
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"   Saved {len(all_selected)} confidence analysis visualizations")

    return conf_stats, uncertainty_stats


def main():
    parser = argparse.ArgumentParser(description="Enhanced Evaluation with Deep Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--phase", type=str, default="1", choices=["1", "2", "3", "all"], help="Which phase to run (1, 2, 3, or all)")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use for evaluation (default: auto-detect)"
    )
    args = parser.parse_args()
    
    device = parse_device(args.device)

    output_dir = get_output_dir(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "evaluation_results.txt")
    tee = TeeOutput(results_file)
    sys.stdout = tee

    try:
        print("=" * 70)
        print("ViT Segmentation - Enhanced Test Set Evaluation")
        print("=" * 70)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Device: {device}")
        print(f"Phase: {args.phase}")

        print("\nLoading test data...")
        test_loader = create_test_loader(root=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
        print(f"Test samples: {len(test_loader.dataset)}")

        print("\nCreating model...")
        model = Segmenter(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=NUM_CLASSES, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO, dropout=DROPOUT)
        model = model.to(device)

        print("\nLoading checkpoint...")
        epoch, train_metrics = load_checkpoint(args.checkpoint, model, device)
        print(f"Loaded model from epoch: {epoch}")

        print(f"\nOutput directory: {output_dir}")

        print("\n" + "=" * 70)
        print("OVERALL METRICS")
        print("=" * 70)
        test_metrics = evaluate_with_metrics(model, test_loader, device, NUM_CLASSES)

        print(f"\nPixel Accuracy: {test_metrics['pixel_accuracy']:.4f}")
        print(f"Mean IoU (mIoU): {test_metrics['miou']:.4f}")
        print("Per-Class IoU:")

        for c in range(NUM_CLASSES):
            class_name = "background" if c == 0 else "pet"
            iou = test_metrics[f"iou_class_{c}"]
            print(f"  Class {c} ({class_name}): {iou:.4f}")

        if args.phase in ["1", "2", "3", "all"]:
            print("\nCollecting all predictions for detailed analysis...")
            all_images, all_masks, all_predictions, all_ious, all_mask_areas, all_logits = collect_all_predictions(model, test_loader, device)

        if args.phase in ["1", "all"]:
            run_phase1_analysis(all_images, all_masks, all_predictions, all_ious, all_mask_areas, output_dir)

        if args.phase in ["2", "all"]:
            selected = select_samples_by_iou(all_ious, num_best=5, num_median=5, num_worst=5)
            run_phase2_analysis(all_images, all_masks, all_predictions, all_ious, selected, output_dir)

        if args.phase in ["3", "all"]:
            selected = select_samples_by_iou(all_ious, num_best=5, num_median=5, num_worst=5)
            run_phase3_analysis(all_images, all_masks, all_predictions, all_logits, all_ious, selected, output_dir)

        print("\n" + "=" * 70)
        print("✓ Enhanced evaluation complete!")
        print(f"Results saved to: {output_dir}/")
        print(f"Full output saved to: {results_file}")
        print("=" * 70)

    finally:
        tee.close()


if __name__ == "__main__":
    main()
