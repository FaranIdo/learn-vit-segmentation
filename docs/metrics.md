# Segmentation Metrics

## Why accuracy is not enough
- Pixel accuracy counts how many pixels are correct.
- With class imbalance it can look high while being wrong (predict all background and still score well).

## Pixel accuracy
- Formula: correct_pixels / total_pixels.
- Good for a quick check, but it ignores imbalance.

## Intersection over Union (IoU)
- For each class: IoU = TP / (TP + FP + FN).
- TP: predicted the class and it was correct.
- FP: predicted the class but it was wrong.
- FN: missed pixels that belong to the class.
- IoU punishes both over-segmentation (FP) and under-segmentation (FN).

## Mean IoU (mIoU)
- Average IoU over all classes.
- For binary masks: mIoU = (IoU_background + IoU_foreground) / 2.
- More fair than accuracy when classes are imbalanced.

## What we will report
- Pixel accuracy for a quick sanity check.
- mIoU as the main metric for model quality.

