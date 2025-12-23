# Segmentation Metrics

## Why Pixel Accuracy Is Not Enough

### The Problem with Class Imbalance

In our Oxford Pets dataset, we have approximately:
- **85% background pixels**
- **15% pet pixels**

A naive model that predicts "all background" would achieve **85% pixel accuracy** without learning anything useful!

"""
Example with 100 pixels:
Ground Truth:  15 pet, 85 background
Prediction:     0 pet, 100 background (predict everything as background)
Pixel Accuracy: 85/100 = 85% ✓ (looks good!)
But: Completely failed to segment the pet! ✗
"""

This is why we need better metrics that account for **per-class performance**.

---

## 1. Pixel Accuracy

The simplest metric - what percentage of pixels are correctly classified?

"""
Pixel Accuracy = (Correct Pixels) / (Total Pixels)
"""

**Use case**: Quick sanity check during training.  
**Problem**: Misleading with class imbalance.

---

## 2. Intersection over Union (IoU)

Also called **Jaccard Index**. Computed separately for each class.

"""
IoU = Intersection / Union
    = TP / (TP + FP + FN)
"""

Where:
- **TP (True Positive)**: Pixels correctly predicted as this class
- **FP (False Positive)**: Pixels incorrectly predicted as this class  
- **FN (False Negative)**: Pixels of this class that were missed

### Visual Example

"""
Ground Truth:          Prediction:           Overlap Analysis:
■■■□□□□               ■■□□□□□               TP: ██ (2 correct)
■■■□□□□               ■■■□□□□               FP: ░  (1 wrong)
■■■□□□□               □■■□□□□               FN: ▓▓▓▓ (4 missed)
□□□□□□□               □□□□□□□
                                            
Pet class (■):                              Union = TP + FP + FN
- TP = 2 (correct pet pixels)                     = 2 + 1 + 4 = 7
- FP = 1 (wrongly predicted as pet)        
- FN = 4 (pet pixels missed)               IoU = 2 / 7 = 0.286
"""

### Why IoU Is Better

IoU penalizes both:
1. **Over-segmentation** (FP): Predicting pet where there's background
2. **Under-segmentation** (FN): Missing pet pixels

A model predicting "all background" would get:
- Pet IoU = 0 / (0 + 0 + all_pet_pixels) = **0.0** (correctly shows failure!)

---

## 3. Mean IoU (mIoU)

The **standard metric** for semantic segmentation. Average IoU across all classes.

"""
mIoU = (IoU_class_0 + IoU_class_1 + ... + IoU_class_N) / N
"""

For our binary segmentation (background, pet):
"""
mIoU = (IoU_background + IoU_pet) / 2
"""

### Why mIoU?

- **Handles class imbalance**: Both classes contribute equally to the final score
- **Industry standard**: Used in PASCAL VOC, COCO, Cityscapes, ADE20K
- **Interpretable**: 
  - mIoU = 1.0: Perfect segmentation
  - mIoU = 0.5: Decent segmentation
  - mIoU < 0.3: Poor segmentation

### Example Comparison

"""
Scenario 1: "All background" prediction
- Background IoU: ~0.85 (mostly correct)
- Pet IoU: 0.0 (completely missed)
- mIoU: (0.85 + 0.0) / 2 = 0.425
- Pixel Accuracy: 85%

Scenario 2: Good segmentation
- Background IoU: 0.92
- Pet IoU: 0.78
- mIoU: (0.92 + 0.78) / 2 = 0.85
- Pixel Accuracy: 89%
"""

Notice how Scenario 2 has only slightly higher pixel accuracy (89% vs 85%), but **much higher mIoU** (0.85 vs 0.425), correctly reflecting the quality difference!

---

## What We Will Report

During training and evaluation, we'll track:

1. **Pixel Accuracy**: Quick sanity check
2. **IoU per class**: See which classes are hard to segment
3. **mIoU**: **Primary metric** for model quality

### Typical Benchmarks

- **Oxford Pets**: mIoU ~0.75-0.85 is good for binary pet segmentation
- **PASCAL VOC** (21 classes): mIoU ~0.80-0.85 is state-of-the-art
- **Cityscapes** (19 classes): mIoU ~0.80-0.85 is state-of-the-art

