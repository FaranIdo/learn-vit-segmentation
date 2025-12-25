# ViT Pet Segmentation - Model Analysis Report

**Vision Transformer for Semantic Segmentation**

**Run ID:** `nightly_second_2025-12-24_06-57-15`  
**Training Date:** December 24, 2025, 06:57 AM - 08:24 AM  
**Total Training Time:** 1h 27m

---

## Executive Summary

This training run achieved **82.74% mean IoU** on the test set, demonstrating strong segmentation performance on the Oxford-IIIT Pet Dataset. The model shows excellent overall accuracy but reveals specific weaknesses in boundary detection and small object segmentation that present clear opportunities for improvement.

### Key Findings

- **Overall Performance:** 91.88% pixel accuracy, 82.74% mIoU
- **Critical Weakness:** 28% accuracy gap between boundaries (63.33%) and interior regions (91.73%)
- **Size Bias:** Large objects (81.5% IoU) significantly outperform small objects (54.9% IoU)
- **Confidence Gap:** 14% difference in confidence between correct (97.3%) and incorrect (83.2%) predictions

---

## Training Configuration

### Model Architecture

"""
Architecture: Vision Transformer (ViT) for Semantic Segmentation
- Total Parameters: 11,911,010
- Trainable Parameters: 11,911,010 (100%)
- Device: Apple Silicon (MPS)
"""

**Model Specifications:**
- Image Size: 224×224 pixels
- Patch Size: 16×16 pixels (196 patches total)
- Embedding Dimension: 384
- Attention Heads: 6
- Transformer Layers: 6
- MLP Ratio: 4.0
- Dropout Rate: 0.1
- Output Classes: 2 (background, pet)

### Training Setup

**Dataset:**
- Training Samples: 2,944
- Validation Samples: 736
- Test Samples: 3,669
- Batch Size: 16

**Optimization:**
- Loss Function: Combined Dice + Cross Entropy
- Optimizer: AdamW
  - Learning Rate: 1e-4 (0.0001)
  - Weight Decay: 0.01
- Scheduler: CosineAnnealingLR (T_max=200)
- Maximum Epochs: 200
- Early Stopping Patience: 10 epochs

---

## Training Progress

### Learning Trajectory

The model trained for **105 epochs** before early stopping was triggered. Training completed in 1 hour 27 minutes, averaging 49 seconds per epoch.

**Key Milestones:**

| Epoch | Val mIoU | Val Accuracy | Status |
|-------|----------|--------------|---------|
| 1 | 52.54% | 71.92% | Initial baseline |
| 12 | 76.80% | 88.90% | First major breakthrough |
| 20 | 79.15% | 89.81% | Reached 79% mIoU |
| 24 | 80.13% | 90.57% | Passed 80% threshold |
| 52 | 82.83% | 91.99% | Reached 82% mIoU |
| 95 | **84.40%** | **92.79%** | **Best model** |
| 105 | 84.10% | 92.61% | Training stopped |

### Training Dynamics

**Early Phase (Epochs 1-20):**
- Rapid initial learning with mIoU jumping from 52.54% to 79.15%
- Steady validation accuracy improvement from 71.92% to 89.81%
- Notable spike at Epoch 4 (val loss: 1.6478) followed by recovery

**Mid Phase (Epochs 21-60):**
- Slower, steadier progress from 79.15% to 83.03% mIoU
- Learning rate gradually decreased from 9.76e-05 to 8.00e-05
- Multiple small improvements with occasional plateaus

**Late Phase (Epochs 61-95):**
- Fine-tuning phase with incremental gains
- Best model achieved at Epoch 95 with 84.40% validation mIoU
- Learning rate reduced to 5.47e-05

**Final Phase (Epochs 96-105):**
- Performance plateaued around 84% mIoU
- Early stopping triggered after 10 epochs without improvement
- Model avoided overfitting while maintaining strong validation performance

### Training Stability

The training showed excellent stability throughout:

- **No catastrophic drops:** Validation metrics remained stable or improved
- **Smooth convergence:** Loss decreased steadily without wild oscillations
- **Proper regularization:** Training and validation metrics tracked closely, indicating minimal overfitting
- **Effective early stopping:** Prevented unnecessary training beyond optimal performance

---

## Test Set Evaluation Results

### Overall Metrics

"""
Test Set Performance (3,669 samples):
- Pixel Accuracy: 91.88%
- Mean IoU (mIoU): 82.74%
- Background IoU: 88.95%
- Pet (Foreground) IoU: 76.53%
"""

**Class-Level Performance:**

The model performs better on background segmentation than foreground:

| Class | IoU | Observation |
|-------|-----|-------------|
| Background | 88.95% | Strong performance |
| Pet (Foreground) | 76.53% | 12.42% gap from background |

This 12% gap indicates the model finds it easier to identify non-pet regions than precisely segment pet boundaries and features.

---

## Deep Analysis Results

### Phase 1: Error Visualization & Size-Based Analysis

#### Error Distribution

The model exhibits a wide performance range across test samples:

**Performance Categories:**
- **Best Predictions (Top 5):** IoU ≈ 97.5-97.8%
  - Near-perfect segmentation
  - Clean boundaries
  - Minimal false positives/negatives

- **Median Predictions (5 samples):** IoU ≈ 79.9-80.0%
  - Representative of typical performance
  - Some boundary roughness
  - Occasional small false regions

- **Worst Predictions (Bottom 5):** IoU ≈ 0.0%
  - Complete segmentation failures
  - Often images with unusual poses, multiple pets, or occlusions
  - High false positive rates

#### Size-Based Performance Analysis

The model shows strong **size bias**, performing significantly worse on small objects:

"""
Object Size Categories (by pixel count):
- Small (< 5000 pixels): 54.9% IoU
- Medium (5000-15000 pixels): 73.8% IoU  
- Large (15000-30000 pixels): 81.5% IoU
- Extra Large (> 30000 pixels): 81.2% IoU
"""

**Key Insights:**
1. **26.6% performance gap** between small and large objects
2. Performance plateaus at ~81% for large objects
3. Small objects (close-up faces, distant pets) are the primary challenge
4. Medium-to-large objects show consistent, strong performance

**Implications:**
- Model architecture or training may need adjustments for multi-scale feature extraction
- Data augmentation focusing on scale variation could help
- Small object detection remains a critical improvement area

---

### Phase 2: Boundary vs Interior Analysis

This analysis reveals the **most significant model weakness**: poor boundary prediction.

"""
Boundary vs Interior Accuracy:
- Boundary Pixels (3-pixel band): 63.33%
- Interior Pixels: 91.73%
- Accuracy Gap: 28.40%
"""

#### How Boundaries Were Defined

Using morphological operations:
1. **Dilation:** Expand mask by 3 pixels
2. **Erosion:** Shrink mask by 3 pixels
3. **Boundary:** Dilated region minus eroded region (creates 3-pixel band around edges)
4. **Interior:** Eroded region (pixels safely inside the object)

#### Analysis of 9 Sample Cases

**Boundary Performance Distribution:**

| Sample | IoU | Boundary Accuracy | Interior Accuracy | Gap |
|--------|-----|-------------------|-------------------|-----|
| 2905 | 97.8% | 73.57% | 96.44% | 22.87% |
| 2661 | 97.8% | 71.13% | 95.89% | 24.76% |
| 2248 | 97.7% | 72.03% | 96.34% | 24.31% |
| 0426 | 79.9% | 52.81% | 88.03% | 35.22% |
| 2241 | 79.9% | 51.60% | 88.45% | 36.85% |
| 2437 | 79.9% | 52.03% | 87.70% | 35.67% |
| 2005 | 0.0% | 57.48% | 88.86% | 31.38% |
| 1858 | 0.0% | 60.75% | 89.81% | 29.06% |
| 3473 | 0.0% | 56.77% | 87.95% | 31.18% |

**Observations:**
- Even in best predictions (97.8% IoU), boundaries achieve only ~72% accuracy
- Worst predictions show slightly better boundary accuracy (57-61%) but fail elsewhere
- Interior accuracy consistently high (87-96%) across all samples
- The model struggles uniformly with edges regardless of overall performance

#### Why This Matters

Boundaries are critical for:
1. **Visual quality:** Rough edges look poor even with high IoU
2. **Object separation:** Essential for distinguishing multiple nearby objects
3. **Fine details:** Ears, paws, tails require precise edge detection
4. **Real-world applications:** Medical imaging, robotics need accurate boundaries

**Recommended Improvements:**
- Add boundary-focused loss term (e.g., boundary IoU loss)
- Use edge-aware data augmentation
- Implement multi-scale feature fusion
- Add dedicated boundary refinement module

---

### Phase 3: Confidence & Uncertainty Analysis

The model shows good calibration overall but reveals interesting confidence patterns.

"""
Confidence Analysis (from softmax probabilities):
- Correct Predictions: Mean 97.31%, Std 7.48%
- Incorrect Predictions: Mean 83.20%, Std 15.28%
- Confidence Gap: 14.11%
"""

#### Confidence Distribution

**Correct Predictions:**
- Very high confidence (97.31% average)
- Low variance (7.48% std) indicates consistency
- Model is generally certain when it's right

**Incorrect Predictions:**
- Significantly lower confidence (83.20% average)
- High variance (15.28% std) shows uncertainty
- Model knows when it might be wrong

#### Uncertainty Zones

"""
Uncertainty Distribution:
- High Uncertainty (confidence < 0.5): 0.0% accuracy, 0 pixels
- Low Uncertainty (confidence >= 0.5): 91.88% accuracy, 184M pixels
"""

**Interpretation:**
- Model never produces highly uncertain predictions (< 50% confidence)
- This suggests possible overconfidence or lack of true uncertainty estimation
- All predictions are made with at least moderate confidence

#### Confidence vs Correctness

Analyzing 9 sample visualizations reveals:

**Best Predictions (IoU 97.5-97.8%):**
- Correct pixels: 98-99% confidence
- Wrong pixels: 72-80% confidence
- Clear confidence separation

**Median Predictions (IoU 79.9%):**
- Correct pixels: 97% confidence
- Wrong pixels: 80-87% confidence  
- Moderate confidence separation

**Worst Predictions (IoU 0.0%):**
- Correct pixels: 84-99% confidence (varies)
- Wrong pixels: 92-96% confidence
- **Little to no confidence separation** - the model doesn't realize it's wrong!

**Key Finding:** The model's confidence calibration breaks down in failure cases. When it completely misses a segmentation, it remains highly confident in its incorrect prediction.

#### Implications

1. **Good calibration for normal cases:** The 14% confidence gap helps identify likely errors
2. **Overconfidence in failures:** Catastrophic failures show high confidence, limiting their detection
3. **Missing uncertainty:** No truly uncertain predictions suggests the model could benefit from:
   - Monte Carlo dropout for uncertainty estimation
   - Bayesian approaches
   - Ensemble methods
   - Temperature scaling for better calibration

---

## Critical Weaknesses Summary

### 1. Boundary Detection (Highest Priority)

**Impact:** Severe - 28% accuracy gap  
**Prevalence:** Affects all predictions  
**Visual Impact:** High - creates rough, unnatural edges

**Root Causes:**
- Standard segmentation losses don't emphasize boundaries
- 16×16 patch size may lose fine spatial detail
- Transformer architecture inherently less suited for pixel-precise tasks

**Recommended Solutions:**
- Add boundary-specific loss (e.g., weighted cross-entropy on edge pixels)
- Implement boundary refinement module
- Use multi-scale features or skip connections
- Consider hybrid CNN-Transformer architecture

### 2. Small Object Performance (High Priority)

**Impact:** Severe - 27% IoU drop for small objects  
**Prevalence:** Affects ~25% of test samples (estimate)  
**Visual Impact:** Medium - small objects are less prominent

**Root Causes:**
- Fixed 224×224 input resolution
- 16×16 patches create 14×14 grid - too coarse for small details
- Limited spatial resolution in feature maps

**Recommended Solutions:**
- Increase input resolution (384×384 or 512×512)
- Use smaller patch sizes (8×8)
- Implement feature pyramid networks
- Add scale-specific augmentations

### 3. Overconfidence in Failures (Medium Priority)

**Impact:** Moderate - prevents error detection  
**Prevalence:** Rare but critical when it occurs  
**Visual Impact:** Low - but high consequence for safety-critical applications

**Root Causes:**
- Softmax produces overconfident predictions
- No explicit uncertainty modeling
- Model trained with hard labels (0 or 1)

**Recommended Solutions:**
- Implement Monte Carlo dropout at inference
- Add temperature scaling for calibration
- Use label smoothing during training
- Consider ensemble methods

### 4. Class Imbalance Effects (Lower Priority)

**Impact:** Moderate - 12% gap between background and foreground  
**Prevalence:** Affects all predictions  
**Visual Impact:** Medium - more false negatives

**Root Causes:**
- Background is easier (more pixels, simpler patterns)
- Foreground requires distinguishing between pet breeds
- Possible training data imbalance

**Recommended Solutions:**
- Use class-weighted losses
- Implement focal loss
- Balance sampling during training
- Add hard example mining

---

## Training Efficiency Analysis

### Computational Performance

"""
Training Metrics:
- Average Epoch Time: 49 seconds
- Training Time per Batch: ~260ms (46s / 184 batches)
- Validation Time per Batch: ~65ms (3s / 46 batches)
- Total Training Time: 1h 27m for 105 epochs
- Hardware: Apple Silicon (MPS)
"""

**Efficiency Observations:**
- Fast training on Apple Silicon
- 5.13 minutes per 1% mIoU improvement (early epochs)
- Final 2% improvement (82% → 84%) took 35 minutes
- Validation is 4× faster than training (expected with no gradient computation)

### Early Stopping Effectiveness

Early stopping triggered after Epoch 105, having seen no improvement for 10 epochs since Epoch 95.

**Saved Time:** ~95 minutes (avoided 95 unnecessary epochs)  
**Performance Loss:** None (best model was from Epoch 95)

This demonstrates excellent training efficiency and proper use of early stopping.

---

## Comparison to Baseline

### Initial Performance (Epoch 1)

| Metric | Epoch 1 | Epoch 95 (Best) | Improvement |
|--------|---------|-----------------|-------------|
| mIoU | 52.54% | 84.40% | +31.86% |
| Pixel Accuracy | 71.92% | 92.79% | +20.87% |
| Background IoU | 65.64% | 90.20% | +24.56% |
| Foreground IoU | 39.44% | 78.59% | +39.15% |

The model achieved nearly **40% improvement in foreground IoU**, showing the training process was highly effective at learning pet segmentation.

---

## Recommendations for Future Runs

### Immediate Improvements (Quick Wins)

1. **Boundary Loss:** Add boundary-weighted cross-entropy
   - Expected gain: +2-3% mIoU, significant visual improvement
   - Implementation time: 1-2 hours
   - Risk: Low

2. **Higher Resolution:** Increase input size to 384×384
   - Expected gain: +3-5% mIoU, especially for small objects
   - Implementation time: 30 minutes
   - Risk: Increased memory usage, longer training

3. **Data Augmentation:** Add scale and boundary-focused augmentations
   - Expected gain: +1-2% mIoU
   - Implementation time: 1 hour
   - Risk: Low

### Medium-Term Enhancements

4. **Smaller Patches:** Reduce from 16×16 to 8×8
   - Expected gain: +2-4% mIoU
   - Implementation time: 2 hours
   - Risk: 4× more patches, may require architecture adjustments

5. **Multi-Scale Features:** Add feature pyramid or U-Net style skip connections
   - Expected gain: +3-5% mIoU
   - Implementation time: 4-8 hours
   - Risk: Medium - architecture redesign

6. **Boundary Refinement Module:** Add dedicated post-processing for edges
   - Expected gain: +2-3% mIoU, large visual improvement
   - Implementation time: 4-6 hours
   - Risk: Medium

### Advanced Improvements

7. **Hybrid Architecture:** Combine CNN encoder with Transformer decoder
   - Expected gain: +4-7% mIoU
   - Implementation time: 8-16 hours
   - Risk: High - major architecture change

8. **Uncertainty Estimation:** Implement Bayesian or ensemble approaches
   - Expected gain: +1-2% mIoU, better confidence calibration
   - Implementation time: 8-12 hours
   - Risk: Medium - inference may be slower

---

## Conclusion

This training run produced a strong baseline model with **82.74% test mIoU** and **91.88% pixel accuracy**. The training process was efficient and stable, reaching peak performance at Epoch 95 after 1h 27m of training.

### Strengths

- Excellent overall accuracy (91.88%)
- Strong performance on large objects (81.5% IoU)
- Good confidence calibration in normal cases
- Efficient training with proper early stopping
- No overfitting observed

### Critical Areas for Improvement

1. **Boundary detection:** 28% accuracy gap is the primary weakness
2. **Small object segmentation:** 27% IoU drop needs addressing
3. **Failure case confidence:** Model overconfident when completely wrong

### Expected Performance with Improvements

With the recommended immediate improvements (boundary loss, higher resolution, augmentation):
- **Projected mIoU:** 88-91% (+5-8%)
- **Boundary accuracy:** 75-80% (+12-17%)
- **Small object IoU:** 65-70% (+10-15%)

The model demonstrates the effectiveness of Vision Transformers for semantic segmentation while highlighting the importance of boundary-aware training and multi-scale feature processing for achieving state-of-the-art performance.

---

**Report Generated:** December 25, 2025  
**Analysis Tools:** Enhanced evaluation pipeline with boundary and confidence analysis  
**Documentation:** `/docs/model_analysis_report.md`

