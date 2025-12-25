# Boundary vs Interior Error Analysis

## Overview

This document explains how we analyze segmentation model performance by separating **boundary errors** from **interior errors**. This helps identify where the model struggles most.

## Key Insight

Your model shows significantly different accuracy at boundaries vs interiors:

- **Boundary Accuracy: 63.33%** ⚠️
- **Interior Accuracy: 91.73%** ✓
- **Gap: 28.40%** - The model struggles at edges!

## How Boundary Extraction Works

### 1. Morphological Operations

We use two fundamental image processing operations:

#### Dilation (Expanding)
- Grows the mask outward by N pixels
- Each iteration adds a 1-pixel-thick ring around the shape
- Formula: For each pixel, if ANY neighbor is 1, set to 1

#### Erosion (Shrinking)
- Shrinks the mask inward by N pixels
- Each iteration removes a 1-pixel-thick outer ring
- Formula: For each pixel, if ANY neighbor is 0, set to 0

### 2. The Algorithm

"""python
def extract_boundary_pixels(mask: np.ndarray, thickness: int = 3):
    mask_binary = mask > 0
    
    # Expand mask by 3 pixels
    dilated = binary_dilation(mask_binary, iterations=3)
    
    # Shrink mask by 3 pixels
    eroded = binary_erosion(mask_binary, iterations=3)
    
    # Boundary = ring between dilated and eroded
    boundary = dilated & ~eroded
    
    # Interior = what remains after erosion
    interior = eroded
    
    return boundary, interior
"""

### 3. Visual Example

Starting with a cat mask (7,943 pixels):

"""
Original Mask:        Dilated (+3px):      Eroded (-3px):
    ▄▄▄▄▄                ▄▄▄▄▄▄▄              ▄▄▄
   ██████               █████████            ████
   ██████               █████████            ████
   ▀▀▀▀▀▀               ▀▀▀▀▀▀▀▀▀            ▀▀▀
    ║ ║                  ║   ║                
    
7,943 pixels          9,693 pixels         6,422 pixels
(original)            (+1,750)              (-1,521)
"""

**Boundary = Dilated & ~Eroded:**
- The yellow ring around the cat
- 3,271 pixels (41.2% of mask)
- Includes all edges, thin structures

**Interior = Eroded:**
- The blue core of the cat
- 6,422 pixels (58.8% of mask)
- At least 3 pixels from any edge

## Why This Matters

### Problem 1: Thin Structures
Small pets have high boundary-to-interior ratio:
- Cat legs: Almost 100% boundary pixels
- Cat tail: Almost 100% boundary pixels
- Result: **Small pets only achieve 54.9% IoU**

### Problem 2: Patch Size Limitation
Your ViT uses 16×16 patches:
- Each patch can only predict one class
- Boundaries between pet and background fall within patches
- The model must make a binary choice for mixed patches

### Problem 3: Boundary Errors Count Double
In IoU calculation:
- A boundary error is counted as both a False Positive AND a False Negative
- Interior errors only affect one class
- This explains why IoU (82.74%) is lower than pixel accuracy (91.88%)

## How We Measure

### Aggregate Statistics (All Test Images)

"""python
def analyze_boundary_vs_interior(all_masks, all_predictions):
    for mask, pred in zip(all_masks, all_predictions):
        boundary, interior = extract_boundary_pixels(mask)
        
        # Check correctness in each region
        boundary_correct = (pred == mask) & boundary
        interior_correct = (pred == mask) & interior
        
    return {
        "boundary_accuracy": boundary_correct / boundary_total,
        "interior_accuracy": interior_correct / interior_total,
    }
"""

### Per-Sample Visualizations

We generate boundary analysis images showing:
1. **Original Image** - The actual photo
2. **Ground Truth** - Where the pet actually is
3. **Prediction** - Where the model thinks the pet is
4. **Boundary Errors** - Green=correct, Red=wrong at edges
5. **Interior Errors** - Green=correct, Red=wrong in center

Filename format: `sample_XXXX_iou_Y.YYY_b_Z.ZZ_i_W.WW.png`
- `XXXX`: Sample index
- `Y.YYY`: Overall IoU score
- `Z.ZZ`: Boundary accuracy
- `W.WW`: Interior accuracy

## Configuration

### Adjustable Parameters

**Boundary Thickness** (default: 3 pixels)
"""python
boundary, interior = extract_boundary_pixels(mask, thickness=3)
"""

Why 3 pixels?
- Balances between capturing edge uncertainty and not being too wide
- Roughly matches the coarseness of 16×16 patches
- Can be adjusted based on your needs

**Sample Selection**
- 5 worst cases (lowest IoU)
- 5 median cases (middle IoU)
- 5 best cases (highest IoU)
- 3 from each shown in boundary analysis (9 total)

## Results Summary

### Overall Metrics
- Test Samples: 3,669
- Boundary Pixels: 13,158,884 (21.3% of all pixels)
- Interior Pixels: 48,481,778 (78.7% of all pixels)

### Accuracy by Region
- Boundary: 63.33% - Major weakness
- Interior: 91.73% - Strong performance
- Gap: 28.40% - Significant difference

### Accuracy by Size
- Small pets (<10%): 54.94% IoU - Poor (lots of boundary)
- Medium pets (10-30%): 73.39% IoU - Decent
- Large pets (>30%): 81.53% IoU - Good (more interior)

## Potential Improvements

### 1. Multi-Scale Features
Add skip connections or feature pyramid to capture fine details at boundaries

### 2. Boundary-Aware Loss
Weight boundary pixels higher in the loss function:
"""python
boundary_weight = 2.0
weights = torch.ones_like(mask)
weights[boundary] = boundary_weight
loss = weighted_cross_entropy(pred, target, weights)
"""

### 3. Post-Processing
Apply CRF or conditional random field to refine boundaries based on image edges

### 4. Smaller Patches
Use 8×8 or 4×4 patches for finer segmentation (increases computation)

### 5. Hybrid Architecture
Combine ViT encoder with CNN decoder (like Segformer) for better boundary handling

## Reproducing the Analysis

### Run Boundary Analysis
"""bash
python evaluate_enhanced.py --checkpoint checkpoints/best_model.pth --phase 2
"""

### Visualize the Process
"""bash
python visualize_boundary_extraction.py
"""

This generates:
- `outputs/boundary_extraction_process.png` - Step-by-step dilation/erosion
- `outputs/boundary_extraction_final.png` - Final boundary vs interior split

## References

- Morphological operations: [scipy.ndimage documentation](https://docs.scipy.org/doc/scipy/reference/ndimage.html)
- Binary dilation: Expands foreground regions
- Binary erosion: Shrinks foreground regions
- Boundary detection: Difference between dilated and eroded masks

