# Data Augmentation for Semantic Segmentation

## The Key Challenge

Unlike image classification, semantic segmentation requires maintaining the **spatial relationship** between the image and its corresponding segmentation mask. If we apply a geometric transformation to the image (like flipping or rotating), we **must apply the exact same transformation** to the mask, otherwise they become misaligned.

### Example of Misalignment

```
Original:
Image: 🐕→        Mask: ■■□□
       (dog facing right)  (mask aligned)

Wrong (only image flipped):
Image: ←🐕        Mask: ■■□□
       (dog facing left)   (mask still on left - MISALIGNED!)

Correct (both flipped):
Image: ←🐕        Mask: □□■■
       (dog facing left)   (mask on right - ALIGNED!)
```

This is why we need **joint transforms** that operate on both the image and mask simultaneously.

---

## Types of Augmentations

### 1. Spatial Transforms (Apply to BOTH Image and Mask)

These transforms change the geometric properties of the data:
- **Horizontal Flip**: Mirrors the image left-to-right
- **Rotation**: Rotates by a small angle
- **Random Resized Crop**: Crops a random portion and resizes back
- **Scaling/Resizing**: Changes the size of the image

**Critical**: Both image and mask must undergo identical spatial transformations to maintain alignment.

### 2. Photometric Transforms (Apply ONLY to Image)

These transforms change the appearance but not the geometry:
- **Brightness**: Adjusts image brightness
- **Contrast**: Adjusts the difference between light and dark
- **Saturation**: Adjusts color intensity
- **Hue**: Shifts colors

**Why not apply to mask?** Segmentation masks contain discrete class labels (0, 1, 2, ...), not pixel intensities. Applying color transforms would corrupt these labels.

---

## Interpolation: Bilinear vs. Nearest

When resizing or rotating images, we need to interpolate new pixel values:

### Bilinear Interpolation (for Images)
- Averages neighboring pixels to create smooth transitions
- Good for natural images (creates realistic intermediate values)
- Example: Resizing a photo from 300×300 to 224×224

```
Original pixels:  [100, 200]
Interpolated:     [100, 120, 140, 160, 180, 200]
                       ↑ smooth gradients
```

### Nearest Neighbor Interpolation (for Masks)
- Picks the closest pixel value without averaging
- **Required for segmentation masks** to preserve class labels
- Prevents invalid intermediate values

```
Original mask:    [0, 1]
Bilinear (BAD):   [0, 0.25, 0.5, 0.75, 1]  ← Invalid class "0.5"!
Nearest (GOOD):   [0, 0, 0, 1, 1]           ← Only valid classes
```

---

## Our Implementation

We implemented a **joint transform pipeline** that ensures image-mask alignment:

### 1. JointResize
Resizes both image and mask to a fixed size (224×224).

```python
image = TF.resize(image, [224, 224])                              # Bilinear
mask = TF.resize(mask, [224, 224], interpolation=NEAREST)         # Nearest
```

**Use case**: Standardizing input size for the model.

---

### 2. JointRandomHorizontalFlip
Randomly mirrors both image and mask horizontally with probability p=0.5.

```
Before:          After (50% chance):
Image: 🐕□□□     Image: □□□🐕
Mask:  ■■□□      Mask:  □□■■
```

**Benefits**:
- Doubles the effective dataset size
- Teaches model that orientation doesn't matter (a left-facing dog is still a dog)
- No information loss (perfect mirror symmetry)

---

### 3. JointRandomRotation
Rotates both image and mask by a random angle in the range ±10°.

```
Before:          After (e.g., +7°):
■■□□             ╱■■□╲
■■□□      →     ╱■■□□╲
□□□□             ╲□□□╱
```

**Benefits**:
- Handles real-world photos that aren't perfectly aligned
- Small angles (±10°) avoid too much distortion
- Mask uses NEAREST interpolation to preserve class labels

**Trade-off**: Rotation can introduce black borders or crop the image slightly.

---

### 4. JointRandomResizedCrop
1. Randomly crops a portion of the image (80-100% of original area)
2. Resizes the crop back to 224×224

```
Step 1: Random crop (90% area)   Step 2: Resize to 224×224
┌───────────────┐                ┌─────────┐
│ 🐕            │                │ 🐕      │
│               │    →           │         │
│  [crop area]  │                │         │
└───────────────┘                └─────────┘
```

**Benefits**:
- Simulates different camera distances (zoom in/out)
- Forces model to recognize objects at various scales
- Adds variability without needing more images

**Implementation detail**: We ensure the same crop coordinates are used for both image and mask.

---

### 5. ColorJitter (Image Only)
Randomly adjusts brightness, contrast, and saturation of the **image only**.

```python
# Brightness: ±20%
image = TF.adjust_brightness(image, 1 + random.uniform(-0.2, 0.2))

# Contrast: ±20%
image = TF.adjust_contrast(image, 1 + random.uniform(-0.2, 0.2))

# Saturation: ±20%
image = TF.adjust_saturation(image, 1 + random.uniform(-0.2, 0.2))
```

**Benefits**:
- Handles different lighting conditions (sunny, cloudy, indoor, outdoor)
- Makes model robust to camera settings
- Prevents overfitting to specific color distributions

**Why not apply to mask?** The mask contains discrete class indices (0 = background, 1 = pet), not color values. Adjusting "brightness" of class labels would corrupt the data.

---

## Training vs. Validation

### Training Transforms (Aggressive)
We apply all augmentations during training to maximize data variety:

```python
train_transforms = JointCompose([
    JointRandomResizedCrop(224, scale=(0.8, 1.0)),  # Scale variation
    JointRandomHorizontalFlip(p=0.5),                # 50% flip
    JointRandomRotation(degrees=10),                 # ±10° rotation
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])
```

**Goal**: Expose the model to as many variations as possible to improve generalization.

### Validation Transforms (Conservative)
We only resize during validation:

```python
val_transforms = JointCompose([
    JointResize(224),  # Just resize, no augmentation
])
```

**Why no augmentation?**
1. **Reproducibility**: We want consistent metrics across runs
2. **Fair comparison**: All models should be evaluated on identical data
3. **Standard practice**: Benchmarks (Pascal VOC, Cityscapes) don't augment test sets

---

## Augmentation Pipeline Flow

```
Raw Data (variable size)
    ↓
JointRandomResizedCrop (80-100% area → 224×224)
    ↓
JointRandomHorizontalFlip (50% chance)
    ↓
JointRandomRotation (±10°)
    ↓
ColorJitter (image only: brightness, contrast, saturation)
    ↓
_finalize_image: to_tensor + normalize (ImageNet stats)
_finalize_mask: to_tensor + convert to binary (0 or 1)
    ↓
Final Tensors: image [3, 224, 224], mask [224, 224]
```

---

## Best Practices

### DO:
✅ Apply spatial transforms to both image and mask  
✅ Use NEAREST interpolation for masks  
✅ Use BILINEAR interpolation for images  
✅ Apply photometric transforms only to images  
✅ Use aggressive augmentation for training  
✅ Use minimal augmentation for validation  

### DON'T:
❌ Apply color transforms to masks  
❌ Use bilinear interpolation on masks  
❌ Use different random parameters for image and mask  
❌ Augment validation/test data  
❌ Use extreme rotation angles (causes too much distortion)  

---

## Impact on Training

**Without augmentation:**
- Model sees only 3,680 training images
- Overfits to specific poses, lighting, backgrounds
- Poor generalization to new data

**With augmentation:**
- Effectively infinite training data (random combinations)
- Learns invariance to pose, lighting, scale
- Better generalization and higher mIoU

**Typical improvement**: +5-10% mIoU on validation set.

---

## Future Enhancements

Potential augmentations not yet implemented:
- **Elastic deformations**: Simulate natural object deformations
- **Cutout/Random erasing**: Occlude random patches (forces model to use context)
- **MixUp**: Blend pairs of images and masks
- **AutoAugment**: Learn optimal augmentation policies from data
- **Vertical flip**: Currently only horizontal (vertical flip is less common for pets)

These are more advanced and can be added if needed for specific datasets or performance improvements.

