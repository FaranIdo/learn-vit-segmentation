# ImageNet Normalization

## What is ImageNet Normalization?

ImageNet normalization is a standard preprocessing step in computer vision that standardizes image pixel values using statistics computed from the ImageNet dataset.

**The values:**
"""python
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB channels
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB channels
"""

These numbers represent:
- **Mean**: Average pixel values across 1.2 million ImageNet training images
- **Std**: Standard deviation of pixel values across ImageNet

---

## Why Normalize Images?

### 1. Training Stability
Neural networks train better when inputs are centered around 0 with similar scales across features.

**Without normalization:**
- Pixel values: [0, 255]
- Large, inconsistent gradients
- Slower convergence

**With normalization:**
- Values centered around 0
- Unit variance across channels
- Faster, more stable training

### 2. Transfer Learning Compatibility
Pretrained models (ResNet, ViT, SegFormer) were trained with ImageNet normalization. To use their weights, we must apply the same preprocessing.

### 3. Industry Standard
Using ImageNet statistics makes our results comparable with other research and production systems.

---

## The Normalization Process

### Step 1: Load Image (0-255 range)
"""python
raw_image = Image.open("cat.jpg")  # Pixel values: 0-255
"""

### Step 2: Convert to Tensor (0-1 range)
"""python
tensor = torchvision.transforms.functional.to_tensor(raw_image)
# Divides by 255: [0, 255] → [0, 1]
"""

### Step 3: Normalize (standardize)
"""python
normalized = torchvision.transforms.functional.normalize(
    tensor,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
# Formula: (x - mean) / std
"""

**Example (Red channel):**
"""
Original pixel: 128
After to_tensor: 128 / 255 = 0.502
After normalize: (0.502 - 0.485) / 0.229 = 0.074
"""

---

## Denormalization for Visualization

To display images, we must reverse the normalization.

**Formula:**
"""
normalized = (original - mean) / std
original = (normalized * std) + mean
"""

**Implementation:**
"""python
def denormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.numpy().transpose(1, 2, 0)  # [C, H, W] → [H, W, C]
    img = std * img + mean  # Reverse normalization
    img = np.clip(img, 0, 1)  # Ensure valid range
    
    return img
"""

---

## Why ImageNet Stats for Oxford Pets?

**Good question!** We train on Oxford Pets, not ImageNet. Why use ImageNet statistics?

### Reasons:

1. **Standard Convention**
   - Everyone in computer vision uses ImageNet normalization
   - Makes results comparable across papers and projects

2. **Transfer Learning**
   - If we fine-tune pretrained models later, they expect ImageNet normalization
   - Enables using powerful pretrained weights (ViT, SegFormer, etc.)

3. **Good Enough**
   - ImageNet covers diverse natural images (animals, objects, scenes)
   - Statistics are similar enough to Oxford Pets

### Alternative: Dataset-Specific Stats

We could compute Oxford Pets statistics:

"""python
# Compute mean/std from Oxford Pets
total_mean = [0, 0, 0]
total_std = [0, 0, 0]

for image in oxford_pets_dataset:
    total_mean += image.mean(dim=[1, 2])
    total_std += image.std(dim=[1, 2])

oxford_mean = total_mean / len(dataset)
oxford_std = total_std / len(dataset)
"""

**But:**
- Results would be very similar to ImageNet
- Breaks compatibility with pretrained models
- Not standard practice

---

## Where We Use These Values

### 1. Dataset Preprocessing
In `data/oxford_pets.py`, we normalize images during loading:

"""python
tensor = TF.to_tensor(pil_image)
normalized = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
"""

### 2. Visualization
In `visualization/predictions.py`, we denormalize before displaying:

"""python
img = tensor.numpy().transpose(1, 2, 0)
img = std * img + mean  # Undo normalization
img = np.clip(img, 0, 1)
"""

---

## Summary

- **ImageNet normalization** is a standard preprocessing step
- **Mean/Std values** come from ImageNet dataset statistics
- **Normalization** improves training stability and enables transfer learning
- **Denormalization** is required to display images correctly
- **Use ImageNet stats** even for other datasets (standard practice)

**Key formula:**
"""
Forward:  normalized = (x - mean) / std
Backward: x = (normalized * std) + mean
"""

