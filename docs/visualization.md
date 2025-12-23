# Visualization

This document explains the visualization tools built for understanding and debugging the ViT segmentation model.

## Overview

We have two main visualization modules:

1. **Prediction Visualization** - Compare model predictions vs ground truth
2. **Attention Visualization** - See what the model focuses on

---

## Prediction Visualization

### Purpose
Quickly assess model performance by visually comparing predictions to ground truth masks.

### Key Functions

#### `plot_prediction_comparison()`
Creates a 4-panel comparison for a single image:
- Original image
- Ground truth mask
- Predicted mask
- Overlay (prediction on image)

"""python
from visualization import plot_prediction_comparison

fig = plot_prediction_comparison(
    image,          # [C, H, W] normalized tensor
    ground_truth,   # [H, W] class indices
    prediction,     # [H, W] class indices
    num_classes=2,
    title="Model Prediction"
)
plt.savefig("prediction.png")
"""

#### `visualize_batch_predictions()`
Creates a grid showing multiple samples:

"""python
from visualization import visualize_batch_predictions

fig = visualize_batch_predictions(
    images,         # [B, C, H, W]
    ground_truths,  # [B, H, W]
    predictions,    # [B, H, W]
    num_classes=2,
    num_samples=4
)
"""

### Usage Example

"""python
# Generate predictions
model.eval()
with torch.no_grad():
    logits = model(images)
    predictions = torch.argmax(logits, dim=1)

# Visualize
from visualization import plot_prediction_comparison
fig = plot_prediction_comparison(images[0], masks[0], predictions[0])
plt.show()
"""

### Running the Script

"""bash
# Untrained model (random predictions)
python tools/visualize_predictions.py

# Trained model
python tools/visualize_predictions.py --checkpoint checkpoints/best_model.pth
"""

---

## Attention Visualization

### Purpose
Understand what patches the model pays attention to when making predictions.

### What is Attention?

Attention weights show how much each patch "looks at" other patches during processing.

**Shape:** `[batch, num_heads, num_patches, num_patches]`
- For our model: `[B, 6, 196, 196]`
- `weights[b, h, i, j]` = "In batch b, head h, patch i attends to patch j"

**Interpretation:**
- **High attention** (red/yellow): Model focuses here
- **Low attention** (blue): Model ignores this
- **Good model**: Attention on relevant regions (pet, not background)

### Key Functions

#### `extract_attention_maps()`
Extracts attention weights from a specific layer:

"""python
from visualization import extract_attention_maps

attention_weights = extract_attention_maps(
    model,
    image,
    layer_idx=-1  # -1 = last layer, 0 = first layer
)
# Returns: [batch, num_heads, num_patches, num_patches]
"""

#### `plot_attention_visualization()`
Creates comprehensive single-image attention visualization:
- Original image
- Attention map at patch resolution (14×14)
- Upsampled attention (224×224)
- Attention overlay on image
- Center patch attention (what center attends to)
- Mean attention per head (bar chart)

"""python
from visualization import plot_attention_visualization

fig = plot_attention_visualization(
    image,              # [C, H, W]
    attention_weights,  # [num_heads, num_patches, num_patches]
    num_patches_per_side=14,
    image_size=224,
    title="Attention Analysis"
)
"""

#### `plot_multi_layer_attention()`
Compares attention across multiple layers:

"""python
from visualization import plot_multi_layer_attention

fig = plot_multi_layer_attention(
    model,
    image,
    layer_indices=[0, 3, 5],  # Early, middle, late layers
    num_patches_per_side=14,
    image_size=224
)
"""

### Usage Example

"""python
from visualization import extract_attention_maps, plot_attention_visualization

model.eval()

# Extract attention from last layer
attention_weights = extract_attention_maps(model, image, layer_idx=-1)

# Visualize
num_patches_per_side = IMAGE_SIZE // PATCH_SIZE  # 224 // 16 = 14
fig = plot_attention_visualization(
    image,
    attention_weights.squeeze(0),  # Remove batch dimension
    num_patches_per_side
)
plt.show()
"""

### Running the Script

"""bash
# Untrained model (random attention)
python tools/visualize_attention.py

# Trained model
python tools/visualize_attention.py --checkpoint checkpoints/best_model.pth
"""

---

## Understanding Attention Patterns

### Untrained Model
- **Random/uniform attention**: All patches attend equally to all other patches
- **No structure**: Model hasn't learned meaningful patterns yet
- **Noisy heatmaps**: Attention is scattered randomly

### Trained Model
- **Focused attention**: High attention on relevant patches (pet)
- **Low attention on background**: Model ignores irrelevant regions
- **Structured patterns**: Attention follows object boundaries

### Layer Progression
Different layers learn different patterns:

**Early layers (0-1):**
- Local attention (neighbors)
- Low-level features (edges, textures)

**Middle layers (2-4):**
- Regional attention
- Object parts (ears, eyes, body)

**Late layers (5):**
- Global attention
- Entire object and context

---

## Utility Functions

### `denormalize_image()`
Converts normalized tensor back to displayable image:

"""python
from visualization import denormalize_image

img = denormalize_image(tensor)  # [H, W, C] in [0, 1]
plt.imshow(img)
"""

### `mask_to_rgb()`
Converts class indices to colored RGB mask:

"""python
from visualization import mask_to_rgb

mask_rgb = mask_to_rgb(mask, num_classes=2)
# Class 0: Black (background)
# Class 1: Green (foreground/pet)
"""

### `overlay_mask_on_image()`
Blends mask onto image with transparency:

"""python
from visualization import overlay_mask_on_image

overlay = overlay_mask_on_image(
    image,
    mask,
    num_classes=2,
    alpha=0.5  # 0=transparent, 1=opaque
)
"""

### `overlay_attention_on_image()`
Blends attention heatmap onto image:

"""python
from visualization import overlay_attention_on_image

overlay = overlay_attention_on_image(
    image,
    attention_map,
    alpha=0.5,
    cmap='jet'  # Color map (jet, viridis, hot, etc.)
)
"""

---

## Integration with Training

### Option 1: Visualize During Training

Add to `Trainer` class:

"""python
def visualize_epoch(self, epoch: int):
    if epoch % 5 == 0:  # Every 5 epochs
        images, masks = next(iter(self.val_loader))
        images = images.to(self.device)
        
        with torch.no_grad():
            logits = self.model(images)
            predictions = torch.argmax(logits, dim=1)
        
        fig = visualize_batch_predictions(images, masks, predictions)
        plt.savefig(f"outputs/epoch_{epoch}_predictions.png")
        plt.close(fig)
"""

### Option 2: Post-Training Analysis

After training completes, analyze best/worst predictions:

"""python
# Find samples with lowest IoU
worst_samples = []
for images, masks in val_loader:
    predictions = model(images).argmax(dim=1)
    ious = compute_iou(predictions, masks)
    worst_samples.append((images, masks, predictions, ious))

# Sort by IoU and visualize worst cases
worst_samples.sort(key=lambda x: x[3].mean())
for img, mask, pred, iou in worst_samples[:5]:
    plot_prediction_comparison(img, mask, pred)
    plot_attention_visualization(...)  # See why it failed
"""

---

## Best Practices

### 1. Always Visualize Before Training
Check that data loading and preprocessing work correctly:

"""bash
python tools/visualize_predictions.py
"""

### 2. Compare Trained vs Untrained
See how much the model has learned:

"""bash
# Before training
python tools/visualize_attention.py

# After training
python tools/visualize_attention.py --checkpoint checkpoints/best_model.pth
"""

### 3. Debug with Attention
When predictions fail, check attention:
- **Attention on wrong object**: Model is confused
- **Uniform attention**: Model hasn't learned features
- **Attention on edges**: Model learned boundaries but not semantic meaning

### 4. Layer Analysis
Compare early vs late layers to understand hierarchical feature learning:

"""python
plot_multi_layer_attention(model, image, layer_indices=[0, 2, 4, 5])
"""

---

## Summary

**Prediction Visualization:**
- Compare predictions vs ground truth
- Identify where model fails
- Quick visual assessment of performance

**Attention Visualization:**
- See what model focuses on
- Understand decision-making process
- Debug failures and understand learned features

**Scripts:**
- `tools/visualize_predictions.py` - Prediction comparison
- `tools/visualize_attention.py` - Attention heatmaps

**Key insight:** Visualization is essential for understanding and debugging deep learning models. Always visualize before, during, and after training.

