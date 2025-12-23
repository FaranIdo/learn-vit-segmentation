# Vision Transformer (ViT) for Semantic Segmentation - Tutorial

A clean, educational implementation of Vision Transformer for semantic segmentation, built from scratch with PyTorch.

## Features

- **Vision Transformer Encoder** - Patch embedding, positional encoding, multi-head self-attention
- **Progressive Decoder** - Upsampling decoder for dense pixel-level predictions
- **Training from Scratch** - Full training pipeline with data augmentation
- **Visualization Tools** - Prediction overlays and attention map visualization
- **Training Monitoring** - Time estimates and TensorBoard integration
- **Timestamped Experiments** - Never overwrite checkpoints

## Quick Start

### 1. Setup Environment

"""bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
"""

### 2. Download Dataset

Download the Oxford-IIIT Pets dataset manually (SSL certificate issues with automatic download):

"""bash
# Create data directory
mkdir -p data/oxford_pets

# Download images
curl -k -L -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
tar -xzf images.tar.gz -C data/oxford_pets/

# Download annotations
curl -k -L -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf annotations.tar.gz -C data/oxford_pets/

# Cleanup
rm images.tar.gz annotations.tar.gz
"""

### 3. Verify Setup

"""bash
python scripts/sanity_check.py
"""

This will verify:
- PyTorch installation
- GPU/MPS availability
- Config loading
- Dataset access

## Training

### Basic Training

"""bash
# Train with default settings (auto-timestamped)
python train.py
# Saves to: checkpoints/run_2024-12-23_22-45-30/
"""

### Named Experiments

"""bash
# Train with experiment name (timestamp always included)
python train.py --name baseline
# Saves to: checkpoints/baseline_2024-12-23_22-45-30/

# Train with different hyperparameters
python train.py --name high_lr
# Saves to: checkpoints/high_lr_2024-12-23_23-10-15/
"""

**Note:** Each run gets a unique timestamped directory, so **you'll never overwrite previous experiments**.

### Training Options

"""bash
# Disable TensorBoard logging
python train.py --no-tensorboard

# Named experiment without TensorBoard
python train.py --name my_experiment --no-tensorboard
"""

### TensorBoard Monitoring

In a separate terminal, launch TensorBoard:

"""bash
# View all experiments
tensorboard --logdir=checkpoints

# Open browser to http://localhost:6006
"""

TensorBoard will automatically discover all runs:
- `checkpoints/run_2024-12-23_22-45-30/tensorboard/`
- `checkpoints/baseline_2024-12-23_23-10-15/tensorboard/`
- `checkpoints/high_lr_2024-12-23_23-15-30/tensorboard/`

### What You'll See During Training

"""
Epoch 5/50
──────────────────────────────────────────────────────────
Training: 100%|████████| 200/200 [01:45<00:00, 1.90it/s, loss=0.3245, miou=0.7234]
Validation: 100%|████████| 50/50 [00:49<00:00, 1.02it/s, loss=0.2891, miou=0.7456]

Epoch 5 Summary:
  Learning Rate: 1.00e-04 (0.000100)
  Train Loss: 0.3245 | Val Loss: 0.2891
  Train Acc:  0.8923 | Val Acc:  0.9012
  Train mIoU: 0.7234 | Val mIoU: 0.7456
  Train IoU [bg, fg]: [0.8234, 0.6234]
  Val IoU   [bg, fg]: [0.8456, 0.6456]

Timing:
  Epoch time: 2m 34s (Train: 1m 45s | Val: 49s)
  Avg epoch time: 2m 36s
  Estimated remaining: 1h 57m (45 epochs × 2m 36s)
  Estimated completion: 2:05 PM
  Total elapsed: 12m 50s
✓ Saved best model (mIoU: 0.7456)
"""

## Configuration

Edit `config.py` to change hyperparameters:

"""python
# Model architecture
IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
DROPOUT = 0.1

# Training
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 50

# Dataset
NUM_CLASSES = 2  # Oxford Pets: background + pet
"""

## Visualization

### Visualize Data Augmentations

"""bash
python scripts/visualize_augmentations.py
# Outputs to: outputs/augmentations/
"""

Shows original image and augmented versions with labeled techniques.

### Visualize Predictions

"""bash
# Visualize predictions from untrained model
python scripts/visualize_predictions.py
# Outputs to: outputs/predictions/

# Visualize predictions from trained model
python scripts/visualize_predictions.py --checkpoint checkpoints/baseline_2024-12-23_22-45-30/best_model.pth
"""

Shows:
- Original image
- Ground truth mask
- Predicted mask
- Prediction overlay

### Visualize Attention Maps

"""bash
# Visualize attention from untrained model
python scripts/visualize_attention.py
# Outputs to: outputs/attention/

# Visualize attention from trained model (specific layer)
python scripts/visualize_attention.py --checkpoint checkpoints/baseline_2024-12-23_22-45-30/best_model.pth --layer 3
"""

Shows:
- Attention maps at different layers
- Mean attention across heads
- Attention overlay on original image

## Key Concepts Explained

### Patch Embedding
Converts 224×224 image into 14×14 grid of 16×16 patches, each embedded to 384 dimensions.

### Positional Encoding
Learnable positional embeddings added to patch tokens to preserve spatial information.

### Transformer Encoder
Stack of 6 transformer blocks with multi-head self-attention and feed-forward networks.

### Progressive Decoder
Upsamples encoder tokens (14×14) back to image resolution (224×224) with 2× upsampling stages.

### Loss Function
Combined Dice Loss (for overlap) + Cross-Entropy Loss (for pixel classification).

### Metrics
- **Pixel Accuracy:** Percentage of correctly classified pixels
- **IoU (Intersection over Union):** Overlap between prediction and ground truth
- **mIoU (mean IoU):** Average IoU across all classes (primary metric)

### Data Augmentation
Joint transforms for image and mask: horizontal flip, rotation, random crop, color jitter.

## Documentation

Comprehensive documentation available in `docs/`:

- **Architecture:** Positional encoding, activation functions, normalization, decoder design
- **Training:** Losses, metrics, augmentation, learning rate schedules
- **Visualization:** Predictions, attention maps, ImageNet normalization
- **Monitoring:** Time estimates, TensorBoard, learning curve interpretation

## Performance

### Expected Results (Oxford Pets)

After 50 epochs of training from scratch:
- **Val mIoU:** ~0.75-0.80 (good)
- **Val Pixel Accuracy:** ~0.90+
- **Background IoU:** ~0.85+
- **Foreground IoU:** ~0.65-0.75

### Training Time

On Apple M-series (MPS):
- **Per epoch:** ~2-3 minutes (3700+ train images, 16 batch size)
- **50 epochs:** ~2-2.5 hours
- **Early stopping:** Often finishes in 20-30 epochs (~1 hour)

## Troubleshooting

### MPS Not Available
If training falls back to CPU:
"""bash
# Check MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
"""

### Out of Memory
Reduce batch size in `config.py`:
"""python
BATCH_SIZE = 8  # or 4
"""

### Training Too Slow
- Reduce `IMAGE_SIZE` to 128 or 160
- Reduce `EMBED_DIM` to 256
- Reduce `NUM_LAYERS` to 4

### TensorBoard Not Showing Data
"""bash
# Ensure training is running with TensorBoard enabled (default)
python train.py

# Launch TensorBoard pointing to checkpoints directory
tensorboard --logdir=checkpoints

# Wait a few seconds and refresh browser
"""

## Next Steps

### Continue Learning
1. Review documentation in `docs/` for deep dives
2. Experiment with different hyperparameters
3. Compare learning curves in TensorBoard
4. Visualize attention maps from trained model

### Future Improvements
- Pretrained model fine-tuning (SegFormer)
- RoPE (Rotary Position Embeddings)
- Swin Transformer architecture
- Multi-scale training
- Production metrics (torchmetrics)

## References

- **ViT Paper:** "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- **Segmentation:** "Segmenter: Transformer for Semantic Segmentation" (Strudel et al., 2021)
- **Dataset:** Oxford-IIIT Pets (Parkhi et al., 2012)

## License

Educational project for learning Vision Transformers.

