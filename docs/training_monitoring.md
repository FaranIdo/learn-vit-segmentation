# Training Monitoring

This guide explains how to monitor training progress using time estimates and TensorBoard learning curves.

## Overview

We have two monitoring systems:

1. **Time Estimates** - Console-based progress tracking with ETA
2. **TensorBoard** - Visual learning curves and metrics tracking

---

## Time Estimates

### What You Get

During training, you'll see detailed timing information after each epoch:

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

### Understanding Time Estimates

**Epoch time:** Time for current epoch (train + validation)
- Helps you know how long each epoch takes
- Useful for comparing training speed across experiments

**Avg epoch time:** Running average of all epochs so far
- More accurate than single epoch time
- Stabilizes after 2-3 epochs

**Estimated remaining:** Time until training completes
- Formula: `avg_epoch_time × remaining_epochs`
- Updates each epoch for better accuracy
- Accounts for variations in epoch time

**Estimated completion:** Wall clock time when training will finish
- Helps you plan when to check results
- Based on current system time + estimated remaining

**Total elapsed:** Time since training started
- Useful for tracking total training cost
- Includes all epochs, validation, and checkpointing

### Why Time Estimates Matter

1. **Planning:** Know when to check back on training
2. **Debugging:** Unusually slow epochs indicate issues
3. **Optimization:** Compare different batch sizes or model configs
4. **Budgeting:** Estimate compute costs for longer runs

---

## TensorBoard

### What is TensorBoard?

TensorBoard is an industry-standard tool for visualizing machine learning experiments. It provides:
- Real-time metric plots
- Interactive charts (zoom, pan, smooth)
- Multiple run comparisons
- Histogram visualizations
- Learning rate schedules

### Starting TensorBoard

**1. Start Training (TensorBoard enabled by default):**
"""bash
python train.py
"""

**2. In another terminal, launch TensorBoard:**
"""bash
tensorboard --logdir=checkpoints/tensorboard
"""

**3. Open browser:**
"""
http://localhost:6006
"""

**4. Disable TensorBoard if needed:**
"""bash
python train.py --no-tensorboard
"""

### Metrics Tracked

TensorBoard automatically logs:

**Loss (Train & Val)**
- Combined Dice + Cross-Entropy loss
- Should decrease over time
- Val loss plateau = potential overfitting

**mIoU (Train & Val)**
- Mean Intersection over Union
- Primary metric for segmentation
- Should increase over time
- Val mIoU is what matters most

**Pixel Accuracy (Train & Val)**
- Percentage of correctly classified pixels
- Can be misleading with class imbalance
- Use mIoU as primary metric

**Per-Class IoU (Train & Val)**
- IoU for each class separately
- For Oxford Pets:
  - Class 0: Background IoU
  - Class 1: Foreground (pet) IoU
- Helps identify which class performs poorly

**Learning Rate**
- Shows LR schedule over time
- For CosineAnnealingLR: smooth decay from initial to 0
- Helps verify scheduler is working correctly

### TensorBoard Interface

**SCALARS Tab:**
- Main tab for viewing metrics
- Select metrics from left sidebar
- Toggle between runs for comparison
- Adjust smoothing slider (0.0 = raw, 0.9 = very smooth)

**Time Series Settings:**
- X-axis: Step (epoch number)
- Y-axis: Metric value
- Hover over points to see exact values
- Click and drag to zoom

**Smoothing:**
- Smoothing = 0.0: Raw values (noisy but accurate)
- Smoothing = 0.6: Recommended (balances noise and trend)
- Smoothing = 0.9: Very smooth (may hide important details)

---

## Interpreting Learning Curves

### Healthy Training

**Loss:**
- Train loss: Steadily decreasing
- Val loss: Decreasing, follows train loss closely
- Gap between train/val: Small and stable

**mIoU:**
- Train mIoU: Steadily increasing
- Val mIoU: Increasing, follows train mIoU
- Val mIoU > 0.7 is good for Oxford Pets

**Learning Rate:**
- Smooth decay (for CosineAnnealingLR)
- No sudden jumps or plateaus

### Warning Signs

**Overfitting:**
- Train loss keeps decreasing
- Val loss starts increasing or plateaus
- Large gap between train and val metrics
- **Fix:** Add more dropout, reduce model size, add data augmentation

**Underfitting:**
- Both train and val loss remain high
- Both train and val mIoU remain low
- Metrics improve very slowly
- **Fix:** Increase model capacity, train longer, reduce regularization

**Training Instability:**
- Loss spikes up suddenly
- Metrics oscillate wildly
- **Fix:** Reduce learning rate, use gradient clipping, check data loading

**Class Imbalance Issues:**
- High pixel accuracy but low mIoU
- One class has much higher IoU than the other
- **Fix:** Adjust loss weights, use focal loss, balance augmentations

### Example Interpretations

**Scenario 1: Good Training**
"""
Train Loss: 0.45 → 0.25 → 0.15 (decreasing)
Val Loss:   0.50 → 0.30 → 0.20 (decreasing, small gap)
Val mIoU:   0.60 → 0.72 → 0.78 (increasing)
✓ Model is learning well, continue training
"""

**Scenario 2: Overfitting**
"""
Train Loss: 0.45 → 0.15 → 0.05 (decreasing)
Val Loss:   0.50 → 0.35 → 0.40 (increasing after epoch 2)
Val mIoU:   0.60 → 0.70 → 0.68 (peaking then declining)
⚠ Overfitting! Use early stopping or regularization
"""

**Scenario 3: Underfitting**
"""
Train Loss: 0.65 → 0.62 → 0.60 (barely improving)
Val Loss:   0.67 → 0.64 → 0.62 (barely improving)
Val mIoU:   0.45 → 0.48 → 0.50 (very slow progress)
⚠ Model too simple! Increase capacity or train longer
"""

---

## Comparing Multiple Runs

### Running Experiments

Each time you train, TensorBoard creates a new run folder with a timestamp.

**Run 1: Baseline**
"""bash
python train.py
"""

**Run 2: Higher learning rate**
"""bash
# Edit config.py: LEARNING_RATE = 2e-4
python train.py
"""

**Run 3: More dropout**
"""bash
# Edit config.py: DROPOUT = 0.2
python train.py
"""

### Viewing Multiple Runs

TensorBoard automatically shows all runs in the same plot:
"""bash
tensorboard --logdir=checkpoints/tensorboard
"""

**Tips:**
- Different colors for each run
- Click run names in legend to toggle on/off
- Use regex to filter runs
- Compare convergence speed and final performance

---

## Advanced Usage

### Custom Metrics

To log additional metrics, modify `training/trainer.py`:

"""python
if self.tensorboard:
    # Your custom metric
    custom_metric = compute_my_metric(predictions, targets)
    self.tensorboard.log_scalar("Custom_Metric", custom_metric, epoch + 1)
"""

### Image Logging

To visualize predictions during training:

"""python
if self.tensorboard and epoch % 5 == 0:  # Every 5 epochs
    # Log sample predictions
    with torch.no_grad():
        sample_images = images[:4]  # First 4 images
        sample_preds = model(sample_images)
        
    # Convert predictions to RGB for visualization
    # (Implementation in visualization/predictions.py)
    self.tensorboard.log_images("Predictions", pred_rgb, epoch + 1)
"""

### Hyperparameter Tracking

TensorBoard supports hyperparameter comparison:

"""python
from torch.utils.tensorboard import SummaryWriter

hparams = {
    "lr": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "dropout": DROPOUT
}
metrics = {"val_miou": best_miou}

self.tensorboard.writer.add_hparams(hparams, metrics)
"""

---

## Best Practices

### During Training

1. **Monitor regularly:** Check TensorBoard every few epochs
2. **Watch val metrics:** Val mIoU is what matters, not train mIoU
3. **Use early stopping:** Patience=10 prevents wasting compute
4. **Save time estimates:** Note estimated completion time

### After Training

1. **Review learning curves:** Identify overfitting or other issues
2. **Compare runs:** Find best hyperparameters
3. **Check final metrics:** Val mIoU > 0.75 is good for this task
4. **Analyze per-class IoU:** Ensure both classes perform well

### Troubleshooting

**TensorBoard not showing data:**
- Check `checkpoints/tensorboard/` folder exists
- Wait a few seconds for TensorBoard to refresh
- Refresh browser page
- Verify training is logging (no `--no-tensorboard` flag)

**Metrics look wrong:**
- Check data loading is correct
- Verify loss function is appropriate
- Check if metrics are computed correctly
- Compare with validation prints in console

**Training too slow:**
- Reduce batch size (faster per epoch, more epochs needed)
- Use smaller image size (IMAGE_SIZE in config.py)
- Reduce model size (EMBED_DIM, NUM_LAYERS in config.py)
- Check MPS/CUDA is being used (prints "Device: mps" or "cuda")

---

## Summary

**Time Estimates:**
- Helps you plan when to check training
- Shows per-epoch and total time
- Estimates completion time

**TensorBoard:**
- Visualize learning curves in real-time
- Compare multiple training runs
- Track comprehensive metrics

**Key Metrics:**
- **Val mIoU:** Primary metric (higher is better)
- **Val Loss:** Should decrease over time
- **Train/Val Gap:** Small gap = healthy training

**Commands:**
"""bash
# Start training
python train.py

# Launch TensorBoard (in another terminal)
tensorboard --logdir=checkpoints/tensorboard

# Open browser
http://localhost:6006
"""

**Pro Tips:**
- Always use TensorBoard for serious experiments
- Save screenshots of good/bad runs for reference
- Compare at least 3 different hyperparameters
- Trust val mIoU over val loss for segmentation

