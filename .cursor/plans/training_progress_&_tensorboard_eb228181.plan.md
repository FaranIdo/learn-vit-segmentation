---
name: Training Progress & TensorBoard
overview: Add time estimates (ETA) for epochs/validation/total training and integrate TensorBoard for real-time learning curves visualization.
todos:
  - id: add-tensorboard-dep
    content: Add tensorboard to requirements.txt
    status: completed
  - id: create-tb-logger
    content: Create training/tensorboard_logger.py wrapper class
    status: completed
  - id: add-time-tracking
    content: Add time estimates to Trainer (epoch, validation, total)
    status: completed
  - id: integrate-tensorboard
    content: Integrate TensorBoard logging into Trainer
    status: completed
  - id: update-train-script
    content: Update train.py with --no-tensorboard flag
    status: completed
  - id: update-gitignore
    content: Add tensorboard/ to .gitignore
    status: completed
  - id: create-monitoring-docs
    content: Create docs/training_monitoring.md
    status: completed
  - id: test-training
    content: Test with scripts/test_training.py
    status: completed
---

# Training Progress & Learning Curves

## Goals

1. Add time estimates (ETA) for epoch, validation, and total training duration
2. Integrate TensorBoard for real-time learning curve visualization
3. Track comprehensive metrics: loss, mIoU, pixel accuracy, per-class IoU, learning rate

## Implementation Plan

### 1. Add TensorBoard Support

**Update [`requirements.txt`](requirements.txt)**

- Add `tensorboard` package

**Create [`training/tensorboard_logger.py`](training/tensorboard_logger.py)**

- Wrapper class `TensorBoardLogger` for easy metric logging
- Methods:
- `log_scalar(tag, value, step)` - Log single metric
- `log_scalars(tag, values_dict, step)` - Log multiple metrics in one plot
- `log_learning_rate(lr, step)` - Log LR
- `log_images(tag, images, step)` - Log images (optional for future)
- `close()` - Close writer

### 2. Add Time Tracking to Trainer

**Modify [`training/trainer.py`](training/trainer.py)**Add time tracking attributes to `__init__`:

```python
import time
self.start_time = None
self.epoch_times = []  # Track time per epoch for better estimates
```

Update `train_epoch()`:

- Add `start_time = time.time()` at the beginning
- Calculate elapsed and ETA in tqdm postfix
- Return elapsed time
- Enhanced tqdm format: `{epoch_time}/{estimated_epoch_time} remaining`

Update `validate_epoch()`:

- Same time tracking as train_epoch
- Return elapsed time

Update `train()` method:

- Track `self.start_time = time.time()` at beginning
- After each epoch, calculate:
- **Epoch time** (train + validation)
- **Average epoch time** (from `self.epoch_times`)
- **Estimated remaining time** = `avg_epoch_time * (num_epochs - current_epoch)`
- Print detailed time estimates:
  ```javascript
      Epoch 5/50
      Epoch time: 2m 34s (Train: 1m 45s, Val: 49s)
      Estimated remaining: 1h 55m (based on avg 2m 36s/epoch)
      Total elapsed: 12m 50s
  ```




### 3. Integrate TensorBoard into Trainer

**Modify [`training/trainer.py`](training/trainer.py)**Add to `__init__`:

```python
from training.tensorboard_logger import TensorBoardLogger
self.tensorboard = TensorBoardLogger(
    log_dir=os.path.join(checkpoint_dir, "tensorboard")
) if use_tensorboard else None
```

Add to `train()` method after each epoch:

```python
if self.tensorboard:
    # Log losses
    self.tensorboard.log_scalars("Loss", {
        "train": train_loss,
        "val": val_loss
    }, epoch)
    
    # Log mIoU
    self.tensorboard.log_scalars("mIoU", {
        "train": train_metrics['miou'],
        "val": val_metrics['miou']
    }, epoch)
    
    # Log pixel accuracy
    self.tensorboard.log_scalars("Pixel_Accuracy", {
        "train": train_metrics['pixel_accuracy'],
        "val": val_metrics['pixel_accuracy']
    }, epoch)
    
    # Log per-class IoU
    for c in range(self.num_classes):
        self.tensorboard.log_scalars(f"IoU_Class_{c}", {
            "train": train_metrics[f'iou_class_{c}'],
            "val": val_metrics[f'iou_class_{c}']
        }, epoch)
    
    # Log learning rate
    self.tensorboard.log_learning_rate(current_lr, epoch)
```

Add to `train()` at the end:

```python
if self.tensorboard:
    self.tensorboard.close()
```



### 4. Update Training Script

**Modify [`train.py`](train.py)**Add argument for TensorBoard:

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--no-tensorboard", action="store_true", 
                   help="Disable TensorBoard logging")
args = parser.parse_args()
```

Pass to Trainer:

```python
trainer = Trainer(
    ...,
    use_tensorboard=not args.no_tensorboard
)
```

Print TensorBoard instructions at the start:

```python
if not args.no_tensorboard:
    print("TensorBoard logging enabled!")
    print(f"  Run: tensorboard --logdir={CHECKPOINT_DIR}/tensorboard")
    print(f"  URL: http://localhost:6006")
    print("=" * 60)
```



### 5. Update Documentation

**Create [`docs/training_monitoring.md`](docs/training_monitoring.md)**Content:

- How to interpret time estimates
- How to launch TensorBoard
- What metrics are tracked
- How to customize TensorBoard logging
- Screenshots/examples of TensorBoard interface
- Tips for monitoring training health:
- Loss should decrease
- mIoU should increase
- Train/val gap indicates overfitting
- LR schedule visualization

**Update [`.gitignore`](.gitignore)**

- Add `**/tensorboard/` to ignore TensorBoard logs

## Files to Modify

1. [`requirements.txt`](requirements.txt) - Add tensorboard
2. [`training/tensorboard_logger.py`](training/tensorboard_logger.py) - New file
3. [`training/trainer.py`](training/trainer.py) - Time tracking + TensorBoard integration
4. [`train.py`](train.py) - Add --no-tensorboard flag
5. [`.gitignore`](.gitignore) - Ignore tensorboard logs
6. [`docs/training_monitoring.md`](docs/training_monitoring.md) - New documentation

## Time Estimate Output Example

```javascript
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
  Total elapsed: 12m 50s
  Estimated completion: 2:05 PM
✓ Saved best model (mIoU: 0.7456)
```



## TensorBoard Benefits

1. **Real-time visualization** - See metrics update live during training
2. **Multiple runs comparison** - Compare different hyperparameters
3. **Smooth curves** - Apply smoothing to noisy metrics
4. **Interactive** - Zoom, pan, download plots
5. **Standard tool** - Industry-standard, well-documented

## Usage

```bash
# Start training (TensorBoard enabled by default)
python train.py

# In another terminal, launch TensorBoard
tensorboard --logdir=checkpoints/tensorboard

# Open browser to http://localhost:6006

# Disable TensorBoard if needed
python train.py --no-tensorboard


```