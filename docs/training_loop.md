# Training Loop

## Overview

The training loop is where all components come together to train the segmentation model. Each epoch consists of a training phase and validation phase.

## Training Flow

"""
For each epoch:
  ├─ Training Phase
  │   ├─ Load batch (images, masks)
  │   ├─ Forward pass → predictions
  │   ├─ Calculate loss (Dice + CrossEntropy)
  │   ├─ Backward pass (compute gradients)
  │   ├─ Optimizer step (update weights)
  │   └─ Track metrics (mIoU, accuracy)
  │
  └─ Validation Phase
      ├─ Load batch (no gradient tracking)
      ├─ Forward pass → predictions
      ├─ Calculate loss
      ├─ Track metrics
      └─ Save checkpoint if best mIoU
"""

---

## Key Components

### 1. Optimizer: AdamW

We use AdamW (Adam with weight decay) instead of regular Adam:

"""python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
"""

**Why AdamW?**
- Decouples weight decay from gradient updates
- Better for transformers (proven in original ViT paper)
- More stable training than SGD for vision transformers

**Alternatives:**
- SGD + momentum: Good for CNNs, less common for ViTs
- Adam: Similar but couples weight decay with gradients

---

### 2. Learning Rate Scheduler

We use **Cosine Annealing** to gradually reduce learning rate:

"""
LR
 │   
 │╲
 │ ╲
 │  ╲___
 │      ╲___
 └──────────────> Epochs
  Start      End
"""

"""python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs
)
"""

**How it works:**
- Starts at initial LR (1e-4)
- Smoothly decreases following cosine curve
- Reaches minimum at final epoch
- Helps fine-tune in later epochs

**Alternative: Linear Warmup + Cosine Decay**
- First few epochs: gradually increase LR (warmup)
- Remaining epochs: cosine decay
- Better for training from scratch, but more complex

---

### 3. Loss Function

Combined loss from `training/losses.py`:

"""python
loss = dice_loss + cross_entropy_loss
"""

- **Dice Loss**: Handles class imbalance, focuses on overlap
- **Cross Entropy**: Pixel-wise classification, standard baseline
- Both contribute equally (weight=1.0 each)

---

### 4. Metrics Tracking

From `training/metrics.py`:

"""python
metrics = SegmentationMetrics(num_classes=2)
metrics.update(preds, targets)

# At end of epoch:
pixel_acc = metrics.get_pixel_accuracy()
miou = metrics.get_miou()
iou_per_class = metrics.get_iou_per_class()
"""

We track:
- **Pixel Accuracy**: Quick sanity check
- **mIoU**: Primary metric for model quality
- **IoU per class**: See which class (background/pet) is harder

---

### 5. Checkpointing

Save model when validation mIoU improves:

"""python
if val_miou > best_miou:
    best_miou = val_miou
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_miou': best_miou,
    }, 'checkpoints/best_model.pth')
"""

**What we save:**
- Model weights
- Optimizer state (for resuming training)
- Best mIoU achieved
- Current epoch number

---

### 6. Early Stopping

Stop training if validation doesn't improve for N epochs:

"""
Validation mIoU per epoch:
Epoch 1: 0.45
Epoch 2: 0.52 ← improvement
Epoch 3: 0.58 ← improvement
Epoch 4: 0.59 ← improvement
Epoch 5: 0.58 ← no improvement (patience = 1)
Epoch 6: 0.57 ← no improvement (patience = 2)
Epoch 7: 0.58 ← no improvement (patience = 3)
Epoch 8: STOP (patience limit reached)
"""

**Why?**
- Prevents overfitting
- Saves training time
- Model likely won't improve further

**Patience parameter:**
- Too low (e.g., 3): Might stop too early
- Too high (e.g., 20): Wastes time
- Sweet spot: 5-10 epochs

---

## MPS (Apple Silicon GPU) Support

### Device Selection

"""python
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
"""

### Moving Data to Device

"""python
# Model
model = model.to(device)

# Data (in training loop)
images = images.to(device)
masks = masks.to(device)
"""

### MPS-Specific Considerations

1. **Gradient Accumulation**: Works normally on MPS
2. **Mixed Precision**: Not yet supported on MPS (use full precision)
3. **Memory**: MPS shares system RAM, so larger batches possible

---

## Training Pseudocode

"""python
for epoch in range(num_epochs):
    # Training
    model.train()
    train_metrics.reset()
    
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        # Forward
        logits = model(images)
        loss = criterion(logits, masks)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        train_metrics.update(preds, masks)
    
    # Validation
    model.eval()
    val_metrics.reset()
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            
            logits = model(images)
            loss = criterion(logits, masks)
            
            preds = torch.argmax(logits, dim=1)
            val_metrics.update(preds, masks)
    
    # Logging
    print(f"Epoch {epoch}")
    print(f"  Train mIoU: {train_metrics.get_miou():.4f}")
    print(f"  Val mIoU: {val_metrics.get_miou():.4f}")
    
    # Checkpoint
    if val_metrics.get_miou() > best_miou:
        save_checkpoint(...)
    
    # Scheduler
    scheduler.step()
"""

---

## Typical Training Times

For Oxford Pets dataset (3680 train images, 224x224):

- **MPS (M1/M2)**: ~3-5 min/epoch (batch_size=16)
- **CUDA (RTX 3090)**: ~1-2 min/epoch (batch_size=32)
- **CPU**: ~15-20 min/epoch (batch_size=8)

**Total training:** 20-50 epochs to convergence

---

## Expected Results

After training from scratch on Oxford Pets:

"""
Epoch 10:
  - Train mIoU: ~0.65
  - Val mIoU: ~0.60

Epoch 30:
  - Train mIoU: ~0.78
  - Val mIoU: ~0.72

Epoch 50 (converged):
  - Train mIoU: ~0.82
  - Val mIoU: ~0.75-0.78
"""

**Note:** These are approximate. With pretrained weights (from ImageNet), we'd get 0.80-0.85 mIoU much faster.

