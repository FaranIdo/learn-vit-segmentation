# Learning Rate Schedulers

## Why Schedule the Learning Rate?

A fixed learning rate throughout training is suboptimal:

- **Early training**: Large LR helps escape bad local minima and learn quickly
- **Late training**: Small LR fine-tunes weights without overshooting the optimal solution

**Analogy**: Finding a parking spot
- Start: Drive fast to get close to the spot
- End: Drive slowly to park precisely

---

## Common Schedulers

### 1. Cosine Annealing (Our Current Choice)

"""
Learning Rate over epochs:

1.0e-4 │╲
       │ ╲___
       │     ╲____
       │          ╲____
1.0e-6 │               ╲___
       └────────────────────> Epochs
       1                    50
"""

**Implementation:**
"""python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs  # Total number of epochs
)
"""

**How it works:**
- Follows a cosine curve from initial LR to near-zero
- Smooth, continuous decrease
- Formula: `lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * epoch / T_max))`

**Example progression (50 epochs, starting at 1e-4):**
"""
Epoch 1:  1.00e-04
Epoch 10: 9.02e-05  (↓10%)
Epoch 20: 6.55e-05  (↓35%)
Epoch 30: 3.45e-05  (↓66%)
Epoch 40: 9.80e-06  (↓90%)
Epoch 50: 1.23e-07  (↓99%)
"""

**Pros:**
- ✓ Smooth transitions (no sudden jumps)
- ✓ Standard for transformer/ViT training
- ✓ Works well for training from scratch
- ✓ Well-studied in papers (ViT, BERT, GPT)

**Cons:**
- ✗ Changes every epoch (harder to debug)
- ✗ Less intuitive than step schedules

---

### 2. Step LR

"""
Learning Rate over epochs:

1.0e-4 │────────────╮
       │            │
5.0e-5 │            ╰──────────╮
       │                       │
2.5e-5 │                       ╰──────────
       └────────────────────────────────> Epochs
       1          10           20        30
"""

**Implementation:**
"""python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,  # Reduce LR every 10 epochs
    gamma=0.5      # Multiply LR by 0.5
)
"""

**Example progression:**
"""
Epochs 1-10:  LR = 1.00e-04
Epochs 11-20: LR = 5.00e-05  (50% of previous)
Epochs 21-30: LR = 2.50e-05  (50% of previous)
Epochs 31-40: LR = 1.25e-05  (50% of previous)
"""

**Pros:**
- ✓ Very stable (constant for many epochs)
- ✓ Easy to understand and debug
- ✓ Traditional choice for CNNs
- ✓ Clear when drops happen

**Cons:**
- ✗ Sudden drops can disrupt training momentarily
- ✗ Need to manually tune `step_size`
- ✗ Less commonly used for transformers

---

### 3. MultiStep LR

"""
Learning Rate over epochs:

1.0e-4 │────────────────╮
       │                │
1.0e-5 │                ╰──────────────────╮
       │                                   │
1.0e-6 │                                   ╰────────
       └────────────────────────────────────────────> Epochs
       1              15                  35        50
"""

**Implementation:**
"""python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[15, 35],  # Specific epochs to reduce LR
    gamma=0.1             # Multiply LR by 0.1 at each milestone
)
"""

**Example progression:**
"""
Epochs 1-14:  LR = 1.00e-04
Epochs 15-34: LR = 1.00e-05  (10% of initial)
Epochs 35-50: LR = 1.00e-06  (10% of previous)
"""

**Pros:**
- ✓ Full control over when to reduce LR
- ✓ Can align with observed training plateaus
- ✓ Very stable between milestones

**Cons:**
- ✗ Requires knowledge of when training plateaus
- ✗ Not adaptive to actual training progress
- ✗ Large sudden drops

---

### 4. ReduceLROnPlateau (Adaptive)

"""
Learning Rate (adapts based on validation metric):

1.0e-4 │─────────╮        ╭─────╮
       │         │        │     │
5.0e-5 │         ╰────────╯     ╰──────
       │
       └────────────────────────────────> Epochs
              ↑ Val loss    ↑ Val loss
                stops         stops
                improving     improving
"""

**Implementation:**
"""python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # 'max' for mIoU, 'min' for loss
    factor=0.5,        # Multiply LR by 0.5
    patience=5,        # Wait 5 epochs before reducing
    verbose=True
)

# In training loop:
scheduler.step(val_miou)  # Pass validation metric
"""

**How it works:**
- Monitors validation metric (e.g., mIoU or loss)
- If metric doesn't improve for `patience` epochs, reduce LR
- Adaptive to actual training progress

**Pros:**
- ✓ Adaptive: responds to actual training dynamics
- ✓ No need to guess when to reduce LR
- ✓ Can help escape plateaus

**Cons:**
- ✗ Less predictable schedule
- ✗ Requires validation metric
- ✗ Can be overly conservative

---

### 5. Linear Warmup + Cosine Decay

"""
Learning Rate over epochs:

1.0e-4 │    ╱╲
       │   ╱  ╲___
       │  ╱      ╲____
       │ ╱            ╲____
1.0e-6 │╱                  ╲___
       └────────────────────────> Epochs
       1  5              50
       Warmup  Cosine Decay
"""

**Implementation:**
"""python
from torch.optim.lr_scheduler import LambdaLR
import math

def warmup_cosine_schedule(epoch, warmup_epochs=5, total_epochs=50):
    if epoch < warmup_epochs:
        # Linear warmup
        return epoch / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
"""

**Why warmup?**
- Large initial LR can destabilize training from random initialization
- Warmup gradually increases LR over first few epochs
- Standard in BERT, GPT, ViT training

**Pros:**
- ✓ Best of both worlds: stable start + smooth decay
- ✓ State-of-the-art for large models
- ✓ Used in most transformer papers

**Cons:**
- ✗ More complex to implement
- ✗ Two hyperparameters (warmup epochs + total epochs)

---

## Which Should You Use?

### For Vision Transformers (ViT)
**Recommended: Cosine Annealing** or **Warmup + Cosine**
- Standard in ViT papers
- Smooth training dynamics
- Works well from random initialization

### For Tutorials / Debugging
**Recommended: StepLR**
- Easy to understand
- Stable for long periods
- Clear when changes happen

### For Production / Research
**Recommended: Warmup + Cosine**
- State-of-the-art results
- Used in published papers
- Can compare with literature

---

## Our Choice: Cosine Annealing

We use cosine annealing because:
1. **Standard for ViTs**: Matches original ViT paper
2. **Simple**: No warmup complexity for tutorial
3. **Smooth**: No disruptive jumps
4. **Works well**: Proven effective for training from scratch

**Trade-off**: Less intuitive than step decay, but better results.

---

## Future Improvements

See `docs/future_work.md` for:
- Adding warmup phase
- Trying ReduceLROnPlateau for adaptive scheduling
- Experimenting with different schedules for your data

