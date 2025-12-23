# Segmentation Losses

## Why Two Losses?

Semantic segmentation is a pixel-level classification problem. We use two complementary losses to train effectively.

---

## Loss Motivations

Each loss function was designed to answer a different question:

| Loss | Origin | Question It Asks |
|------|--------|------------------|
| **Cross-Entropy** | Information theory (Shannon, 1948) | "How surprised am I by the prediction?" |
| **Dice** | Medical imaging (~2016, based on 1940s coefficient) | "How much do the regions overlap?" |
| **IoU (Jaccard)** | Set theory (Jaccard, 1901) | "What fraction of the combined area is overlap?" |

---

## Cross-Entropy Loss

**Origin:** Information theory - measuring "surprise" or "information content"

**Motivation:** If you're confident about the wrong answer, you should be heavily penalized.

### How It Works

For each pixel, Cross-Entropy measures how far the prediction is from the truth:

"""python
pixel_loss = -log(predicted_probability_of_correct_class)
"""

**Example:**
- True class: cat (class 2)
- Prediction probabilities: [0.1, 0.2, 0.7] for [background, dog, cat]
- Loss = -log(0.7) = 0.36 (low loss, good prediction)

If the prediction was wrong:
- Prediction: [0.7, 0.2, 0.1]
- Loss = -log(0.1) = 2.3 (high loss)

### Why Logarithm?

The log creates an asymmetric penalty:
- Probability 0.9 → Loss = 0.1 (confident and right = small penalty)
- Probability 0.1 → Loss = 2.3 (confident and wrong = big penalty)
- Probability 0.01 → Loss = 4.6 (very wrong = very big penalty)

Being *confidently wrong* is much more expensive than being *uncertain*.

### The Class Imbalance Problem

Consider segmenting a small object in a large image:
- 95% of pixels = background
- 5% of pixels = object

If the model predicts "background everywhere":
- 95% of pixels are correct
- Cross-Entropy averages over all pixels
- The model thinks it's doing great!

This is why we need Dice Loss.

---

## Dice Loss

**Origin:** Medical image segmentation - doctors segmenting tumors faced a problem:
- Tumor = 2% of image
- Healthy tissue = 98%
- A model predicting "all healthy" would be 98% accurate but useless!

**Motivation:** Measure overlap, not per-pixel accuracy.

### The Dice Coefficient

The coefficient (from 1940s) measures similarity between two regions:

"""
Dice = 2 × |Intersection| / (|Region A| + |Region B|)
"""

- Ranges from 0 (no overlap) to 1 (perfect overlap)
- The "2×" ensures perfect overlap gives 1.0

**Visual example:**

"""
Ground Truth:              Prediction:
┌─────────────────┐       ┌─────────────────┐
│    ████████     │       │      ██████████ │
│    ████████     │       │      ██████████ │
└─────────────────┘       └─────────────────┘

Intersection (overlap):
┌─────────────────┐
│      ██████     │  ← Only this part is in BOTH
│      ██████     │
└─────────────────┘

Dice = 2 × 12 / (24 + 20) = 0.55
Dice Loss = 1 - 0.55 = 0.45
"""

### Why Dice Solves Class Imbalance

Back to our small object example:

**Cross-Entropy view:**
- 95% background correct → 95% of loss is "satisfied"
- Model learns to ignore the small object

**Dice view:**
- Object overlap = 0 (model predicted all background)
- Dice = 0 → Dice Loss = 1.0 (maximum penalty!)

Dice ignores the correctly classified background. It only asks: "Did you find the object?"

---

## Combined Loss

We use both losses together:

"""python
total_loss = cross_entropy_loss + dice_loss
"""

This gives us:
1. **Pixel-level gradients** from Cross-Entropy (stable training)
2. **Region-level optimization** from Dice (handles class imbalance)

### Implementation

"""python
class SegmentationLoss(nn.Module):
    def __init__(self, dice_weight=1.0, ce_weight=1.0):
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        return dice_weight * dice + ce_weight * ce
"""

The weights can be tuned based on the dataset characteristics.

---

## Smooth Factor in Dice Loss

The implementation includes a `smooth` parameter (default 1.0):

"""python
dice = (2 * intersection + smooth) / (union + smooth)
"""

This prevents division by zero when a class has no pixels in either the prediction or ground truth.

---

## IoU (Jaccard) Loss

**Origin:** Paul Jaccard (1901) was comparing plant species between regions!

- Region A species: {oak, pine, birch}
- Region B species: {oak, maple, birch}
- Intersection: {oak, birch} = 2
- Union: {oak, pine, birch, maple} = 4
- Similarity = 2/4 = 0.5

### Formula

"""
IoU = Intersection / Union
    = |A ∩ B| / |A ∪ B|
    = |A ∩ B| / (|A| + |B| - |A ∩ B|)
"""

### IoU vs Dice Comparison

"""
Example with 12 pixel overlap:
- Ground truth: 24 pixels
- Prediction: 20 pixels
- Intersection: 12 pixels
- Union: 24 + 20 - 12 = 32 pixels

IoU  = 12 / 32 = 0.375
Dice = 2 × 12 / (24 + 20) = 0.545
"""

They are mathematically related:
"""
Dice = 2 × IoU / (1 + IoU)
IoU  = Dice / (2 - Dice)
"""

### Why We Use Dice for Training, IoU for Evaluation

| Aspect | Dice Loss | IoU Loss |
|--------|-----------|----------|
| Gradient behavior | Smoother | Harsher at low overlap |
| Training stability | More stable | Can be unstable early |
| Industry standard | Training | Evaluation metric (mIoU) |

When overlap is small (early training), IoU's gradient can be very steep. Dice provides more consistent gradients throughout training.

We train with Dice+CE, but report mIoU as the evaluation metric (the standard benchmark).

---

## File Location

Losses are in `training/losses.py` (not `vit/`) because they are **general segmentation losses**, not specific to the Vision Transformer architecture. They work with any segmentation model.

