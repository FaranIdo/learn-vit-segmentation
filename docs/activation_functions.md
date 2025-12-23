# Activation Functions in Transformers

## Why GELU?

GELU (Gaussian Error Linear Unit) is the standard activation function for Transformers, used in BERT, GPT, and ViT.

## Comparison

| Activation | Formula | Behavior | Used In |
|------------|---------|----------|---------|
| ReLU | max(0, x) | Hard cutoff at 0 | CNNs, older models |
| LeakyReLU | max(0.01x, x) | Small slope for negatives | GANs |
| GELU | x · Φ(x) | Smooth, probabilistic | Transformers |
| SiLU/Swish | x · σ(x) | Similar to GELU | EfficientNet, some LLMs |

Φ(x) = cumulative distribution function of standard normal
σ(x) = sigmoid function

## Key Differences

### ReLU: Hard Cutoff

"""python
ReLU(-0.1) = 0      # Completely killed
ReLU(0.1)  = 0.1
"""

Problems:
- Abrupt transition at x=0
- "Dead neurons": negative inputs get zero gradient
- Information loss for negative values

### GELU: Smooth Transition

"""python
GELU(-0.1) ≈ -0.046  # Small negative allowed
GELU(0.1)  ≈ 0.054
"""

Benefits:
- Smooth gradient flow
- Small negative values still contribute
- No dead neurons

## The GELU Intuition

GELU asks: "What's the probability this input is important?"

- Large positive → probably important → pass through
- Large negative → probably not important → suppress  
- Near zero → uncertain → partial pass-through

This probabilistic decision-making aligns well with attention, which also makes soft decisions about relevance.

## Why Transformers Prefer GELU

1. **Smoother gradients** - No hard cutoff means stable training
2. **Preserves small negatives** - More information flows through
3. **Empirical results** - Consistent ~0.5-1% improvement over ReLU
4. **Self-regularizing** - Probabilistic nature provides mild regularization

## SiLU/Swish Alternative

SiLU (Sigmoid Linear Unit), also called Swish:

"""python
SiLU(x) = x * sigmoid(x)
"""

Very similar to GELU in practice. Some newer models (LLaMA 2) use SiLU. The difference is minimal - both are smooth and allow small negatives.

## Computational Cost

GELU is slightly more expensive than ReLU:
- ReLU: 1 comparison
- GELU: requires computing the Gaussian CDF (or approximation)

In practice, the extra computation is negligible compared to attention and matrix multiplications. The performance gain is worth it.

